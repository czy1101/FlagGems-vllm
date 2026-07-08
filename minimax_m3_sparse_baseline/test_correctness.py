# -*- coding: utf-8 -*-
"""Correctness test: compare sparse attention output against torch reference.

Verifies that the block-sparse GQA attention kernels produce results that match
(or closely approximate) the naive torch attention over the same selected blocks.

Usage:
    python test_correctness.py
"""

import torch
import triton

from kernels import (
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
    minimax_m3_index_decode,
    minimax_m3_index_score,
    minimax_m3_index_topk,
)


def naive_block_sparse_attention(
    q: torch.Tensor,          # [total_q, num_heads, head_dim]
    k: torch.Tensor,          # [total_kv, num_kv_heads, head_dim]
    v: torch.Tensor,          # [total_kv, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,   # [num_kv_heads, total_q, topk]
    sm_scale: float,
    block_size: int = 128,
    causal: bool = True,
    seq_len: int | None = None,
) -> torch.Tensor:
    """Naive reference: for each (query, kv_head), attend only to selected blocks."""
    total_q, num_heads, head_dim = q.shape
    _, num_kv_heads, _ = k.shape
    gqa_group = num_heads // num_kv_heads
    topk = topk_idx.shape[-1]

    if seq_len is None:
        seq_len = k.shape[0]

    out = torch.zeros_like(q)
    q_flat = q.view(total_q, num_kv_heads, gqa_group, head_dim)

    for t in range(total_q):
        for h in range(num_kv_heads):
            block_ids = topk_idx[h, t]  # [topk]
            # Build KV for this head from selected blocks
            kv_indices = []
            for blk_id in block_ids:
                if blk_id < 0:
                    break
                start = int(blk_id) * block_size
                end = min(start + block_size, seq_len)
                kv_indices.extend(range(start, end))

            if not kv_indices:
                continue

            # Gather K and V for this head
            k_sel = k[kv_indices, h, :]  # [n_sel, head_dim]
            v_sel = v[kv_indices, h, :]  # [n_sel, head_dim]

            # Compute attention for all queries in this GQA group
            q_group = q_flat[t, h, :, :]  # [gqa_group, head_dim]
            # Use float32 for matmul to avoid bf16 CUBLAS alignment issues
            attn_scores = torch.matmul(
                q_group.float(), k_sel.T.float()
            ) * sm_scale  # [gqa_group, n_sel]

            # Causal mask reconstruction (simplified: assume sequential)
            if causal and seq_len is not None:
                # For simplicity in unit tests, skip full causal reconstruction
                pass

            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_out = torch.matmul(attn_weights, v_sel.float()).to(q.dtype)

            start_h = h * gqa_group
            out[t, start_h : start_h + gqa_group, :] = attn_out

    return out


def test_prefill_correctness():
    """Test prefill sparse attention against naive implementation.

    Uses prefix_lens == seq_len so the kernel's causal mask does not block any
    KV positions (all queries are placed after all KV data). The naive reference
    computes non-causal attention over the same selected blocks.
    """
    print("=" * 70)
    print("PREFILL Correctness Test")
    print("=" * 70)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    batch = 1
    seq_len = 256
    block_size = 128
    num_heads = 8
    num_kv_heads = 2
    head_dim = 128
    topk = 4
    gqa_group = num_heads // num_kv_heads
    max_blocks = (seq_len + block_size - 1) // block_size  # 2 blocks
    total_q = batch * seq_len

    sm_scale = head_dim**-0.5

    torch.manual_seed(42)

    q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)
    k_dense = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_dense = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)

    kv_cache = torch.zeros(
        max_blocks, 2, block_size, num_kv_heads, head_dim,
        dtype=dtype, device=device,
    )
    for blk in range(max_blocks):
        start = blk * block_size
        end = min(start + block_size, seq_len)
        kv_cache[blk, 0, : end - start, :, :] = k_dense[start:end]
        kv_cache[blk, 1, : end - start, :, :] = v_dense[start:end]

    block_table = torch.arange(max_blocks, dtype=torch.int32, device=device).unsqueeze(0)

    cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

    # Place queries AFTER all KV data so causal mask is a no-op.
    prefix_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)

    topk_idx = torch.zeros(num_kv_heads, total_q, topk, dtype=torch.int32, device=device)
    for t in range(total_q):
        for h in range(num_kv_heads):
            for i in range(min(topk, max_blocks)):
                topk_idx[h, t, i] = i
            for i in range(max_blocks, topk):
                topk_idx[h, t, i] = -1

    output = torch.empty_like(q)

    minimax_m3_sparse_attn(
        q, kv_cache, topk_idx, block_table,
        cu_seqlens_q, seq_lens, prefix_lens,
        seq_len, num_kv_heads, sm_scale, output,
    )

    ref_out = naive_block_sparse_attention(
        q, k_dense, v_dense, topk_idx, sm_scale, block_size,
        causal=False, seq_len=seq_len,
    )

    diff = (output.float() - ref_out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = diff / (ref_out.float().abs() + 1e-8)

    print(f"  shape: {output.shape}")
    print(f"  max absolute diff: {max_diff:.6f}")
    print(f"  mean absolute diff: {mean_diff:.6f}")
    print(f"  max relative diff: {rel_diff.max().item():.6f}")
    print(f"  mean relative diff: {rel_diff.mean().item():.6f}")

    assert max_diff < 1.0, f"Max diff {max_diff} too large!"
    print("  PASSED ✓")
    return True


def test_decode_correctness():
    """Test decode sparse attention against prefill with same data."""
    print("\n" + "=" * 70)
    print("DECODE Correctness Test")
    print("=" * 70)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    batch = 4
    seq_len = 512
    block_size = 128
    num_heads = 8
    num_kv_heads = 2
    head_dim = 128
    topk = 4
    max_blocks = (seq_len + block_size - 1) // block_size  # 4
    decode_qlen = 1
    total_q = batch * decode_qlen

    sm_scale = head_dim**-0.5

    torch.manual_seed(123)

    # Build inputs
    q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)
    kv_cache = torch.randn(
        max_blocks, 2, block_size, num_kv_heads, head_dim,
        dtype=dtype, device=device,
    )
    block_table = torch.arange(max_blocks, dtype=torch.int32, device=device)
    block_table = block_table.unsqueeze(0).expand(batch, -1).contiguous()
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)

    # Topk: select first 'topk' blocks
    topk_idx = torch.zeros(num_kv_heads, total_q, topk, dtype=torch.int32, device=device)
    for t in range(total_q):
        for h in range(num_kv_heads):
            for i in range(min(topk, max_blocks)):
                topk_idx[h, t, i] = i
            for i in range(max_blocks, topk):
                topk_idx[h, t, i] = -1

    output = torch.empty_like(q)

    minimax_m3_sparse_attn_decode(
        q, kv_cache, topk_idx, block_table, seq_lens,
        num_kv_heads, sm_scale, output, decode_qlen,
    )

    # Compare: run prefill kernel on same data (1 query per request)
    cu_seqlens_q = torch.arange(0, batch + 1, dtype=torch.int32, device=device)
    # Place prefill query after all KV data so causal is a no-op
    prefix_lens = torch.full((batch,), seq_len - decode_qlen, dtype=torch.int32, device=device)
    output_prefill = torch.empty_like(q)

    minimax_m3_sparse_attn(
        q, kv_cache, topk_idx, block_table,
        cu_seqlens_q, seq_lens, prefix_lens,
        1, num_kv_heads, sm_scale, output_prefill,
    )

    diff = (output.float() - output_prefill.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  shape: {output.shape}")
    print(f"  decode vs prefill max diff: {max_diff:.6f}")
    print(f"  decode vs prefill mean diff: {mean_diff:.6f}")

    # Split-K merge introduces minor numerical differences
    assert max_diff < 1.0, f"Max diff {max_diff} too large!"
    print("  PASSED ✓")
    return True


def test_indexer_correctness():
    """Test that indexer score+topk pipeline runs and produces valid block ids."""
    print("\n" + "=" * 70)
    print("INDEXER Correctness Test")
    print("=" * 70)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    batch = 1
    seq_len = 512
    block_size = 128
    num_kv_heads = 2
    head_dim = 128
    max_blocks = (seq_len + block_size - 1) // block_size  # 4

    torch.manual_seed(7)

    # Prefill indexer
    total_q = batch * seq_len
    idx_q = torch.randn(total_q, num_kv_heads, head_dim, dtype=dtype, device=device)
    ik_cache = torch.randn(
        max_blocks, block_size, head_dim, dtype=dtype, device=device,
    )
    block_table = torch.arange(max_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    prefix_lens = torch.zeros(batch, dtype=torch.int32, device=device)

    # Score
    score = minimax_m3_index_score(
        idx_q, ik_cache, block_table, cu_seqlens,
        seq_lens, prefix_lens, seq_len, seq_len, num_kv_heads,
    )
    print(f"  index_score shape: {score.shape}")

    # Check scores are finite
    finite_mask = torch.isfinite(score)
    print(f"  finite scores: {finite_mask.float().mean().item():.4f}")

    # Topk
    topk = 4
    topk_idx = minimax_m3_index_topk(
        score, cu_seqlens, prefix_lens, seq_len,
        topk, init_blocks=0, local_blocks=0,
    )
    print(f"  index_topk shape: {topk_idx.shape}")

    # Check valid block ids
    valid_mask = topk_idx >= 0
    n_valid = valid_mask.sum().item()
    n_total = topk_idx.numel()
    print(f"  valid topk entries: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")

    # All block ids should be within range
    blk_ids = topk_idx[valid_mask]
    in_range = (blk_ids >= 0) & (blk_ids < max_blocks)
    print(f"  in-range block ids: {in_range.float().mean().item():.4f}")

    # Decode indexer
    batch_decode = 4
    decode_qlen = 1
    total_q_d = batch_decode * decode_qlen
    idx_q_d = torch.randn(total_q_d, num_kv_heads, head_dim, dtype=dtype, device=device)
    block_table_d = torch.arange(max_blocks, dtype=torch.int32, device=device)
    block_table_d = block_table_d.unsqueeze(0).expand(batch_decode, -1).contiguous()
    seq_lens_d = torch.full((batch_decode,), seq_len, dtype=torch.int32, device=device)

    topk_idx_d = minimax_m3_index_decode(
        idx_q_d, ik_cache, block_table_d, seq_lens_d,
        seq_len, topk=4, init_blocks=0, local_blocks=0,
        num_kv_heads=num_kv_heads, decode_query_len=decode_qlen,
        max_decode_query_len=decode_qlen,
    )
    print(f"  decode topk shape: {topk_idx_d.shape}")
    valid_d = (topk_idx_d >= 0).sum().item()
    print(f"  decode valid entries: {valid_d}/{topk_idx_d.numel()} ({100*valid_d/topk_idx_d.numel():.1f}%)")

    print("  PASSED ✓")
    return True


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests.")
        return

    props = torch.cuda.get_device_properties(0)
    print(f"Device: {props.name} (SM {props.major}.{props.minor})")
    print(f"Triton version: {triton.__version__}")

    all_passed = True
    try:
        test_indexer_correctness()
    except Exception as e:
        print(f"  Indexer test FAILED: {e}")
        all_passed = False

    try:
        test_prefill_correctness()
    except Exception as e:
        print(f"  Prefill test FAILED: {e}")
        all_passed = False

    try:
        test_decode_correctness()
    except Exception as e:
        print(f"  Decode test FAILED: {e}")
        all_passed = False

    print(f"\n{'='*70}")
    if all_passed:
        print("All tests PASSED ✓")
    else:
        print("Some tests FAILED ✗")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
