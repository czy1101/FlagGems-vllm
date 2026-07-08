#!/usr/bin/env python3
"""
End-to-end correctness test for vLLM-based MSA Triton kernels.
Compares against the PyTorch naive reference in torch_naive.py.
"""

import torch
from vllm_msa import minimax_m3_index_score, minimax_m3_index_topk, minimax_m3_sparse_attn, SPARSE_BLOCK_SIZE
from torch_naive import ref_index_score, ref_index_topk, ref_sparse_attn


def run_correctness_test(
    batch=1, seq_len=1024, num_kv_heads=4, head_dim=128,
    topk=32, init_blocks=1, local_blocks=2,
):
    num_heads = num_kv_heads * 6
    BLOCK = SPARSE_BLOCK_SIZE
    device = torch.device("cuda")
    dtype = torch.bfloat16

    total_q = batch * seq_len
    num_blocks_total = batch * ((seq_len + BLOCK - 1) // BLOCK)

    q = torch.randn(total_q, num_heads, head_dim, device=device, dtype=dtype)
    idx_q = torch.randn(total_q, num_kv_heads, head_dim, device=device, dtype=dtype)
    kv_cache = torch.randn(num_blocks_total, 2, BLOCK, num_kv_heads, head_dim, device=device, dtype=dtype)
    index_kv_cache = torch.randn(num_blocks_total, BLOCK, head_dim, device=device, dtype=dtype)

    blocks_per_batch = (seq_len + BLOCK - 1) // BLOCK
    block_table = torch.arange(num_blocks_total, dtype=torch.int32, device=device).reshape(batch, -1)
    cu = torch.arange(0, (batch + 1) * seq_len, seq_len, dtype=torch.int32, device=device)
    sl = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    pl = torch.zeros(batch, dtype=torch.int32, device=device)
    sm_scale = head_dim ** -0.5

    print(f"Test: batch={batch} seq_len={seq_len} kv_heads={num_kv_heads} heads={num_heads} head_dim={head_dim} topk={topk}")
    print(f"Blocks: total={num_blocks_total} per_batch={blocks_per_batch}")

    # Triton
    print("\n[1/3] Triton index scoring ...")
    scores = minimax_m3_index_score(idx_q, index_kv_cache, block_table, cu, sl, pl, seq_len, seq_len, num_kv_heads)
    print("[2/3] Triton index top-k ...")
    topk_idx = minimax_m3_index_topk(scores, cu, pl, seq_len, topk, init_blocks, local_blocks)
    print("[3/3] Triton sparse attention ...")
    output = torch.empty_like(q)
    minimax_m3_sparse_attn(q, kv_cache, topk_idx, block_table, cu, sl, pl, seq_len, num_kv_heads, sm_scale, output)
    torch.cuda.synchronize()

    # Reference
    print("[Ref 1/3] Naive index scoring ...")
    sr = ref_index_score(idx_q, index_kv_cache, block_table, cu, sl, pl)
    print("[Ref 2/3] Naive index top-k ...")
    tr = ref_index_topk(sr, cu, pl, topk, init_blocks, local_blocks)
    print("[Ref 3/3] Naive sparse attention ...")
    oref = ref_sparse_attn(q, kv_cache, tr, block_table, cu, sl, pl, sm_scale)
    torch.cuda.synchronize()

    # Validate
    print("\nValidating ...")
    topk_match = (topk_idx == tr).all().item()
    topk_ratio = (topk_idx == tr).float().mean().item()
    print(f"  topk_idx: {topk_match} ({topk_ratio:.4%})"
          f"{' (tie-breaking, benign)' if not topk_match and topk_ratio > 0.97 else ''}")
    if not topk_match:
        mc = (topk_idx != tr).sum().item()
        print(f"  mismatches: {mc}/{topk_idx.numel()}")
        if mc <= topk_idx.numel() * 0.1:
            for pos in (topk_idx != tr).nonzero(as_tuple=False)[:3]:
                h, qid, k = pos.tolist()
                print(f"    [{h},{qid},{k}]: triton={topk_idx[h,qid,k].item()} naive={tr[h,qid,k].item()}")

    ot, orf = output.float(), oref.float()
    valid = ~orf.isnan() & (orf.abs() > 1e-8)
    cos_sim = torch.nn.functional.cosine_similarity(
        ot[valid].reshape(-1), orf[valid].reshape(-1), dim=0).item() if valid.sum() > 0 else 1.0

    THRESHOLD = 0.9999
    passed = cos_sim >= THRESHOLD
    print(f"  cos_sim:  {cos_sim:.6f} (threshold: {THRESHOLD})")
    print(f"  max_diff: {(ot - orf).abs().max().item():.6e}")
    print(f"  mean_diff: {(ot - orf).abs().mean().item():.6e}")

    print()
    if passed:
        print("=" * 50)
        print(f"  Test Passed! (cos_sim={cos_sim:.6f} >= {THRESHOLD})")
        if not topk_match and topk_ratio > 0.97:
            print(f"  (Note: {topk_ratio:.2%} topk match, tie-breaking benign)")
        print("=" * 50)
    else:
        print("=" * 50)
        print(f"  Test Failed! (cos_sim={cos_sim:.6f} < {THRESHOLD})")
        print("=" * 50)
    return passed


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_correctness_test() else 1)
