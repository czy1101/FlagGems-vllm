#!/usr/bin/env python3
"""Correctness test for FlagGems MSA (paged KV) vs PyTorch reference."""
import warnings
warnings.filterwarnings("ignore", message="tl.make_block_ptr is deprecated")

import sys
from pathlib import Path

import torch
import triton.knobs
triton.knobs.autotuning.adjust_block_size = False

# Allow running this migrated test directly from a source checkout without
# requiring an editable installation of FlagGems-vllm.
_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from flaggems_vllm.ops.MSA import (
    minimax_m3_index_score,
    minimax_m3_index_topk,
    minimax_m3_index_decode,
    SPARSE_BLOCK_SIZE,
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)
from .ref_torch import (
    ref_index_score,
    ref_index_topk,
    ref_sparse_attn,
    ref_index_decode,
    ref_sparse_attn_decode,
)
from benchmark.test_MSA._bench_common import make_data, BLOCK

COS_SIM_THRESHOLD = 0.9999
TOPK_SET_MATCH_THRESHOLD = 1.0


def _cos_sim(a, b):
    a, b = a.float(), b.float()
    valid = ~b.isnan() & (b.abs() > 1e-8)
    if valid.sum() == 0:
        return 1.0
    return torch.nn.functional.cosine_similarity(
        a[valid].reshape(-1), b[valid].reshape(-1), dim=0
    ).item()


def _topk_set_match(triton_idx, ref_idx):
    triton_sorted = triton_idx.sort(dim=-1).values
    ref_sorted = ref_idx.sort(dim=-1).values
    return (triton_sorted == ref_sorted).float().mean().item()


def _topk_pos_match(triton_idx, ref_idx):
    return (triton_idx == ref_idx).float().mean().item()


def test_prefill(batch=2, seq_len=1024, num_kv_heads=4, head_dim=128,
                 topk=16, init_blocks=1, local_blocks=2, seed=42):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_heads = num_kv_heads * 16

    q, idx_q, kv_cache, index_kv_cache, block_table, cu_q, sl, pl, sm_scale = \
        make_data(batch, seq_len, num_kv_heads, num_heads, device, dtype,
                  head_dim=head_dim, randomize_pages=True)

    print(f"\n[Test] prefill b={batch} seq={seq_len} kv_h={num_kv_heads} "
          f"q_h={num_heads} d={head_dim} topk={topk}")

    scores = minimax_m3_index_score(idx_q, index_kv_cache, block_table, cu_q,
                                    sl, pl, seq_len, seq_len, num_kv_heads)
    topk_idx = minimax_m3_index_topk(scores, cu_q, pl, seq_len, topk,
                                     init_blocks, local_blocks)
    output = torch.empty_like(q)
    minimax_m3_sparse_attn(q, kv_cache, topk_idx, block_table, cu_q, sl, pl,
                           seq_len, num_kv_heads, sm_scale, output)
    torch.cuda.synchronize()

    sr = ref_index_score(idx_q, index_kv_cache, block_table, cu_q, sl, pl, seq_len)
    tr = ref_index_topk(sr, cu_q, pl, topk, init_blocks, local_blocks)
    oref = ref_sparse_attn(q, kv_cache, tr, block_table, cu_q, sl, pl,
                            sm_scale, seq_len)

    topk_set = _topk_set_match(topk_idx, tr)
    topk_pos = _topk_pos_match(topk_idx, tr)
    cs = _cos_sim(output, oref)
    max_diff = (output.float() - oref.float()).abs().max().item()

    topk_ok = topk_set >= TOPK_SET_MATCH_THRESHOLD
    attn_ok = cs >= COS_SIM_THRESHOLD

    print(f"  topk set:    {topk_set:.4%}  {'OK' if topk_ok else 'FAIL'}")
    print(f"  topk pos:    {topk_pos:.4%}  (info: tie-breaking reorder)")
    print(f"  cos_sim:     {cs:.6f}  {'OK' if attn_ok else 'FAIL'}")
    print(f"  max_diff:    {max_diff:.6e}")

    if topk_ok and attn_ok:
        print("  PASSED")
        return True
    else:
        print("  FAILED")
        return False


def test_decode(batch=4, seq_len=2048, num_kv_heads=4, head_dim=128,
                topk=32, init_blocks=1, local_blocks=2, decode_qlen=1, seed=42):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_heads = num_kv_heads * 16

    q, idx_q, kv_cache, index_kv_cache, block_table, cu_q, sl, pl, sm_scale = \
        make_data(batch, seq_len, num_kv_heads, num_heads, device, dtype,
                  decode=True, decode_qlen=decode_qlen, head_dim=head_dim,
                  randomize_pages=True)

    print(f"\n[Test] decode b={batch} seq={seq_len} kv_h={num_kv_heads} "
          f"q_h={num_heads} d={head_dim} topk={topk} qlen={decode_qlen}")

    topk_idx = minimax_m3_index_decode(
        idx_q, index_kv_cache, block_table, sl, seq_len, topk,
        init_blocks, local_blocks, num_kv_heads, decode_qlen, decode_qlen,
    )
    output = torch.empty_like(q)
    minimax_m3_sparse_attn_decode(q, kv_cache, topk_idx, block_table, sl,
                                  num_kv_heads, sm_scale, output, decode_qlen)
    torch.cuda.synchronize()

    tr = ref_index_decode(idx_q, index_kv_cache, block_table, sl, seq_len,
                          topk, init_blocks, local_blocks, decode_qlen)
    oref = ref_sparse_attn_decode(q, kv_cache, tr, block_table, sl,
                                  sm_scale, decode_qlen)

    topk_set = _topk_set_match(topk_idx, tr)
    topk_pos = _topk_pos_match(topk_idx, tr)
    cs = _cos_sim(output, oref)
    max_diff = (output.float() - oref.float()).abs().max().item()

    topk_ok = topk_set >= TOPK_SET_MATCH_THRESHOLD
    attn_ok = cs >= COS_SIM_THRESHOLD

    print(f"  topk set:    {topk_set:.4%}  {'OK' if topk_ok else 'FAIL'}")
    print(f"  topk pos:    {topk_pos:.4%}  (info: tie-breaking reorder)")
    print(f"  cos_sim:     {cs:.6f}  {'OK' if attn_ok else 'FAIL'}")
    print(f"  max_diff:    {max_diff:.6e}")

    if topk_ok and attn_ok:
        print("  PASSED")
        return True
    else:
        print("  FAILED")
        return False


# Keep the original configurable test functions for the standalone runner while
# exposing the same fixed case matrix through one pytest-compatible entry point.
test_prefill.__test__ = False
test_decode.__test__ = False


def test_msa_correctness():
    results = [
        test_prefill(batch=2, seq_len=512, num_kv_heads=4, topk=16),
        test_prefill(batch=2, seq_len=1024, num_kv_heads=4, topk=16),
        test_prefill(batch=1, seq_len=4096, num_kv_heads=4, topk=32),
        test_prefill(batch=4, seq_len=512, num_kv_heads=8, topk=16),
        test_prefill(batch=2, seq_len=2048, num_kv_heads=4, topk=48),
        test_decode(batch=1, seq_len=512, num_kv_heads=4, topk=16),
        test_decode(batch=4, seq_len=2048, num_kv_heads=4, topk=32),
        test_decode(batch=1, seq_len=8192, num_kv_heads=4, topk=48),
        test_decode(batch=8, seq_len=1024, num_kv_heads=8, topk=32),
        test_decode(
            batch=4,
            seq_len=2048,
            num_kv_heads=4,
            topk=32,
            decode_qlen=1,
        ),
    ]
    assert all(results)


def main():
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping.")
        sys.exit(0)

    results = []
    print("=" * 70)
    print("  Prefill Correctness Tests")
    print("=" * 70)
    results.append(("prefill_small", test_prefill(batch=2, seq_len=512, num_kv_heads=4, topk=16)))
    results.append(("prefill_med",   test_prefill(batch=2, seq_len=1024, num_kv_heads=4, topk=16)))
    results.append(("prefill_long",  test_prefill(batch=1, seq_len=4096, num_kv_heads=4, topk=32)))
    results.append(("prefill_batch", test_prefill(batch=4, seq_len=512, num_kv_heads=8, topk=16)))
    results.append(("prefill_big",   test_prefill(batch=2, seq_len=2048, num_kv_heads=4, topk=48)))

    print("\n" + "=" * 70)
    print("  Decode Correctness Tests")
    print("=" * 70)
    results.append(("decode_small", test_decode(batch=1, seq_len=512, num_kv_heads=4, topk=16)))
    results.append(("decode_med",   test_decode(batch=4, seq_len=2048, num_kv_heads=4, topk=32)))
    results.append(("decode_long",  test_decode(batch=1, seq_len=8192, num_kv_heads=4, topk=48)))
    results.append(("decode_batch", test_decode(batch=8, seq_len=1024, num_kv_heads=8, topk=32)))
    results.append(("decode_spec",  test_decode(batch=4, seq_len=2048, num_kv_heads=4, topk=32, decode_qlen=1)))

    print("\n" + "=" * 70)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        print(f"  {name:20s}  {'PASS' if ok else 'FAIL'}")
    print(f"\n  {passed}/{total} tests passed")
    print("=" * 70)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
