#!/usr/bin/env python3
"""Benchmark: FlagGems MSA (paged) vs vLLM MSA.

Usage:
  python -m benchmark.test_MSA.bench_vs_vllm --all-shapes --per-step --no-vllm
  python -m benchmark.test_MSA.bench_vs_vllm --all-shapes --vllm-path /path/to/vllm-main
  python -m benchmark.test_MSA.bench_vs_vllm --decode --all-shapes --vllm-path /path/to/vllm-main
"""
import os
import sys
import argparse
import torch
import triton.knobs
triton.knobs.autotuning.adjust_block_size = False

from ._bench_common import (
    make_data, bench_fn, run_triton_prefill, run_triton_decode,
    bench_triton_prefill_steps, bench_triton_decode_steps,
    fmt_shape, parse_common_args, get_shapes,
)


def main():
    p = argparse.ArgumentParser(description="FlagGems vs vLLM MSA benchmark")
    p = parse_common_args(p)
    p.add_argument("--vllm-path", type=str, default=None,
                   help="Path to vLLM source root (default: sibling ../vllm-main)")
    p.add_argument("--no-vllm", action="store_true", help="Skip vLLM baseline")
    args = p.parse_args()
    if args.vllm_path:
        os.environ["VLLM_HOME"] = args.vllm_path
    shapes = get_shapes(args)
    run_vllm = not args.no_vllm
    if run_vllm:
        from .vllm_adapter import clear_vllm_cache, vllm_prefill, vllm_decode

    mode_str = f"Decode (qlen={args.decode_qlen})" if args.decode else "Prefill"
    impl_str = "FlagGems vs vLLM MSA" if run_vllm else "FlagGems only"
    print("=" * 90)
    print(f"  MiniMax M3 Sparse Attention -- {mode_str}")
    print(f"  TopK={args.topk}, Init={args.init_blocks}, Local={args.local_blocks}")
    print(f"  {impl_str}")
    print(f"  Warmup={args.warmup}, Rep={args.rep}")
    if args.per_step:
        print("  Per-step: ENABLED (FlagGems only)")
    print("=" * 90)

    cols = f"{'Shape':>22s}  {'triton(ms)':>10s}"
    if run_vllm:
        cols += f"  {'vllm(ms)':>10s}  {'speedup':>8s}"
    if args.per_step:
        if args.decode:
            cols += f"  {'IdxDec':>8s}  {'AttnDec':>8s}"
        else:
            cols += f"  {'Score':>8s}  {'TopK':>8s}  {'Attn':>8s}  {'Attn%':>6s}"
    sep = "-" * len(cols)
    print(sep); print(cols); print(sep); sys.stdout.flush()

    for shape in shapes:
        batch, seq_len, n_kv_h, n_h = shape
        device = torch.device("cuda")
        dtype = torch.bfloat16

        if args.decode:
            data = make_data(batch, seq_len, n_kv_h, n_h, device, dtype, decode=True, decode_qlen=args.decode_qlen)
            q, idx_q, kv_cache, index_kv_cache, block_table, cu_q, sl, pl, sm_scale = data
            out = torch.empty_like(q)
            def run_triton():
                run_triton_decode(q, idx_q, kv_cache, index_kv_cache, block_table, cu_q, sl,
                                 seq_len, n_kv_h, args.topk, args.init_blocks, args.local_blocks, sm_scale, out, args.decode_qlen)
            lat_triton = bench_fn(run_triton, args.warmup, args.rep)
            steps = {}
            if args.per_step:
                steps = bench_triton_decode_steps(q, idx_q, kv_cache, index_kv_cache, block_table, cu_q, sl,
                    seq_len, n_kv_h, args.topk, args.init_blocks, args.local_blocks, sm_scale, out, args.decode_qlen, args.warmup, args.rep)
            lat_vllm = None
            if run_vllm:
                def run_vllm_fn():
                    vllm_decode(q, idx_q, kv_cache, index_kv_cache, block_table, cu_q, sl,
                               seq_len, n_kv_h, args.topk, args.init_blocks, args.local_blocks, sm_scale, args.decode_qlen)
                lat_vllm = bench_fn(run_vllm_fn, args.warmup, args.rep)
        else:
            data = make_data(batch, seq_len, n_kv_h, n_h, device, dtype)
            q, idx_q, kv_cache, index_kv_cache, block_table, cu_q, sl, pl, sm_scale = data
            out = torch.empty_like(q)
            def run_triton():
                run_triton_prefill(q, idx_q, kv_cache, index_kv_cache, block_table, cu_q, sl, pl,
                                  seq_len, n_kv_h, args.topk, args.init_blocks, args.local_blocks, sm_scale, out)
            lat_triton = bench_fn(run_triton, args.warmup, args.rep)
            steps = {}
            if args.per_step:
                steps = bench_triton_prefill_steps(q, idx_q, kv_cache, index_kv_cache, block_table, cu_q, sl, pl,
                    seq_len, n_kv_h, args.topk, args.init_blocks, args.local_blocks, sm_scale, out, args.warmup, args.rep)
            lat_vllm = None
            if run_vllm:
                def run_vllm_fn():
                    vllm_prefill(q, idx_q, kv_cache, index_kv_cache, block_table, cu_q, sl, pl,
                                seq_len, n_kv_h, args.topk, args.init_blocks, args.local_blocks, sm_scale)
                lat_vllm = bench_fn(run_vllm_fn, args.warmup, args.rep)

        speedup = lat_vllm / lat_triton if (lat_triton > 0 and lat_vllm is not None) else 0
        line = f"{fmt_shape(shape):>22s}  {lat_triton:10.4f}"
        if run_vllm:
            line += f"  {lat_vllm:10.4f}  {speedup:7.2f}x"
        if args.per_step:
            if args.decode:
                line += f"  {steps.get('decode_idx', 0):8.4f}  {steps.get('decode_attn', 0):8.4f}"
            else:
                ss = steps.get("index_score", 0); st = steps.get("index_topk", 0); sa = steps.get("sparse_attn", 0)
                pct = sa / (ss + st + sa) * 100 if (ss + st + sa) > 0 else 0
                line += f"  {ss:8.4f}  {st:8.4f}  {sa:8.4f}  {pct:5.0f}%"
        print(line); sys.stdout.flush()
        if run_vllm:
            clear_vllm_cache()
    print(sep); print()


if __name__ == "__main__":
    main()
