#!/usr/bin/env python3
"""One-table benchmark for FlagGems, vLLM, and sglang MSA."""
import argparse
import os
import sys

import torch
import triton.knobs
triton.knobs.autotuning.adjust_block_size = False

from ._bench_common import (
    bench_fn,
    fmt_shape,
    get_shapes,
    make_data,
    parse_common_args,
    run_triton_decode,
    run_triton_prefill,
)


def main():
    parser = parse_common_args(
        argparse.ArgumentParser(description=__doc__)
    )
    parser.add_argument("--vllm-path", default=None)
    parser.add_argument("--no-vllm", action="store_true")
    parser.add_argument("--no-sglang", action="store_true")
    args = parser.parse_args()

    if args.vllm_path:
        os.environ["VLLM_HOME"] = args.vllm_path
    if args.per_step:
        parser.error("--per-step is available in bench_vs_vllm.py and bench_vs_sglang.py")
    if args.decode and args.decode_qlen != 1 and not args.no_sglang:
        parser.error("sglang decode supports --decode-qlen 1 only; pass --no-sglang otherwise")

    if not args.no_vllm:
        from .vllm_adapter import clear_vllm_cache, vllm_decode, vllm_prefill
    if not args.no_sglang:
        from .sglang_adapter import clear_sglang_cache, sglang_decode, sglang_prefill

    columns = f"{'Shape':>22s}  {'ours(ms)':>10s}"
    if not args.no_vllm:
        columns += f"  {'vLLM(ms)':>10s}  {'vs vLLM':>9s}"
    if not args.no_sglang:
        columns += f"  {'sglang(ms)':>10s}  {'vs sglang':>9s}"
    separator = "-" * len(columns)
    mode = f"decode qlen={args.decode_qlen}" if args.decode else "prefill"
    print(f"MiniMax M3 paged sparse attention ({mode})")
    print(separator)
    print(columns)
    print(separator)

    for shape in get_shapes(args):
        batch, seq_len, num_kv_heads, num_heads = shape
        data = make_data(
            batch, seq_len, num_kv_heads, num_heads,
            torch.device("cuda"), torch.bfloat16,
            decode=args.decode, decode_qlen=args.decode_qlen,
        )
        q, idx_q, kv, idx_kv, block_table, cu_q, sl, pl, scale = data
        output = torch.empty_like(q)

        if args.decode:
            def ours():
                run_triton_decode(
                    q, idx_q, kv, idx_kv, block_table, cu_q, sl, seq_len,
                    num_kv_heads, args.topk, args.init_blocks,
                    args.local_blocks, scale, output, args.decode_qlen,
                )

            if not args.no_vllm:
                def baseline_vllm():
                    vllm_decode(
                        q, idx_q, kv, idx_kv, block_table, cu_q, sl, seq_len,
                        num_kv_heads, args.topk, args.init_blocks,
                        args.local_blocks, scale, args.decode_qlen, output=output,
                    )
            if not args.no_sglang:
                def baseline_sglang():
                    sglang_decode(
                        q, idx_q, kv, idx_kv, block_table, cu_q, sl, seq_len,
                        num_kv_heads, args.topk, args.init_blocks,
                        args.local_blocks, scale, args.decode_qlen,
                    )
        else:
            def ours():
                run_triton_prefill(
                    q, idx_q, kv, idx_kv, block_table, cu_q, sl, pl, seq_len,
                    num_kv_heads, args.topk, args.init_blocks,
                    args.local_blocks, scale, output,
                )

            if not args.no_vllm:
                def baseline_vllm():
                    vllm_prefill(
                        q, idx_q, kv, idx_kv, block_table, cu_q, sl, pl,
                        seq_len, num_kv_heads, args.topk, args.init_blocks,
                        args.local_blocks, scale, output=output,
                    )
            if not args.no_sglang:
                def baseline_sglang():
                    sglang_prefill(
                        q, idx_q, kv, idx_kv, block_table, cu_q, sl, pl,
                        seq_len, num_kv_heads, args.topk, args.init_blocks,
                        args.local_blocks, scale,
                    )

        ours_ms = bench_fn(ours, args.warmup, args.rep)
        line = f"{fmt_shape(shape):>22s}  {ours_ms:10.4f}"
        if not args.no_vllm:
            vllm_ms = bench_fn(baseline_vllm, args.warmup, args.rep)
            line += f"  {vllm_ms:10.4f}  {vllm_ms / ours_ms:8.2f}x"
        if not args.no_sglang:
            sglang_ms = bench_fn(baseline_sglang, args.warmup, args.rep)
            line += f"  {sglang_ms:10.4f}  {sglang_ms / ours_ms:8.2f}x"
        print(line)
        sys.stdout.flush()
        if not args.no_vllm:
            clear_vllm_cache()
        if not args.no_sglang:
            clear_sglang_cache()

    print(separator)


if __name__ == "__main__":
    main()
