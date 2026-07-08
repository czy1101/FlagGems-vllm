# -*- coding: utf-8 -*-
"""Benchmark: MiniMax M3 Sparse Attention Triton kernels.

Runs both the lightning-indexer + block-sparse-attention pipeline and measures
performance on realistic MiniMax M3 workloads (prefill and decode).

Usage:
    # Quick test (small sizes)
    python benchmark.py --mode quick

    # Full benchmark
    python benchmark.py --mode full

    # Profiling (adds torch.profiler)
    python benchmark.py --mode full --profile
"""

import argparse
import time

import torch

from kernels import (
    minimax_m3_index_decode,
    minimax_m3_index_score,
    minimax_m3_index_topk,
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)


# ---------------------------------------------------------------------------
# Config: MiniMax M3 typical hyperparameters
# ---------------------------------------------------------------------------
# These match the M3 architecture documented at:
#   MiniMaxAI/MiniMax-M3 on HuggingFace
# We use smaller numbers to make benchmarks fast for quick mode.
DEFAULT_CONFIGS = {
    "small": {
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "index_head_dim": 128,
        "topk_blocks": 4,
        "init_blocks": 0,
        "local_blocks": 1,
        "dtype": torch.bfloat16,
    },
    "large": {
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "index_head_dim": 128,
        "topk_blocks": 8,
        "init_blocks": 1,
        "local_blocks": 3,
        "dtype": torch.bfloat16,
    },
}


def get_device():
    """Return cuda device with properties."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"Device: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"Total Memory: {props.total_memory / 1024**3:.1f} GiB")
    return device


def warmup(fn_iters=5):
    """Warmup GPU."""
    for _ in range(fn_iters):
        _ = torch.empty(1, device="cuda")


def time_ms(start_event, end_event):
    """Return elapsed time in ms."""
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event)


# ---------------------------------------------------------------------------
# Prefill benchmark
# ---------------------------------------------------------------------------
def bench_prefill(
    device: torch.device,
    cfg: dict,
    batch_size: int = 1,
    seq_len: int = 4096,
    num_iters: int = 100,
    profile: bool = False,
):
    """Benchmark the full prefill pipeline: index_score -> index_topk -> sparse_attn."""
    dtype = cfg["dtype"]
    num_heads = cfg["num_heads"]
    num_kv_heads = cfg["num_kv_heads"]
    head_dim = cfg["head_dim"]
    index_head_dim = cfg["index_head_dim"]
    topk = cfg["topk_blocks"]
    init_blocks = cfg["init_blocks"]
    local_blocks = cfg["local_blocks"]
    block_size = 128

    total_q = batch_size * seq_len
    max_blocks = (seq_len + block_size - 1) // block_size

    sm_scale = head_dim**-0.5

    # ---- Build input tensors ----
    q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)
    idx_q = torch.randn(
        total_q, num_kv_heads, index_head_dim, dtype=dtype, device=device
    )

    # Paged KV cache for main attention
    kv_cache = torch.randn(
        max_blocks, 2, block_size, num_kv_heads, head_dim,
        dtype=dtype, device=device,
    )
    # Index-K cache (side cache)
    ik_cache = torch.randn(
        max_blocks, block_size, index_head_dim, dtype=dtype, device=device,
    )

    # Block table: consecutive mapping for simplicity
    block_table = torch.arange(max_blocks, dtype=torch.int32, device=device)
    block_table = block_table.unsqueeze(0).expand(batch_size, -1).contiguous()

    # cu_seqlens / seq_lens / prefix_lens
    cu_seqlens = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * seq_len
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    prefix_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)

    max_query_len = seq_len
    max_seq_len = seq_len

    output = torch.empty_like(q)

    # ---- Pipeline ----
    # Step 1: index_score
    idx_score = minimax_m3_index_score(
        idx_q, ik_cache, block_table, cu_seqlens,
        seq_lens, prefix_lens, max_query_len, max_seq_len, num_kv_heads,
    )

    # Step 2: index_topk
    topk_idx = minimax_m3_index_topk(
        idx_score, cu_seqlens, prefix_lens, max_query_len,
        topk, init_blocks, local_blocks,
    )

    # Warmup
    warmup(10)
    for _ in range(5):
        minimax_m3_sparse_attn(
            q, kv_cache, topk_idx, block_table,
            cu_seqlens, seq_lens, prefix_lens,
            max_query_len, num_kv_heads, sm_scale, output,
        )

    torch.cuda.synchronize()

    print(f"\n{'='*70}")
    print(f"PREFILL Benchmark")
    print(f"{'='*70}")
    print(f"batch={batch_size}  seq_len={seq_len}  num_heads={num_heads}")
    print(f"num_kv_heads={num_kv_heads}  head_dim={head_dim}  topk={topk}")
    print(f"KV blocks: {max_blocks}  total_q: {total_q}")
    print(f"dtype: {dtype}")

    # Benchmark sparse attention
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    if profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            start.record()
            for i in range(num_iters):
                minimax_m3_sparse_attn(
                    q, kv_cache, topk_idx, block_table,
                    cu_seqlens, seq_lens, prefix_lens,
                    max_query_len, num_kv_heads, sm_scale, output,
                )
            end.record()
        torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    else:
        start.record()
        for i in range(num_iters):
            minimax_m3_sparse_attn(
                q, kv_cache, topk_idx, block_table,
                cu_seqlens, seq_lens, prefix_lens,
                max_query_len, num_kv_heads, sm_scale, output,
            )
        end.record()

    elapsed_ms = time_ms(start, end)
    avg_ms = elapsed_ms / num_iters
    print(f"\nSparseAttn: {elapsed_ms:.2f}ms total ({num_iters} iters) -> {avg_ms:.3f}ms/iter")

    # Also measure indexer time separately
    start.record()
    for i in range(num_iters):
        _ = minimax_m3_index_score(
            idx_q, ik_cache, block_table, cu_seqlens,
            seq_lens, prefix_lens, max_query_len, max_seq_len, num_kv_heads,
        )
    end.record()
    score_ms = time_ms(start, end) / num_iters
    print(f"IndexScore: {score_ms:.3f}ms/iter")

    start.record()
    for i in range(num_iters):
        _ = minimax_m3_index_topk(
            idx_score, cu_seqlens, prefix_lens, max_query_len,
            topk, init_blocks, local_blocks,
        )
    end.record()
    topk_ms = time_ms(start, end) / num_iters
    print(f"IndexTopK:  {topk_ms:.3f}ms/iter")
    print(f"Total pipeline: {score_ms + topk_ms + avg_ms:.3f}ms/iter")

    # GFLOPs estimate (rough: only compute on topk selected blocks)
    gqa_group = num_heads // num_kv_heads
    flops_per_token = 2 * gqa_group * head_dim * block_size * topk * 2  # QK + PV
    total_gflops = total_q * flops_per_token / 1e9
    elapsed_s = avg_ms / 1000
    tflops = total_gflops / 1000 / elapsed_s if elapsed_s > 0 else 0
    print(f"Estimated compute: {total_gflops:.1f} GFLOPs -> {tflops:.1f} TFLOPs/s")


# ---------------------------------------------------------------------------
# Decode benchmark
# ---------------------------------------------------------------------------
def bench_decode(
    device: torch.device,
    cfg: dict,
    batch_size: int = 32,
    seq_len: int = 4096,
    num_iters: int = 200,
    decode_query_len: int = 1,
    profile: bool = False,
):
    """Benchmark decode: index_decode + sparse_attn_decode (both split-K)."""
    dtype = cfg["dtype"]
    num_heads = cfg["num_heads"]
    num_kv_heads = cfg["num_kv_heads"]
    head_dim = cfg["head_dim"]
    index_head_dim = cfg["index_head_dim"]
    topk = cfg["topk_blocks"]
    init_blocks = cfg["init_blocks"]
    local_blocks = cfg["local_blocks"]
    block_size = 128

    sm_scale = head_dim**-0.5
    total_q = batch_size * decode_query_len
    max_blocks = (seq_len + block_size - 1) // block_size

    # ---- Build input tensors ----
    q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)
    idx_q = torch.randn(
        total_q, num_kv_heads, index_head_dim, dtype=dtype, device=device,
    )
    kv_cache = torch.randn(
        max_blocks, 2, block_size, num_kv_heads, head_dim,
        dtype=dtype, device=device,
    )
    ik_cache = torch.randn(
        max_blocks, block_size, index_head_dim, dtype=dtype, device=device,
    )
    block_table = torch.arange(max_blocks, dtype=torch.int32, device=device)
    block_table = block_table.unsqueeze(0).expand(batch_size, -1).contiguous()
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    max_seq_len = seq_len
    max_decode_query_len = decode_query_len

    output = torch.empty_like(q)

    # Step 1: index_decode
    topk_idx = minimax_m3_index_decode(
        idx_q, ik_cache, block_table, seq_lens,
        max_seq_len, topk, init_blocks, local_blocks,
        num_kv_heads, decode_query_len, max_decode_query_len,
    )

    # Warmup
    warmup(10)
    for _ in range(5):
        minimax_m3_sparse_attn_decode(
            q, kv_cache, topk_idx, block_table, seq_lens,
            num_kv_heads, sm_scale, output, decode_query_len,
        )
    torch.cuda.synchronize()

    print(f"\n{'='*70}")
    print(f"DECODE Benchmark")
    print(f"{'='*70}")
    print(f"batch={batch_size}  seq_len={seq_len}  decode_qlen={decode_query_len}")
    print(f"num_heads={num_heads}  num_kv_heads={num_kv_heads}  topk={topk}")
    print(f"dtype: {dtype}")

    # Benchmark sparse attention decode
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    if profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            start.record()
            for i in range(num_iters):
                minimax_m3_sparse_attn_decode(
                    q, kv_cache, topk_idx, block_table, seq_lens,
                    num_kv_heads, sm_scale, output, decode_query_len,
                )
            end.record()
        torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    else:
        start.record()
        for i in range(num_iters):
            minimax_m3_sparse_attn_decode(
                q, kv_cache, topk_idx, block_table, seq_lens,
                num_kv_heads, sm_scale, output, decode_query_len,
            )
        end.record()

    elapsed_ms = time_ms(start, end)
    avg_ms = elapsed_ms / num_iters
    print(f"\nSparseAttnDecode: {elapsed_ms:.2f}ms total ({num_iters} iters) -> {avg_ms:.3f}ms/iter")

    # Index decode time
    start.record()
    for i in range(num_iters):
        _ = minimax_m3_index_decode(
            idx_q, ik_cache, block_table, seq_lens,
            max_seq_len, topk, init_blocks, local_blocks,
            num_kv_heads, decode_query_len, max_decode_query_len,
        )
    end.record()
    idx_ms = time_ms(start, end) / num_iters
    print(f"IndexDecode: {idx_ms:.3f}ms/iter")
    print(f"Attention throughput: {batch_size / (avg_ms / 1000):.0f} tokens/s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MiniMax M3 Sparse Attention Triton Benchmark"
    )
    parser.add_argument(
        "--mode", type=str, default="quick",
        choices=["quick", "full"],
        help="Benchmark mode: quick (small configs) or full (realistic configs)",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Enable torch.profiler CUDA trace",
    )
    args = parser.parse_args()

    device = get_device()
    cfg = DEFAULT_CONFIGS["small"] if args.mode == "quick" else DEFAULT_CONFIGS["large"]

    # Prefill benchmarks (varying seq_len)
    for seq_len in [1024, 2048, 4096] if args.mode == "full" else [1024]:
        bench_prefill(
            device, cfg,
            batch_size=1,
            seq_len=seq_len,
            num_iters=50 if args.mode == "full" else 5,
            profile=args.profile,
        )

    # Decode benchmarks (varying batch)
    for batch_sz in [1, 8, 32, 64] if args.mode == "full" else [8]:
        try:
            bench_decode(
                device, cfg,
                batch_size=batch_sz,
                seq_len=4096,
                num_iters=100 if args.mode == "full" else 5,
                decode_query_len=1,
                profile=args.profile,
            )
        except RuntimeError as e:
            print(f"Decode b={batch_sz} failed (OOM?): {e}")

    print(f"\n{'='*70}")
    print("Benchmark complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
