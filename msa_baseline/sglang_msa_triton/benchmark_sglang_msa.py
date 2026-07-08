#!/usr/bin/env python3
"""Standalone benchmark for sglang MSA triton operators — decode pipeline.

Usage:
  python benchmark_sglang_msa.py
  python benchmark_sglang_msa.py --per-step   # per-step breakdown
  python benchmark_sglang_msa.py --warmup 20 --rep 200
"""

import csv
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import triton.testing as triton_testing

from sglang_msa import flash_decode_with_topk_idx, flash_decode_with_gqa_share_sparse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BLOCK_SIZE = 128
HEAD_DIM = 128
SHAPES = [
    (1, 1024, 1, 8),
    (2, 2048, 1, 8),
    (4, 4096, 1, 8),
    (8, 8192, 1, 8),
    (1, 32768, 1, 8),
    (1, 65536, 1, 8),
    (32, 1024, 1, 8),
    (128, 1024, 1, 8),
]
TOPK, INIT_BLOCKS, LOCAL_BLOCKS = 32, 1, 2
DEFAULT_WARMUP, DEFAULT_REP = 10, 100
CSV_FILE = os.path.join(os.path.dirname(__file__), "sglang_msa_bench.csv")


def parse_cli():
    per_step = "--per-step" in sys.argv
    warmup, rep = DEFAULT_WARMUP, DEFAULT_REP
    for i, arg in enumerate(sys.argv):
        if arg == "--warmup" and i + 1 < len(sys.argv):
            warmup = int(sys.argv[i + 1])
        if arg == "--rep" and i + 1 < len(sys.argv):
            rep = int(sys.argv[i + 1])
    return per_step, warmup, rep


def build_decode_inputs(batch, seq_len, n_kv_h, n_h, d=HEAD_DIM, bs=BLOCK_SIZE):
    device = torch.device("cuda")
    max_kv_len = seq_len
    max_slots = batch * max_kv_len
    req_to_token = torch.zeros(batch, max_kv_len, dtype=torch.int32, device=device)
    slot_ids = torch.zeros(batch, dtype=torch.int64, device=device)
    for i in range(batch):
        base = i * max_kv_len
        slot_ids[i] = i
        req_to_token[i] = torch.arange(base, base + max_kv_len, device=device)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    q = torch.randn(batch, n_h, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(max_slots, n_kv_h, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(max_slots, n_kv_h, d, dtype=torch.bfloat16, device=device)
    iq = torch.randn(batch, n_kv_h, d, dtype=torch.bfloat16, device=device)
    ik = torch.randn(max_slots, 1, d, dtype=torch.bfloat16, device=device)
    iv = torch.randn(max_slots, 1, d, dtype=torch.bfloat16, device=device)
    return q, k, v, iq, ik, iv, req_to_token, slot_ids, seq_lens, seq_len, bs


def benchmark_shape(batch, seq_len, n_kv_h, n_h, per_step, warmup, rep):
    q, k, v, iq, ik, iv, r2t, sid, sl, max_seqlen, bs = build_decode_inputs(
        batch, seq_len, n_kv_h, n_h
    )
    n_cache_elements = batch * seq_len * n_kv_h * HEAD_DIM * 4
    cache_mb = n_cache_elements * 2 / (1024 * 1024)

    def step1():
        return flash_decode_with_topk_idx(
            iq, None, ik, iv, r2t, sl, max_seqlen, sid,
            bs, TOPK, INIT_BLOCKS, LOCAL_BLOCKS,
            score_type="max",
        )

    step1()
    torch.cuda.synchronize()
    _, topk_idx, _ = step1()
    torch.cuda.synchronize()

    def step2():
        return flash_decode_with_gqa_share_sparse(
            q, None, k, v, r2t, sl, sid, bs, topk_idx,
        )

    def full():
        _, ti, _ = flash_decode_with_topk_idx(
            iq, None, ik, iv, r2t, sl, max_seqlen, sid,
            bs, TOPK, INIT_BLOCKS, LOCAL_BLOCKS,
            score_type="max",
        )
        return flash_decode_with_gqa_share_sparse(
            q, None, k, v, r2t, sl, sid, bs, ti,
        )

    for _ in range(warmup):
        full()
    torch.cuda.synchronize()
    latency = triton_testing.do_bench(full, warmup=0, rep=rep)

    lats = []
    for _ in range(rep):
        torch.cuda.synchronize()
        s_e = torch.cuda.Event(enable_timing=True)
        e_e = torch.cuda.Event(enable_timing=True)
        s_e.record()
        full()
        e_e.record()
        torch.cuda.synchronize()
        lats.append(s_e.elapsed_time(e_e))
    lats.sort()
    p50 = lats[len(lats) // 2]

    step_times = {}
    if per_step:
        for _ in range(warmup):
            step1(); step2()
        torch.cuda.synchronize()
        step_times["index"] = round(triton_testing.do_bench(step1, warmup=0, rep=rep), 4)
        step_times["sparse"] = round(triton_testing.do_bench(step2, warmup=0, rep=rep), 4)

    return {
        "batch": batch, "seq_len": seq_len, "num_kv_heads": n_kv_h, "num_q_heads": n_h,
        "head_dim": HEAD_DIM, "cache_mb": round(cache_mb, 2),
        "topk": TOPK, "init_blocks": INIT_BLOCKS, "local_blocks": LOCAL_BLOCKS,
        "latency_ms": round(latency, 4), "p50_ms": round(p50, 4),
        "timestamp": datetime.now().isoformat(),
        **({"step_idx_ms": step_times.get("index"),
            "step_attn_ms": step_times.get("sparse")} if step_times else {}),
    }


def main():
    per_step, warmup, rep = parse_cli()
    print("=" * 90)
    print("  sglang MSA Triton Decode Benchmark")
    print(f"  TopK={TOPK}, Init={INIT_BLOCKS}, Local={LOCAL_BLOCKS}")
    print(f"  Warmup={warmup}, Rep={rep}")
    print("=" * 90)

    results = []
    cols = f"{'B':>3s}  {'SeqLen':>7s}  {'KVH':>4s}  {'QH':>4s}  {'CacheMB':>8s}  {'Total':>9s}  {'P50':>8s}"
    if per_step:
        cols += f"  {'Idx':>8s}  {'Attn':>8s}"
    print(cols)
    print("-" * len(cols))

    for batch, seq_len, n_kv_h, n_h in SHAPES:
        r = benchmark_shape(batch, seq_len, n_kv_h, n_h, per_step, warmup, rep)
        results.append(r)
        line = (f"{r['batch']:3d}  {r['seq_len']:7d}  {r['num_kv_heads']:4d}  "
                f"{r['num_q_heads']:4d}  {r['cache_mb']:8.2f}  {r['latency_ms']:9.4f}  "
                f"{r['p50_ms']:8.4f}")
        if per_step:
            line += f"  {r.get('step_idx_ms', 0):8.4f}  {r.get('step_attn_ms', 0):8.4f}"
        print(line)
    print()

    if results:
        fns = list(results[0].keys())
        existed = os.path.exists(CSV_FILE)
        with open(CSV_FILE, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fns)
            if not existed:
                w.writeheader()
            w.writerows(results)
        print(f"Results: {CSV_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()
