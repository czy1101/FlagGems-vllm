#!/usr/bin/env python3
"""
End-to-end benchmark for vLLM-based MSA pipeline.
Directly uses vllm_msa implementation.
"""

import os
import sys
import csv
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

import torch
import triton
import triton.testing as triton_testing

from vllm_msa import (
    minimax_m3_index_score,
    minimax_m3_index_topk,
    minimax_m3_sparse_attn,
    minimax_m3_index_decode,
    minimax_m3_sparse_attn_decode,
    SPARSE_BLOCK_SIZE,
)
from torch_naive import ref_index_score, ref_index_topk, ref_sparse_attn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BLOCK_SIZE = SPARSE_BLOCK_SIZE
HEAD_DIM = 128
SHAPES = [
    # (1, 8192, 16, 96),
    # (2, 16384, 16, 96),
    # (4, 2048, 16, 96),
    # (4, 4096, 64, 384),
    # (8, 2048, 32, 192),
    # (2, 2048, 16, 96),
    # (4, 1024, 8, 48),
    (8, 1024, 8, 48),
]
KV_DTYPES = [torch.bfloat16, torch.float8_e4m3fn]
TOPK, INIT_BLOCKS, LOCAL_BLOCKS = 32, 1, 2
DEFAULT_WARMUP, DEFAULT_REP = 10, 50
CSV_FILE = os.path.join(_PROJECT_ROOT, "msa_e2e_benchmark_results.csv")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_cli():
    with_ref = "--with-ref" in sys.argv
    per_step = "--per-step" in sys.argv
    decode_mode = "--decode" in sys.argv
    decode_qlen, warmup, rep = 1, DEFAULT_WARMUP, DEFAULT_REP
    for i, arg in enumerate(sys.argv):
        if arg == "--warmup" and i + 1 < len(sys.argv): warmup = int(sys.argv[i + 1])
        if arg == "--rep" and i + 1 < len(sys.argv): rep = int(sys.argv[i + 1])
        if arg == "--decode-qlen" and i + 1 < len(sys.argv): decode_qlen = int(sys.argv[i + 1])
    return with_ref, per_step, warmup, rep, decode_mode, decode_qlen


# ---------------------------------------------------------------------------
# Prefill pipeline wrappers
# ---------------------------------------------------------------------------
def make_step_wrappers(idx_q, idx_kv, q, kv, bt, cu_q, sl, pl, mql, msl, nkv, nh, sc, tk, ib, lb):
    out = torch.empty_like(q)
    _s, _t = None, None

    def s1():
        nonlocal _s
        _s = minimax_m3_index_score(idx_q, idx_kv, bt, cu_q, sl, pl, mql, msl, nkv)
        return _s
    def s2():
        nonlocal _t
        _t = minimax_m3_index_topk(_s, cu_q, pl, mql, tk, ib, lb)
        return _t
    def s3():
        minimax_m3_sparse_attn(q, kv, _t, bt, cu_q, sl, pl, mql, nkv, sc, out)
        return out
    return s1, s2, s3

def make_full_pipeline(idx_q, idx_kv, q, kv, bt, cu_q, sl, pl, mql, msl, nkv, nh, sc, tk, ib, lb):
    out = torch.empty_like(q)
    def _p():
        s = minimax_m3_index_score(idx_q, idx_kv, bt, cu_q, sl, pl, mql, msl, nkv)
        t = minimax_m3_index_topk(s, cu_q, pl, mql, tk, ib, lb)
        minimax_m3_sparse_attn(q, kv, t, bt, cu_q, sl, pl, mql, nkv, sc, out)
        return out
    return _p


# ---------------------------------------------------------------------------
# Decode pipeline wrappers
# ---------------------------------------------------------------------------
def make_decode_step_wrappers(idx_q, idx_kv, q, kv, bt, sl, nkv, nh, sc, tk, ib, lb, dql, mdql, msl):
    out = torch.empty_like(q)
    _t = None
    def s1():
        nonlocal _t
        _t = minimax_m3_index_decode(idx_q, idx_kv, bt, sl, msl, tk, ib, lb, nkv, dql, mdql)
        return _t
    def s2():
        minimax_m3_sparse_attn_decode(q, kv, _t, bt, sl, nkv, sc, out, dql)
        return out
    return s1, s2

def make_decode_pipeline(idx_q, idx_kv, q, kv, bt, sl, nkv, nh, sc, tk, ib, lb, dql, mdql, msl):
    out = torch.empty_like(q)
    def _p():
        t = minimax_m3_index_decode(idx_q, idx_kv, bt, sl, msl, tk, ib, lb, nkv, dql, mdql)
        minimax_m3_sparse_attn_decode(q, kv, t, bt, sl, nkv, sc, out, dql)
        return out
    return _p


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def benchmark_shape(batch, seq_len, n_kv_h, n_h, kv_dtype,
                    with_ref=False, per_step=False, warmup=DEFAULT_WARMUP, rep=DEFAULT_REP,
                    decode_mode=False, decode_qlen=1):
    device, q_dtype = torch.device("cuda"), torch.bfloat16
    nb = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    tblocks = batch * nb
    tq = batch * decode_qlen if decode_mode else batch * seq_len

    q = torch.randn(tq, n_h, HEAD_DIM, device=device, dtype=q_dtype)
    idx_q = torch.randn(tq, n_kv_h, HEAD_DIM, device=device, dtype=q_dtype)
    kv = torch.randn(tblocks, 2, BLOCK_SIZE, n_kv_h, HEAD_DIM, device=device, dtype=torch.bfloat16).to(kv_dtype)
    idx_kv = torch.randn(tblocks, BLOCK_SIZE, HEAD_DIM, device=device, dtype=torch.bfloat16)
    bt = torch.arange(tblocks, dtype=torch.int32, device=device).reshape(batch, nb)
    cu_q = torch.arange(0, (batch + 1) * seq_len, seq_len, dtype=torch.int32, device=device)
    sl = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    sc = HEAD_DIM ** -0.5

    kv_b = tblocks * 2 * BLOCK_SIZE * n_kv_h * HEAD_DIM * kv.element_size()
    ix_b = tblocks * BLOCK_SIZE * HEAD_DIM * idx_kv.element_size()
    cache_mb = (kv_b + ix_b) / (1024 ** 2)

    if decode_mode:
        pl = None
        full_pipeline = make_decode_pipeline(
            idx_q, idx_kv, q, kv, bt, sl, n_kv_h, n_h, sc, TOPK, INIT_BLOCKS, LOCAL_BLOCKS,
            decode_qlen, decode_qlen, seq_len)
    else:
        pl = torch.zeros(batch, dtype=torch.int32, device=device)
        full_pipeline = make_full_pipeline(
            idx_q, idx_kv, q, kv, bt, cu_q, sl, pl, seq_len, seq_len, n_kv_h, n_h,
            sc, TOPK, INIT_BLOCKS, LOCAL_BLOCKS)

    for _ in range(warmup): full_pipeline()
    torch.cuda.synchronize()
    latency_ms = triton_testing.do_bench(full_pipeline, warmup=0, rep=rep)

    lats = []
    for _ in range(rep):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); full_pipeline(); e.record()
        torch.cuda.synchronize(); lats.append(s.elapsed_time(e))
    lats.sort(); p50 = lats[len(lats) // 2]

    step_times = {}
    if per_step:
        if decode_mode:
            s1, s2 = make_decode_step_wrappers(
                idx_q, idx_kv, q, kv, bt, sl, n_kv_h, n_h, sc, TOPK, INIT_BLOCKS, LOCAL_BLOCKS,
                decode_qlen, decode_qlen, seq_len)
            s1(); s2(); torch.cuda.synchronize()
            for name, fn in [("index_decode", s1), ("sparse_attn_decode", s2)]:
                for _ in range(warmup): fn()
                torch.cuda.synchronize()
                step_times[name] = round(triton_testing.do_bench(fn, warmup=0, rep=rep), 4)
        else:
            s1, s2, s3 = make_step_wrappers(
                idx_q, idx_kv, q, kv, bt, cu_q, sl, pl, seq_len, seq_len, n_kv_h, n_h,
                sc, TOPK, INIT_BLOCKS, LOCAL_BLOCKS)
            s1(); s2(); s3(); torch.cuda.synchronize()
            for name, fn in [("index_score", s1), ("index_topk", s2), ("sparse_attn", s3)]:
                for _ in range(warmup): fn()
                torch.cuda.synchronize()
                step_times[name] = round(triton_testing.do_bench(fn, warmup=0, rep=rep), 4)

    ref_ms = None
    if with_ref and not decode_mode:
        kvr = kv.to(torch.bfloat16) if kv.dtype != torch.bfloat16 else kv
        torch.cuda.synchronize()
        rs = torch.cuda.Event(enable_timing=True); re_ = torch.cuda.Event(enable_timing=True)
        rs.record()
        sr = ref_index_score(idx_q, idx_kv, bt, cu_q, sl, pl)
        tr = ref_index_topk(sr, cu_q, pl, TOPK, INIT_BLOCKS, LOCAL_BLOCKS)
        ref_sparse_attn(q, kvr, tr, bt, cu_q, sl, pl, sc)
        re_.record(); torch.cuda.synchronize()
        ref_ms = rs.elapsed_time(re_)

    r = {"batch": batch, "seq_len": seq_len, "num_kv_heads": n_kv_h, "num_heads": n_h,
         "head_dim": HEAD_DIM, "kv_dtype": str(kv_dtype).split(".")[-1],
         "total_q": tq, "total_blocks": tblocks, "cache_mb": round(cache_mb, 2),
         "latency_ms": round(latency_ms, 4), "p50_ms": round(p50, 4),
         "ref_ms": round(ref_ms, 2) if ref_ms else None,
         "speedup": round(ref_ms / latency_ms, 1) if ref_ms else None,
         "topk": TOPK, "timestamp": datetime.now().isoformat()}
    if step_times:
        if decode_mode:
            r["step_idx_decode_ms"] = step_times.get("index_decode")
            r["step_attn_decode_ms"] = step_times.get("sparse_attn_decode")
        else:
            r["step_score_ms"] = step_times.get("index_score")
            r["step_topk_ms"] = step_times.get("index_topk")
            r["step_attn_ms"] = step_times.get("sparse_attn")
    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    with_ref, per_step, warmup, rep, decode_mode, decode_qlen = _parse_cli()
    mode_str = f"Decode (qlen={decode_qlen})" if decode_mode else "Prefill"
    print("=" * 90)
    print(f"  MiniMax M3 Block-Sparse Attention (MSA) E2E Benchmark — {mode_str}")
    print(f"  TopK={TOPK}, InitBlocks={INIT_BLOCKS}, LocalBlocks={LOCAL_BLOCKS}")
    print(f"  Warmup={warmup}, Rep={rep}")
    if with_ref and not decode_mode: print("  Reference baseline: ENABLED")
    if per_step: print("  Per-step breakdown:  ENABLED")
    print("=" * 90); print()

    all_results = []
    for kv_dtype in KV_DTYPES:
        dn = str(kv_dtype).split(".")[-1]
        print(f"{'=' * 90}\n  KV Cache dtype: {dn}\n{'=' * 90}")
        cols = (f"{'Batch':>5s}  {'SeqLen':>7s}  {'KVHeads':>5s}  {'QHeads':>6s}  "
                f"{'TotalQ':>8s}  {'CacheMB':>7s}  {'Total(ms)':>10s}  {'P50(ms)':>9s}")
        if per_step:
            if decode_mode: cols += f"  {'IdxDec':>8s}  {'AttnDec':>8s}"
            else: cols += f"  {'Score':>8s}  {'TopK':>8s}  {'Attn':>8s}"
        if with_ref and not decode_mode: cols += f"  {'Ref(ms)':>10s}  {'Speedup':>7s}"
        print("-" * len(cols.expandtabs())); print(cols); print("-" * len(cols.expandtabs()))

        for batch, seq_len, n_kv_h, n_h in SHAPES:
            r = benchmark_shape(batch, seq_len, n_kv_h, n_h, kv_dtype,
                                with_ref=with_ref, per_step=per_step, warmup=warmup, rep=rep,
                                decode_mode=decode_mode, decode_qlen=decode_qlen)
            all_results.append(r)
            line = (f"{r['batch']:5d}  {r['seq_len']:7d}  {r['num_kv_heads']:5d}  "
                    f"{r['num_heads']:6d}  {r['total_q']:8d}  {r['cache_mb']:7.2f}  "
                    f"{r['latency_ms']:10.4f}  {r['p50_ms']:9.4f}")
            if per_step:
                if decode_mode:
                    line += f"  {r.get('step_idx_decode_ms', 0):8.4f}  {r.get('step_attn_decode_ms', 0):8.4f}"
                else:
                    line += f"  {r.get('step_score_ms', 0):8.4f}  {r.get('step_topk_ms', 0):8.4f}  {r.get('step_attn_ms', 0):8.4f}"
            if with_ref and not decode_mode:
                ref_s = f"{r['ref_ms']:10.2f}" if r['ref_ms'] else f"{'N/A':>10s}"
                spd_s = f"{r['speedup']:6.1f}x" if r['speedup'] else f"{'N/A':>7s}"
                line += f"  {ref_s}  {spd_s}"
            print(line)
        print()

    if all_results:
        fns = list(all_results[0].keys())
        fe = os.path.exists(CSV_FILE)
        with open(CSV_FILE, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fns)
            if not fe: w.writeheader()
            w.writerows(all_results)
        print(f"Results appended to: {CSV_FILE}")
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
