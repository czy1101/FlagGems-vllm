"""Shared benchmark logic for paged KV cache format.

Data format (same as vLLM):
  kv_cache: [num_blocks, 2, 128, num_kv_heads, head_dim]  K=[:,0] V=[:,1]
  index_kv_cache: [num_blocks, 128, head_dim]
  block_table: [batch, max_blocks]  (identity mapping)
"""
import warnings
warnings.filterwarnings("ignore", message="tl.make_block_ptr is deprecated")

import sys
from pathlib import Path

import torch
import triton.testing as triton_testing
import triton.knobs
triton.knobs.autotuning.adjust_block_size = False

# Allow running migrated benchmarks directly from a source checkout without
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

BLOCK = SPARSE_BLOCK_SIZE
HEAD_DIM = 128
DEFAULT_WARMUP, DEFAULT_REP = 5, 30

ALL_SHAPES = [
    (1, 8192, 16, 96),
    (2, 16384, 8, 96),
    (4, 2048, 16, 96),
    (4, 4096, 16, 384),
    (8, 2048, 32, 192),
    (2, 2048, 16, 96),
    (4, 1024, 8, 96),
    (8, 1024, 8, 48),
]

DECODE_SHAPES = [
    (1, 4096, 16, 96),
    (1, 16384, 16, 96),
    (1, 65536, 16, 96),
    (4, 4096, 8, 96),
    (4, 16384, 8, 96),
    (16, 4096, 8, 96),
    (32, 2048, 4, 48),
    (64, 1024, 4, 48),
]


def make_data(batch, seq_len, num_kv_heads, num_heads, device, dtype,
              decode=False, decode_qlen=1, head_dim=HEAD_DIM,
              randomize_pages=False):
    """Generate paged KV cache + block_table + Q tensors."""
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            "GQA requires num_heads to be divisible by num_kv_heads, "
            f"but got num_heads={num_heads}, num_kv_heads={num_kv_heads}."
        )
    blocks_per_batch = (seq_len + BLOCK - 1) // BLOCK
    total_blocks = batch * blocks_per_batch

    if decode:
        total_q = batch * decode_qlen
    else:
        total_q = batch * seq_len

    # Q and index Q (same as before, continuous)
    q = torch.randn(total_q, num_heads, head_dim, device=device, dtype=dtype)
    idx_q = torch.randn(total_q, num_kv_heads, head_dim, device=device, dtype=dtype)

    # Generate continuous K/V then pack into paged format
    k_cont = torch.randn(total_blocks * BLOCK, num_kv_heads, head_dim,
                         device=device, dtype=dtype)
    v_cont = torch.randn(total_blocks * BLOCK, num_kv_heads, head_dim,
                         device=device, dtype=dtype)
    index_k_cont = torch.randn(total_blocks * BLOCK, head_dim,
                               device=device, dtype=dtype)

    # Paged KV cache: [num_blocks, 2, 128, num_kv_heads, head_dim]
    kv_cache = torch.empty(
        total_blocks, 2, BLOCK, num_kv_heads, head_dim,
        device=device, dtype=dtype,
    )
    k_paged = k_cont.reshape(total_blocks, BLOCK, num_kv_heads, head_dim)
    kv_cache[:, 0] = k_paged
    v_paged = v_cont.reshape(total_blocks, BLOCK, num_kv_heads, head_dim)
    kv_cache[:, 1] = v_paged

    # Index KV cache: [num_blocks, 128, head_dim]
    index_kv_cache = index_k_cont.reshape(total_blocks, BLOCK, head_dim)

    # block_table: identity mapping [batch, max_blocks]
    physical_pages = (
        torch.randperm(total_blocks, device=device, dtype=torch.int64)
        if randomize_pages
        else torch.arange(total_blocks, device=device, dtype=torch.int64)
    )
    block_table = physical_pages.to(torch.int32).reshape(batch, blocks_per_batch)
    if randomize_pages:
        # Keep each request's logical K/V sequence unchanged while scattering
        # logical blocks across physical pages. This catches kernels that
        # accidentally assume an identity block table.
        kv_cache = kv_cache[physical_pages.argsort()].contiguous()
        index_kv_cache = index_kv_cache[physical_pages.argsort()].contiguous()

    # Metadata
    if decode:
        cu_q = torch.arange(0, (batch + 1) * decode_qlen, decode_qlen,
                            dtype=torch.int32, device=device)
    else:
        cu_q = torch.arange(0, (batch + 1) * seq_len, seq_len,
                            dtype=torch.int32, device=device)
    sl = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    pl = torch.zeros(batch, dtype=torch.int32, device=device)
    sm_scale = head_dim ** -0.5
    return q, idx_q, kv_cache, index_kv_cache, block_table, cu_q, sl, pl, sm_scale


def bench_fn(fn, warmup=DEFAULT_WARMUP, rep=DEFAULT_REP):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    return triton_testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def run_triton_prefill(q, idx_q, kv_cache, index_kv_cache, block_table,
                       cu_q, sl, pl, seq_len, n_kv_h, topk,
                       init_blocks, local_blocks, sm_scale, out):
    s = minimax_m3_index_score(idx_q, index_kv_cache, block_table, cu_q,
                               sl, pl, seq_len, seq_len, n_kv_h)
    t = minimax_m3_index_topk(s, cu_q, pl, seq_len, topk,
                              init_blocks, local_blocks)
    minimax_m3_sparse_attn(q, kv_cache, t, block_table, cu_q, sl, pl,
                           seq_len, n_kv_h, sm_scale, out)


def run_triton_decode(q, idx_q, kv_cache, index_kv_cache, block_table,
                      cu_q, sl, seq_len, n_kv_h, topk,
                      init_blocks, local_blocks, sm_scale, out, decode_qlen):
    t = minimax_m3_index_decode(
        idx_q, index_kv_cache, block_table, sl, seq_len, topk,
        init_blocks, local_blocks, n_kv_h, decode_qlen, decode_qlen,
    )
    minimax_m3_sparse_attn_decode(q, kv_cache, t, block_table, sl,
                                  n_kv_h, sm_scale, out, decode_qlen)


def bench_triton_prefill_steps(q, idx_q, kv_cache, index_kv_cache, block_table,
                               cu_q, sl, pl, seq_len, n_kv_h, topk,
                               init_blocks, local_blocks, sm_scale, out,
                               warmup=DEFAULT_WARMUP, rep=DEFAULT_REP):
    _s = None
    _t = None
    def s1():
        nonlocal _s
        _s = minimax_m3_index_score(idx_q, index_kv_cache, block_table, cu_q,
                                    sl, pl, seq_len, seq_len, n_kv_h)
    def s2():
        nonlocal _t
        _t = minimax_m3_index_topk(_s, cu_q, pl, seq_len, topk,
                                   init_blocks, local_blocks)
    def s3():
        minimax_m3_sparse_attn(q, kv_cache, _t, block_table, cu_q, sl, pl,
                               seq_len, n_kv_h, sm_scale, out)
    s1(); s2(); s3()
    torch.cuda.synchronize()
    return {
        "index_score": round(bench_fn(s1, warmup, rep), 4),
        "index_topk": round(bench_fn(s2, warmup, rep), 4),
        "sparse_attn": round(bench_fn(s3, warmup, rep), 4),
    }


def bench_triton_decode_steps(q, idx_q, kv_cache, index_kv_cache, block_table,
                              cu_q, sl, seq_len, n_kv_h, topk,
                              init_blocks, local_blocks, sm_scale, out, decode_qlen,
                              warmup=DEFAULT_WARMUP, rep=DEFAULT_REP):
    def s1():
        return minimax_m3_index_decode(
            idx_q, index_kv_cache, block_table, sl, seq_len, topk,
            init_blocks, local_blocks, n_kv_h, decode_qlen, decode_qlen,
        )
    def s2():
        t = s1()
        minimax_m3_sparse_attn_decode(q, kv_cache, t, block_table, sl,
                                      n_kv_h, sm_scale, out, decode_qlen)
    s1(); s2()
    torch.cuda.synchronize()
    return {
        "decode_idx": round(bench_fn(s1, warmup, rep), 4),
        "decode_attn": round(bench_fn(s2, warmup, rep), 4),
    }


def fmt_shape(shape):
    return f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}"


def parse_common_args(p):
    p.add_argument("--shape", type=str, default="8,1024,8,48")
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--init-blocks", type=int, default=1)
    p.add_argument("--local-blocks", type=int, default=2)
    p.add_argument("--all-shapes", action="store_true")
    p.add_argument("--per-step", action="store_true")
    p.add_argument("--decode", action="store_true")
    p.add_argument("--decode-qlen", type=int, default=1)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--rep", type=int, default=DEFAULT_REP)
    return p


def get_shapes(args):
    if args.decode:
        return DECODE_SHAPES if args.all_shapes else [tuple(map(int, args.shape.split(",")))]
    return ALL_SHAPES if args.all_shapes else [tuple(map(int, args.shape.split(",")))]
