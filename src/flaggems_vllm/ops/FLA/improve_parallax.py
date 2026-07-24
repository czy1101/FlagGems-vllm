# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from einops import reduce

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import (
    IS_NVIDIA_BLACKWELL,
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    check_shared_mem,
    contiguous,
)

PARALLAX_TILE_PAIRS = (
    (32, 32),
    (32, 64),
    (64, 32),
    (64, 64),
    (64, 128),
    (128, 64),
    (128, 128),
)

PARALLAX_AUTOTUNE_CONFIGS = [
    triton.Config({'BT': BT, 'BS': BS}, num_warps=num_warps, num_stages=2)
    for BT, BS in PARALLAX_TILE_PAIRS
    for num_warps in (2, 4, 8)
]


def parallax_prune_configs(configs, named_args, **kwargs):
    args = {**(named_args or {}), **kwargs}
    BK = args.get('BK')
    if BK is None:
        K = args.get('K')
        BK = triton.next_power_of_2(K) if K is not None else 64

    # cap the largest live tile dimension before compilation; Blackwell tolerates the existing
    # BT=BS=128 fallback at BK=256, while older architectures need a tighter bound.
    max_tile_elements = 32768 if IS_NVIDIA_BLACKWELL else 16384
    valid = [
        config
        for config in configs
        if max(config.kwargs['BT'], config.kwargs['BS']) * BK <= max_tile_elements
        and not (BK >= 128 and config.num_warps == 2)
    ]
    if valid:
        return valid
    return [min(configs, key=lambda config: max(config.kwargs['BT'], config.kwargs['BS']))]


def _block_size(head_dim: int, device_index: int) -> int:
    """
    Fallback block size for short and variable-length Parallax paths.

    Constraints:
    - Short and variable-length paths keep BT/BS coupled because chunk indices
      are prepared with this block size.
    - Long dense paths autotune BT and BS independently.
    - Larger BT improves GEMM efficiency.
    - Larger BT increases:
        * fp32 accumulator size
        * register pressure
        * shared memory usage

    H100 shared memory limit:
        228 KB per SM

    Avoid configurations that exceed ~200KB.
    """

    is_hopper = check_shared_mem('hopper', device_index)

    # Blackwell has different resource balance.
    if IS_NVIDIA_BLACKWELL:
        return 128

    # H100 / Hopper
    if is_hopper:

        # Small head:
        # Q/K/V tile is cheap, keep larger tile.
        if head_dim <= 64:
            return 128

        # Medium dimension:
        # 128 causes large accumulator footprint.
        if head_dim <= 128:
            return 64

        # Large head dimension:
        # e.g. D=256
        # must avoid:
        # BT=128, BK=256 combination
        return 32

    # Other GPUs safer choice
    return 64


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def improve_parallax_fwd_kernel_short(
    q, r, k, v, o, barv, d1, bart, m, scale,
    cu_seqlens, chunk_indices, T,
    HQ: tl.constexpr, H: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_bh = tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + 0).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)
        
    RCP_LN2: tl.constexpr = 1.4426950216
    scale_log2 = scale * RCP_LN2

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (0, 0), (BT, BK), (1, 0))
    p_r = tl.make_block_ptr(r + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (0, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (0, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * K, (T, K), (H * K, 1), (0, 0), (BT, BK), (1, 0))
    
    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_r = tl.load(p_r, boundary_check=(0, 1), padding_option="zero")
    b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")

    offs_q = tl.arange(0, BT)
    offs_k = tl.arange(0, BT)
    mask = offs_q[:, None] >= offs_k[None, :]
    if WINDOW_SIZE_LEFT >= 0:
        mask = mask & (offs_k[None, :] >= offs_q[:, None] - WINDOW_SIZE_LEFT + 1)
    mask = mask & (offs_q[:, None] < T) & (offs_k[None, :] < T)

    qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
    qk = tl.where(mask, qk, -float("inf"))

    m_max = tl.max(qk, axis=1, keep_dims=True)
    safe_m = tl.where(m_max == -float("inf"), 0.0, m_max)
    w = exp2(qk - safe_m)

    rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
    wr = w * rk

    # 修复 1：重命名局部变量，避免覆写函数参数 d1 和 barv
    d1_sum = tl.sum(w, axis=1, keep_dims=True)       # 形状: (BT, 1)
    d2_sum = tl.sum(wr, axis=1, keep_dims=True)      # 形状: (BT, 1)

    barv_sum = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32) # 形状: (BT, BK)
    Rv = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32)      # 形状: (BT, BK)

    row_mask = offs_q[:, None] < T
    inv_d1 = tl.where(row_mask, 1.0 / d1_sum, 0.0)
    
    b_barv = barv_sum * inv_d1
    b_bart = d2_sum * inv_d1
    b_o = b_barv + b_bart * b_barv - Rv * inv_d1

    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (0, 0), (BT, BK), (1, 0))
    p_barv = tl.make_block_ptr(barv + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (0, 0), (BT, BK), (1, 0))
    p_d1 = tl.make_block_ptr(d1 + bos * HQ + i_hq, (T, 1), (HQ, 1), (0, 0), (BT, 1), (1, 0))
    p_bart = tl.make_block_ptr(bart + bos * HQ + i_hq, (T, 1), (HQ, 1), (0, 0), (BT, 1), (1, 0))
    p_m = tl.make_block_ptr(m + bos * HQ + i_hq, (T, 1), (HQ, 1), (0, 0), (BT, 1), (1, 0))

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_barv, b_barv.to(p_barv.dtype.element_ty), boundary_check=(0, 1))
    
    # 修复 2：原样传入 2D 的 d1_sum, b_bart, m_max，保持与 block_ptr 维度一致
    tl.store(p_d1, d1_sum, boundary_check=(0, 1))
    tl.store(p_bart, b_bart, boundary_check=(0, 1))
    tl.store(p_m, m_max, boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def improve_parallax_fwd_kernel_short_multi(
    q, r, k, v, o, barv, d1, bart, m, scale, T,
    HQ: tl.constexpr, H: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr, BT: tl.constexpr,
    NT: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    offs_t = i_t * BT + tl.arange(0, BT).to(tl.int64)
    offs_k = tl.arange(0, BK).to(tl.int64)
    row_mask = offs_t[:, None] < T
    head_mask = offs_k[None, :] < K

    p_q = q + ((bos + offs_t[:, None]) * HQ + i_hq) * K + offs_k[None, :]
    p_r = r + ((bos + offs_t[:, None]) * HQ + i_hq) * K + offs_k[None, :]
    b_q = tl.load(p_q, mask=row_mask & head_mask, other=0.0)
    b_r = tl.load(p_r, mask=row_mask & head_mask, other=0.0)

    m_acc = tl.full((BT, 1), -float("inf"), dtype=tl.float32)
    d1_acc = tl.zeros((BT, 1), dtype=tl.float32)
    d2_acc = tl.zeros((BT, 1), dtype=tl.float32)
    barv_acc = tl.zeros((BT, BK), dtype=tl.float32)
    rv_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * 1.4426950216

    # the causal tile index bounds this loop to the active short-sequence tiles.
    for i_s in range(0, tl.minimum(i_t + 1, NT)):
        offs_s = (i_s * BT + tl.arange(0, BT)).to(tl.int64)
        col_mask = offs_s[:, None] < T
        p_k = k + ((bos + offs_s[:, None]) * H + i_h) * K + offs_k[None, :]
        p_v = v + ((bos + offs_s[:, None]) * H + i_h) * K + offs_k[None, :]
        b_k = tl.load(p_k, mask=col_mask & head_mask, other=0.0)
        b_v = tl.load(p_v, mask=col_mask & head_mask, other=0.0)

        mask = (offs_t[:, None] >= offs_s[None, :]) & row_mask & (offs_s[None, :] < T)
        if WINDOW_SIZE_LEFT >= 0:
            mask = mask & (offs_s[None, :] >= offs_t[:, None] - WINDOW_SIZE_LEFT + 1)

        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        qk = exp2(qk - safe_m)
        rk = qk * tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)

        d1_acc = alpha * d1_acc + tl.sum(qk, axis=1, keep_dims=True)
        d2_acc = alpha * d2_acc + tl.sum(rk, axis=1, keep_dims=True)
        barv_acc = alpha * barv_acc
        rv_acc = alpha * rv_acc
        barv_acc = tl.dot(qk.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=barv_acc)
        rv_acc = tl.dot(rk.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=rv_acc)
        m_acc = m_new

    inv_d1 = tl.where(row_mask, 1.0 / d1_acc, 0.0)
    b_barv = barv_acc * inv_d1
    b_bart = d2_acc * inv_d1
    b_o = b_barv + b_bart * b_barv - rv_acc * inv_d1

    p_o = o + ((bos + offs_t[:, None]) * HQ + i_hq) * K + offs_k[None, :]
    p_barv = barv + ((bos + offs_t[:, None]) * HQ + i_hq) * K + offs_k[None, :]
    scalar_offsets = (bos + offs_t[:, None]) * HQ + i_hq
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=row_mask & head_mask)
    tl.store(p_barv, b_barv.to(p_barv.dtype.element_ty), mask=row_mask & head_mask)
    tl.store(d1 + scalar_offsets, d1_acc, mask=row_mask)
    tl.store(bart + scalar_offsets, b_bart, mask=row_mask)
    tl.store(m + scalar_offsets, m_acc, mask=row_mask)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=PARALLAX_AUTOTUNE_CONFIGS,
    key=['T', 'HQ', 'G', 'K', 'BK', 'WINDOW_SIZE_LEFT', 'IS_VARLEN'],
    reset_to_zero=['o'],
    prune_configs_by={'early_config_prune': parallax_prune_configs},
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def improve_parallax_fwd_kernel(
    q,
    r,
    k,
    v,
    o,
    barv,
    d1,
    bart,
    m,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

# ✅ 修复：用新变量名保存 tile 索引
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        tile_idx = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)
        tile_idx = i_t
    RCP_LN2: tl.constexpr = 1.4426950216

    row_offset = tile_idx * BT  # ✅ 使用正确的 tile 索引
    row_indices = row_offset + tl.arange(0, BT)
    row_mask = row_indices[:, None] < T
    NUM_TOTAL_BLOCKS = tl.cdiv(tl.minimum(T, row_offset + BT), BS)
    NUM_SAFE_BLOCKS = tl.minimum(row_offset, T) // BS

    # SWA col-block boundaries. WINDOW_SIZE_LEFT < 0 disables SWA.
    if WINDOW_SIZE_LEFT >= 0:
        leftmost_valid = tl.maximum(0, row_offset - WINDOW_SIZE_LEFT + 1)
        FIRST_COL_BLOCK = leftmost_valid // BS
        # Phase A is unmasked, so the safe zone must clear the window's left edge for
        # the tile's LAST row (row_offset + BT - 1), not its first.
        safe_left_valid = tl.maximum(0, row_offset + BT - WINDOW_SIZE_LEFT)
        SAFE_LEFT_START = (safe_left_valid + BS - 1) // BS
    else:
        FIRST_COL_BLOCK = 0
        SAFE_LEFT_START = 0
    LEFT_BORDER_END = tl.minimum(SAFE_LEFT_START, NUM_SAFE_BLOCKS)
    SAFE_MIDDLE_START = tl.maximum(FIRST_COL_BLOCK, SAFE_LEFT_START)
    RIGHT_BORDER_START = tl.maximum(FIRST_COL_BLOCK, NUM_SAFE_BLOCKS)

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_r = tl.make_block_ptr(r + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (FIRST_COL_BLOCK * BS, 0), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * K, (T, K), (H * K, 1), (FIRST_COL_BLOCK * BS, 0), (BS, BK), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_barv = tl.make_block_ptr(barv + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_d1 = tl.make_block_ptr(d1 + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))
    p_bart = tl.make_block_ptr(bart + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))
    p_m = tl.make_block_ptr(m + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_r = tl.load(p_r, boundary_check=(0, 1), padding_option="zero")
    m_acc = tl.zeros((BT, 1), dtype=tl.float32) - float("inf")
    d1_acc = tl.zeros((BT, 1), dtype=tl.float32)
    d2_acc = tl.zeros((BT, 1), dtype=tl.float32)
    barv_acc = tl.zeros((BT, BK), dtype=tl.float32)
    Rv_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    # Phase 0: left-border blocks (SWA only). Window mask only.
    for col_block_id in range(FIRST_COL_BLOCK, LEFT_BORDER_END):
        col_indices = col_block_id * BS + tl.arange(0, BS)
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        mask = (
            (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
            & row_mask
            & (col_indices[None, :] < T)
        )
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        w = exp2(qk - safe_m)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        wr = w * rk
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        barv_acc = alpha * barv_acc
        Rv_acc = alpha * Rv_acc
        barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=barv_acc)
        Rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (BS, 0))

    # Phase A: safe blocks (no mask).
    for _safe in range(SAFE_MIDDLE_START, NUM_SAFE_BLOCKS):
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        w = exp2(qk - safe_m)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        wr = w * rk
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        barv_acc = alpha * barv_acc
        Rv_acc = alpha * Rv_acc
        barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=barv_acc)
        Rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (BS, 0))

    # Phase B: right-border blocks (causal + boundary + window mask).
    for col_block_id in range(RIGHT_BORDER_START, NUM_TOTAL_BLOCKS):
        col_indices = col_block_id * BS + tl.arange(0, BS)
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        if WINDOW_SIZE_LEFT >= 0:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                & row_mask
                & (col_indices[None, :] < T)
            )
        else:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & row_mask
                & (col_indices[None, :] < T)
            )
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        w = exp2(qk - safe_m)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        wr = w * rk
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        barv_acc = alpha * barv_acc
        Rv_acc = alpha * Rv_acc
        barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=barv_acc)
        Rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (BS, 0))

    inv_d1 = tl.where(row_mask, 1.0 / d1_acc, 0.0)
    b_barv = barv_acc * inv_d1
    b_bart = d2_acc * inv_d1
    b_o = b_barv + b_bart * b_barv - Rv_acc * inv_d1

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_barv, b_barv.to(p_barv.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_d1, d1_acc, boundary_check=(0, 1))
    tl.store(p_bart, b_bart, boundary_check=(0, 1))
    tl.store(p_m, m_acc, boundary_check=(0, 1))

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def improve_parallax_fwd_kernel_varlen(
    q, r, k, v, o, barv, d1, bart, m, scale,
    cu_seqlens, chunk_indices, T,
    HQ: tl.constexpr, H: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

# ✅ 修复：用新变量名保存 tile 索引
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        tile_idx = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)
        tile_idx = i_t
    RCP_LN2: tl.constexpr = 1.4426950216

    row_offset = tile_idx * BT  # ✅ 使用正确的 tile 索引
    row_indices = row_offset + tl.arange(0, BT)
    row_mask = row_indices[:, None] < T
    NUM_TOTAL_BLOCKS = tl.cdiv(tl.minimum(T, row_offset + BT), BS)
    NUM_SAFE_BLOCKS = tl.minimum(row_offset, T) // BS

    # SWA col-block boundaries. WINDOW_SIZE_LEFT < 0 disables SWA.
    if WINDOW_SIZE_LEFT >= 0:
        leftmost_valid = tl.maximum(0, row_offset - WINDOW_SIZE_LEFT + 1)
        FIRST_COL_BLOCK = leftmost_valid // BS
        # Phase A is unmasked, so the safe zone must clear the window's left edge for
        # the tile's LAST row (row_offset + BT - 1), not its first.
        safe_left_valid = tl.maximum(0, row_offset + BT - WINDOW_SIZE_LEFT)
        SAFE_LEFT_START = (safe_left_valid + BS - 1) // BS
    else:
        FIRST_COL_BLOCK = 0
        SAFE_LEFT_START = 0
    LEFT_BORDER_END = tl.minimum(SAFE_LEFT_START, NUM_SAFE_BLOCKS)
    SAFE_MIDDLE_START = tl.maximum(FIRST_COL_BLOCK, SAFE_LEFT_START)
    RIGHT_BORDER_START = tl.maximum(FIRST_COL_BLOCK, NUM_SAFE_BLOCKS)

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_r = tl.make_block_ptr(r + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (FIRST_COL_BLOCK * BS, 0), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * K, (T, K), (H * K, 1), (FIRST_COL_BLOCK * BS, 0), (BS, BK), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_barv = tl.make_block_ptr(barv + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_d1 = tl.make_block_ptr(d1 + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))
    p_bart = tl.make_block_ptr(bart + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))
    p_m = tl.make_block_ptr(m + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_r = tl.load(p_r, boundary_check=(0, 1), padding_option="zero")
    m_acc = tl.zeros((BT, 1), dtype=tl.float32) - float("inf")
    d1_acc = tl.zeros((BT, 1), dtype=tl.float32)
    d2_acc = tl.zeros((BT, 1), dtype=tl.float32)
    barv_acc = tl.zeros((BT, BK), dtype=tl.float32)
    Rv_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    # Phase 0: left-border blocks (SWA only). Window mask only.
    for col_block_id in range(FIRST_COL_BLOCK, LEFT_BORDER_END):
        col_indices = col_block_id * BS + tl.arange(0, BS)
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        mask = (
            (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
            & row_mask
            & (col_indices[None, :] < T)
        )
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        w = exp2(qk - safe_m)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        wr = w * rk
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        barv_acc = alpha * barv_acc
        Rv_acc = alpha * Rv_acc
        barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=barv_acc)
        Rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (BS, 0))

    # Phase A: safe blocks (no mask).
    for _safe in range(SAFE_MIDDLE_START, NUM_SAFE_BLOCKS):
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        w = exp2(qk - safe_m)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        wr = w * rk
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        barv_acc = alpha * barv_acc
        Rv_acc = alpha * Rv_acc
        barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=barv_acc)
        Rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (BS, 0))

    # Phase B: right-border blocks (causal + boundary + window mask).
    for col_block_id in range(RIGHT_BORDER_START, NUM_TOTAL_BLOCKS):
        col_indices = col_block_id * BS + tl.arange(0, BS)
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        if WINDOW_SIZE_LEFT >= 0:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                & row_mask
                & (col_indices[None, :] < T)
            )
        else:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & row_mask
                & (col_indices[None, :] < T)
            )
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        w = exp2(qk - safe_m)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        wr = w * rk
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        barv_acc = alpha * barv_acc
        Rv_acc = alpha * Rv_acc
        barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=barv_acc)
        Rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (BS, 0))

    inv_d1 = tl.where(row_mask, 1.0 / d1_acc, 0.0)
    b_barv = barv_acc * inv_d1
    b_bart = d2_acc * inv_d1
    b_o = b_barv + b_bart * b_barv - Rv_acc * inv_d1

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_barv, b_barv.to(p_barv.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_d1, d1_acc, boundary_check=(0, 1))
    tl.store(p_bart, b_bart, boundary_check=(0, 1))
    tl.store(p_m, m_acc, boundary_check=(0, 1))
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def improve_parallax_bwd_kernel_short(
    q, r, k, v, o, barv, d1, bart, m, grad_o, 
    grad_q, grad_r, grad_k_buf, grad_v_buf, scale,
    cu_seqlens, chunk_indices, T,
    HQ: tl.constexpr, H: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    针对小序列的极简反向 Kernel。
    将 preprocess, dqr, dkv 三个 Kernel 融为一体，消灭显存读写和 Launch 开销。
    """
    i_bh = tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + 0).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)
        
    RCP_LN2: tl.constexpr = 1.4426950216
    scale_log2 = scale * RCP_LN2

    # 1. 一次性加载所有必需数据到 SRAM
    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (0, 0), (BT, BK), (1, 0))
    p_r = tl.make_block_ptr(r + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (0, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (0, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * K, (T, K), (H * K, 1), (0, 0), (BT, BK), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (0, 0), (BT, BK), (1, 0))
    p_barv = tl.make_block_ptr(barv + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (0, 0), (BT, BK), (1, 0))
    p_grad_o = tl.make_block_ptr(grad_o + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (0, 0), (BT, BK), (1, 0))
    
    p_d1 = tl.make_block_ptr(d1 + bos * HQ + i_hq, (T, 1), (HQ, 1), (0, 0), (BT, 1), (1, 0))
    p_bart = tl.make_block_ptr(bart + bos * HQ + i_hq, (T, 1), (HQ, 1), (0, 0), (BT, 1), (1, 0))
    p_m = tl.make_block_ptr(m + bos * HQ + i_hq, (T, 1), (HQ, 1), (0, 0), (BT, 1), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_r = tl.load(p_r, boundary_check=(0, 1), padding_option="zero")
    b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
    b_o = tl.load(p_o, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    b_barv = tl.load(p_barv, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    b_grad_o = tl.load(p_grad_o, boundary_check=(0, 1), padding_option="zero").to(b_q.dtype)
    
    b_d1 = tl.load(p_d1, boundary_check=(0, 1), padding_option="zero")
    b_bart = tl.load(p_bart, boundary_check=(0, 1), padding_option="zero")
    b_m = tl.load(p_m, boundary_check=(0, 1), padding_option="zero")

    # 2. 替代 preprocess: 在寄存器中直接计算 delta_t 和 delta_b
    delta_t = tl.sum(b_grad_o * b_o, axis=1, keep_dims=True)
    delta_b = tl.sum(b_grad_o * b_barv, axis=1, keep_dims=True)

    # 3. 构造 Mask
    offs_q = tl.arange(0, BT)
    offs_k = tl.arange(0, BT)
    row_mask = offs_q[:, None] < T
    mask = offs_q[:, None] >= offs_k[None, :]
    if WINDOW_SIZE_LEFT >= 0:
        mask = mask & (offs_k[None, :] >= offs_q[:, None] - WINDOW_SIZE_LEFT + 1)
    mask = mask & row_mask & (offs_k[None, :] < T)

    # 4. 重算 Attention 权重矩阵
    qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
    qk = tl.where(mask, qk, -float("inf"))
    w = exp2(qk - b_m)
    
    inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
    p = w * inv_d1

    # 5. 核心反向推导逻辑
    a = tl.dot(b_grad_o, tl.trans(b_v), out_dtype=tl.float32)
    rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
    delta = a - delta_b
    bart_minus_rk = b_bart - rk
    
    gl = p * (a - delta_t + bart_minus_rk * delta)
    gu = -p * delta

    # 6. 计算 Q, R, K, V 的梯度
    grad_q_acc = tl.dot(gl.to(b_k.dtype), b_k, out_dtype=tl.float32) * scale
    grad_r_acc = tl.dot(gu.to(b_k.dtype), b_k, out_dtype=tl.float32)
    
    gl_scale = gl * scale
    grad_k_acc = tl.dot(tl.trans(gl_scale).to(b_q.dtype), b_q, out_dtype=tl.float32)
    grad_k_acc += tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32)
    
    weights = p * (1.0 + bart_minus_rk)
    grad_v_acc = tl.dot(tl.trans(weights).to(b_grad_o.dtype), b_grad_o, out_dtype=tl.float32)

    # 7. 写回梯度到 Global Memory
    p_grad_q = tl.make_block_ptr(grad_q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (0, 0), (BT, BK), (1, 0))
    p_grad_r = tl.make_block_ptr(grad_r + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (0, 0), (BT, BK), (1, 0))
    p_grad_k = tl.make_block_ptr(grad_k_buf + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (0, 0), (BT, BK), (1, 0))
    p_grad_v = tl.make_block_ptr(grad_v_buf + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (0, 0), (BT, BK), (1, 0))

    tl.store(p_grad_q, grad_q_acc.to(p_grad_q.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_grad_r, grad_r_acc.to(p_grad_r.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_grad_k, grad_k_acc.to(p_grad_k.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_grad_v, grad_v_acc.to(p_grad_v.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def improve_parallax_bwd_kernel_dqr_short_multi(
    q, r, k, v, o, barv, d1, bart, m, grad_o,
    grad_q, grad_r, scale, T,
    HQ: tl.constexpr, H: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr, BT: tl.constexpr,
    NT: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    offs_t = i_t * BT + tl.arange(0, BT).to(tl.int64)
    offs_k = tl.arange(0, BK).to(tl.int64)
    row_mask = offs_t[:, None] < T
    head_mask = offs_k[None, :] < K
    q_offsets = ((bos + offs_t[:, None]) * HQ + i_hq) * K + offs_k[None, :]

    b_q = tl.load(q + q_offsets, mask=row_mask & head_mask, other=0.0)
    b_r = tl.load(r + q_offsets, mask=row_mask & head_mask, other=0.0)
    b_o = tl.load(o + q_offsets, mask=row_mask & head_mask, other=0.0).to(tl.float32)
    b_barv = tl.load(barv + q_offsets, mask=row_mask & head_mask, other=0.0).to(tl.float32)
    b_grad_o = tl.load(grad_o + q_offsets, mask=row_mask & head_mask, other=0.0)
    scalar_offsets = (bos + offs_t) * HQ + i_hq
    b_d1 = tl.load(d1 + scalar_offsets, mask=offs_t < T, other=1.0)[:, None]
    b_bart = tl.load(bart + scalar_offsets, mask=offs_t < T, other=0.0)[:, None]
    b_m = tl.load(m + scalar_offsets, mask=offs_t < T, other=0.0)[:, None]

    delta_t = tl.sum(b_grad_o * b_o, axis=1, keep_dims=True)
    delta_b = tl.sum(b_grad_o * b_barv, axis=1, keep_dims=True)
    inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
    grad_q_acc = tl.zeros((BT, BK), dtype=tl.float32)
    grad_r_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * 1.4426950216

    for i_s in range(0, tl.minimum(i_t + 1, NT)):
        offs_s = (i_s * BT + tl.arange(0, BT)).to(tl.int64)
        col_mask = offs_s[:, None] < T
        kv_offsets = ((bos + offs_s[:, None]) * H + i_h) * K + offs_k[None, :]
        b_k = tl.load(k + kv_offsets, mask=col_mask & head_mask, other=0.0)
        b_v = tl.load(v + kv_offsets, mask=col_mask & head_mask, other=0.0)

        mask = (offs_t[:, None] >= offs_s[None, :]) & row_mask & (offs_s[None, :] < T)
        if WINDOW_SIZE_LEFT >= 0:
            mask = mask & (offs_s[None, :] >= offs_t[:, None] - WINDOW_SIZE_LEFT + 1)

        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        p = exp2(qk - b_m) * inv_d1
        a = tl.dot(b_grad_o, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - delta_b
        bart_minus_rk = b_bart - tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        gl = p * (a - delta_t + bart_minus_rk * delta) * scale
        gu = -p * delta
        grad_q_acc = tl.dot(gl.to(b_k.dtype), b_k, out_dtype=tl.float32, acc=grad_q_acc)
        grad_r_acc = tl.dot(gu.to(b_k.dtype), b_k, out_dtype=tl.float32, acc=grad_r_acc)

    tl.store(grad_q + q_offsets, grad_q_acc.to(grad_q.dtype.element_ty), mask=row_mask & head_mask)
    tl.store(grad_r + q_offsets, grad_r_acc.to(grad_r.dtype.element_ty), mask=row_mask & head_mask)


@triton.jit(do_not_specialize=['T'])
def improve_parallax_bwd_kernel_dkv_short_multi(
    q, r, k, v, o, barv, d1, bart, m, grad_o,
    grad_k_buf, grad_v_buf, scale, T,
    HQ: tl.constexpr, H: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr, BT: tl.constexpr,
    NT: tl.constexpr,
):
    i_s = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    offs_s = i_s * BT + tl.arange(0, BT).to(tl.int64)
    offs_k = tl.arange(0, BK).to(tl.int64)
    col_mask = offs_s[:, None] < T
    head_mask = offs_k[None, :] < K
    kv_offsets = ((bos + offs_s[:, None]) * H + i_h) * K + offs_k[None, :]
    b_k = tl.load(k + kv_offsets, mask=col_mask & head_mask, other=0.0)
    b_v = tl.load(v + kv_offsets, mask=col_mask & head_mask, other=0.0)

    grad_k_acc = tl.zeros((BT, BK), dtype=tl.float32)
    grad_v_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * 1.4426950216

    for i_t in range(i_s, NT):
        offs_t = (i_t * BT + tl.arange(0, BT)).to(tl.int64)
        row_mask = offs_t[:, None] < T
        q_offsets = ((bos + offs_t[:, None]) * HQ + i_hq) * K + offs_k[None, :]
        b_q = tl.load(q + q_offsets, mask=row_mask & head_mask, other=0.0)
        b_r = tl.load(r + q_offsets, mask=row_mask & head_mask, other=0.0)
        b_o = tl.load(o + q_offsets, mask=row_mask & head_mask, other=0.0).to(tl.float32)
        b_barv = tl.load(barv + q_offsets, mask=row_mask & head_mask, other=0.0).to(tl.float32)
        b_grad_o = tl.load(grad_o + q_offsets, mask=row_mask & head_mask, other=0.0)
        scalar_offsets = (bos + offs_t) * HQ + i_hq
        b_d1 = tl.load(d1 + scalar_offsets, mask=offs_t < T, other=1.0)[:, None]
        b_bart = tl.load(bart + scalar_offsets, mask=offs_t < T, other=0.0)[:, None]
        b_m = tl.load(m + scalar_offsets, mask=offs_t < T, other=0.0)[:, None]

        mask = (offs_t[:, None] >= offs_s[None, :]) & row_mask & (offs_s[None, :] < T)
        if WINDOW_SIZE_LEFT >= 0:
            mask = mask & (offs_s[None, :] >= offs_t[:, None] - WINDOW_SIZE_LEFT + 1)

        delta_t = tl.sum(b_grad_o * b_o, axis=1, keep_dims=True)
        delta_b = tl.sum(b_grad_o * b_barv, axis=1, keep_dims=True)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        p = exp2(qk - b_m) * tl.where(row_mask, 1.0 / b_d1, 0.0)
        a = tl.dot(b_grad_o, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - delta_b
        bart_minus_rk = b_bart - tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        gl = p * (a - delta_t + bart_minus_rk * delta) * scale
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1.0 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(weights).to(b_grad_o.dtype), b_grad_o, out_dtype=tl.float32, acc=grad_v_acc)

    grad_offsets = ((bos + offs_s[:, None]) * HQ + i_hq) * K + offs_k[None, :]
    tl.store(grad_k_buf + grad_offsets, grad_k_acc.to(grad_k_buf.dtype.element_ty), mask=col_mask & head_mask)
    tl.store(grad_v_buf + grad_offsets, grad_v_acc.to(grad_v_buf.dtype.element_ty), mask=col_mask & head_mask)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=PARALLAX_AUTOTUNE_CONFIGS,
    key=['T', 'HQ', 'G', 'K', 'BK', 'WINDOW_SIZE_LEFT', 'IS_VARLEN'],
    reset_to_zero=['grad_q', 'grad_r'],
    prune_configs_by={'early_config_prune': parallax_prune_configs},
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def improve_parallax_bwd_kernel_dqr_fused(
    q, r, k, v, 
    o, barv, d1, bart, m,
    delta_t, delta_b,               # 恢复为原名，无缝兼容外部传入
    grad_o, grad_q, grad_r, scale,
    cu_seqlens, chunk_indices, T,   # 统一保留变长参数位
    HQ: tl.constexpr, H: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        tile_idx = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)
        tile_idx = i_t
        
    RCP_LN2: tl.constexpr = 1.4426950216

    row_offset = tile_idx * BT
    row_indices = row_offset + tl.arange(0, BT)
    row_mask = row_indices[:, None] < T
    NUM_TOTAL_BLOCKS = tl.cdiv(tl.minimum(T, row_offset + BT), BS)
    NUM_SAFE_BLOCKS = tl.minimum(row_offset, T) // BS

    if WINDOW_SIZE_LEFT >= 0:
        leftmost_valid = tl.maximum(0, row_offset - WINDOW_SIZE_LEFT + 1)
        FIRST_COL_BLOCK = leftmost_valid // BS
        safe_left_valid = tl.maximum(0, row_offset + BT - WINDOW_SIZE_LEFT)
        SAFE_LEFT_START = (safe_left_valid + BS - 1) // BS
    else:
        FIRST_COL_BLOCK = 0
        SAFE_LEFT_START = 0
        
    LEFT_BORDER_END = tl.minimum(SAFE_LEFT_START, NUM_SAFE_BLOCKS)
    SAFE_MIDDLE_START = tl.maximum(FIRST_COL_BLOCK, SAFE_LEFT_START)
    RIGHT_BORDER_START = tl.maximum(FIRST_COL_BLOCK, NUM_SAFE_BLOCKS)

    # ----------------------------------------------------------------
    # 步骤 1: 预处理逻辑融合
    # ----------------------------------------------------------------
    p_grad_o = tl.make_block_ptr(grad_o + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_barv = tl.make_block_ptr(barv + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))

    grad_o_tile_fp32 = tl.load(p_grad_o, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    b_o = tl.load(p_o, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    b_barv_fp32 = tl.load(p_barv, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    # 计算标量 delta，供后续 dkv 阶段使用
    b_t = tl.sum(grad_o_tile_fp32 * b_o, axis=1, keep_dims=True)
    b_b = tl.sum(grad_o_tile_fp32 * b_barv_fp32, axis=1, keep_dims=True)

    # 将 b_t, b_b 写入全局内存
    p_t = tl.make_block_ptr(delta_t + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))
    p_b = tl.make_block_ptr(delta_b + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))
    tl.store(p_t, b_t, boundary_check=(0, 1))
    tl.store(p_b, b_b, boundary_check=(0, 1))

    # 声明 grad_o_tile 供下方主循环使用
    grad_o_tile = grad_o_tile_fp32.to(p_grad_o.dtype.element_ty)

    # ----------------------------------------------------------------
    # 步骤 2: 主 DQR 阶段逻辑
    # ----------------------------------------------------------------
    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_r = tl.make_block_ptr(r + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_d1 = tl.make_block_ptr(d1 + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))
    p_bart = tl.make_block_ptr(bart + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))
    p_m = tl.make_block_ptr(m + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))
    
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (FIRST_COL_BLOCK * BS, 0), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * K, (T, K), (H * K, 1), (FIRST_COL_BLOCK * BS, 0), (BS, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_r = tl.load(p_r, boundary_check=(0, 1), padding_option="zero")
    b_d1 = tl.load(p_d1, boundary_check=(0, 1), padding_option="zero")
    b_bart = tl.load(p_bart, boundary_check=(0, 1), padding_option="zero")
    b_m = tl.load(p_m, boundary_check=(0, 1), padding_option="zero")

    grad_q_acc = tl.zeros((BT, BK), dtype=tl.float32)
    grad_r_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2
    inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)

    # Phase 0
    for col_block_id in range(FIRST_COL_BLOCK, LEFT_BORDER_END):
        col_indices = col_block_id * BS + tl.arange(0, BS)
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        mask = (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1) & row_mask & (col_indices[None, :] < T)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        w = exp2(qk - b_m)
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        p = w * inv_d1
        bart_minus_rk = b_bart - rk
        
        delta = a - b_b
        grad_q_acc = tl.dot(
            (p * (a - b_t + (b_bart - rk) * delta)).to(b_k.dtype), 
            b_k, 
            out_dtype=tl.float32, 
            acc=grad_q_acc
        )
        grad_r_acc = tl.dot(
            (-p * delta).to(b_k.dtype), 
            b_k, 
            out_dtype=tl.float32, 
            acc=grad_r_acc
        )
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (BS, 0))

    # Phase A
    for _ in range(SAFE_MIDDLE_START, NUM_SAFE_BLOCKS):
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        w = exp2(qk - b_m)
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        p = w * inv_d1
        bart_minus_rk = b_bart - rk
        
        delta = a - b_b
        grad_q_acc = tl.dot(
            (p * (a - b_t + (b_bart - rk) * delta)).to(b_k.dtype), 
            b_k, 
            out_dtype=tl.float32, 
            acc=grad_q_acc
        )
        grad_r_acc = tl.dot(
            (-p * delta).to(b_k.dtype), 
            b_k, 
            out_dtype=tl.float32, 
            acc=grad_r_acc
        )
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (BS, 0))

    # Phase B
    for col_block_id in range(RIGHT_BORDER_START, NUM_TOTAL_BLOCKS):
        col_indices = col_block_id * BS + tl.arange(0, BS)
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        if WINDOW_SIZE_LEFT >= 0:
            mask = (row_indices[:, None] >= col_indices[None, :]) & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1) & row_mask & (col_indices[None, :] < T)
        else:
            mask = (row_indices[:, None] >= col_indices[None, :]) & row_mask & (col_indices[None, :] < T)
            
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        w = exp2(qk - b_m)
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        p = w * inv_d1
        bart_minus_rk = b_bart - rk
        
        delta = a - b_b
        grad_q_acc = tl.dot(
            (p * (a - b_t + (b_bart - rk) * delta)).to(b_k.dtype), 
            b_k, 
            out_dtype=tl.float32, 
            acc=grad_q_acc
        )
        grad_r_acc = tl.dot(
            (-p * delta).to(b_k.dtype), 
            b_k, 
            out_dtype=tl.float32, 
            acc=grad_r_acc
        )
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (BS, 0))

    # 声明 p_grad_q 和 p_grad_r 供最终写回
    grad_q_acc = scale * grad_q_acc
    p_grad_q = tl.make_block_ptr(grad_q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_grad_r = tl.make_block_ptr(grad_r + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    
    tl.store(p_grad_q, grad_q_acc.to(p_grad_q.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_grad_r, grad_r_acc.to(p_grad_r.dtype.element_ty), boundary_check=(0, 1))
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def improve_parallax_bwd_kernel_dqr_varlen_fused(
    q, r, k, v, 
    o, barv, d1, bart, m,
    delta_t, delta_b, 
    grad_o, grad_q, grad_r, scale,
    cu_seqlens, chunk_indices, T,
    HQ: tl.constexpr, H: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        tile_idx = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)
        tile_idx = i_t
        
    RCP_LN2: tl.constexpr = 1.4426950216

    row_offset = tile_idx * BT
    row_indices = row_offset + tl.arange(0, BT)
    row_mask = row_indices[:, None] < T
    NUM_TOTAL_BLOCKS = tl.cdiv(tl.minimum(T, row_offset + BT), BS)
    NUM_SAFE_BLOCKS = tl.minimum(row_offset, T) // BS

    if WINDOW_SIZE_LEFT >= 0:
        leftmost_valid = tl.maximum(0, row_offset - WINDOW_SIZE_LEFT + 1)
        FIRST_COL_BLOCK = leftmost_valid // BS
        safe_left_valid = tl.maximum(0, row_offset + BT - WINDOW_SIZE_LEFT)
        SAFE_LEFT_START = (safe_left_valid + BS - 1) // BS
    else:
        FIRST_COL_BLOCK = 0
        SAFE_LEFT_START = 0
        
    LEFT_BORDER_END = tl.minimum(SAFE_LEFT_START, NUM_SAFE_BLOCKS)
    SAFE_MIDDLE_START = tl.maximum(FIRST_COL_BLOCK, SAFE_LEFT_START)
    RIGHT_BORDER_START = tl.maximum(FIRST_COL_BLOCK, NUM_SAFE_BLOCKS)

    # ----------------------------------------------------------------
    # 步骤 1: 预处理逻辑融合
    # ----------------------------------------------------------------
    p_grad_o = tl.make_block_ptr(grad_o + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_barv = tl.make_block_ptr(barv + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))

    grad_o_tile_fp32 = tl.load(p_grad_o, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    b_o = tl.load(p_o, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    b_barv_fp32 = tl.load(p_barv, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    b_t = tl.sum(grad_o_tile_fp32 * b_o, axis=1, keep_dims=True)
    b_b = tl.sum(grad_o_tile_fp32 * b_barv_fp32, axis=1, keep_dims=True)

    p_t = tl.make_block_ptr(delta_t + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))
    p_b = tl.make_block_ptr(delta_b + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))
    tl.store(p_t, b_t, boundary_check=(0, 1))
    tl.store(p_b, b_b, boundary_check=(0, 1))

    grad_o_tile = grad_o_tile_fp32.to(p_grad_o.dtype.element_ty)

    # ----------------------------------------------------------------
    # 步骤 2: 主 DQR 阶段逻辑
    # ----------------------------------------------------------------
    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_r = tl.make_block_ptr(r + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_d1 = tl.make_block_ptr(d1 + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))
    p_bart = tl.make_block_ptr(bart + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))
    p_m = tl.make_block_ptr(m + bos * HQ + i_hq, (T, 1), (HQ, 1), (row_offset, 0), (BT, 1), (1, 0))
    
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (FIRST_COL_BLOCK * BS, 0), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * K, (T, K), (H * K, 1), (FIRST_COL_BLOCK * BS, 0), (BS, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_r = tl.load(p_r, boundary_check=(0, 1), padding_option="zero")
    b_d1 = tl.load(p_d1, boundary_check=(0, 1), padding_option="zero")
    b_bart = tl.load(p_bart, boundary_check=(0, 1), padding_option="zero")
    b_m = tl.load(p_m, boundary_check=(0, 1), padding_option="zero")

    grad_q_acc = tl.zeros((BT, BK), dtype=tl.float32)
    grad_r_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2
    inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)

    # Phase 0
    for col_block_id in range(FIRST_COL_BLOCK, LEFT_BORDER_END):
        col_indices = col_block_id * BS + tl.arange(0, BS)
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        mask = (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1) & row_mask & (col_indices[None, :] < T)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        w = exp2(qk - b_m)
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        p = w * inv_d1
        bart_minus_rk = b_bart - rk
        
        delta = a - b_b
        grad_q_acc = tl.dot(
            (p * (a - b_t + (b_bart - rk) * delta)).to(b_k.dtype), 
            b_k, 
            out_dtype=tl.float32, 
            acc=grad_q_acc
        )
        grad_r_acc = tl.dot(
            (-p * delta).to(b_k.dtype), 
            b_k, 
            out_dtype=tl.float32, 
            acc=grad_r_acc
        )
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (BS, 0))

    # Phase A
    for _ in range(SAFE_MIDDLE_START, NUM_SAFE_BLOCKS):
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        w = exp2(qk - b_m)
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        p = w * inv_d1
        bart_minus_rk = b_bart - rk
        
        delta = a - b_b
        grad_q_acc = tl.dot(
            (p * (a - b_t + (b_bart - rk) * delta)).to(b_k.dtype), 
            b_k, 
            out_dtype=tl.float32, 
            acc=grad_q_acc
        )
        grad_r_acc = tl.dot(
            (-p * delta).to(b_k.dtype), 
            b_k, 
            out_dtype=tl.float32, 
            acc=grad_r_acc
        )
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (BS, 0))

    # Phase B
    for col_block_id in range(RIGHT_BORDER_START, NUM_TOTAL_BLOCKS):
        col_indices = col_block_id * BS + tl.arange(0, BS)
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        if WINDOW_SIZE_LEFT >= 0:
            mask = (row_indices[:, None] >= col_indices[None, :]) & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1) & row_mask & (col_indices[None, :] < T)
        else:
            mask = (row_indices[:, None] >= col_indices[None, :]) & row_mask & (col_indices[None, :] < T)
            
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        w = exp2(qk - b_m)
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        p = w * inv_d1
        bart_minus_rk = b_bart - rk
        
        delta = a - b_b
        grad_q_acc = tl.dot(
            (p * (a - b_t + (b_bart - rk) * delta)).to(b_k.dtype), 
            b_k, 
            out_dtype=tl.float32, 
            acc=grad_q_acc
        )
        grad_r_acc = tl.dot(
            (-p * delta).to(b_k.dtype), 
            b_k, 
            out_dtype=tl.float32, 
            acc=grad_r_acc
        )
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (BS, 0))

    grad_q_acc = scale * grad_q_acc
    p_grad_q = tl.make_block_ptr(grad_q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    p_grad_r = tl.make_block_ptr(grad_r + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (row_offset, 0), (BT, BK), (1, 0))
    
    tl.store(p_grad_q, grad_q_acc.to(p_grad_q.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_grad_r, grad_r_acc.to(p_grad_r.dtype.element_ty), boundary_check=(0, 1))
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=PARALLAX_AUTOTUNE_CONFIGS,
    key=['T', 'HQ', 'G', 'K', 'BK', 'WINDOW_SIZE_LEFT', 'IS_VARLEN'],
    reset_to_zero=['grad_k', 'grad_v'],
    prune_configs_by={'early_config_prune': parallax_prune_configs},
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def improve_parallax_bwd_kernel_dkv(
    q,
    r,
    k,
    v,
    d1,
    bart,
    m,
    delta_t,
    delta_b,
    grad_o,
    grad_k,
    grad_v,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

# ✅ 修复
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        tile_idx = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)
        tile_idx = i_t
    RCP_LN2: tl.constexpr = 1.4426950216

    col_offset = tile_idx * BS  # ✅
    col_indices = col_offset + tl.arange(0, BS)

    start_row_block = col_offset // BT
    start_row_offset = start_row_block * BT

    num_row_blocks_qbound = tl.cdiv(T, BT)
    if WINDOW_SIZE_LEFT >= 0:
        last_row_window = tl.cdiv(col_offset + BS + WINDOW_SIZE_LEFT - 1, BT)
        num_row_blocks = tl.minimum(num_row_blocks_qbound, last_row_window)
        WINDOW_SAFE_END = (col_offset + WINDOW_SIZE_LEFT) // BT
    else:
        num_row_blocks = num_row_blocks_qbound
        WINDOW_SAFE_END = num_row_blocks

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (start_row_offset, 0), (BT, BK), (1, 0))
    p_r = tl.make_block_ptr(r + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (start_row_offset, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (col_offset, 0), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * K, (T, K), (H * K, 1), (col_offset, 0), (BS, BK), (1, 0))
    p_d1 = tl.make_block_ptr(d1 + bos * HQ + i_hq, (T, 1), (HQ, 1), (start_row_offset, 0), (BT, 1), (1, 0))
    p_bart = tl.make_block_ptr(bart + bos * HQ + i_hq, (T, 1), (HQ, 1), (start_row_offset, 0), (BT, 1), (1, 0))
    p_m = tl.make_block_ptr(m + bos * HQ + i_hq, (T, 1), (HQ, 1), (start_row_offset, 0), (BT, 1), (1, 0))
    p_t = tl.make_block_ptr(delta_t + bos * HQ + i_hq, (T, 1), (HQ, 1), (start_row_offset, 0), (BT, 1), (1, 0))
    p_b = tl.make_block_ptr(delta_b + bos * HQ + i_hq, (T, 1), (HQ, 1), (start_row_offset, 0), (BT, 1), (1, 0))
    p_grad_o = tl.make_block_ptr(grad_o + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (start_row_offset, 0), (BT, BK), (1, 0))
    p_grad_k = tl.make_block_ptr(grad_k + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (col_offset, 0), (BS, BK), (1, 0))
    p_grad_v = tl.make_block_ptr(grad_v + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (col_offset, 0), (BS, BK), (1, 0))

    b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
    grad_k_acc = tl.zeros((BS, BK), dtype=tl.float32)
    grad_v_acc = tl.zeros((BS, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    first_safe_row_block = tl.cdiv(col_offset + BS, BT)
    SAFE_MIDDLE_END = tl.minimum(WINDOW_SAFE_END, num_row_blocks)
    WINDOW_BORDER_START = tl.maximum(first_safe_row_block, WINDOW_SAFE_END)

    # Phase A: causal-border row blocks.
    causal_end = tl.minimum(first_safe_row_block, num_row_blocks)
    for row_block_id in range(start_row_block, causal_end):
        row_offset = row_block_id * BT
        row_indices = row_offset + tl.arange(0, BT)
        row_mask = row_indices[:, None] < T
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_r = tl.load(p_r, boundary_check=(0, 1), padding_option="zero")
        b_d1 = tl.load(p_d1, boundary_check=(0, 1), padding_option="zero")
        b_bart = tl.load(p_bart, boundary_check=(0, 1), padding_option="zero")
        b_m = tl.load(p_m, boundary_check=(0, 1), padding_option="zero")
        b_t = tl.load(p_t, boundary_check=(0, 1), padding_option="zero")
        b_b = tl.load(p_b, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
        if WINDOW_SIZE_LEFT >= 0:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                & row_mask
                & (col_indices[None, :] < T)
            )
        else:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & row_mask
                & (col_indices[None, :] < T)
            )
        qk = tl.where(mask, qk, -float("inf"))
        w = exp2(qk - b_m)
        p = w * inv_d1
        grad_o_tile = tl.load(p_grad_o, boundary_check=(0, 1), padding_option="zero")

        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        bart_minus_rk = b_bart - rk
        grad_k_acc = tl.dot(
            tl.trans(p * (a - b_t + bart_minus_rk * delta) * scale).to(b_q.dtype), 
            b_q, 
            out_dtype=tl.float32, 
            acc=grad_k_acc
        )
        grad_k_acc = tl.dot(
            tl.trans(-p * delta).to(b_r.dtype), 
            b_r, 
            out_dtype=tl.float32, 
            acc=grad_k_acc
        )        
        grad_v_acc = tl.dot(
            tl.trans(p * (1.0 + bart_minus_rk)).to(grad_o_tile.dtype), 
            grad_o_tile, 
            out_dtype=tl.float32, 
            acc=grad_v_acc
        )

        p_q = tl.advance(p_q, (BT, 0))
        p_r = tl.advance(p_r, (BT, 0))
        p_d1 = tl.advance(p_d1, (BT, 0))
        p_bart = tl.advance(p_bart, (BT, 0))
        p_m = tl.advance(p_m, (BT, 0))
        p_t = tl.advance(p_t, (BT, 0))
        p_b = tl.advance(p_b, (BT, 0))
        p_grad_o = tl.advance(p_grad_o, (BT, 0))

    # Phase B: safe row blocks (no causal/col/window mask).
    safe_b_start = tl.maximum(first_safe_row_block, start_row_block)
    for row_block_id in range(safe_b_start, SAFE_MIDDLE_END):
        row_offset = row_block_id * BT
        row_indices = row_offset + tl.arange(0, BT)
        row_mask = row_indices[:, None] < T
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_r = tl.load(p_r, boundary_check=(0, 1), padding_option="zero")
        b_d1 = tl.load(p_d1, boundary_check=(0, 1), padding_option="zero")
        b_bart = tl.load(p_bart, boundary_check=(0, 1), padding_option="zero")
        b_m = tl.load(p_m, boundary_check=(0, 1), padding_option="zero")
        b_t = tl.load(p_t, boundary_check=(0, 1), padding_option="zero")
        b_b = tl.load(p_b, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
        w = exp2(qk - b_m)
        p = w * inv_d1
        grad_o_tile = tl.load(p_grad_o, boundary_check=(0, 1), padding_option="zero")

        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        bart_minus_rk = b_bart - rk
        grad_k_acc = tl.dot(
            tl.trans(p * (a - b_t + bart_minus_rk * delta) * scale).to(b_q.dtype), 
            b_q, 
            out_dtype=tl.float32, 
            acc=grad_k_acc
        )
        grad_k_acc = tl.dot(
            tl.trans(-p * delta).to(b_r.dtype), 
            b_r, 
            out_dtype=tl.float32, 
            acc=grad_k_acc
        )        
        grad_v_acc = tl.dot(
            tl.trans(p * (1.0 + bart_minus_rk)).to(grad_o_tile.dtype), 
            grad_o_tile, 
            out_dtype=tl.float32, 
            acc=grad_v_acc
        )

        p_q = tl.advance(p_q, (BT, 0))
        p_r = tl.advance(p_r, (BT, 0))
        p_d1 = tl.advance(p_d1, (BT, 0))
        p_bart = tl.advance(p_bart, (BT, 0))
        p_m = tl.advance(p_m, (BT, 0))
        p_t = tl.advance(p_t, (BT, 0))
        p_b = tl.advance(p_b, (BT, 0))
        p_grad_o = tl.advance(p_grad_o, (BT, 0))

    # Phase C: window-border row blocks (SWA only).
    window_border_start = tl.maximum(WINDOW_BORDER_START, start_row_block)
    for row_block_id in range(window_border_start, num_row_blocks):
        row_offset = row_block_id * BT
        row_indices = row_offset + tl.arange(0, BT)
        row_mask = row_indices[:, None] < T
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_r = tl.load(p_r, boundary_check=(0, 1), padding_option="zero")
        b_d1 = tl.load(p_d1, boundary_check=(0, 1), padding_option="zero")
        b_bart = tl.load(p_bart, boundary_check=(0, 1), padding_option="zero")
        b_m = tl.load(p_m, boundary_check=(0, 1), padding_option="zero")
        b_t = tl.load(p_t, boundary_check=(0, 1), padding_option="zero")
        b_b = tl.load(p_b, boundary_check=(0, 1), padding_option="zero")

        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
        mask = (
            (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
            & row_mask
            & (col_indices[None, :] < T)
        )
        qk = tl.where(mask, qk, -float("inf"))
        w = exp2(qk - b_m)
        p = w * inv_d1
        grad_o_tile = tl.load(p_grad_o, boundary_check=(0, 1), padding_option="zero")

        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        bart_minus_rk = b_bart - rk
        grad_k_acc = tl.dot(
            tl.trans(p * (a - b_t + bart_minus_rk * delta) * scale).to(b_q.dtype), 
            b_q, 
            out_dtype=tl.float32, 
            acc=grad_k_acc
        )
        grad_k_acc = tl.dot(
            tl.trans(-p * delta).to(b_r.dtype), 
            b_r, 
            out_dtype=tl.float32, 
            acc=grad_k_acc
        )        
        grad_v_acc = tl.dot(
            tl.trans(p * (1.0 + bart_minus_rk)).to(grad_o_tile.dtype), 
            grad_o_tile, 
            out_dtype=tl.float32, 
            acc=grad_v_acc
        )
        p_q = tl.advance(p_q, (BT, 0))
        p_r = tl.advance(p_r, (BT, 0))
        p_d1 = tl.advance(p_d1, (BT, 0))
        p_bart = tl.advance(p_bart, (BT, 0))
        p_m = tl.advance(p_m, (BT, 0))
        p_t = tl.advance(p_t, (BT, 0))
        p_b = tl.advance(p_b, (BT, 0))
        p_grad_o = tl.advance(p_grad_o, (BT, 0))

    tl.store(p_grad_k, grad_k_acc.to(p_grad_k.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_grad_v, grad_v_acc.to(p_grad_v.dtype.element_ty), boundary_check=(0, 1))
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def improve_parallax_bwd_kernel_dkv_varlen(
    q, r, k, v, d1, bart, m, delta_t, delta_b, grad_o, grad_k, grad_v,
    scale, cu_seqlens, chunk_indices, T,
    HQ: tl.constexpr, H: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

# ✅ 修复
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        tile_idx = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)
        tile_idx = i_t
    RCP_LN2: tl.constexpr = 1.4426950216

    col_offset = tile_idx * BS  # ✅
    col_indices = col_offset + tl.arange(0, BS)

    start_row_block = col_offset // BT
    start_row_offset = start_row_block * BT

    num_row_blocks_qbound = tl.cdiv(T, BT)
    if WINDOW_SIZE_LEFT >= 0:
        last_row_window = tl.cdiv(col_offset + BS + WINDOW_SIZE_LEFT - 1, BT)
        num_row_blocks = tl.minimum(num_row_blocks_qbound, last_row_window)
        WINDOW_SAFE_END = (col_offset + WINDOW_SIZE_LEFT) // BT
    else:
        num_row_blocks = num_row_blocks_qbound
        WINDOW_SAFE_END = num_row_blocks

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (start_row_offset, 0), (BT, BK), (1, 0))
    p_r = tl.make_block_ptr(r + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (start_row_offset, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (col_offset, 0), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * K, (T, K), (H * K, 1), (col_offset, 0), (BS, BK), (1, 0))
    p_d1 = tl.make_block_ptr(d1 + bos * HQ + i_hq, (T, 1), (HQ, 1), (start_row_offset, 0), (BT, 1), (1, 0))
    p_bart = tl.make_block_ptr(bart + bos * HQ + i_hq, (T, 1), (HQ, 1), (start_row_offset, 0), (BT, 1), (1, 0))
    p_m = tl.make_block_ptr(m + bos * HQ + i_hq, (T, 1), (HQ, 1), (start_row_offset, 0), (BT, 1), (1, 0))
    p_t = tl.make_block_ptr(delta_t + bos * HQ + i_hq, (T, 1), (HQ, 1), (start_row_offset, 0), (BT, 1), (1, 0))
    p_b = tl.make_block_ptr(delta_b + bos * HQ + i_hq, (T, 1), (HQ, 1), (start_row_offset, 0), (BT, 1), (1, 0))
    p_grad_o = tl.make_block_ptr(grad_o + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (start_row_offset, 0), (BT, BK), (1, 0))
    p_grad_k = tl.make_block_ptr(grad_k + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (col_offset, 0), (BS, BK), (1, 0))
    p_grad_v = tl.make_block_ptr(grad_v + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (col_offset, 0), (BS, BK), (1, 0))

    b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
    grad_k_acc = tl.zeros((BS, BK), dtype=tl.float32)
    grad_v_acc = tl.zeros((BS, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    first_safe_row_block = tl.cdiv(col_offset + BS, BT)
    SAFE_MIDDLE_END = tl.minimum(WINDOW_SAFE_END, num_row_blocks)
    WINDOW_BORDER_START = tl.maximum(first_safe_row_block, WINDOW_SAFE_END)

    # Phase A: causal-border row blocks.
    causal_end = tl.minimum(first_safe_row_block, num_row_blocks)
    for row_block_id in range(start_row_block, causal_end):
        row_offset = row_block_id * BT
        row_indices = row_offset + tl.arange(0, BT)
        row_mask = row_indices[:, None] < T
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_r = tl.load(p_r, boundary_check=(0, 1), padding_option="zero")
        b_d1 = tl.load(p_d1, boundary_check=(0, 1), padding_option="zero")
        b_bart = tl.load(p_bart, boundary_check=(0, 1), padding_option="zero")
        b_m = tl.load(p_m, boundary_check=(0, 1), padding_option="zero")
        b_t = tl.load(p_t, boundary_check=(0, 1), padding_option="zero")
        b_b = tl.load(p_b, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
        if WINDOW_SIZE_LEFT >= 0:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                & row_mask
                & (col_indices[None, :] < T)
            )
        else:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & row_mask
                & (col_indices[None, :] < T)
            )
        qk = tl.where(mask, qk, -float("inf"))
        w = exp2(qk - b_m)
        p = w * inv_d1
        grad_o_tile = tl.load(p_grad_o, boundary_check=(0, 1), padding_option="zero")

        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        bart_minus_rk = b_bart - rk
        grad_k_acc = tl.dot(
            tl.trans(p * (a - b_t + bart_minus_rk * delta) * scale).to(b_q.dtype), 
            b_q, 
            out_dtype=tl.float32, 
            acc=grad_k_acc
        )
        grad_k_acc = tl.dot(
            tl.trans(-p * delta).to(b_r.dtype), 
            b_r, 
            out_dtype=tl.float32, 
            acc=grad_k_acc
        )        
        grad_v_acc = tl.dot(
            tl.trans(p * (1.0 + bart_minus_rk)).to(grad_o_tile.dtype), 
            grad_o_tile, 
            out_dtype=tl.float32, 
            acc=grad_v_acc
        )

        p_q = tl.advance(p_q, (BT, 0))
        p_r = tl.advance(p_r, (BT, 0))
        p_d1 = tl.advance(p_d1, (BT, 0))
        p_bart = tl.advance(p_bart, (BT, 0))
        p_m = tl.advance(p_m, (BT, 0))
        p_t = tl.advance(p_t, (BT, 0))
        p_b = tl.advance(p_b, (BT, 0))
        p_grad_o = tl.advance(p_grad_o, (BT, 0))

    # Phase B: safe row blocks (no causal/col/window mask).
    safe_b_start = tl.maximum(first_safe_row_block, start_row_block)
    for row_block_id in range(safe_b_start, SAFE_MIDDLE_END):
        row_offset = row_block_id * BT
        row_indices = row_offset + tl.arange(0, BT)
        row_mask = row_indices[:, None] < T
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_r = tl.load(p_r, boundary_check=(0, 1), padding_option="zero")
        b_d1 = tl.load(p_d1, boundary_check=(0, 1), padding_option="zero")
        b_bart = tl.load(p_bart, boundary_check=(0, 1), padding_option="zero")
        b_m = tl.load(p_m, boundary_check=(0, 1), padding_option="zero")
        b_t = tl.load(p_t, boundary_check=(0, 1), padding_option="zero")
        b_b = tl.load(p_b, boundary_check=(0, 1), padding_option="zero")

        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
        w = exp2(qk - b_m)
        p = w * inv_d1
        grad_o_tile = tl.load(p_grad_o, boundary_check=(0, 1), padding_option="zero")

        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        bart_minus_rk = b_bart - rk
        grad_k_acc = tl.dot(
            tl.trans(p * (a - b_t + bart_minus_rk * delta) * scale).to(b_q.dtype), 
            b_q, 
            out_dtype=tl.float32, 
            acc=grad_k_acc
        )
        grad_k_acc = tl.dot(
            tl.trans(-p * delta).to(b_r.dtype), 
            b_r, 
            out_dtype=tl.float32, 
            acc=grad_k_acc
        )        
        grad_v_acc = tl.dot(
            tl.trans(p * (1.0 + bart_minus_rk)).to(grad_o_tile.dtype), 
            grad_o_tile, 
            out_dtype=tl.float32, 
            acc=grad_v_acc
        )

        p_q = tl.advance(p_q, (BT, 0))
        p_r = tl.advance(p_r, (BT, 0))
        p_d1 = tl.advance(p_d1, (BT, 0))
        p_bart = tl.advance(p_bart, (BT, 0))
        p_m = tl.advance(p_m, (BT, 0))
        p_t = tl.advance(p_t, (BT, 0))
        p_b = tl.advance(p_b, (BT, 0))
        p_grad_o = tl.advance(p_grad_o, (BT, 0))

    # Phase C: window-border row blocks (SWA only).
    window_border_start = tl.maximum(WINDOW_BORDER_START, start_row_block)
    for row_block_id in range(window_border_start, num_row_blocks):
        row_offset = row_block_id * BT
        row_indices = row_offset + tl.arange(0, BT)
        row_mask = row_indices[:, None] < T
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_r = tl.load(p_r, boundary_check=(0, 1), padding_option="zero")
        b_d1 = tl.load(p_d1, boundary_check=(0, 1), padding_option="zero")
        b_bart = tl.load(p_bart, boundary_check=(0, 1), padding_option="zero")
        b_m = tl.load(p_m, boundary_check=(0, 1), padding_option="zero")
        b_t = tl.load(p_t, boundary_check=(0, 1), padding_option="zero")
        b_b = tl.load(p_b, boundary_check=(0, 1), padding_option="zero")

        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
        mask = (
            (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
            & row_mask
            & (col_indices[None, :] < T)
        )
        qk = tl.where(mask, qk, -float("inf"))
        w = exp2(qk - b_m)
        p = w * inv_d1
        grad_o_tile = tl.load(p_grad_o, boundary_check=(0, 1), padding_option="zero")

        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        bart_minus_rk = b_bart - rk
        grad_k_acc = tl.dot(
            tl.trans(p * (a - b_t + bart_minus_rk * delta) * scale).to(b_q.dtype), 
            b_q, 
            out_dtype=tl.float32, 
            acc=grad_k_acc
        )
        grad_k_acc = tl.dot(
            tl.trans(-p * delta).to(b_r.dtype), 
            b_r, 
            out_dtype=tl.float32, 
            acc=grad_k_acc
        )        
        grad_v_acc = tl.dot(
            tl.trans(p * (1.0 + bart_minus_rk)).to(grad_o_tile.dtype), 
            grad_o_tile, 
            out_dtype=tl.float32, 
            acc=grad_v_acc
        )

        p_q = tl.advance(p_q, (BT, 0))
        p_r = tl.advance(p_r, (BT, 0))
        p_d1 = tl.advance(p_d1, (BT, 0))
        p_bart = tl.advance(p_bart, (BT, 0))
        p_m = tl.advance(p_m, (BT, 0))
        p_t = tl.advance(p_t, (BT, 0))
        p_b = tl.advance(p_b, (BT, 0))
        p_grad_o = tl.advance(p_grad_o, (BT, 0))

    tl.store(p_grad_k, grad_k_acc.to(p_grad_k.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_grad_v, grad_v_acc.to(p_grad_v.dtype.element_ty), boundary_check=(0, 1))


def improve_parallax_fwd(q, r, k, v, scale, cu_seqlens=None, chunk_indices=None, window_size_left=-1):
    """Parallax forward (Triton). `(B, T, HQ, D)` / packed `(1, T_total, HQ, D)` inputs.

    Returns `(o, barv, d1, bart, m)`: `o`/`barv` in the input dtype and layout;
    `d1`/`bart`/`m` are fp32 per-(position, query-head) scalars `(B, T, HQ)`.
    """
    B, T, HQ, K = q.shape
    H = k.shape[2]
    G = HQ // H
    BK = triton.next_power_of_2(K)
    BT = _block_size(K, q.device.index)
    o = torch.empty_like(q)
    barv = torch.empty_like(q)
    d1 = torch.empty((B, T, HQ), device=q.device, dtype=torch.float32)
    bart = torch.empty((B, T, HQ), device=q.device, dtype=torch.float32)
    m = torch.empty((B, T, HQ), device=q.device, dtype=torch.float32)

    NT = triton.cdiv(T, BT) if cu_seqlens is None else None
    is_short_seq = (T <= BT) and (cu_seqlens is None)

    if is_short_seq:
        # 【小序列路线】：NT 固定为 1，使用极简内核
        grid = (1, B * HQ)
        improve_parallax_fwd_kernel_short[grid](
            q, r, k, v, o, barv, d1, bart, m,
            scale, cu_seqlens, chunk_indices, T,
            HQ=HQ, H=H, G=G, K=K, BK=BK,
            WINDOW_SIZE_LEFT=window_size_left, BT=BT,
            num_warps=4,
            num_stages=1
        )
    elif cu_seqlens is None and NT <= 4:
        grid = (NT, B * HQ)
        improve_parallax_fwd_kernel_short_multi[grid](
            q, r, k, v, o, barv, d1, bart, m, scale, T,
            HQ=HQ, H=H, G=G, K=K, BK=BK,
            WINDOW_SIZE_LEFT=window_size_left, BT=BT, NT=NT,
            num_warps=4 if BT <= 64 else 8,
            num_stages=2,
        )
    else:
        # varlen 场景：chunk_indices 的 tile 划分必须与 kernel 的 BT 一致
        if cu_seqlens is not None:
            BS = BT
            NT = len(chunk_indices)
            grid = (NT, B * HQ)
            improve_parallax_fwd_kernel_varlen[grid](  # 无 autotune
                q, r, k, v, o, barv, d1, bart, m,
                scale, cu_seqlens, chunk_indices, T,
                HQ=HQ, H=H, G=G, K=K, BK=BK,
                WINDOW_SIZE_LEFT=window_size_left, BT=BT, BS=BS,
                num_warps=4,
                num_stages=2,
            )
        else:
            improve_parallax_fwd_kernel[lambda meta: (triton.cdiv(T, meta['BT']), B * HQ)](
                q, r, k, v, o, barv, d1, bart, m,
                scale, cu_seqlens, chunk_indices, T,
                HQ=HQ, H=H, G=G, K=K, BK=BK,
                WINDOW_SIZE_LEFT=window_size_left,
            )
    return o, barv, d1, bart, m

def improve_parallax_bwd(q, r, k, v, o, barv, d1, bart, m, grad_o, scale, cu_seqlens=None, chunk_indices=None, window_size_left=-1):
    """Parallax backward (Triton). Returns grads matching `q, r, k, v`."""
    B, T, HQ, K = q.shape
    H = k.shape[2]
    G = HQ // H
    BK = triton.next_power_of_2(K)
    BT = _block_size(K, q.device.index)

    grad_q = torch.empty_like(q)
    grad_r = torch.empty_like(r)
    grad_k_buf = torch.empty((B, T, HQ, K), device=q.device, dtype=q.dtype)
    grad_v_buf = torch.empty((B, T, HQ, K), device=q.device, dtype=q.dtype)

    NT = triton.cdiv(T, BT) if cu_seqlens is None else None
    is_short_seq = (T <= BT) and (cu_seqlens is None)

    if is_short_seq:
        # === 反向传播快车道：Launch 1 次 ===
        grid = (1, B * HQ)
        improve_parallax_bwd_kernel_short[grid](
            q, r, k, v, o, barv, d1, bart, m, grad_o,
            grad_q, grad_r, grad_k_buf, grad_v_buf, scale,
            cu_seqlens, chunk_indices, T,
            HQ=HQ, H=H, G=G, K=K, BK=BK,
            WINDOW_SIZE_LEFT=window_size_left, BT=BT,
            num_warps=4,
            num_stages=1
        )
    elif cu_seqlens is None and NT <= 32:
        grid = (NT, B * HQ)
        improve_parallax_bwd_kernel_dqr_short_multi[grid](
            q, r, k, v, o, barv, d1, bart, m, grad_o,
            grad_q, grad_r, scale, T,
            HQ=HQ, H=H, G=G, K=K, BK=BK,
            WINDOW_SIZE_LEFT=window_size_left, BT=BT, NT=NT,
            num_warps=4 if BT <= 64 else 8,
            num_stages=2,
        )
        improve_parallax_bwd_kernel_dkv_short_multi[grid](
            q, r, k, v, o, barv, d1, bart, m, grad_o,
            grad_k_buf, grad_v_buf, scale, T,
            HQ=HQ, H=H, G=G, K=K, BK=BK,
            WINDOW_SIZE_LEFT=window_size_left, BT=BT, NT=NT,
            num_warps=4 if BT <= 64 else 8,
            num_stages=2,
        )
    else:
        delta_t = torch.empty((B, T, HQ), device=q.device, dtype=torch.float32)
        delta_b = torch.empty((B, T, HQ), device=q.device, dtype=torch.float32)

        # ✅ 修复：varlen 场景下强制使用与 chunk_indices 一致的 block size
        if cu_seqlens is not None:
            BS = BT
            NT = len(chunk_indices)
            grid = (NT, B * HQ)
            
            improve_parallax_bwd_kernel_dqr_varlen_fused[grid]( 
                q, r, k, v,o, barv, d1, bart, m, delta_t, delta_b, grad_o, grad_q, grad_r,
                scale, cu_seqlens, chunk_indices, T,
                HQ=HQ, H=H, G=G, K=K, BK=BK,
                WINDOW_SIZE_LEFT=window_size_left, BT=BT, BS=BS,
                num_warps=4,
                num_stages=2,
            )
            improve_parallax_bwd_kernel_dkv_varlen[grid]( 
                q, r, k, v, d1, bart, m, delta_t, delta_b, grad_o, grad_k_buf, grad_v_buf,
                scale, cu_seqlens, chunk_indices, T,
                HQ=HQ, H=H, G=G, K=K, BK=BK,
                WINDOW_SIZE_LEFT=window_size_left, BT=BT, BS=BS,
                num_warps=4,
                num_stages=2,
            )
        else:
            improve_parallax_bwd_kernel_dqr_fused[lambda meta: (triton.cdiv(T, meta['BT']), B * HQ)](
                q, r, k, v,o, barv, d1, bart, m, delta_t, delta_b, grad_o, grad_q, grad_r,
                scale, cu_seqlens, chunk_indices, T,
                HQ=HQ, H=H, G=G, K=K, BK=BK,
                WINDOW_SIZE_LEFT=window_size_left,
            )
            improve_parallax_bwd_kernel_dkv[lambda meta: (triton.cdiv(T, meta['BS']), B * HQ)](
                q, r, k, v, d1, bart, m, delta_t, delta_b, grad_o, grad_k_buf, grad_v_buf,
                scale, cu_seqlens, chunk_indices, T,
                HQ=HQ, H=H, G=G, K=K, BK=BK,
                WINDOW_SIZE_LEFT=window_size_left,
            )

    if G == 1:
        grad_k = grad_k_buf
        grad_v = grad_v_buf
    else:
        grad_k = reduce(grad_k_buf, 'b t (h g) k -> b t h k', g=G, reduction='sum')
        grad_v = reduce(grad_v_buf, 'b t (h g) k -> b t h k', g=G, reduction='sum')
    return grad_q, grad_r, grad_k, grad_v
class ParallaxFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, r, k, v, scale, window_size_left, cu_seqlens):
        # ✅ 修复：统一处理 cu_seqlens 和 chunk_indices 的 dtype 和对齐
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.contiguous().to(torch.int32)
            chunk_indices = prepare_chunk_indices(cu_seqlens, _block_size(q.shape[-1], q.device.index))
            if chunk_indices is not None:
                chunk_indices = chunk_indices.contiguous().to(torch.int32)
        else:
            chunk_indices = None
        # ...
        o, barv, d1, bart, m = improve_parallax_fwd(q, r, k, v, scale, cu_seqlens, chunk_indices, window_size_left)
        ctx.save_for_backward(q, r, k, v, o, barv, d1, bart, m)
        ctx.scale = scale
        ctx.window_size_left = window_size_left
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_indices = chunk_indices
        return o

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do):
        q, r, k, v, o, barv, d1, bart, m = ctx.saved_tensors
        # ✅ ctx.cu_seqlens 在 forward 中已被转换为 int32，直接复用
        gq, gr, gk, gv = improve_parallax_bwd(
            q, r, k, v, o, barv, d1, bart, m, do,
            ctx.scale, ctx.cu_seqlens, ctx.chunk_indices, ctx.window_size_left,
        )
        return gq.to(q), gr.to(r), gk.to(k), gv.to(v), None, None, None

def improve_parallax(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
) -> torch.Tensor:
    r"""
    Causal Parallax (parameterized local linear attention) with autograd,
    backed by Triton kernels. See `fla.ops.parallax.naive.naive_parallax` for
    the reference math.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, D]`.
        r (torch.Tensor):
            secondary queries of shape `[B, T, HQ, D]` (same shape as `q`). NOTE:
            `r` is *not* scaled by `scale`; pass it un-pre-scaled.
        k (torch.Tensor):
            keys of shape `[B, T, H, D]`. GQA is applied when `HQ` is divisible by `H`.
        v (torch.Tensor):
            values of shape `[B, T, H, D]`.
        scale (float, Optional):
            Scale applied to the `q @ k^T` logits only. If `None`, defaults to `1 / sqrt(D)`.
            Default: `None`.
        window_size (int, Optional):
            Sliding-window length. If provided, each query at position `i` only attends to
            keys in `[i - window_size + 1, i]`. If `None`, full causal attention is used.
            Default: `None`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` for variable-length training
            (FlashAttention convention). The batch size must be 1 when packing. Default: `None`.

    Returns:
        o (torch.Tensor):
            output of shape `[B, T, HQ, D]`.
    """
    if 'head_first' in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"improve_parallax requires bf16 or fp16 inputs, got q.dtype={q.dtype}")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
            f"Please flatten variable-length inputs before processing.",
        )
    # The kernel keeps cols [i - W + 1, i] (W keys total, diagonal included),
    # matching FLA's `window_size=W` semantics exactly (no off-by-one).
    window_size_left = -1 if window_size is None else window_size
    return ParallaxFunction.apply(q, r, k, v, float(scale), window_size_left, cu_seqlens)
