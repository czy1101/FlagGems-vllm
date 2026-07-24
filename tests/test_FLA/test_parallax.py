# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os
import logging
import warnings
from functools import lru_cache

import pytest
import torch
import triton
import triton.language as tl
from src.flaggems_vllm.ops.FLA.improve_parallax import improve_parallax

def _get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except Exception:
        return 'cpu'

device = _get_available_device() if _get_available_device() != 'hip' else 'cuda'
IS_NVIDIA = device == 'cuda'
IS_NVIDIA_BLACKWELL = (
    IS_NVIDIA
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] in (10, 12)
)

@lru_cache(maxsize=1)
def _get_max_shared_mem(device_idx: int = 0) -> int:
    try:
        return triton.runtime.driver.active.utils.get_device_properties(device_idx)['max_shared_mem']
    except Exception:
        return -1

def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        max_shared_memory = _get_max_shared_mem(tensor_idx)
        shared_mem_map = {
            'ADA': 101376,
            'AMPERE': 166912,
            'HOPPER': 232448,
            'DEFAULT': 102400,
        }
        shared_mem = shared_mem_map.get(arch.upper(), shared_mem_map['DEFAULT'])
        return max_shared_memory >= shared_mem
    except Exception:
        return False

def _block_size(head_dim: int, device_index: int) -> int:
    # A single square tile size shared by all kernels so one `chunk_indices`
    # (built host-side for varlen) matches every grid. Kept modest to bound the
    # fp32 accumulator footprint Parallax carries (barv/Rv/grad accumulators).
    if check_shared_mem('hopper', device_index) and not IS_NVIDIA_BLACKWELL and head_dim <= 64:
        return 128
    return 64

FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"
FLA_CACHE_RESULTS = os.getenv('FLA_CACHE_RESULTS', '1') == '1'

FLA_DISABLE_TENSOR_CACHE = os.getenv('FLA_DISABLE_TENSOR_CACHE', '0') == '1'
try:
    FLA_TENSOR_CACHE_SIZE = int(os.getenv('FLA_TENSOR_CACHE_SIZE', "4"))
except ValueError:
    FLA_TENSOR_CACHE_SIZE = 4

logger = logging.getLogger(__name__)

def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()

def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)

def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    error_rate = get_err_ratio(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {error_rate:.6f}"
    logger.info(msg)
    if abs_atol <= err_atol:
        return
    assert not torch.isnan(ref).any(), f"{prefix}: NaN detected in ref"
    assert not torch.isnan(tri).any(), f"{prefix}: NaN detected in tri"
    if warning or (FLA_CI_ENV and (error_rate < 0.01 or abs_atol <= 0.3)):
        if error_rate > ratio:
            warnings.warn(msg)
    else:
        assert error_rate < ratio, msg
@triton.jit
def exp2(x): return tl.math.exp2(x.to(tl.float32))
@triton.jit(do_not_specialize=['Sq', 'Skv'])
def parallax_decode_kernel(
    q,
    r,
    k,
    v,
    o,
    scale,
    cache_start,
    Sq,
    Skv,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    USE_CACHE_START: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    """Forward-only Parallax over a cached KV (prefill / chunked decode).

    The ``Sq`` query tokens are the last ``Sq`` positions of a length-``Skv``
    sequence, so query row ``i`` is at absolute position ``Skv - Sq + i`` and
    attends to keys ``[0, Skv - Sq + i]`` (causal), further restricted to the
    sliding window and to ``[cache_start, Skv)`` when set. One program owns a
    ``BT``-row query block; see ``naive_parallax`` for the output formula.
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G                               # kv head shared by this q head (GQA)
    RCP_LN2: tl.constexpr = 1.4426950216

    # First valid key index (left-padding); 0 when the whole cache is valid.
    if USE_CACHE_START:
        kv_lo = tl.load(cache_start + i_b).to(tl.int32)
    else:
        kv_lo = 0

    q_off = i_t * BT
    kv_offset = Skv - Sq                          # absolute position of query row 0
    rows = q_off + tl.arange(0, BT)
    abs_q = (kv_offset + rows)[:, None]           # [BT, 1] absolute position per query row
    row_mask = (rows < Sq)[:, None]               # [BT, 1] real (non-padded) query rows

    # Restrict the key-block loop to what this query block can reach: up to the
    # causal diagonal (KV_END_BLOCK) and down to the window's left edge (else 0).
    max_abs = kv_offset + tl.minimum(Sq, q_off + BT) - 1
    KV_END_BLOCK = tl.cdiv(tl.minimum(Skv, max_abs + 1), BS)
    if WINDOW_SIZE_LEFT >= 0:
        leftmost = tl.maximum(kv_lo, kv_offset + q_off - WINDOW_SIZE_LEFT + 1)
    else:
        leftmost = kv_lo
    KV_START_BLOCK = leftmost // BS

    p_q = tl.make_block_ptr(q + (i_b * Sq * HQ + i_hq) * K, (Sq, K), (HQ * K, 1), (q_off, 0), (BT, BK), (1, 0))
    p_r = tl.make_block_ptr(r + (i_b * Sq * HQ + i_hq) * K, (Sq, K), (HQ * K, 1), (q_off, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (i_b * Skv * H + i_h) * K, (Skv, K), (H * K, 1), (KV_START_BLOCK * BS, 0), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (i_b * Skv * H + i_h) * K, (Skv, K), (H * K, 1), (KV_START_BLOCK * BS, 0), (BS, BK), (1, 0))
    p_o = tl.make_block_ptr(o + (i_b * Sq * HQ + i_hq) * K, (Sq, K), (HQ * K, 1), (q_off, 0), (BT, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_r = tl.load(p_r, boundary_check=(0, 1), padding_option="zero")
    m_acc = tl.zeros((BT, 1), dtype=tl.float32) - float("inf")
    d1_acc = tl.zeros((BT, 1), dtype=tl.float32)
    d2_acc = tl.zeros((BT, 1), dtype=tl.float32)
    barv_acc = tl.zeros((BT, BK), dtype=tl.float32)
    Rv_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    for col_block_id in range(KV_START_BLOCK, KV_END_BLOCK):
        col = (col_block_id * BS + tl.arange(0, BS))[None, :]   # [1, BS] key positions
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")   # [BS, BK]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")   # [BS, BK]
        # [BT, BS]: causal, in-cache, past left-padding; optionally inside the window.
        mask = (abs_q >= col) & row_mask & (col < Skv) & (col >= kv_lo)
        if WINDOW_SIZE_LEFT >= 0:
            mask = mask & (col >= abs_q - WINDOW_SIZE_LEFT + 1)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2   # [BT, BS], base-2 logits
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        # finite pivot so a row with no valid key yet doesn't hit exp2(-inf - -inf) = NaN
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)                           # online-softmax rescale of running state
        w = exp2(qk - safe_m)                                  # [BT, BS] = p1 (unnormalized softmax)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)    # [BT, BS] = r @ k^T (unscaled)
        wr = w * rk                                            # [BT, BS] = p2 (unnormalized)
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)    # running sum(p1)
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)   # running sum(p2)
        barv_acc = alpha * barv_acc                            # running p1 @ v
        Rv_acc = alpha * Rv_acc                                # running p2 @ v
        barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=barv_acc)
        Rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (BS, 0))

    # Rows that see no valid key (e.g. left-padded query positions) have d1 == 0;
    # emit a finite zero instead of inf/NaN so padding can't poison valid rows.
    inv_d1 = tl.where(row_mask & (d1_acc > 0.0), 1.0 / d1_acc, 0.0)
    b_barv = barv_acc * inv_d1                                 # O1 / d1
    b_bart = d2_acc * inv_d1                                   # d2 / d1
    b_o = b_barv + b_bart * b_barv - Rv_acc * inv_d1           # O1/d1 * (1 + d2/d1) - O2/d1

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

def parallax_decode(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    window_size: int | None = None,
    cache_start: torch.LongTensor | None = None,
) -> torch.Tensor:
    r"""
    Forward-only Parallax decode/prefill over a cached KV (inference; no autograd).

    The ``Sq`` query tokens are treated as the *last* ``Sq`` positions of a
    length-``Skv`` sequence: query ``i`` sits at absolute position ``Skv - Sq + i``
    and attends causally to keys ``[0, Skv - Sq + i]`` (and, with `window_size`,
    only the most recent `window_size` of them). `Sq == Skv` reduces to a full
    causal prefill; `Sq == 1` is a single decode step.

    Args:
        q (torch.Tensor):
            new queries of shape `[B, Sq, HQ, D]`.
        r (torch.Tensor):
            new secondary queries of shape `[B, Sq, HQ, D]`.
        k (torch.Tensor):
            cached keys of shape `[B, Skv, H, D]` (`Skv >= Sq`).
        v (torch.Tensor):
            cached values of shape `[B, Skv, H, D]`.
        scale (float, Optional):
            Scale applied to `q @ k^T` only. Defaults to `1 / sqrt(D)`. Default: `None`.
        window_size (int, Optional):
            Sliding-window length. Default: `None`.
        cache_start (torch.LongTensor, Optional):
            Per-batch first valid key index of shape `[B]` (to mask left-padding
            in the cache). `None` means the whole cache is valid. Default: `None`.

    Returns:
        o (torch.Tensor):
            output of shape `[B, Sq, HQ, D]`.
    """
    B, Sq, HQ, K = q.shape
    Skv, H = k.shape[1], k.shape[2]
    G = HQ // H
    if scale is None:
        scale = K ** -0.5
    window_size_left = -1 if window_size is None else window_size

    q, r, k, v = (x.contiguous() for x in (q, r, k, v))
    if cache_start is not None:
        cache_start = cache_start.to(device=q.device, dtype=torch.int32).contiguous()
    BK = triton.next_power_of_2(K)
    BT = _block_size(K, q.device.index)
    o = torch.empty_like(q)
    grid = (triton.cdiv(Sq, BT), B * HQ)
    parallax_decode_kernel[grid](
        q, r, k, v, o, float(scale), cache_start, Sq, Skv,
        HQ=HQ, H=H, G=G, K=K, BK=BK,
        WINDOW_SIZE_LEFT=window_size_left,
        USE_CACHE_START=cache_start is not None,
        BT=BT, BS=BT,
        num_warps=8, num_stages=2,
    )
    return o

@triton.heuristics({
    'USE_CACHE_START': lambda args: args['cache_start'] is not None,
})
@triton.jit(do_not_specialize=['Skv'])
def parallax_decode_one_step_kernel(
    q,
    r,
    k,
    v,
    o,
    scale,
    cache_start,
    Skv,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    USE_CACHE_START: tl.constexpr,
    BS: tl.constexpr,
):
    """Single-token Parallax decode: one query per (batch, head) over its cached KV.

    The query is the sequence's last position, so causality is implicit (it sees
    every key) and only the lower bound from the window / left-padding matters.
    The query is held as a *vector* and reduced against the cache with an online
    softmax, so there is none of the wasted-tile compute the prefill-shaped
    ``parallax_decode_kernel`` incurs at ``Sq == 1``. One program per
    (batch, head); see ``naive_parallax`` for the output formula.
    """
    i_bh = tl.program_id(0)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G                               # kv head shared by this q head (GQA)
    RCP_LN2: tl.constexpr = 1.4426950216

    # Lowest key the query attends to: left-padding start, raised to the window edge.
    kv_lo = 0
    if USE_CACHE_START:
        kv_lo = tl.load(cache_start + i_b).to(tl.int32)
    if WINDOW_SIZE_LEFT >= 0:
        kv_lo = tl.maximum(kv_lo, Skv - WINDOW_SIZE_LEFT)
    kv_lo = tl.maximum(kv_lo, 0)

    p_q = tl.make_block_ptr(q + i_bh * K, (K,), (1,), (0,), (BK,), (0,))
    p_r = tl.make_block_ptr(r + i_bh * K, (K,), (1,), (0,), (BK,), (0,))
    p_o = tl.make_block_ptr(o + i_bh * K, (K,), (1,), (0,), (BK,), (0,))
    b_q = tl.load(p_q, boundary_check=(0,), padding_option="zero").to(tl.float32)   # [BK] query vector
    b_r = tl.load(p_r, boundary_check=(0,), padding_option="zero").to(tl.float32)   # [BK] secondary query
    scale_log2 = scale * RCP_LN2

    # Running online-softmax state for the single query: pivot m, denominators
    # d1/d2 = sum(p1)/sum(p2), and unnormalized outputs o1/o2 = p1@v / p2@v.
    m = tl.full((1,), -float("inf"), dtype=tl.float32)
    d1 = tl.zeros((1,), dtype=tl.float32)
    d2 = tl.zeros((1,), dtype=tl.float32)
    o1 = tl.zeros((BK,), dtype=tl.float32)
    o2 = tl.zeros((BK,), dtype=tl.float32)

    start_block = kv_lo // BS
    p_k = tl.make_block_ptr(k + (i_b * Skv * H + i_h) * K, (Skv, K), (H * K, 1), (start_block * BS, 0), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (i_b * Skv * H + i_h) * K, (Skv, K), (H * K, 1), (start_block * BS, 0), (BS, BK), (1, 0))
    for i_s in range(start_block * BS, tl.cdiv(Skv, BS) * BS, BS):
        col = i_s + tl.arange(0, BS)
        mask = (col >= kv_lo) & (col < Skv)                          # [BS] valid keys
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")   # [BS, BK]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")   # [BS, BK]
        s1 = tl.sum(b_q[None, :] * b_k, axis=1) * scale_log2          # [BS] = scale * (q . k), base-2
        s2 = tl.sum(b_r[None, :] * b_k, axis=1)                        # [BS] = r . k (unscaled)
        s1 = tl.where(mask, s1, -float("inf"))
        m_new = tl.maximum(m, tl.max(s1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)         # finite pivot (empty cache -> 0)
        alpha = exp2(m - m_safe)                                      # rescale running state
        p1 = exp2(s1 - m_safe)                                         # [BS]
        p2 = p1 * s2                                                   # [BS]
        d1 = d1 * alpha + tl.sum(p1)
        d2 = d2 * alpha + tl.sum(p2)
        o1 = o1 * alpha + tl.sum(p1[:, None] * b_v, axis=0)           # [BK]
        o2 = o2 * alpha + tl.sum(p2[:, None] * b_v, axis=0)
        m = m_new
        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (BS, 0))

    inv_d1 = tl.where(d1 > 0.0, 1.0 / d1, 0.0)                        # 0 when no valid key (avoid NaN)
    out = o1 * inv_d1 * (1.0 + d2 * inv_d1) - o2 * inv_d1             # [BK] O1/d1*(1 + d2/d1) - O2/d1
    tl.store(p_o, out.to(p_o.dtype.element_ty), boundary_check=(0,))

def parallax_decode_one_step(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    window_size: int | None = None,
    cache_start: torch.LongTensor | None = None,
) -> torch.Tensor:
    r"""
    Single-token Parallax decode (one query per sequence over the cached KV).

    Optimized for the `Sq == 1` autoregressive step: the query is loaded as a
    vector (not a `[BT, D]` tile) and reduced against the cache with an online
    softmax, avoiding the wasted-tile compute of the prefill-shaped
    :func:`parallax_decode`. Forward-only (inference).

    Args:
        q (torch.Tensor):
            new query of shape `[B, 1, HQ, D]`.
        r (torch.Tensor):
            new secondary query of shape `[B, 1, HQ, D]`.
        k (torch.Tensor):
            cached keys of shape `[B, Skv, H, D]`.
        v (torch.Tensor):
            cached values of shape `[B, Skv, H, D]`.
        scale (float, Optional):
            Scale applied to `q @ k^T` only. Defaults to `1 / sqrt(D)`. Default: `None`.
        window_size (int, Optional):
            Sliding-window length; the query attends to the most recent `window_size` keys.
            Default: `None`.
        cache_start (torch.LongTensor, Optional):
            Per-batch first valid key index of shape `[B]` (left-padding). Default: `None`.

    Returns:
        o (torch.Tensor):
            output of shape `[B, 1, HQ, D]`.
    """
    B, Sq, HQ, K = q.shape
    if Sq != 1:
        raise ValueError(f"parallax_decode_one_step expects a single query (Sq=1), got Sq={Sq}")
    Skv, H = k.shape[1], k.shape[2]
    G = HQ // H
    if scale is None:
        scale = K ** -0.5
    window_size_left = -1 if window_size is None else window_size

    q, r, k, v = (x.contiguous() for x in (q, r, k, v))
    if cache_start is not None:
        cache_start = cache_start.to(device=q.device, dtype=torch.int32).contiguous()
    BK = triton.next_power_of_2(K)
    o = torch.empty_like(q)
    grid = (B * HQ,)
    parallax_decode_one_step_kernel[grid](
        q, r, k, v, o, float(scale), cache_start, Skv,
        HQ=HQ, H=H, G=G, K=K, BK=BK,
        WINDOW_SIZE_LEFT=window_size_left,
        USE_CACHE_START=cache_start is not None,
        BS=128,
        num_warps=4, num_stages=2,
    )
    return o

def naive_parallax(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    window_size: int | None = None,
    causal: bool = True,
) -> torch.Tensor:
    B, T, HQ, D = q.shape
    H = k.shape[2]
    G = HQ // H

    if scale is None:
        scale = D ** -0.5

    dtype = q.dtype
    q = q.float().reshape(B, T, H, G, D)
    r = r.float().reshape(B, T, H, G, D)
    k = k.float()
    v = v.float()

    # Explicit loops to avoid Blackwell GPU cuBLAS bug (CUBLAS_STATUS_INVALID_VALUE
    # on strided batched gemm). Performance is secondary for a reference impl.
    s1 = torch.empty((B, H, G, T, T), dtype=torch.float32, device=q.device)
    s2 = torch.empty((B, H, G, T, T), dtype=torch.float32, device=q.device)
    for b in range(B):
        for h in range(H):
            for g in range(G):
                s1[b, h, g] = q[b, :, h, g] @ k[b, :, h].T * scale
                s2[b, h, g] = r[b, :, h, g] @ k[b, :, h].T

    if causal:
        row_idx = torch.arange(T, device=q.device)[:, None]
        col_idx = torch.arange(T, device=q.device)[None, :]
        mask = col_idx > row_idx
        if window_size is not None:
            mask = mask | (row_idx - col_idx >= window_size)
        s1 = s1.masked_fill(mask[None, None, None], float('-inf'))

    m = s1.amax(dim=-1, keepdim=True)
    m = torch.where(torch.isneginf(m), torch.zeros_like(m), m)
    p1 = (s1 - m).exp()
    d1 = p1.sum(dim=-1)
    p2 = p1 * s2
    d2 = p2.sum(dim=-1)

    o1 = torch.empty((B, T, H, G, D), dtype=torch.float32, device=q.device)
    o2 = torch.empty((B, T, H, G, D), dtype=torch.float32, device=q.device)
    for b in range(B):
        for h in range(H):
            for g in range(G):
                o1[b, :, h, g] = p1[b, h, g] @ v[b, :, h]
                o2[b, :, h, g] = p2[b, h, g] @ v[b, :, h]

    c_norm = (d2 / d1).permute(0, 3, 1, 2)
    inv_d1 = (1.0 / d1).permute(0, 3, 1, 2)
    out = o1 * inv_d1[..., None] * (1.0 + c_norm[..., None]) - o2 * inv_d1[..., None]
    return out.reshape(B, T, HQ, D).to(dtype)

def _ref_varlen(q, r, k, v, cu_seqlens, window_size=None):
    out = q.new_empty(q.shape)
    for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False):
        out[:, bos:eos] = naive_parallax(
            q=q[:, bos:eos].float(),
            r=r[:, bos:eos].float(),
            k=k[:, bos:eos].float(),
            v=v[:, bos:eos].float(),
            window_size=window_size,
        ).to(q.dtype)
    return out

# bf16 carries fewer mantissa bits than fp16, and the `r` correction amplifies
# rounding, so it gets a looser ratio (bf16 backward grads run ~1e-2 relative).
TOL = {torch.float16: 0.005, torch.bfloat16: 0.02}

@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'D', 'scale'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-D{}-scale{}".format(*test))
        for test in [
            (1, 63, 1, 1, 64, 1.0),
            (3, 111, 2, 2, 100, 1.0),
            (3, 1024, 2, 8, 60, 0.1),
            (3, 1024, 2, 8, 128, 0.1),
            (4, 2048, 2, 8, 64, 0.1),
        ]
    ],
)
def test_improve(
    B: int,
    T: int,
    H: int,
    HQ: int,
    D: int,
    scale: float,
    dtype: torch.dtype,
):
    if not check_shared_mem('hopper') and D > 128:
        pytest.skip(reason="Skip test, do not have enough shared mem")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    tol = TOL[dtype]
    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    r = torch.randn((B, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)

    ref = naive_parallax(q=q.float(), r=r.float(), k=k.float(), v=v.float(), scale=scale)
    ref = ref.to(dtype)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dr, r.grad = r.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = improve_parallax(q=q, r=r, k=k, v=v, scale=scale)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dr, r.grad = r.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close(" o", ref, tri, tol)
    assert_close("dq", ref_dq, tri_dq, tol)
    assert_close("dr", ref_dr, tri_dr, tol)
    assert_close("dk", ref_dk, tri_dk, tol)
    assert_close("dv", ref_dv, tri_dv, tol)

@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'D', 'W'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-D{}-W{}".format(*test))
        for test in [
            (1, 63, 1, 1, 64, 16),
            (3, 111, 2, 2, 100, 32),
            (3, 1024, 2, 8, 128, 64),
            (2, 2048, 2, 8, 64, 256),
            (2, 1024, 2, 2, 64, 200),    # W > tile_size and W % tile_size != 0 (safe-zone boundary)
        ]
    ],
)
def test_improve_swa(
    B: int,
    T: int,
    H: int,
    HQ: int,
    D: int,
    W: int,
    dtype: torch.dtype,
):
    if not check_shared_mem('hopper') and D > 128:
        pytest.skip(reason="Skip test, do not have enough shared mem")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    tol = TOL[dtype]
    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    r = torch.randn((B, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)

    ref = naive_parallax(q=q.float(), r=r.float(), k=k.float(), v=v.float(), window_size=W)
    ref = ref.to(dtype)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dr, r.grad = r.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = improve_parallax(q=q, r=r, k=k, v=v, window_size=W)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dr, r.grad = r.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close(" o", ref, tri, tol)
    assert_close("dq", ref_dq, tri_dq, tol)
    assert_close("dr", ref_dr, tri_dr, tol)
    assert_close("dk", ref_dk, tri_dk, tol)
    assert_close("dv", ref_dv, tri_dv, tol)

@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ('H', 'HQ', 'D', 'cu_seqlens'),
    [
        pytest.param(*test, id="H{}-HQ{}-D{}-cu{}".format(*test))
        for test in [
            (2, 2, 64, [0, 15]),
            (2, 8, 64, [0, 256, 500, 1000]),
            (2, 2, 100, [0, 15, 100, 300, 1200, 2000]),
        ]
    ],
)
def test_improve_varlen(H: int, HQ: int, D: int, cu_seqlens: list[int], dtype: torch.dtype):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    tol = TOL[dtype]
    T = cu_seqlens[-1]
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    r = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    do = torch.randn((1, T, HQ, D), dtype=dtype, device=device)

    ref = _ref_varlen(q, r, k, v, cu_seqlens)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dr, r.grad = r.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = improve_parallax(q=q, r=r, k=k, v=v, cu_seqlens=cu)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dr, r.grad = r.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close(" o", ref, tri, tol)
    assert_close("dq", ref_dq, tri_dq, tol)
    assert_close("dr", ref_dr, tri_dr, tol)
    assert_close("dk", ref_dk, tri_dk, tol)
    assert_close("dv", ref_dv, tri_dv, tol)

@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ('H', 'HQ', 'D', 'W', 'cu_seqlens'),
    [
        pytest.param(*test, id="H{}-HQ{}-D{}-W{}-cu{}".format(*test))
        for test in [
            (2, 2, 64, 16, [0, 111]),
            (2, 8, 100, 32, [0, 256, 500, 1000]),
        ]
    ],
)
def test_improve_swa_varlen(H: int, HQ: int, D: int, W: int, cu_seqlens: list[int], dtype: torch.dtype):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    tol = TOL[dtype]
    T = cu_seqlens[-1]
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    r = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    do = torch.randn((1, T, HQ, D), dtype=dtype, device=device)

    ref = _ref_varlen(q, r, k, v, cu_seqlens, window_size=W)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dr, r.grad = r.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = improve_parallax(q=q, r=r, k=k, v=v, window_size=W, cu_seqlens=cu)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dr, r.grad = r.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close(" o", ref, tri, tol)
    assert_close("dq", ref_dq, tri_dq, tol)
    assert_close("dr", ref_dr, tri_dr, tol)
    assert_close("dk", ref_dk, tri_dk, tol)
    assert_close("dv", ref_dv, tri_dv, tol)

def _decode_ref(q, r, k, v, scale, window_size=None):
    """End-aligned causal reference: query i sits at absolute position Skv-Sq+i."""
    B, Sq, HQ, D = q.shape
    Skv, H = k.shape[1], k.shape[2]
    G = HQ // H
    kv_off = Skv - Sq
    q = q.float().reshape(B, Sq, H, G, D)
    r = r.float().reshape(B, Sq, H, G, D)
    k = k.float()
    v = v.float()
    s1 = torch.empty((B, H, G, Sq, Skv), dtype=torch.float32, device=q.device)
    s2 = torch.empty((B, H, G, Sq, Skv), dtype=torch.float32, device=q.device)
    for b in range(B):
        for h in range(H):
            for g in range(G):
                s1[b, h, g] = q[b, :, h, g] @ k[b, :, h].T * scale
                s2[b, h, g] = r[b, :, h, g] @ k[b, :, h].T
    i = torch.arange(Sq, device=q.device)[:, None]
    j = torch.arange(Skv, device=q.device)[None, :]
    absq = kv_off + i
    mask = j <= absq
    if window_size is not None:
        mask = mask & (j >= absq - window_size + 1)
    s1 = s1.masked_fill(~mask[None, None, None], float('-inf'))
    m = s1.amax(dim=-1, keepdim=True)
    p1 = (s1 - m).exp()
    d1 = p1.sum(dim=-1)
    p2 = p1 * s2
    d2 = p2.sum(dim=-1)
    o1 = torch.empty((B, Sq, H, G, D), dtype=torch.float32, device=q.device)
    o2 = torch.empty((B, Sq, H, G, D), dtype=torch.float32, device=q.device)
    for b in range(B):
        for h in range(H):
            for g in range(G):
                o1[b, :, h, g] = p1[b, h, g] @ v[b, :, h]
                o2[b, :, h, g] = p2[b, h, g] @ v[b, :, h]
    c_norm = (d2 / d1).permute(0, 3, 1, 2)
    inv_d1 = (1.0 / d1).permute(0, 3, 1, 2)
    out = o1 * inv_d1[..., None] * (1.0 + c_norm[..., None]) - o2 * inv_d1[..., None]
    return out.reshape(B, Sq, HQ, D).to(q.dtype)

@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ('B', 'Sq', 'Skv', 'H', 'HQ', 'D', 'W'),
    [
        pytest.param(*test, id="B{}-Sq{}-Skv{}-H{}-HQ{}-D{}-W{}".format(*test))
        for test in [
            (2, 1, 1, 2, 2, 64, None),       # single token, empty-ish cache
            (2, 1, 137, 2, 2, 64, None),     # single decode step over a cache
            (2, 1, 500, 2, 8, 128, None),    # GQA decode, D128
            (2, 1, 300, 2, 2, 100, 64),      # windowed decode, non-pow2 D
            (2, 64, 64, 2, 2, 64, None),     # full prefill == training causal
            (3, 200, 200, 2, 8, 64, 32),     # windowed prefill (GQA)
        ]
    ],
)
def test_decode(B: int, Sq: int, Skv: int, H: int, HQ: int, D: int, W, dtype: torch.dtype):
    if not check_shared_mem('hopper') and D > 128:
        pytest.skip(reason="Skip test, do not have enough shared mem")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    tol = TOL[dtype]
    q = torch.randn((B, Sq, HQ, D), dtype=dtype, device=device)
    r = torch.randn((B, Sq, HQ, D), dtype=dtype, device=device)
    k = torch.randn((B, Skv, H, D), dtype=dtype, device=device)
    v = torch.randn((B, Skv, H, D), dtype=dtype, device=device)

    ref = _decode_ref(q, r, k, v, scale=D ** -0.5, window_size=W)
    tri = parallax_decode(q, r, k, v, window_size=W)
    assert_close(" o", ref, tri, tol)

@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'D', 'W'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-D{}-W{}".format(*test))
        for test in [
            (2, 256, 2, 2, 64, None),
            (2, 256, 2, 8, 64, None),     # GQA
            (2, 200, 2, 2, 128, None),    # D128
            (2, 256, 2, 4, 64, 64),       # sliding window
        ]
    ],
)
def test_decode_matches_improve(B: int, T: int, H: int, HQ: int, D: int, W, dtype: torch.dtype):
    """The decode kernel must reproduce the (reference-verified) training kernel
    for the corresponding positions: decode(last m queries | full KV) == improve[last m]."""
    if not check_shared_mem('hopper') and D > 128:
        pytest.skip(reason="Skip test, do not have enough shared mem")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    tol = TOL[dtype]
    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
    r = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)
    o_full = improve_parallax(q, r, k, v, window_size=W)
    for m in (1, T // 2, T):
        o_dec = parallax_decode(q[:, T - m:], r[:, T - m:], k, v, window_size=W)
        assert_close(f"o(m={m})", o_full[:, T - m:], o_dec, tol)

@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HQ', 'D', 'W'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-D{}-W{}".format(*test))
        for test in [
            (2, 200, 2, 2, 64, None),
            (2, 200, 2, 8, 64, None),     # GQA
            (2, 200, 2, 2, 128, None),    # D128
            (2, 200, 2, 4, 64, 64),       # sliding window
        ]
    ],
)
def test_decode_one_step(B: int, T: int, H: int, HQ: int, D: int, W, dtype: torch.dtype):
    """Single-query decode kernel: must match the training kernel at the last position
    and the prefill-shaped decode kernel."""
    if not check_shared_mem('hopper') and D > 128:
        pytest.skip(reason="Skip test, do not have enough shared mem")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    tol = TOL[dtype]
    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
    r = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)

    o_train = improve_parallax(q, r, k, v, window_size=W)[:, -1:]
    o_one = parallax_decode_one_step(q[:, -1:], r[:, -1:], k, v, window_size=W)
    o_tile = parallax_decode(q[:, -1:], r[:, -1:], k, v, window_size=W)
    assert_close("one_step vs train", o_train, o_one, tol)
    assert_close("one_step vs tile ", o_tile, o_one, tol)

    # left-padding: one_step(cache_start=p) must equal training on the unpadded suffix
    p = 23
    cs = torch.full((B,), p, device=device, dtype=torch.int32)
    o_one_p = parallax_decode_one_step(q[:, -1:], r[:, -1:], k, v, window_size=W, cache_start=cs)
    o_ref_p = improve_parallax(q[:, p:], r[:, p:], k[:, p:], v[:, p:], window_size=W)[:, -1:]
    assert_close("one_step cache_start", o_ref_p, o_one_p, tol)
