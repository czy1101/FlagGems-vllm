"""Adapter for sglang's MiniMax MSA kernels.

The benchmark uses vLLM's paged cache layout:
  kv_cache [num_blocks, num_kv_heads, 128, 2 * head_dim]
  index_kv_cache [num_blocks, 128, head_dim]

sglang uses flattened token-major K/V caches and a request-to-token table, so
the page layout is converted once before timing the kernel calls.
"""
import os
import sys

import torch

_here = os.path.dirname(os.path.abspath(__file__))
_workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
_sglang_python = os.path.join(_workspace_root, "sglang", "python")
if os.path.isdir(os.path.join(_sglang_python, "sglang")):
    sys.path.insert(0, _sglang_python)

from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
    minimax_sparse_prefill,
    minimax_sparse_decode,
)

SPARSE_BLOCK_SIZE = 128
_cached_source = None
_cached_sglang_inputs = None


def _paged_to_sglang(kv_cache, index_kv_cache, block_table, batch, seq_len):
    num_blocks = kv_cache.shape[0]
    num_kv_heads = kv_cache.shape[1]
    block_size = kv_cache.shape[2]
    packed_dim = kv_cache.shape[3]
    if block_size != SPARSE_BLOCK_SIZE or packed_dim % 2:
        raise ValueError(
            "Expected kv_cache shape "
            "[num_blocks, num_kv_heads, 128, 2 * head_dim], got "
            f"{tuple(kv_cache.shape)}"
        )
    head_dim = packed_dim // 2
    expected_index_shape = (num_blocks, block_size, head_dim)
    if tuple(index_kv_cache.shape) != expected_index_shape:
        raise ValueError(
            "Expected index_kv_cache shape "
            f"{expected_index_shape}, got {tuple(index_kv_cache.shape)}"
        )
    blocks_per_batch = (seq_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    total_kv = num_blocks * SPARSE_BLOCK_SIZE
    max_kv_len = blocks_per_batch * SPARSE_BLOCK_SIZE

    # vLLM stores heads before tokens inside each page. sglang indexes a
    # flattened token-major cache: [page * block_size + token, head, dim].
    k_cache = (
        kv_cache[..., :head_dim]
        .permute(0, 2, 1, 3)
        .reshape(total_kv, num_kv_heads, head_dim)
        .contiguous()
    )
    v_cache = (
        kv_cache[..., head_dim:]
        .permute(0, 2, 1, 3)
        .reshape(total_kv, num_kv_heads, head_dim)
        .contiguous()
    )
    idx_k_cache = (
        index_kv_cache.reshape(total_kv, 1, head_dim).contiguous()
    )

    token_offsets = torch.arange(
        SPARSE_BLOCK_SIZE, device=kv_cache.device, dtype=torch.int64
    )
    req_to_token = (
        block_table.to(torch.int64)[..., None] * SPARSE_BLOCK_SIZE
        + token_offsets
    ).reshape(batch, max_kv_len).to(torch.int32)
    slot_ids = torch.arange(batch, device=kv_cache.device, dtype=torch.int32)
    return k_cache, v_cache, idx_k_cache, req_to_token, slot_ids


def _get_sglang_inputs(kv_cache, index_kv_cache, block_table, batch, seq_len):
    """Prepare/cache layout metadata outside the measured kernel region."""
    global _cached_source, _cached_sglang_inputs
    source = (kv_cache, index_kv_cache, block_table, batch, seq_len)
    if _cached_source is None or any(
        old is not new for old, new in zip(_cached_source[:3], source[:3])
    ) or _cached_source[3:] != source[3:]:
        _cached_source = source
        _cached_sglang_inputs = _paged_to_sglang(
            kv_cache, index_kv_cache, block_table, batch, seq_len
        )
    return _cached_sglang_inputs


def clear_sglang_cache():
    """Release the one-entry conversion cache between benchmark shapes."""
    global _cached_source, _cached_sglang_inputs
    _cached_source = None
    _cached_sglang_inputs = None


def sglang_prefill(q, idx_q, kv_cache, index_kv_cache, block_table, cu_q, sl, pl,
                   seq_len, n_kv_h, topk, init_blocks, local_blocks, sm_scale):
    batch = cu_q.shape[0] - 1
    k_cache, v_cache, idx_k_cache, req_to_token, slot_ids = _get_sglang_inputs(
        kv_cache, index_kv_cache, block_table, batch, seq_len
    )
    _idx_o, o = minimax_sparse_prefill(
        q=q, k_cache=k_cache, v_cache=v_cache, sink=None,
        idx_q=idx_q, idx_k_cache=idx_k_cache, idx_v_cache=None, idx_sink=None,
        req_to_token=req_to_token, slot_ids=slot_ids, cu_seqlens=cu_q,
        seq_lens=sl, prefix_lens=pl, max_seqlen_q=seq_len, max_seqlen_k=seq_len,
        block_size_q=1, block_size_k=SPARSE_BLOCK_SIZE, topk=topk,
        init_blocks=init_blocks, local_blocks=local_blocks,
        sm_scale=sm_scale, idx_sm_scale=sm_scale, score_type="max",
        disable_index_value=True, use_msa=False,
    )
    return o


def sglang_decode(q, idx_q, kv_cache, index_kv_cache, block_table, cu_q, sl,
                  seq_len, n_kv_h, topk, init_blocks, local_blocks, sm_scale,
                  decode_qlen=1):
    if decode_qlen != 1:
        raise ValueError(
            "The sglang minimax_sparse_decode baseline accepts one query "
            "token per request; use --decode-qlen 1."
        )
    batch = sl.shape[0]
    k_cache, v_cache, idx_k_cache, req_to_token, slot_ids = _get_sglang_inputs(
        kv_cache, index_kv_cache, block_table, batch, seq_len
    )
    q_decode = q[:batch]
    idx_q_decode = idx_q[:batch]
    _idx_o, o = minimax_sparse_decode(
        q=q_decode, sink=None, k_cache=k_cache, v_cache=v_cache,
        idx_q=idx_q_decode, idx_sink=None, idx_k_cache=idx_k_cache, idx_v_cache=None,
        req_to_token=req_to_token, slot_ids=slot_ids, seq_lens=sl, max_seqlen=seq_len,
        block_size_q=1, block_size_k=SPARSE_BLOCK_SIZE, topk=topk,
        init_blocks=init_blocks, local_blocks=local_blocks,
        sm_scale=sm_scale, idx_sm_scale=sm_scale, score_type="max",
        disable_index_value=True, use_msa=False,
    )
    return o
