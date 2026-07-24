"""Adapter: call vLLM's MSA Triton kernels using paged KV cache.

Shared paged: kv_cache [num_blocks, H, 128, 2*D], index_kv_cache [num_blocks, 128, D]
vLLM paged: kv_cache [num_blocks, H, 128, 2*D], index_kv_cache [num_blocks, 128, D]

Uses importlib + stubs to bypass vLLM framework (no vllm._C needed).
"""
import os
import sys
import types
import importlib.util

import torch
import triton
import triton.language as tl

SPARSE_BLOCK_SIZE = 128
_vllm_ops = None
_cached_kv_source = None
_cached_vllm_kv = None


def _install_vllm_stubs():
    if "vllm" not in sys.modules:
        m = types.ModuleType("vllm"); m.__path__ = []; sys.modules["vllm"] = m
    if "vllm.platforms" not in sys.modules:
        m = types.ModuleType("vllm.platforms")
        class _P:
            def is_arch_support_pdl(self): return True
        m.current_platform = _P()
        sys.modules["vllm.platforms"] = m
        setattr(sys.modules["vllm"], "platforms", m)
    if "vllm.triton_utils" not in sys.modules:
        m = types.ModuleType("vllm.triton_utils")
        m.tl = tl; m.triton = triton
        sys.modules["vllm.triton_utils"] = m
        setattr(sys.modules["vllm"], "triton_utils", m)
    if "vllm.utils" not in sys.modules:
        m = types.ModuleType("vllm.utils"); m.__path__ = []
        sys.modules["vllm.utils"] = m
        setattr(sys.modules["vllm"], "utils", m)
    if "vllm.utils.math_utils" not in sys.modules:
        m = types.ModuleType("vllm.utils.math_utils")
        m.round_up = lambda n, d: (n + d - 1) // d * d
        sys.modules["vllm.utils.math_utils"] = m
        setattr(sys.modules["vllm.utils"], "math_utils", m)


def _ensure_vllm_imported():
    global _vllm_ops
    if _vllm_ops is not None:
        return
    vllm_path = os.environ.get("VLLM_HOME", "")
    if not vllm_path:
        _here = os.path.dirname(os.path.abspath(__file__))
        _workspace_root = os.path.dirname(
            os.path.dirname(os.path.dirname(_here))
        )
        _c = os.path.join(_workspace_root, "vllm-main")
        if os.path.isfile(os.path.join(_c, "vllm", "__init__.py")):
            vllm_path = _c
    if not vllm_path:
        raise FileNotFoundError("vLLM source not found. Set --vllm-path or VLLM_HOME.")
    _install_vllm_stubs()
    ops_dir = os.path.join(vllm_path, "vllm", "models", "minimax_m3", "common", "ops")
    for name, path in [("idx", os.path.join(ops_dir, "index_topk.py")),
                       ("sp", os.path.join(ops_dir, "sparse_attn.py"))]:
        spec = importlib.util.spec_from_file_location(f"vllm_msa_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"vllm_msa_{name}"] = mod
        spec.loader.exec_module(mod)
        if name == "idx":
            _vllm_ops = {
                "index_score": mod.minimax_m3_index_score,
                "index_topk": mod.minimax_m3_index_topk,
                "index_decode": mod.minimax_m3_index_decode,
            }
        else:
            _vllm_ops["sparse_attn"] = mod.minimax_m3_sparse_attn
            _vllm_ops["sparse_attn_decode"] = mod.minimax_m3_sparse_attn_decode


def _convert_kv_cache(kv_cache):
    # The input already uses the vLLM paged KV layout.
    return kv_cache


def _get_vllm_kv_cache(kv_cache):
    global _cached_kv_source, _cached_vllm_kv
    if _cached_kv_source is not kv_cache:
        _cached_kv_source = kv_cache
        _cached_vllm_kv = _convert_kv_cache(kv_cache)
    return _cached_vllm_kv

def clear_vllm_cache():
    """Release the one-entry conversion cache between benchmark shapes."""
    global _cached_kv_source, _cached_vllm_kv
    _cached_kv_source = None
    _cached_vllm_kv = None


def vllm_prefill(q, idx_q, kv_cache, index_kv_cache, block_table, cu_q, sl, pl,
                 seq_len, n_kv_h, topk, init_blocks, local_blocks, sm_scale,
                 output: torch.Tensor | None = None):
    _ensure_vllm_imported()
    ops = _vllm_ops
    vllm_kv = _get_vllm_kv_cache(kv_cache)
    scores = ops["index_score"](idx_q, index_kv_cache, block_table, cu_q, sl, pl, seq_len, seq_len, n_kv_h)
    topk_idx = ops["index_topk"](scores, cu_q, pl, seq_len, topk, init_blocks, local_blocks)
    if output is None:
        output = torch.empty_like(q)
    ops["sparse_attn"](q, vllm_kv, topk_idx, block_table, cu_q, sl, pl, seq_len, n_kv_h, sm_scale, output)
    return output

def vllm_decode(q, idx_q, kv_cache, index_kv_cache, block_table, cu_q, sl,
               seq_len, n_kv_h, topk, init_blocks, local_blocks, sm_scale,
               decode_qlen=1, output: torch.Tensor | None = None):
    _ensure_vllm_imported()
    ops = _vllm_ops
    vllm_kv = _get_vllm_kv_cache(kv_cache)
    topk_idx = ops["index_decode"](idx_q, index_kv_cache, block_table, sl, seq_len, topk,
                                   init_blocks, local_blocks, n_kv_h, decode_qlen, decode_qlen)
    if output is None:
        output = torch.empty_like(q)
    ops["sparse_attn_decode"](q, vllm_kv, topk_idx, block_table, sl, n_kv_h, sm_scale, output, decode_qlen)
    return output
