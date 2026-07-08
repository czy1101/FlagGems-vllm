# -*- coding: utf-8 -*-
"""MiniMax M3 Sparse Attention Triton Kernels (standalone baseline).

Extracted and de-vLLM'ified from:
  vllm/models/minimax_m3/common/ops/sparse_attn.py
  vllm/models/minimax_m3/common/ops/index_topk.py
"""

from kernels.common import SPARSE_BLOCK_SIZE, is_pdl_supported, round_up
from kernels.index_topk import (
    minimax_m3_index_decode,
    minimax_m3_index_score,
    minimax_m3_index_topk,
)
from kernels.sparse_attn import (
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)

__all__ = [
    "SPARSE_BLOCK_SIZE",
    "is_pdl_supported",
    "round_up",
    "minimax_m3_sparse_attn",
    "minimax_m3_sparse_attn_decode",
    "minimax_m3_index_score",
    "minimax_m3_index_topk",
    "minimax_m3_index_decode",
]
