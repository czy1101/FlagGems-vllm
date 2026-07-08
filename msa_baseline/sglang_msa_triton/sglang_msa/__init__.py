# MiniMax Sparse Ops (sglang edition) — Standalone
#
# Extracted from sglang for independent operator benchmarking.
# Logic is byte-identical to the original sglang implementation.
#
# The decode index scorer supports JIT radix-select and dense page-table
# code paths only when sglang is installed. Standalone uses the 2-stage
# Triton fallback (equivalent results, different speed).

from .decode.flash_with_topk_idx import flash_decode_with_topk_idx
from .decode.topk_sparse import flash_decode_with_gqa_share_sparse
from .prefill.flash_with_topk_idx import flash_prefill_with_topk_index
from .prefill.topk_sparse import flash_prefill_with_gqa_share_sparse
from .minimax_sparse import minimax_sparse_decode, minimax_sparse_prefill
from .common.index import topk_index_reduce
from .common.utils import get_cu_seqblocks

SPARSE_BLOCK_SIZE = 128

__all__ = [
    "flash_decode_with_topk_idx",
    "flash_decode_with_gqa_share_sparse",
    "flash_prefill_with_topk_index",
    "flash_prefill_with_gqa_share_sparse",
    "minimax_sparse_decode",
    "minimax_sparse_prefill",
    "topk_index_reduce",
    "get_cu_seqblocks",
    "SPARSE_BLOCK_SIZE",
]
