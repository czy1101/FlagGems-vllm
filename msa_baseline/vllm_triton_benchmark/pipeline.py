"""
MSA pipeline interface.

To switch implementations, change the import below.
Each implementation must expose the same 5 functions + SPARSE_BLOCK_SIZE.
"""

# === Current implementation: vllm_msa ===
from vllm_msa import (
    minimax_m3_index_score as index_score,
    minimax_m3_index_topk as index_topk,
    minimax_m3_sparse_attn as sparse_attn,
    minimax_m3_index_decode as index_decode,
    minimax_m3_sparse_attn_decode as sparse_attn_decode,
    SPARSE_BLOCK_SIZE,
)
