"""
Pure PyTorch reference implementation for MSA pipeline.
Correctness-first (Python for-loops), NOT optimized for performance.
"""

import torch
from pipeline import SPARSE_BLOCK_SIZE
def ref_index_score(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> torch.Tensor:
    """Reference: per-token index block scores via max-pool over 128-token blocks."""
    batch = cu_seqlens_q.shape[0] - 1
    total_q, num_idx_heads, head_dim = idx_q.shape
    max_blocks_per_seq = (seq_lens.max().item() + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE

    scores = torch.full(
        (num_idx_heads, total_q, max_blocks_per_seq),
        float("-inf"), device=idx_q.device, dtype=torch.float32,
    )

    for b in range(batch):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        seq_len = seq_lens[b].item()
        prefix_len = prefix_lens[b].item()
        num_blocks = (seq_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE

        # Reconstruct full index-K from paged cache
        idx_K = torch.zeros(seq_len, head_dim, device=idx_q.device, dtype=idx_q.dtype)
        for blk in range(num_blocks):
            page = block_table[b, blk].item()
            ks, ke = blk * SPARSE_BLOCK_SIZE, min(blk * SPARSE_BLOCK_SIZE + SPARSE_BLOCK_SIZE, seq_len)
            idx_K[ks:ke] = index_kv_cache[page, : ke - ks, :]

        for qi in range(q_end - q_start):
            query_pos = prefix_len + qi
            max_k = min(query_pos + 1, seq_len)
            if max_k == 0:
                continue
            for hi in range(num_idx_heads):
                q_vec = idx_q[q_start + qi, hi].float()
                qk = q_vec @ idx_K[:max_k].float().T  # [max_k]
                for blk in range(num_blocks):
                    bs = blk * SPARSE_BLOCK_SIZE
                    be = min(bs + SPARSE_BLOCK_SIZE, max_k)
                    if bs < max_k:
                        scores[hi, q_start + qi, blk] = qk[bs:be].max()
    return scores


@torch.no_grad()
def ref_index_topk(
    scores: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    prefix_lens: torch.Tensor,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Reference: top-k block selection with forced init/local blocks.

    Priority (matching Triton tl.where order):
      local (1e29) > init (1e30) > original score
    """
    batch = cu_seqlens_q.shape[0] - 1
    num_idx_heads, total_q, _ = scores.shape
    topk_idx = torch.full((num_idx_heads, total_q, topk), -1, dtype=torch.int32, device=scores.device)

    for b in range(batch):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        prefix_len = prefix_lens[b].item()
        for qi in range(q_end - q_start):
            query_pos = prefix_len + qi
            valid_blocks = (query_pos + 1 + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
            local_start = max(0, valid_blocks - local_blocks)
            for hi in range(num_idx_heads):
                s = scores[hi, q_start + qi].clone()

                # Apply forced scores: local >> init >> original (matching Triton's tl.where)
                for blk in range(valid_blocks):
                    if blk >= local_start:  # is_local
                        s[blk] = 1e29
                    elif blk < init_blocks:  # is_init (only if not already local)
                        s[blk] = 1e30
                    # else: keep original score

                n_select = min(topk, valid_blocks)
                _, idx = s[:valid_blocks].topk(n_select)
                topk_idx[hi, q_start + qi, :n_select] = idx
    return topk_idx


@torch.no_grad()
def ref_sparse_attn(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Reference: block-sparse GQA attention with causal mask."""
    batch = cu_seqlens_q.shape[0] - 1
    total_q, num_heads, head_dim = q.shape
    num_kv_heads = kv_cache.shape[3]
    gqa_group_size = num_heads // num_kv_heads

    output = torch.zeros_like(q)

    for b in range(batch):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        seq_len = seq_lens[b].item()
        prefix_len = prefix_lens[b].item()
        num_blocks = (seq_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE

        # Reconstruct full K, V from paged cache
        K_full = torch.zeros(seq_len, num_kv_heads, head_dim, device=q.device, dtype=torch.float32)
        V_full = torch.zeros(seq_len, num_kv_heads, head_dim, device=q.device, dtype=torch.float32)
        for blk in range(num_blocks):
            page = block_table[b, blk].item()
            ks = blk * SPARSE_BLOCK_SIZE
            ke = min(ks + SPARSE_BLOCK_SIZE, seq_len)
            valid = ke - ks
            K_full[ks:ke] = kv_cache[page, 0, :valid, :, :].float()
            V_full[ks:ke] = kv_cache[page, 1, :valid, :, :].float()

        for qi in range(q_end - q_start):
            query_pos = prefix_len + qi
            max_k = min(query_pos + 1, seq_len)
            if max_k == 0:
                continue
            for hi in range(num_heads):
                kv_hi = hi // gqa_group_size
                q_vec = q[q_start + qi, hi].float()
                selected = topk_idx[kv_hi, q_start + qi]
                selected = selected[selected >= 0].tolist()

                all_scores, all_v = [], []
                for blk in selected:
                    bs = blk * SPARSE_BLOCK_SIZE
                    be = min(bs + SPARSE_BLOCK_SIZE, max_k)
                    k_blk = K_full[bs:be, kv_hi]
                    v_blk = V_full[bs:be, kv_hi]
                    all_scores.append(q_vec @ k_blk.T * sm_scale)
                    all_v.append(v_blk)

                if not all_scores:
                    continue
                scores = torch.cat(all_scores)
                v_cat = torch.cat(all_v, dim=0)
                scores = scores - scores.max()
                attn = torch.softmax(scores, dim=-1)
                output[q_start + qi, hi] = (attn @ v_cat).to(q.dtype)
    return output


# ---------------------------------------------------------------------------
# End-to-end correctness test
# ---------------------------------------------------------------------------

