"""PyTorch reference for MSA pipeline (paged KV cache format).

kv_cache: [num_blocks, 2, 128, num_kv_heads, head_dim]  K=[:,0] V=[:,1]
index_kv_cache: [num_blocks, 128, head_dim]
block_table: [batch, max_blocks]
"""
import torch

SPARSE_BLOCK_SIZE = 128
BLOCK = SPARSE_BLOCK_SIZE


def ref_index_score(idx_q, index_kv_cache, block_table, cu_seqlens_q,
                    seq_lens, prefix_lens, max_seqlen_q):
    """Score: max over 128-token blocks of (idx_q . index_k)."""
    batch = cu_seqlens_q.shape[0] - 1
    total_q, num_idx_heads, head_dim = idx_q.shape
    blocks_per_batch = (max_seqlen_q + BLOCK - 1) // BLOCK
    device = idx_q.device
    scores = torch.full(
        (num_idx_heads, total_q, blocks_per_batch),
        float("-inf"), device=device, dtype=torch.float32,
    )
    for b in range(batch):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        seq_len = seq_lens[b].item()
        prefix_len = prefix_lens[b].item()
        Q_b = q_end - q_start
        if Q_b == 0:
            continue
        num_blocks = (seq_len + BLOCK - 1) // BLOCK
        idx_q_b = idx_q[q_start:q_end].float()
        for blk in range(num_blocks):
            page = block_table[b, blk].item()
            ik_block = index_kv_cache[page, :BLOCK].float()  # [128, D]
            qk = torch.einsum('qhd,kd->qhk', idx_q_b, ik_block)  # [Q_b, H, 128]
            q_pos = torch.arange(Q_b, device=device) + prefix_len
            k_pos = torch.arange(BLK_START := blk * BLOCK,
                                 min(BLK_START + BLOCK, seq_len), device=device)
            causal = k_pos[None, :] <= q_pos[:, None]
            qk[:, :, :len(k_pos)] = qk[:, :, :len(k_pos)].masked_fill(
                ~causal[:, None, :], float('-inf'))
            if len(k_pos) > 0:
                # [Q, H] -> [H, Q], matching scores[:, token, block].
                block_score = qk[:, :, :len(k_pos)].amax(dim=-1).transpose(0, 1)
                scores[:, q_start:q_end, blk] = block_score
    return scores


def ref_index_topk(scores, cu_seqlens_q, prefix_lens, topk,
                   init_blocks, local_blocks):
    """Top-K selection with init/local forcing."""
    batch = cu_seqlens_q.shape[0] - 1
    num_idx_heads, total_q, _ = scores.shape
    device = scores.device
    topk_idx = torch.full(
        (num_idx_heads, total_q, topk), -1, dtype=torch.int32, device=device)
    for b in range(batch):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        prefix_len = prefix_lens[b].item()
        Q_b = q_end - q_start
        if Q_b == 0:
            continue
        scores_b = scores[:, q_start:q_end].clone()
        for qi in range(Q_b):
            query_pos = prefix_len + qi
            valid_blocks = (query_pos + 1 + BLOCK - 1) // BLOCK
            if valid_blocks == 0:
                continue
            local_start = max(0, valid_blocks - local_blocks)
            n_select = min(topk, valid_blocks)
            s = scores_b[:, qi, :valid_blocks].clone()
            if local_start < valid_blocks:
                s[:, local_start:valid_blocks] = 1e29
            init_end = min(init_blocks, valid_blocks)
            if init_end > 0:
                mask = s[:, :init_end] < 1e29
                s[:, :init_end] = torch.where(mask, 1e30, s[:, :init_end])
            _, idx = s.topk(n_select, dim=-1)
            topk_idx[:, q_start + qi, :n_select] = idx
    return topk_idx


def ref_sparse_attn(q, kv_cache, topk_idx, block_table, cu_seqlens_q,
                    seq_lens, prefix_lens, sm_scale, max_seqlen_q):
    """Sparse attention over selected blocks (paged KV)."""
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
        Q_b = q_end - q_start
        if Q_b == 0:
            continue
        for qi in range(Q_b):
            query_pos = prefix_len + qi
            max_k = min(query_pos + 1, seq_len)
            if max_k == 0:
                continue
            for kv_hi in range(num_kv_heads):
                selected = topk_idx[kv_hi, q_start + qi]
                selected = selected[selected >= 0]
                if len(selected) == 0:
                    continue
                k_chunks, v_chunks = [], []
                for blk in selected:
                    blk_i = blk.item()
                    bs = blk_i * BLOCK
                    be = min(bs + BLOCK, max_k)
                    page = block_table[b, blk_i].item()
                    k_chunks.append(kv_cache[page, 0, :be - bs, kv_hi].float())
                    v_chunks.append(kv_cache[page, 1, :be - bs, kv_hi].float())
                k_cat = torch.cat(k_chunks, dim=0)
                v_cat = torch.cat(v_chunks, dim=0)
                hi_start = kv_hi * gqa_group_size
                hi_end = hi_start + gqa_group_size
                q_vecs = q[q_start + qi, hi_start:hi_end].float()
                scores = (q_vecs @ k_cat.T) * sm_scale
                scores = scores - scores.amax(dim=-1, keepdim=True)
                attn = torch.softmax(scores.float(), dim=-1)
                output[q_start + qi, hi_start:hi_end] = (attn @ v_cat).to(q.dtype)
    return output


def ref_index_decode(idx_q, index_kv_cache, block_table, seq_lens,
                     max_seq_len, topk, init_blocks, local_blocks, decode_qlen=1):
    """Decode index scoring + topk (paged)."""
    num_reqs = seq_lens.shape[0]
    total_q, num_idx_heads, head_dim = idx_q.shape
    device = idx_q.device
    max_block = (max_seq_len + BLOCK - 1) // BLOCK
    scores = torch.full(
        (num_idx_heads, total_q, max_block),
        float("-inf"), device=device, dtype=torch.float32)
    for r in range(num_reqs):
        seq_len = seq_lens[r].item()
        for qi in range(decode_qlen):
            gq = r * decode_qlen + qi
            query_pos = seq_len - decode_qlen + qi
            kv_len = max(query_pos + 1, 0)
            if kv_len == 0:
                continue
            num_blocks = (kv_len + BLOCK - 1) // BLOCK
            iq = idx_q[gq].float()
            for blk in range(num_blocks):
                page = block_table[r, blk].item()
                valid_tokens = min(BLOCK, kv_len - blk * BLOCK)
                ik_block = index_kv_cache[page, :valid_tokens].float()
                qk = torch.einsum('hd,kd->hk', iq, ik_block)
                scores[:, gq, blk] = qk.max(dim=-1).values if ik_block.shape[0] > 0 else float('-inf')
            local_start = max(0, num_blocks - local_blocks)
            for blk in range(num_blocks):
                if blk < init_blocks:
                    scores[:, gq, blk] = 1e30
                elif blk >= local_start:
                    scores[:, gq, blk] = 1e29
    topk_idx = torch.full(
        (num_idx_heads, total_q, topk), -1, dtype=torch.int32, device=device)
    for r in range(num_reqs):
        seq_len = seq_lens[r].item()
        for qi in range(decode_qlen):
            gq = r * decode_qlen + qi
            query_pos = seq_len - decode_qlen + qi
            kv_len = max(query_pos + 1, 0)
            num_blocks = (kv_len + BLOCK - 1) // BLOCK
            n_select = min(topk, num_blocks)
            if n_select == 0:
                continue
            _, idx = scores[:, gq, :num_blocks].topk(n_select, dim=-1)
            topk_idx[:, gq, :n_select] = idx
    return topk_idx


def ref_sparse_attn_decode(q, kv_cache, topk_idx, block_table, seq_lens,
                            sm_scale, decode_qlen=1):
    """Decode sparse attention (paged)."""
    num_reqs = seq_lens.shape[0]
    total_q, num_heads, head_dim = q.shape
    num_kv_heads = kv_cache.shape[3]
    gqa_group_size = num_heads // num_kv_heads
    output = torch.zeros_like(q)
    for r in range(num_reqs):
        seq_len = seq_lens[r].item()
        for qi in range(decode_qlen):
            gq = r * decode_qlen + qi
            query_pos = seq_len - decode_qlen + qi
            kv_len = max(query_pos + 1, 0)
            if kv_len == 0:
                continue
            for kv_hi in range(num_kv_heads):
                selected = topk_idx[kv_hi, gq]
                selected = selected[selected >= 0]
                if len(selected) == 0:
                    continue
                k_chunks, v_chunks = [], []
                for blk in selected:
                    blk_i = blk.item()
                    bs = blk_i * BLOCK
                    be = min(bs + BLOCK, kv_len)
                    page = block_table[r, blk_i].item()
                    k_chunks.append(kv_cache[page, 0, :be - bs, kv_hi].float())
                    v_chunks.append(kv_cache[page, 1, :be - bs, kv_hi].float())
                k_cat = torch.cat(k_chunks, dim=0)
                v_cat = torch.cat(v_chunks, dim=0)
                hi_start = kv_hi * gqa_group_size
                hi_end = hi_start + gqa_group_size
                q_vecs = q[gq, hi_start:hi_end].float()
                scores = (q_vecs @ k_cat.T) * sm_scale
                scores = scores - scores.amax(dim=-1, keepdim=True)
                attn = torch.softmax(scores.float(), dim=-1)
                output[gq, hi_start:hi_end] = (attn @ v_cat).to(q.dtype)
    return output
