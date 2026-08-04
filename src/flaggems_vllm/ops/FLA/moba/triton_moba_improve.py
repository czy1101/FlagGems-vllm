import torch
import triton
import triton.language as tl

try:
    import flash_attn_interface as flash_attn_3
except ImportError:
    flash_attn_3 = None
from functools import lru_cache
from einops import rearrange

def _flash_attn_3_varlen_forward(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
):
    if hasattr(flash_attn_3, "_flash_attn_varlen_forward"):
        result = flash_attn_3._flash_attn_varlen_forward(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        return result[0], result[5]

    if hasattr(flash_attn_3, "_flash_attn_forward"):
        out, softmax_lse, *_ = flash_attn_3._flash_attn_forward(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        return out, softmax_lse

    raise RuntimeError(
        "The installed FlashAttention-3 does not expose a supported varlen forward API"
    )

def _flash_attn_3_varlen_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
    deterministic,
):
    if hasattr(flash_attn_3, "_flash_attn_varlen_backward"):
        flash_attn_3._flash_attn_varlen_backward(
            dout=dout,
            q=q,
            k=k,
            v=v,
            out=out,
            softmax_lse=softmax_lse,
            dq=dq,
            dk=dk,
            dv=dv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=(-1, -1),
            deterministic=deterministic,
        )
        return

    if hasattr(flash_attn_3, "_flash_attn_backward"):
        flash_attn_3._flash_attn_backward(
            dout=dout,
            q=q,
            k=k,
            v=v,
            out=out,
            softmax_lse=softmax_lse,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dq=dq,
            dk=dk,
            dv=dv,
            softmax_scale=softmax_scale,
            is_causal=causal,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            deterministic=deterministic,
        )
        return

    raise RuntimeError(
        "The installed FlashAttention-3 does not expose a supported varlen backward API"
    )

def _flash_attn_3_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    causal,
):
    result = flash_attn_3.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=causal,
    )
    return result if torch.is_tensor(result) else result[0]

# Helper for triton.cdiv compatibility
def cdiv(x, y):
    return (x + y - 1) // y


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_ROWS": 4, "BYPASS_L1": False},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_ROWS": 4, "BYPASS_L1": True},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_ROWS": 8, "BYPASS_L1": False},
            num_warps=8,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_ROWS": 8, "BYPASS_L1": True},
            num_warps=8,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_ROWS": 16, "BYPASS_L1": False},
            num_warps=8,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_ROWS": 16, "BYPASS_L1": True},
            num_warps=8,
            num_stages=1,
        ),
    ],
    key=["head_dim"],
)
@triton.jit
def _gather_moba_q_kernel(
    q_ptr,
    indices_ptr,
    output_ptr,
    num_rows,
    stride_q_row: tl.constexpr,
    stride_q_dim: tl.constexpr,
    stride_output_row: tl.constexpr,
    stride_output_dim: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    BYPASS_L1: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    dim_offsets = tl.arange(0, BLOCK_DIM)
    row_mask = row_offsets < num_rows
    source_rows = tl.load(
        indices_ptr + row_offsets,
        mask=row_mask,
        other=0,
    ).to(tl.int64)

    q_offsets = (
        source_rows[:, None] * stride_q_row
        + dim_offsets[None, :] * stride_q_dim
    )
    mask = row_mask[:, None] & (dim_offsets[None, :] < head_dim)
    if BYPASS_L1:
        values = tl.load(
            q_ptr + q_offsets,
            mask=mask,
            other=0.0,
            cache_modifier=".cg",
        )
    else:
        values = tl.load(q_ptr + q_offsets, mask=mask, other=0.0)

    output_offsets = (
        row_offsets[:, None] * stride_output_row
        + dim_offsets[None, :] * stride_output_dim
    )
    tl.store(output_ptr + output_offsets, values, mask=mask)


def _gather_moba_q_forward(q: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    head_dim = q.shape[-1]
    q_2d = q.reshape(-1, head_dim)
    output = torch.empty(
        (indices.numel(), 1, head_dim),
        device=q.device,
        dtype=q.dtype,
    )
    if indices.numel() == 0:
        return output

    block_dim = triton.next_power_of_2(head_dim)
    grid = lambda meta: (triton.cdiv(indices.numel(), meta["BLOCK_ROWS"]),)
    _gather_moba_q_kernel[grid](
        q_2d,
        indices,
        output,
        indices.numel(),
        q_2d.stride(0),
        q_2d.stride(1),
        output.stride(0),
        output.stride(2),
        head_dim=head_dim,
        BLOCK_DIM=block_dim,
    )
    return output


class _GatherMobaQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(indices)
        ctx.q_shape = q.shape
        ctx.head_dim = q.shape[-1]
        return _gather_moba_q_forward(q, indices)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (indices,) = ctx.saved_tensors
        grad_q = torch.zeros(
            ctx.q_shape,
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_q.view(-1, ctx.head_dim).index_add_(
            0,
            indices.long(),
            grad_output.reshape(-1, ctx.head_dim),
        )
        return grad_q, None


def gather_moba_q(q: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather flattened query rows into the [selected, 1, head_dim] MoBA layout."""
    if not q.is_cuda or not indices.is_cuda:
        raise ValueError("gather_moba_q requires CUDA tensors")
    if q.ndim != 3:
        raise ValueError(
            f"q must have shape [seqlen, heads, head_dim], got {q.shape}"
        )
    if indices.ndim != 1:
        raise ValueError(f"indices must be one-dimensional, got {indices.shape}")
    if q.device != indices.device:
        raise ValueError("q and indices must be on the same CUDA device")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"indices must use int32 or int64, got {indices.dtype}")

    indices = indices.contiguous()
    if torch.is_grad_enabled() and q.requires_grad:
        return _GatherMobaQ.apply(q, indices)
    return _gather_moba_q_forward(q, indices)


@lru_cache(maxsize=16)
def calc_chunks(cu_seqlen, moba_chunk_size):
    """
    Pre-calculates chunk metadata based on cumulative sequence lengths.
    Determines how sequences are split into fixed-size chunks for MoBA.
    """
    # batch_sizes[batch_idx] = batch size ( seqlen ) of batch idx
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]
    # batch_num_chunk[batch_idx] = how many chunk in batch idx
    batch_num_chunk = (batch_sizes + (moba_chunk_size - 1)) // moba_chunk_size
    # cu_num_chunk[batch_idx] = first chunk id of this batch
    cu_num_chunk = torch.ones(
        batch_num_chunk.numel() + 1,
        device=cu_seqlen.device,
        dtype=batch_num_chunk.dtype,
    )
    cu_num_chunk[1:] = batch_num_chunk.cumsum(dim=0)
    # total chunk ( for all batch )
    num_chunk = cu_num_chunk[-1]
    # chunk_sizes[chunk_idx] = chunk_size of chunk idx
    chunk_sizes = torch.full(
        (num_chunk + 1,), moba_chunk_size, dtype=torch.int32, device=cu_seqlen.device
    )
    chunk_sizes[0] = 0  # for calc cu chunk
    batch_last_chunk_size = batch_sizes - (batch_num_chunk - 1) * moba_chunk_size
    chunk_sizes[cu_num_chunk[1:]] = batch_last_chunk_size
    # cu_chunk[chunk_idx] = the start chunk offset of chunk idx
    cu_chunk = chunk_sizes.cumsum(dim=-1, dtype=torch.int32)
    # chunk_to_batch[chunk_idx] = batch idx of the chunk idx
    chunk_to_batch = torch.zeros(
        (num_chunk,), dtype=torch.int32, device=cu_seqlen.device
    )
    chunk_to_batch[cu_num_chunk[1:-1]] = 1
    chunk_to_batch = chunk_to_batch.cumsum(dim=0, dtype=torch.int32)

    """ filter chunks that need moba attn """
    # filter chunks ( remove last chunk of each batch as it's processed by self-attn )
    chunk_to_remove = cu_num_chunk[1:] - 1
    chunk_to_remain = torch.ones(
        (num_chunk,), dtype=torch.bool, device=cu_seqlen.device
    )
    chunk_to_remain[chunk_to_remove] = False
    filtered_chunk_indices = chunk_to_remain.nonzero(as_tuple=True)[0]
    num_filtered_chunk = len(filtered_chunk_indices)

    return (
        cu_chunk,
        filtered_chunk_indices,
        num_filtered_chunk,
        chunk_to_batch,
    )


@triton.jit
def _chunk_mean_kernel(
    k_ptr,
    out_ptr,
    stride_chunk,
    stride_token,
    stride_head,
    stride_d,
    out_stride_chunk,
    out_stride_head,
    out_stride_d,
    num_chunks,
    num_heads,
    CHUNK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Kernel to compute the mean of Key vectors within each chunk.
    Replacing: filtered_kv.view(...).mean(dim=1)
    """
    chunk_id = tl.program_id(0)
    head_id = tl.program_id(1)
    if chunk_id >= num_chunks or head_id >= num_heads:
        return

    offs_d = tl.arange(0, BLOCK_D)

    for d_start in range(0, HEAD_DIM, BLOCK_D):
        mask_d = (d_start + offs_d) < HEAD_DIM
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for t_start in range(0, CHUNK_SIZE, BLOCK_TOK):
            offs_t = tl.arange(0, BLOCK_TOK)
            mask_t = (t_start + offs_t) < CHUNK_SIZE

            ptrs = (
                k_ptr
                + chunk_id * stride_chunk
                + (t_start + offs_t)[:, None] * stride_token
                + head_id * stride_head
                + (d_start + offs_d)[None, :] * stride_d
            )
            vals = tl.load(ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0)
            vals = vals.to(tl.float32)
            acc += tl.sum(vals, axis=0)

        acc = acc / CHUNK_SIZE
        out_ptrs = (
            out_ptr
            + chunk_id * out_stride_chunk
            + head_id * out_stride_head
            + (d_start + offs_d) * out_stride_d
        )
        tl.store(out_ptrs, acc, mask=mask_d)

@triton.jit
def _streaming_topk_float_to_key(values):
    value_bits = values.to(tl.uint32, bitcast=True)
    sign_mask = tl.full(value_bits.shape, 0x80000000, dtype=tl.uint32)
    full_mask = tl.full(value_bits.shape, 0xFFFFFFFF, dtype=tl.uint32)
    flip_mask = tl.where((value_bits & sign_mask) != 0, full_mask, sign_mask)
    return value_bits ^ flip_mask


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_TOK": 32, "BLOCK_CHUNK": 32},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_TOK": 16, "BLOCK_CHUNK": 64},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_TOK": 32, "BLOCK_CHUNK": 64},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_TOK": 32, "BLOCK_CHUNK": 64},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_TOK": 64, "BLOCK_CHUNK": 64},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_TOK": 16, "BLOCK_CHUNK": 128},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_TOK": 32, "BLOCK_CHUNK": 128},
            num_warps=4,
            num_stages=1,
        ),
    ],
    key=[
        "num_chunks",
        "num_heads",
        "seqlen",
        "HEAD_DIM",
        "TOPK",
    ],
)
@triton.jit
def _chunk_topk_streaming_kernel(
    q_ptr,
    chunk_means_ptr,
    chunk_end_ptr,
    batch_end_ptr,
    selected_chunks_ptr,
    num_chunks: tl.constexpr,
    num_heads,
    seqlen,
    stride_q_seq,
    stride_q_head,
    stride_q_d,
    stride_chunk_mean_chunk,
    stride_chunk_mean_head,
    stride_chunk_mean_d,
    stride_selected_rank,
    stride_selected_head,
    stride_selected_token,
    HEAD_DIM: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_tok = tl.program_id(0)
    pid_head = tl.program_id(1)

    if pid_head >= num_heads:
        return

    offs_token = pid_tok * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
    token_mask = offs_token < seqlen
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    chunk_rows = tl.arange(0, BLOCK_CHUNK)

    q_ptrs = (
        q_ptr
        + offs_token[:, None] * stride_q_seq
        + pid_head * stride_q_head
        + offs_d[None, :] * stride_q_d
    )
    q_values = tl.load(
        q_ptrs,
        mask=token_mask[:, None] & mask_d[None, :],
        other=0.0,
    )

    neg_inf = -1.0e38
    best_packed = tl.zeros(
        [BLOCK_TOK, BLOCK_TOPK],
        dtype=tl.uint64,
    )
    max_index_key = tl.full(
        [BLOCK_CHUNK],
        0xFFFFFFFF,
        dtype=tl.uint32,
    )

    for chunk_start in range(0, num_chunks, BLOCK_CHUNK):
        chunk_offsets = chunk_start + chunk_rows
        chunk_mask = chunk_offsets < num_chunks
        safe_chunk_offsets = tl.where(chunk_mask, chunk_offsets, 0)

        chunk_ptrs = (
            chunk_means_ptr
            + safe_chunk_offsets[:, None] * stride_chunk_mean_chunk
            + pid_head * stride_chunk_mean_head
            + offs_d[None, :] * stride_chunk_mean_d
        )
        chunk_values = tl.load(
            chunk_ptrs,
            mask=chunk_mask[:, None] & mask_d[None, :],
            other=0.0,
        )
        gating = tl.dot(
            chunk_values,
            tl.trans(q_values),
            out_dtype=tl.float32,
        )

        chunk_end = tl.load(
            chunk_end_ptr + safe_chunk_offsets,
            mask=chunk_mask,
            other=0,
        )
        batch_end = tl.load(
            batch_end_ptr + safe_chunk_offsets,
            mask=chunk_mask,
            other=0,
        )
        valid = (
            chunk_mask[:, None]
            & token_mask[None, :]
            & (offs_token[None, :] >= chunk_end[:, None])
            & (offs_token[None, :] < batch_end[:, None])
        )
        gating = tl.where(valid, gating, neg_inf)

        scores = tl.trans(gating)
        score_keys = _streaming_topk_float_to_key(scores)
        index_keys = max_index_key - chunk_offsets.to(tl.uint32)
        packed = (
            score_keys.to(tl.uint64) << 32
        ) | index_keys[None, :].to(tl.uint64)
        packed = tl.where(
            tl.trans(valid),
            packed,
            tl.zeros(packed.shape, dtype=tl.uint64),
        )
        local_packed = tl.topk(packed, BLOCK_TOPK)

        if chunk_start == 0:
            best_packed = local_packed
        else:
            best_packed = tl.bitonic_merge(best_packed)
            best_packed = tl.maximum(best_packed, local_packed)

    best_packed = tl.sort(best_packed, descending=True)
    best_rank_major = tl.trans(best_packed)
    selected_index_keys = best_rank_major.to(tl.uint32)
    selected_indices = (
        tl.full(best_rank_major.shape, 0xFFFFFFFF, dtype=tl.uint32)
        - selected_index_keys
    ).to(tl.int32)
    rank_offsets = tl.arange(0, BLOCK_TOPK)
    selected_ptrs = (
        selected_chunks_ptr
        + rank_offsets[:, None] * stride_selected_rank
        + pid_head * stride_selected_head
        + offs_token[None, :] * stride_selected_token
    )
    tl.store(
        selected_ptrs,
        selected_indices,
        mask=(
            (rank_offsets[:, None] < TOPK)
            & token_mask[None, :]
            & (best_rank_major != 0)
        ),
    )


def _compute_chunk_means_triton(filtered_k: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Wrapper for chunk mean kernel"""
    num_chunks, _, num_heads, head_dim = filtered_k.shape
    if num_chunks == 0:
        return filtered_k.new_zeros((0, num_heads, head_dim))

    filtered_k = filtered_k.contiguous()
    chunk_means = torch.empty(
        (num_chunks, num_heads, head_dim),
        device=filtered_k.device,
        dtype=filtered_k.dtype,
    )

    block_tok = min(64, triton.next_power_of_2(chunk_size))
    block_d = max(16, triton.next_power_of_2(head_dim))
    num_warps = 4 if block_d >= 64 else 2
    num_stages = 4 if chunk_size >= 256 else 2

    grid = (num_chunks, num_heads)

    _chunk_mean_kernel[grid](
        filtered_k,
        chunk_means,
        filtered_k.stride(0),
        filtered_k.stride(1),
        filtered_k.stride(2),
        filtered_k.stride(3),
        chunk_means.stride(0),
        chunk_means.stride(1),
        chunk_means.stride(2),
        num_chunks,
        num_heads,
        CHUNK_SIZE=chunk_size,
        HEAD_DIM=head_dim,
        BLOCK_TOK=block_tok,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return chunk_means

def _compute_selected_chunks_streaming_triton(
    q: torch.Tensor,
    chunk_means: torch.Tensor,
    chunk_end: torch.Tensor,
    batch_end: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    num_chunks, num_heads, head_dim = chunk_means.shape
    seqlen = q.shape[0]

    if topk <= 0 or num_chunks == 0:
        return torch.empty(
            (0, num_heads, seqlen),
            device=q.device,
            dtype=torch.int32,
        )
    if topk > num_chunks:
        raise ValueError(f"topk ({topk}) cannot exceed num_chunks ({num_chunks})")

    block_topk = triton.next_power_of_2(topk)
    if block_topk > 32:
        raise ValueError(
            "streaming Triton Top-K currently supports topk <= 32, "
            f"got topk={topk}"
        )

    q = q.contiguous()
    chunk_means = chunk_means.contiguous()
    chunk_end = chunk_end.to(dtype=torch.int32).contiguous()
    batch_end = batch_end.to(dtype=torch.int32).contiguous()
    selected_chunks = torch.full(
        (topk, num_heads, seqlen),
        -1,
        device=q.device,
        dtype=torch.int32,
    )

    block_d = max(16, triton.next_power_of_2(head_dim))
    grid = lambda meta: (
        cdiv(seqlen, meta["BLOCK_TOK"]),
        num_heads,
    )

    _chunk_topk_streaming_kernel[grid](
        q,
        chunk_means,
        chunk_end,
        batch_end,
        selected_chunks,
        num_chunks,
        num_heads,
        seqlen,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        chunk_means.stride(0),
        chunk_means.stride(1),
        chunk_means.stride(2),
        selected_chunks.stride(0),
        selected_chunks.stride(1),
        selected_chunks.stride(2),
        HEAD_DIM=head_dim,
        TOPK=topk,
        BLOCK_TOPK=block_topk,
        BLOCK_D=block_d,
    )

    return selected_chunks


def _build_sparse_routes(
    selected_chunks: torch.Tensor,
    num_chunks: int,
    num_heads: int,
    seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    topk = selected_chunks.shape[0]
    selected_flat = selected_chunks.view(-1)
    valid_slots = torch.nonzero(
        selected_flat >= 0,
        as_tuple=True,
    )[0]
    num_experts = num_chunks * num_heads

    if valid_slots.numel() == 0:
        route_positions = torch.full_like(selected_chunks, -1)
        expert_counts = torch.zeros(
            num_experts,
            device=selected_chunks.device,
            dtype=torch.int32,
        )
        return (
            valid_slots,
            route_positions.permute(0, 2, 1).contiguous().view(topk, -1),
            expert_counts,
        )

    selected_valid = selected_flat.index_select(0, valid_slots)
    slots_per_rank = num_heads * seqlen
    slots_within_rank = valid_slots % slots_per_rank
    head_indices = slots_within_rank // seqlen
    token_indices = slots_within_rank % seqlen

    expert_indices = selected_valid.to(torch.long) * num_heads + head_indices
    query_indices = token_indices * num_heads + head_indices
    expert_order = torch.argsort(expert_indices)

    expert_indices = expert_indices.index_select(0, expert_order)
    query_indices = query_indices.index_select(0, expert_order)
    route_slots = valid_slots.index_select(0, expert_order)

    route_positions = torch.full_like(selected_chunks, -1)
    route_positions.view(-1).scatter_(
        0,
        route_slots,
        torch.arange(
            route_slots.numel(),
            device=selected_chunks.device,
            dtype=torch.int32,
        ),
    )
    expert_counts = torch.bincount(
        expert_indices,
        minlength=num_experts,
    ).to(torch.int32)

    return (
        query_indices,
        route_positions.permute(0, 2, 1).contiguous().view(topk, -1),
        expert_counts,
    )


@triton.jit
def _fused_merge_softmax_kernel(
    self_out_ptr,
    self_lse_ptr,
    moba_out_ptr,
    moba_lse_ptr,
    route_positions_ptr,
    out_ptr,
    final_lse_ptr,
    num_rows,
    head_dim,
    stride_self_row,
    stride_self_d,
    stride_moba_row,
    stride_moba_d,
    stride_route_rank,
    stride_route_row,
    stride_out_row,
    stride_out_d,
    BLOCK_D: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_ROUTES: tl.constexpr,
):
    """
    Fused kernel to combine Self-Attention and MoBA-Attention results.
    
    It performs the standard FlashAttention output merging:
    O = (O_self * exp(LSE_self - LSE_max) + sum(O_moba * exp(LSE_moba - LSE_max))) / exp(LSE_new - LSE_max)
    
    This avoids multiple passes of reading/writing large output tensors.
    """
    row_id = tl.program_id(0)
    if row_id >= num_rows:
        return

    offs_k = tl.arange(0, TOPK)
    route_mask = offs_k < NUM_ROUTES
    positions = tl.load(
        route_positions_ptr
        + offs_k * stride_route_rank
        + row_id * stride_route_row,
        mask=route_mask,
        other=-1,
    )
    mask_k = route_mask & (positions >= 0)
    safe_positions = tl.where(mask_k, positions, 0)

    # Load LSEs and compute global Max LSE
    self_lse = tl.load(self_lse_ptr + row_id).to(tl.float32)
    moba_lse = tl.load(
        moba_lse_ptr + safe_positions,
        mask=mask_k,
        other=-float("inf"),
    ).to(tl.float32)

    max_moba = tl.max(moba_lse, axis=0)
    max_lse = tl.maximum(self_lse, max_moba)

    # Compute weights
    self_se = tl.exp(self_lse - max_lse)
    moba_se = tl.exp(moba_lse - max_lse)
    moba_se = tl.where(mask_k, moba_se, 0.0)
    total_se = self_se + tl.sum(moba_se, axis=0)
    merged_lse = tl.log(total_se) + max_lse
    tl.store(final_lse_ptr + row_id, merged_lse)

    # Compute weighted output
    inv_total = 1.0 / total_se
    self_factor = self_se * inv_total
    moba_factor = moba_se * inv_total

    offs_d = tl.arange(0, BLOCK_D)

    for d_start in range(0, head_dim, BLOCK_D):
        mask_d = (d_start + offs_d) < head_dim

        self_ptrs = (
            self_out_ptr
            + row_id * stride_self_row
            + (d_start + offs_d) * stride_self_d
        )
        self_vals = tl.load(self_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        acc = self_vals * self_factor

        moba_ptrs = (
            moba_out_ptr
            + safe_positions[:, None] * stride_moba_row
            + (d_start + offs_d)[None, :] * stride_moba_d
        )
        moba_vals = tl.load(
            moba_ptrs,
            mask=mask_k[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        scaled = moba_vals * moba_factor[:, None]
        acc = acc + tl.sum(scaled, axis=0)

        out_ptrs = (
            out_ptr
            + row_id * stride_out_row
            + (d_start + offs_d) * stride_out_d
        )
        tl.store(out_ptrs, acc, mask=mask_d)


def _fused_merge_softmax_triton(
    self_out: torch.Tensor,
    self_lse: torch.Tensor,
    moba_out: torch.Tensor,
    moba_lse: torch.Tensor,
    route_positions: torch.Tensor,
    final_out: torch.Tensor,
    final_lse: torch.Tensor,
):
    """Wrapper for the fused merge kernel"""
    num_rows, head_dim = self_out.shape
    topk = route_positions.shape[0]

    if topk == 0 or moba_out.numel() == 0:
        final_out.copy_(self_out.to(final_out.dtype))
        final_lse.copy_(self_lse)
        return

    route_positions = route_positions.contiguous()
    block_topk = triton.next_power_of_2(topk)

    block_d = min(128, (head_dim + 31) // 32 * 32)
    if block_d == 0:
        block_d = head_dim

    grid = (num_rows,)

    _fused_merge_softmax_kernel[grid](
        self_out,
        self_lse,
        moba_out,
        moba_lse,
        route_positions,
        final_out,
        final_lse,
        num_rows,
        head_dim,
        self_out.stride(0),
        self_out.stride(1),
        moba_out.stride(0),
        moba_out.stride(1),
        route_positions.stride(0),
        route_positions.stride(1),
        final_out.stride(0),
        final_out.stride(1),
        BLOCK_D=block_d,
        TOPK=block_topk,
        NUM_ROUTES=topk,
    )


@triton.jit
def _gather_moba_backward_inputs_kernel(
    dout_ptr,
    out_ptr,
    lse_ptr,
    indices_ptr,
    gathered_dout_ptr,
    gathered_out_ptr,
    gathered_lse_ptr,
    num_indices,
    head_dim,
    stride_dout_row,
    stride_dout_d,
    stride_out_row,
    stride_out_d,
    stride_gdout_row,
    stride_gdout_d,
    stride_gout_row,
    stride_gout_d,
    BLOCK_D: tl.constexpr,
):
    """
    Kernel to gather backward pass inputs for the MoBA branch.
    Since MoBA works on a sparse subset of Q, we need to gather gradients (dout), 
    outputs (out), and LSE scores corresponding to those sparse indices.
    """
    pid = tl.program_id(0)
    if pid >= num_indices:
        return

    sh_idx = tl.load(indices_ptr + pid)
    lse_val = tl.load(lse_ptr + sh_idx)
    tl.store(gathered_lse_ptr + pid, lse_val)

    offs_d = tl.arange(0, BLOCK_D)

    for d_start in range(0, head_dim, BLOCK_D):
        mask_d = (d_start + offs_d) < head_dim

        dout_ptrs = (
            dout_ptr
            + sh_idx * stride_dout_row
            + (d_start + offs_d) * stride_dout_d
        )
        dout_vals = tl.load(dout_ptrs, mask=mask_d, other=0.0)
        gdout_ptrs = (
            gathered_dout_ptr
            + pid * stride_gdout_row
            + (d_start + offs_d) * stride_gdout_d
        )
        tl.store(gdout_ptrs, dout_vals, mask=mask_d)

        out_ptrs = (
            out_ptr
            + sh_idx * stride_out_row
            + (d_start + offs_d) * stride_out_d
        )
        out_vals = tl.load(out_ptrs, mask=mask_d, other=0.0)
        gout_ptrs = (
            gathered_out_ptr
            + pid * stride_gout_row
            + (d_start + offs_d) * stride_gout_d
        )
        tl.store(gout_ptrs, out_vals, mask=mask_d)


def _gather_moba_backward_inputs_triton(
    d_output_2d: torch.Tensor,
    output_2d: torch.Tensor,
    mixed_attn_vlse_flat: torch.Tensor,
    moba_indices: torch.Tensor,
    gathered_d_output: torch.Tensor,
    gathered_output: torch.Tensor,
    gathered_lse: torch.Tensor,
):
    """Wrapper for the backward gather kernel"""
    num_indices = moba_indices.numel()
    if num_indices == 0:
        return

    head_dim = d_output_2d.shape[1]
    indices_i32 = moba_indices.to(torch.int32)

    block_d = min(128, (head_dim + 31) // 32 * 32)
    if block_d == 0:
        block_d = head_dim

    grid = (num_indices,)

    _gather_moba_backward_inputs_kernel[grid](
        d_output_2d,
        output_2d,
        mixed_attn_vlse_flat,
        indices_i32,
        gathered_d_output,
        gathered_output,
        gathered_lse,
        num_indices,
        head_dim,
        d_output_2d.stride(0),
        d_output_2d.stride(1),
        output_2d.stride(0),
        output_2d.stride(1),
        gathered_d_output.stride(0),
        gathered_d_output.stride(1),
        gathered_output.stride(0),
        gathered_output.stride(1),
        BLOCK_D=block_d,
    )

class MixedAttention(torch.autograd.Function):
    """
    Custom Autograd Function handling the mixed attention mechanism.
    Integrates Self-Attention and MoBA-Attention using Triton-optimized kernels.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        self_attn_max_seqlen,
        moba_max_seqlen_q,
        moba_chunk_size,
        moba_q_sh_indices,
        route_positions,
    ):
        ctx.self_attn_max_seqlen = self_attn_max_seqlen
        ctx.moba_max_seqlen_q = moba_max_seqlen_q
        ctx.moba_chunk_size = moba_chunk_size
        ctx.softmax_scale = softmax_scale = q.shape[-1] ** (-0.5)

        # 1. Self Attention (FlashAttention-3)
        self_attn_out_sh, self_attn_lse_hs = _flash_attn_3_varlen_forward(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self_attn_cu_seqlen,
            cu_seqlens_k=self_attn_cu_seqlen,
            max_seqlen_q=self_attn_max_seqlen,
            max_seqlen_k=self_attn_max_seqlen,
            softmax_scale=softmax_scale,
            causal=True,
        )

        # 2. MoBA Attention (FlashAttention-3 on selected sparse chunks)
        moba_attn_out, moba_attn_lse_hs = _flash_attn_3_varlen_forward(
            q=moba_q,
            k=moba_kv[:, 0],
            v=moba_kv[:, 1],
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv,
            max_seqlen_q=moba_max_seqlen_q,
            max_seqlen_k=moba_chunk_size,
            softmax_scale=softmax_scale,
            causal=False,
        )

        # 3. Output Merging
        # Optimization: Replaced Python-loop based LSE combination with Fused Triton Kernel
        self_attn_lse_sh = self_attn_lse_hs.t().contiguous()
        moba_attn_lse = moba_attn_lse_hs.t().contiguous()

        output = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=q.dtype
        )

        output_2d = output.view(-1, q.shape[2])
        self_attn_out_2d = self_attn_out_sh.reshape(-1, q.shape[2]).contiguous()
        self_attn_lse_flat = self_attn_lse_sh.view(-1).contiguous()
        moba_out_flat = moba_attn_out.view(-1, moba_attn_out.shape[-1]).contiguous()
        moba_lse_flat = moba_attn_lse.view(-1).contiguous()
        mixed_attn_lse_flat = torch.empty_like(self_attn_lse_flat)

        # Fused Kernel call
        _fused_merge_softmax_triton(
            self_attn_out_2d,
            self_attn_lse_flat,
            moba_out_flat,
            moba_lse_flat,
            route_positions,
            output_2d,
            mixed_attn_lse_flat,
        )


        mixed_attn_lse_sh = mixed_attn_lse_flat.view_as(self_attn_lse_sh)
        
        ctx.save_for_backward(
            output,
            mixed_attn_lse_sh,
            q,
            k,
            v,
            self_attn_cu_seqlen,
            moba_q,
            moba_kv,
            moba_cu_seqlen_q,
            moba_cu_seqlen_kv,
            moba_q_sh_indices,
        )

        return output

    @staticmethod
    def backward(ctx, d_output):
        self_attn_max_seqlen = ctx.self_attn_max_seqlen
        moba_max_seqlen_q = ctx.moba_max_seqlen_q
        moba_chunk_size = ctx.moba_chunk_size
        softmax_scale = ctx.softmax_scale
        (
            output, mixed_attn_vlse_sh, q, k, v, self_attn_cu_seqlen, moba_q,
            moba_kv, moba_cu_seqlen_q, moba_cu_seqlen_kv, moba_q_sh_indices,
        ) = ctx.saved_tensors
        d_output = d_output.contiguous()

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        # 1. Self Attention Backward (FlashAttention-3)
        _flash_attn_3_varlen_backward(
            dout=d_output, q=q, k=k, v=v, out=output, softmax_lse=mixed_attn_vlse_sh.t().contiguous(),
            dq=dq, dk=dk, dv=dv,
            cu_seqlens_q=self_attn_cu_seqlen, cu_seqlens_k=self_attn_cu_seqlen,
            max_seqlen_q=self_attn_max_seqlen, max_seqlen_k=self_attn_max_seqlen,
            softmax_scale=softmax_scale, causal=True,
            deterministic=True,
        )

        # 2. Gather inputs for MoBA Backward
        # Optimization: Triton kernel to sparsely gather gradients/outputs needed for MoBA backward
        headdim = q.shape[-1]
        d_output_2d = d_output.view(-1, headdim).contiguous()
        output_2d = output.view(-1, headdim).contiguous()
        mixed_attn_vlse_flat = mixed_attn_vlse_sh.view(-1).contiguous()

        num_selected = moba_q_sh_indices.numel()
        gathered_d_moba = torch.empty(
            (num_selected, headdim), device=d_output.device, dtype=d_output.dtype
        )
        gathered_moba_out = torch.empty(
            (num_selected, headdim), device=output.device, dtype=output.dtype
        )
        gathered_lse = torch.empty(
            (num_selected,), device=mixed_attn_vlse_flat.device, dtype=mixed_attn_vlse_flat.dtype
        )

        _gather_moba_backward_inputs_triton(
            d_output_2d,
            output_2d,
            mixed_attn_vlse_flat,
            moba_q_sh_indices,
            gathered_d_moba,
            gathered_moba_out,
            gathered_lse,
        )

        d_moba_output = gathered_d_moba.unsqueeze(1)
        moba_output = gathered_moba_out.unsqueeze(1)
        mixed_attn_vlse = gathered_lse.unsqueeze(0)
        
        dmq = torch.empty_like(moba_q)
        dmk = torch.empty_like(moba_kv[:, 0])
        dmv = torch.empty_like(moba_kv[:, 1])

        # 3. MoBA Attention Backward (FlashAttention-3)
        _flash_attn_3_varlen_backward(
            dout=d_moba_output, q=moba_q, k=moba_kv[:, 0], v=moba_kv[:, 1], out=moba_output,
            softmax_lse=mixed_attn_vlse,
            dq=dmq, dk=dmk, dv=dmv, 
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv, max_seqlen_q=moba_max_seqlen_q,
            max_seqlen_k=moba_chunk_size,
            softmax_scale=softmax_scale, causal=False,
            deterministic=True,
        )

        dmkv = torch.stack((dmk, dmv), dim=1)
        
        # Return gradients in order. 
        # Note: 'dmq' (sparse) will be scattered back to 'q.grad' by gather_moba_q's backward.
        return dq, dk, dv, None, dmq, dmkv, None, None, None, None, None, None, None


def moba_attn_varlen_triton_improve(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    moba_chunk_size: int,
    moba_topk: int,
) -> torch.Tensor:
    """
    Entry point for Efficient MoBA with Triton optimizations.
    
    Args:
        q (torch.Tensor): [seqlen, head, head_dim]
        k (torch.Tensor): [seqlen, head, head_dim]
        v (torch.Tensor): [seqlen, head, head_dim]
        cu_seqlens (torch.Tensor): Cumulative sequence length (FlashAttention format)
        max_seqlen (int): Max sequence length in batch
        moba_chunk_size (int): Size of chunks
        moba_topk (int): Number of chunks to attend to
    """

    kv = torch.stack((k, v), dim=1)

    """ some basic variables """
    # qkv shape = [ S, H, D ]
    seqlen, num_head, head_dim = q.shape

    """ prepare chunk meta """
    (
        cu_chunk,
        filtered_chunk_indices,
        num_filtered_chunk,
        chunk_to_batch,
    ) = calc_chunks(cu_seqlens, moba_chunk_size)

    # we will adjust selective topk to moba_topk - 1, as the last chunk is always chosen
    moba_topk = min(moba_topk - 1, num_filtered_chunk)
    need_moba_attn = moba_topk > 0

    # corner case: if no moba attn needed, just return self attn
    if not need_moba_attn:
        return _flash_attn_3_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=True,
        )

    self_attn_cu_seqlen = cu_chunk
    self_attn_max_seqlen = min(max_seqlen, moba_chunk_size)

    # filtered_kv is a dense matrix that only contains filtered chunk of kv
    filtered_kv_indices = torch.arange(
        0, moba_chunk_size, dtype=torch.int32, device=q.device
    )[None, :].repeat(num_filtered_chunk, 1)
    filtered_kv_indices += cu_chunk[filtered_chunk_indices][:, None]
    filtered_kv = kv.index_select(0, filtered_kv_indices.view(-1))

    """ calc key_gate_weight and gate """
    # Optimization: Use Triton Kernel to calculate chunk means
    filtered_k = filtered_kv[:, 0].reshape(
        num_filtered_chunk, moba_chunk_size, num_head, head_dim
    )
    chunk_means = _compute_chunk_means_triton(filtered_k, moba_chunk_size)

    chunk_end = cu_chunk.index_select(0, filtered_chunk_indices + 1)
    batch_indices = chunk_to_batch.index_select(0, filtered_chunk_indices)
    batch_end = cu_seqlens.index_select(0, batch_indices + 1)

    # Optimization: Use compact Top-K chunk indices instead of a dense gate mask.
    selected_chunks = _compute_selected_chunks_streaming_triton(
        q,
        chunk_means,
        chunk_end,
        batch_end,
        moba_topk,
    )

    (
        moba_q_sh_indices,
        route_positions,
        expert_counts,
    ) = _build_sparse_routes(
        selected_chunks,
        num_filtered_chunk,
        num_head,
        seqlen,
    )

    if moba_q_sh_indices.numel() == 0:
        return _flash_attn_3_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=True,
        )

    moba_q = gather_moba_q(q, moba_q_sh_indices)
    moba_q_sh_indices = moba_q_sh_indices.to(torch.int32)

    valid_expert_mask = expert_counts > 0
    active_expert_counts = expert_counts[valid_expert_mask]
    moba_max_seqlen_q = int(active_expert_counts.max().item())
    moba_cu_seqlen_q = torch.empty(
        active_expert_counts.numel() + 1,
        device=q.device,
        dtype=torch.int32,
    )
    moba_cu_seqlen_q[0] = 0
    moba_cu_seqlen_q[1:] = active_expert_counts.cumsum(
        dim=0,
        dtype=torch.int32,
    )

    moba_kv = (
        filtered_kv.view(
            num_filtered_chunk,
            moba_chunk_size,
            2,
            num_head,
            head_dim,
        )
        .permute(0, 3, 1, 2, 4)
        .reshape(
            num_filtered_chunk * num_head,
            moba_chunk_size,
            2,
            head_dim,
        )
    )
    moba_kv = moba_kv[valid_expert_mask]
    moba_kv = moba_kv.reshape(-1, 2, 1, head_dim)
    moba_cu_seqlen_kv = (
        torch.arange(
            active_expert_counts.numel() + 1,
            dtype=torch.int32,
            device=q.device,
        )
        * moba_chunk_size
    )

    return MixedAttention.apply(
        q,
        k,
        v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        self_attn_max_seqlen,
        moba_max_seqlen_q,
        moba_chunk_size,
        moba_q_sh_indices,
        route_positions,
    )
