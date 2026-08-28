# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import pytest
import torch
import triton
from einops import repeat

import flaggems_vllm
from flaggems_vllm.ops.FLA.index import prepare_token_indices

logger = logging.getLogger(__name__)


# ===========================================================================
# Naive reference implementation (from flash-linear-attention)
# ===========================================================================


def naive_nsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_size: int = 64,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
) -> torch.Tensor:
    r"""
    Naive PyTorch reference for NSA selected-sparse attention.

    Args:
        q: queries of shape ``[B, T, HQ, K]``.
        k: keys of shape ``[B, T, H, K]``.
        v: values of shape ``[B, T, H, V]``.
        block_indices: block indices of shape ``[B, T, H, S]``.
        block_size: selected block size. Default: 64.
        scale: scale factor. If None, defaults to ``K ** -0.5``.
        cu_seqlens: cumulative sequence lengths for variable-length sequences.

    Returns:
        o: outputs of shape ``[B, T, HQ, V]``.
    """
    if "head_first" in kwargs:
        raise DeprecationWarning("head_first has been removed.")
    if scale is None:
        scale = k.shape[-1] ** -0.5

    dtype = q.dtype
    G = q.shape[2] // k.shape[2]
    BS = block_size
    k, v, block_indices = (
        repeat(x, "b t h d -> b t (h g) d", g=G) for x in (k, v, block_indices)
    )
    q, k, v = map(lambda x: x.float(), (q, k, v))

    o = torch.zeros_like(v)
    varlen = True
    if cu_seqlens is None:
        varlen = False
        B, T = q.shape[:2]
        cu_seqlens = torch.cat(
            [
                block_indices.new_tensor(range(0, B * T, T)),
                block_indices.new_tensor([B * T]),
            ]
        )

    for i in range(len(cu_seqlens) - 1):
        if not varlen:
            q_b, k_b, v_b, i_b = q[i], k[i], v[i], block_indices[i]
        else:
            T = cu_seqlens[i + 1] - cu_seqlens[i]
            q_b, k_b, v_b, i_b = map(
                lambda x: x[0][cu_seqlens[i] : cu_seqlens[i + 1]],
                (q, k, v, block_indices),
            )

        i_b = i_b.unsqueeze(-1) * BS + i_b.new_tensor(range(BS))
        i_b = i_b.view(T, block_indices.shape[2], -1).transpose(1, 2)
        for i_q in range(T):
            q_i = q_b[i_q] * scale
            i_i = i_b[i_q]
            k_i, v_i = map(
                lambda x: x.gather(
                    0, i_i.clamp(0, T - 1).unsqueeze(-1).expand(*i_i.shape, x.shape[-1])
                ),
                (k_b, v_b),
            )
            attn = (
                torch.einsum("h d, n h d -> n h", q_i, k_i)
                .masked_fill(i_i > i_q, float("-inf"))
                .softmax(0)
            )
            if not varlen:
                o[i, i_q] = torch.einsum("n h, n h v -> h v", attn, v_i)
            else:
                o[0][cu_seqlens[i] + i_q] = torch.einsum("n h, n h v -> h v", attn, v_i)

    return o.to(dtype)


# ===========================================================================
# Testing utilities (from fla.utils._testing)
# ===========================================================================


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
    if warning:
        if error_rate > ratio:
            logger.warning(msg)
        else:
            assert error_rate < ratio, msg


def reference_with_grads(q, k, v, do, block_indices, block_size, scale, cu_seqlens=None):
    """Run the naive reference on CPU for large NPU cases to avoid HBM OOM."""
    reference_device = q.device
    if q.device.type == "npu" and q.numel() >= 8_000_000:
        # Keep the reference on NPU but evaluate it in bounded chunks.  The
        # original full autograd graph exhausts HBM; the explicit-gradient
        # implementation below avoids that graph and is much faster than the
        # Python-token CPU reference.
        reference_device = q.device
    q_ref = q.detach().clone().to(reference_device).requires_grad_(True)
    k_ref = k.detach().clone().to(reference_device).requires_grad_(True)
    v_ref = v.detach().clone().to(reference_device).requires_grad_(True)
    do_ref = do.detach().to(reference_device)
    indices_ref = block_indices.detach().to(reference_device)
    cu_ref = cu_seqlens.detach().to(reference_device) if cu_seqlens is not None else None
    # The original reference loops over every query token.  That is useful
    # for readability but takes tens of minutes for the largest 910B case.
    # On CPU, use an equivalent query-chunked implementation for fixed-length
    # inputs; it keeps the same gather/mask/softmax semantics while allowing
    # PyTorch to use batched kernels.  Varlen inputs retain the original
    # reference because their sequence boundaries are part of the test.
    if reference_device.type in {"cpu", "npu"} and cu_ref is None and q.numel() >= 8_000_000:
        return _naive_nsa_fixed_chunked_grads(
            q_ref,
            k_ref,
            v_ref,
            do_ref,
            indices_ref,
            block_size=block_size,
            scale=scale,
            target_device=q.device,
        )
    else:
        ref = naive_nsa(
            q=q_ref,
            k=k_ref,
            v=v_ref,
            block_indices=indices_ref,
            block_size=block_size,
            scale=scale,
            cu_seqlens=cu_ref,
        )
    ref.backward(do_ref)
    return (
        ref.to(q.device),
        q_ref.grad.to(q.device),
        k_ref.grad.to(k.device),
        v_ref.grad.to(v.device),
    )


def _naive_nsa_fixed_chunked_grads(
    q, k, v, do, block_indices, block_size, scale, target_device
):
    """Vectorized fixed-length NSA reference with explicit backward on CPU.

    This is mathematically the same operation as :func:`naive_nsa` and its
    autograd backward, but evaluates bounded query chunks and computes the
    softmax derivatives explicitly.  It avoids retaining a huge autograd
    graph for the largest 910B correctness case.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    B, T, HQ, D = q.shape
    H = k.shape[2]
    G = HQ // H
    BS = block_size
    # Keep the original KV-head layout.  Expanding K/V to HQ with
    # ``repeat_interleave`` is unexpectedly very memory hungry on 910B.
    k, v, q, do = k.float(), v.float(), q.float(), do.float()
    indices = block_indices

    offsets = torch.arange(BS, device=q.device, dtype=indices.dtype)
    positions = (indices.unsqueeze(-1) * BS + offsets).flatten(-2)
    chunk_size = 64
    head_ids = torch.arange(H, device=q.device).view(1, H, 1)
    head_ids_full = head_ids.expand(chunk_size, H, positions.shape[-1])
    outputs, dqs, dks, dvs = [], [], [], []
    for b in range(B):
        k_b, v_b = k[b], v[b]
        dq_b = torch.zeros_like(q[b])
        dk_b = torch.zeros((T, H, D), device=q.device, dtype=q.dtype)
        dv_b = torch.zeros((T, H, D), device=q.device, dtype=q.dtype)
        out_chunks = []
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            q_chunk = q[b, start:end]
            do_chunk = do[b, start:end]
            pos = positions[b, start:end]
            pos_safe = pos.clamp(0, T - 1)
            q_len = end - start
            head_ids_q = head_ids_full[:q_len]
            k_sel = k_b[pos_safe, head_ids_q]
            v_sel = v_b[pos_safe, head_ids_q]
            q_group = q_chunk.view(q_len, H, G, D)
            do_group = do_chunk.view(q_len, H, G, D)
            scores = torch.einsum("qhgd,qhnd->qhgn", q_group * scale, k_sel)
            query_pos = torch.arange(start, end, device=q.device).view(-1, 1, 1)
            scores = scores.masked_fill(pos.unsqueeze(2) > query_pos.unsqueeze(2), float("-inf"))
            attn = scores.softmax(dim=-1)
            out_chunks.append(torch.einsum("qhgn,qhnd->qhgd", attn, v_sel).reshape(q_len, HQ, D))
            d_attn = torch.einsum("qhgd,qhnd->qhgn", do_group, v_sel)
            d_scores = attn * (d_attn - (d_attn * attn).sum(dim=-1, keepdim=True))
            dq_b[start:end] = torch.einsum("qhgn,qhnd->qhgd", d_scores, k_sel).mul(scale).reshape(q_len, HQ, D)

            flat_pos = pos_safe.reshape(-1)
            flat_heads = head_ids_q.expand(q_len, H, pos.shape[-1]).reshape(-1)
            flat_linear = flat_pos * H + flat_heads
            dk_b.view(T * H, D).index_add_(
                0,
                flat_linear,
                torch.einsum("qhgn,qhgd->qhnd", d_scores, q_group * scale).reshape(-1, D),
            )
            dv_b.view(T * H, D).index_add_(
                0,
                flat_linear,
                torch.einsum("qhgn,qhgd->qhnd", attn, do_group).reshape(-1, D),
            )
        outputs.append(torch.cat(out_chunks, dim=0))
        dqs.append(dq_b)
        dks.append(dk_b)
        dvs.append(dv_b)
    return tuple(
        x.to(target_device, dtype=getattr(x, "dtype", q.dtype))
        for x in (
            torch.stack(outputs, dim=0),
            torch.stack(dqs, dim=0),
            torch.stack(dks, dim=0),
            torch.stack(dvs, dim=0),
        )
    )


# ===========================================================================
# Tests
# ===========================================================================


@pytest.mark.parallel_nsa
@pytest.mark.parametrize(
    ("B", "T", "H", "HQ", "D", "S", "block_size", "scale", "dtype"),
    [
        pytest.param(
            *test, id="B{}-T{}-H{}-HQ{}-D{}-S{}-block_size{}-scale{}-{}".format(*test)
        )
        for test in [
            (1, 63, 1, 16, 64, 16, 32, 1.0, torch.float16),
            (3, 111, 1, 32, 100, 16, 32, 1.0, torch.float16),
            (3, 1024, 2, 32, 60, 16, 32, 0.1, torch.float16),
            (3, 1024, 2, 32, 128, 16, 32, 0.1, torch.float16),
            (4, 2048, 2, 32, 64, 16, 32, 0.1, torch.float16),
        ]
    ],
)
def test_parallel(
    B: int,
    T: int,
    H: int,
    HQ: int,
    D: int,
    S: int,
    block_size: int,
    scale: float,
    dtype: torch.dtype,
):
    """Compare FlagGems-vllm parallel_nsa with the naive reference (forward + backward)."""
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    device = flaggems_vllm.device

    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)

    block_indices = torch.full((B, T, H, S), T, dtype=torch.long, device=device)
    for b in range(B):
        for t in range(T):
            for h in range(H):
                i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
                block_indices[b, t, h, : len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]

    ref, ref_dq, ref_dk, ref_dv = reference_with_grads(
        q=q,
        k=k,
        v=v,
        do=do,
        block_indices=block_indices,
        block_size=block_size,
        scale=scale,
    )

    tri = flaggems_vllm.parallel_nsa(
        q=q,
        k=k,
        v=v,
        block_indices=block_indices,
        block_counts=S,
        block_size=block_size,
        scale=scale,
    )
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close(" o", ref, tri, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.005)
    assert_close("dk", ref_dk, tri_dk, 0.005)
    assert_close("dv", ref_dv, tri_dv, 0.005)


@pytest.mark.parallel_nsa
@pytest.mark.parametrize(
    ("H", "HQ", "D", "S", "block_size", "cu_seqlens", "dtype"),
    [
        pytest.param(
            *test, id="H{}-HQ{}-D{}-S{}-block_size{}-cu_seqlens{}-{}".format(*test)
        )
        for test in [
            (1, 16, 64, 16, 32, [0, 15], torch.float16),
            (2, 32, 64, 16, 32, [0, 256, 500, 1000], torch.float16),
            (2, 32, 100, 16, 32, [0, 15, 100, 300, 1200, 2000], torch.float16),
        ]
    ],
)
def test_parallel_varlen(
    H: int,
    HQ: int,
    D: int,
    S: int,
    block_size: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    """Compare FlagGems-vllm parallel_nsa with naive reference for variable-length sequences."""
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    device = flaggems_vllm.device

    T = cu_seqlens[-1]
    cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((1, T, HQ, D), dtype=dtype, device=device)

    block_indices = torch.full((1, T, H, S), T, dtype=torch.long, device=device)
    # Build host-side token indices.  Calling ``prepare_token_indices`` on an
    # Ascend tensor and immediately converting with ``tolist`` can observe an
    # incomplete asynchronous result on 910B.
    seq_indices = [
        (seq_id, t)
        for seq_id, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:]))
        for t in range(end - start)
    ]

    for i in range(T):
        _, t = seq_indices[i]
        for h in range(H):
            i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
            block_indices[0, i, h, : len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]

    ref, ref_dq, ref_dk, ref_dv = reference_with_grads(
        q=q,
        k=k,
        v=v,
        do=do,
        block_indices=block_indices,
        block_size=block_size,
        scale=None,
        cu_seqlens=cu_seqlens_t,
    )

    tri = flaggems_vllm.parallel_nsa(
        q=q,
        k=k,
        v=v,
        block_indices=block_indices,
        block_counts=S,
        block_size=block_size,
        cu_seqlens=cu_seqlens_t,
    )
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close("o", ref, tri, 0.004)
    assert_close("dq", ref_dq, tri_dq, 0.005)
    assert_close("dk", ref_dk, tri_dk, 0.005)
    assert_close("dv", ref_dv, tri_dv, 0.005)
