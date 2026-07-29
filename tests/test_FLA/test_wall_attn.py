from __future__ import annotations

import math
import os

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.FLA.wall_attn import parallel_wall_attn


# The migrated target is the upstream parallel/prefill operator. Upstream
# decode/cache APIs live in separate source modules and are not exported by
# flaggems_vllm.ops.FLA.wall_attn, so they are outside this test file's scope.
os.environ["TRITON_F32_DEFAULT"] = "ieee"

RCP_LN2 = 1.4426950216
DEVICE = torch.device(flaggems_vllm.device)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Wall Attention tests require CUDA",
)


def _log_decay(*shape: int, scale: float = 0.05) -> torch.Tensor:
    return -torch.randn(*shape, device=DEVICE, dtype=torch.float32).abs() * scale


def _segment_cumsum(
    value: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    if cu_seqlens is None:
        return torch.cumsum(value.float(), dim=1)

    result = torch.empty_like(value, dtype=torch.float32)
    for segment in range(cu_seqlens.numel() - 1):
        start = int(cu_seqlens[segment].item())
        end = int(cu_seqlens[segment + 1].item())
        result[:, start:end] = torch.cumsum(
            value[:, start:end].float(),
            dim=1,
        )
    return result


def _naive_wall_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    *,
    scale: float,
    g_scalar: torch.Tensor | None = None,
    sink_bias: torch.Tensor | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable eager reference for the migrated parallel API."""
    B, T, HQ, K = q.shape
    H = k.shape[2]
    if g.shape != q.shape:
        raise ValueError("g must match q")
    if HQ % H != 0:
        raise ValueError("HQ must be divisible by H")

    group_size = HQ // H
    prefix = _segment_cumsum(g, cu_seqlens) * RCP_LN2
    k_expanded = k.repeat_interleave(group_size, dim=2)
    prefix_diff = prefix.unsqueeze(2) - prefix.unsqueeze(1)
    scores = (
        q.unsqueeze(2).float()
        * k_expanded.unsqueeze(1).float()
        * torch.exp2(prefix_diff)
    ).sum(-1)
    scores = scores.permute(0, 3, 1, 2).contiguous()

    query_index = torch.arange(T, device=DEVICE).view(1, T, 1)
    key_index = torch.arange(T, device=DEVICE).view(1, 1, T)
    valid = key_index <= query_index
    if window_size is not None:
        valid = valid & (query_index - key_index < window_size)
    if cu_seqlens is not None:
        segment_ids = torch.zeros(T, device=DEVICE, dtype=torch.long)
        for segment in range(cu_seqlens.numel() - 1):
            start = int(cu_seqlens[segment].item())
            end = int(cu_seqlens[segment + 1].item())
            segment_ids[start:end] = segment
        valid = valid & (segment_ids.view(1, T, 1) == segment_ids.view(1, 1, T))

    valid = valid.unsqueeze(1).expand(B, HQ, T, T)
    scores = scores.masked_fill(~valid, float("-inf"))
    scores = scores * (scale * RCP_LN2)

    if g_scalar is not None:
        scalar_prefix = _segment_cumsum(g_scalar, cu_seqlens) * RCP_LN2
        scalar_prefix = scalar_prefix.permute(0, 2, 1).float()
        scores = scores + scalar_prefix.unsqueeze(-1) - scalar_prefix.unsqueeze(-2)

    row_max = scores.amax(dim=-1, keepdim=True)
    stable_max = torch.where(
        torch.isfinite(row_max),
        row_max,
        torch.zeros_like(row_max),
    )
    weights = torch.exp2(scores - stable_max)
    denominator = weights.sum(dim=-1)
    if sink_bias is not None:
        sink_l2 = (sink_bias * RCP_LN2).view(1, HQ, 1)
        denominator = denominator + torch.exp2(sink_l2 - stable_max.squeeze(-1))
    weights = weights / denominator.unsqueeze(-1)

    v_expanded = (
        v.repeat_interleave(group_size, dim=2).permute(0, 2, 1, 3).float().contiguous()
    )
    # Avoid the known FP32 strided-batched SGEMM failure in this environment.
    output = (weights.unsqueeze(-1) * v_expanded.unsqueeze(-3)).sum(dim=-2)
    return output.transpose(1, 2).contiguous()


def _make_inputs(
    B: int,
    T: int,
    H: int,
    HQ: int,
    K: int,
    V: int,
    *,
    dtype: torch.dtype,
    seed: int,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    q = torch.randn(B, T, HQ, K, device=DEVICE, dtype=dtype)
    k = torch.randn(B, T, H, K, device=DEVICE, dtype=dtype)
    v = torch.randn(B, T, H, V, device=DEVICE, dtype=dtype)
    g = _log_decay(B, T, HQ, K).to(dtype)

    if requires_grad:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        g.requires_grad_(True)
    return q, k, v, g


def _assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    limit: float,
) -> None:
    actual = actual.detach().float()
    expected = expected.detach().float()
    assert torch.isfinite(actual).all(), f"{name}: actual contains NaN/Inf"
    assert torch.isfinite(expected).all(), f"{name}: reference contains NaN/Inf"

    diff = actual - expected
    max_abs = diff.abs().max().item()
    relative_l2 = (
        torch.linalg.vector_norm(diff) / (torch.linalg.vector_norm(expected) + 1e-8)
    ).item()
    print(
        f"{name}: max_abs={max_abs:.6e}, "
        f"relative_l2={relative_l2:.6e}, limit={limit:.6e}"
    )
    assert relative_l2 < limit


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("window_size", [None, 8])
@pytest.mark.parametrize(
    "B,T,H,HQ,K,V",
    [
        pytest.param(1, 48, 2, 4, 32, 16, id="gqa"),
        pytest.param(2, 31, 1, 1, 24, 8, id="mha"),
    ],
)
def test_parallel_matches_reference(
    dtype: torch.dtype,
    window_size: int | None,
    B: int,
    T: int,
    H: int,
    HQ: int,
    K: int,
    V: int,
) -> None:
    q, k, v, g = _make_inputs(
        B,
        T,
        H,
        HQ,
        K,
        V,
        dtype=dtype,
        seed=0,
    )
    scale = K**-0.5
    reference = _naive_wall_attn(
        q,
        k,
        v,
        g,
        scale=scale,
        window_size=window_size,
    )
    actual = parallel_wall_attn(
        q,
        k,
        v,
        g,
        scale=scale,
        window_size=window_size,
    )
    limit = 5e-3 if dtype == torch.float32 else 2e-2
    _assert_close("forward", actual, reference, limit)


def test_parallel_varlen_matches_reference() -> None:
    T1, T2 = 17, 23
    q, k, v, g = _make_inputs(
        1,
        T1 + T2,
        1,
        2,
        16,
        12,
        dtype=torch.float32,
        seed=2,
    )
    cu_seqlens = torch.tensor(
        [0, T1, T1 + T2],
        device=DEVICE,
        dtype=torch.long,
    )
    scale = q.shape[-1] ** -0.5
    reference = _naive_wall_attn(
        q,
        k,
        v,
        g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    actual = parallel_wall_attn(
        q,
        k,
        v,
        g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    _assert_close("varlen", actual, reference, 5e-3)


def test_parallel_sink_bias_matches_reference() -> None:
    q, k, v, g = _make_inputs(
        1,
        29,
        1,
        2,
        20,
        10,
        dtype=torch.float32,
        seed=3,
    )
    sink_bias = torch.randn(2, device=DEVICE, dtype=torch.float32) * 0.1
    scale = q.shape[-1] ** -0.5
    reference = _naive_wall_attn(
        q,
        k,
        v,
        g,
        scale=scale,
        sink_bias=sink_bias,
    )
    actual = parallel_wall_attn(
        q,
        k,
        v,
        g,
        scale=scale,
        sink_bias=sink_bias,
    )
    _assert_close("sink_bias", actual, reference, 5e-3)


def test_parallel_aggressive_gates_long_sequence() -> None:
    q, k, v, _ = _make_inputs(
        1,
        512,
        1,
        1,
        32,
        32,
        dtype=torch.float32,
        seed=42,
    )
    g = torch.full_like(q, math.log2(0.9))
    scale = q.shape[-1] ** -0.5
    reference = _naive_wall_attn(q, k, v, g, scale=scale)
    actual = parallel_wall_attn(q, k, v, g, scale=scale)
    _assert_close("aggressive_gate", actual, reference, 5e-3)


@pytest.mark.parametrize(
    "dtype,B,T,H,HQ,K,V",
    [
        pytest.param(
            torch.float32,
            1,
            24,
            2,
            4,
            16,
            12,
            id="gqa-fp32",
        ),
        pytest.param(
            torch.float32,
            1,
            32,
            2,
            2,
            32,
            48,
            id="mha-fp32",
        ),
        pytest.param(
            torch.bfloat16,
            1,
            16,
            2,
            2,
            64,
            64,
            id="mha-bf16",
        ),
    ],
)
def test_backward_matches_reference(
    dtype: torch.dtype,
    B: int,
    T: int,
    H: int,
    HQ: int,
    K: int,
    V: int,
) -> None:
    q, k, v, g = _make_inputs(
        B,
        T,
        H,
        HQ,
        K,
        V,
        dtype=dtype,
        seed=11,
        requires_grad=True,
    )
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    g_ref = g.detach().clone().requires_grad_(True)
    scale = K**-0.5

    actual = parallel_wall_attn(q, k, v, g, scale=scale)
    reference = _naive_wall_attn(
        q_ref,
        k_ref,
        v_ref,
        g_ref,
        scale=scale,
    )
    torch.manual_seed(12)
    output_gradient = torch.randn_like(actual)
    (actual.float() * output_gradient.float()).sum().backward()
    (reference.float() * output_gradient.float()).sum().backward()

    for name, actual_grad, reference_grad in [
        ("dq", q.grad, q_ref.grad),
        ("dk", k.grad, k_ref.grad),
        ("dv", v.grad, v_ref.grad),
        ("dg", g.grad, g_ref.grad),
    ]:
        assert actual_grad is not None
        assert reference_grad is not None
        if dtype == torch.bfloat16:
            limit = 5e-2 if name == "dg" else 2e-2
        else:
            limit = 1e-2
        _assert_close(name, actual_grad, reference_grad, limit)


def test_gate_gradient_is_finite_and_nonzero() -> None:
    q, k, v, g = _make_inputs(
        1,
        16,
        1,
        1,
        8,
        8,
        dtype=torch.float32,
        seed=3,
        requires_grad=True,
    )
    parallel_wall_attn(
        q,
        k,
        v,
        g,
        scale=q.shape[-1] ** -0.5,
    ).sum().backward()
    assert g.grad is not None
    assert torch.isfinite(g.grad).all()
    assert torch.count_nonzero(g.grad).item() > 0


def test_gate_gradient_matches_finite_difference() -> None:
    q, k, v, g0 = _make_inputs(
        1,
        4,
        1,
        1,
        3,
        3,
        dtype=torch.float32,
        seed=7,
    )
    scale = q.shape[-1] ** -0.5
    epsilon = 3e-3
    output_gradient = torch.randn(1, 4, 1, 3, device=DEVICE)

    g = g0.clone().requires_grad_(True)
    (parallel_wall_attn(q, k, v, g, scale=scale) * output_gradient).sum().backward()
    assert g.grad is not None
    analytical = g.grad.detach().clone()

    finite_difference = torch.empty_like(g0).flatten()
    flat = g0.flatten()
    for index in range(flat.numel()):
        positive = flat.clone()
        negative = flat.clone()
        positive[index] += epsilon
        negative[index] -= epsilon
        output_positive = parallel_wall_attn(
            q,
            k,
            v,
            positive.view_as(g0),
            scale=scale,
        )
        output_negative = parallel_wall_attn(
            q,
            k,
            v,
            negative.view_as(g0),
            scale=scale,
        )
        loss_positive = (output_positive * output_gradient).sum().double()
        loss_negative = (output_negative * output_gradient).sum().double()
        finite_difference[index] = (
            (loss_positive - loss_negative) / (2 * epsilon)
        ).float()

    _assert_close(
        "dg_finite_difference",
        analytical,
        finite_difference.view_as(analytical),
        2e-2,
    )


@pytest.mark.parametrize(
    "B,T,H,HQ,K,V",
    [
        pytest.param(1, 48, 2, 4, 32, 16, id="gqa"),
        pytest.param(2, 31, 1, 1, 24, 8, id="mha"),
    ],
)
def test_scalar_gate_matches_reference(
    B: int,
    T: int,
    H: int,
    HQ: int,
    K: int,
    V: int,
) -> None:
    q, k, v, g = _make_inputs(
        B,
        T,
        H,
        HQ,
        K,
        V,
        dtype=torch.float32,
        seed=42,
    )
    g_scalar = _log_decay(B, T, HQ)
    scale = K**-0.5
    reference = _naive_wall_attn(
        q,
        k,
        v,
        g,
        scale=scale,
        g_scalar=g_scalar,
    )
    actual = parallel_wall_attn(
        q,
        k,
        v,
        g,
        scale=scale,
        g_scalar=g_scalar,
    )
    _assert_close("scalar_gate", actual, reference, 5e-3)


def test_scalar_gate_gradient_matches_finite_difference() -> None:
    q, k, v, g = _make_inputs(
        1,
        4,
        1,
        1,
        3,
        3,
        dtype=torch.float32,
        seed=13,
    )
    g_scalar0 = _log_decay(1, 4, 1)
    output_gradient = torch.randn(1, 4, 1, 3, device=DEVICE)
    scale = q.shape[-1] ** -0.5
    epsilon = 3e-3

    g_scalar = g_scalar0.clone().requires_grad_(True)
    output = parallel_wall_attn(
        q,
        k,
        v,
        g,
        scale=scale,
        g_scalar=g_scalar,
    )
    (output * output_gradient).sum().backward()
    assert g_scalar.grad is not None
    analytical = g_scalar.grad.detach().clone()

    finite_difference = torch.empty_like(g_scalar0).flatten()
    flat = g_scalar0.flatten()
    for index in range(flat.numel()):
        positive = flat.clone()
        negative = flat.clone()
        positive[index] += epsilon
        negative[index] -= epsilon
        output_positive = parallel_wall_attn(
            q,
            k,
            v,
            g,
            scale=scale,
            g_scalar=positive.view_as(g_scalar0),
        )
        output_negative = parallel_wall_attn(
            q,
            k,
            v,
            g,
            scale=scale,
            g_scalar=negative.view_as(g_scalar0),
        )
        loss_positive = (output_positive * output_gradient).sum().double()
        loss_negative = (output_negative * output_gradient).sum().double()
        finite_difference[index] = (
            (loss_positive - loss_negative) / (2 * epsilon)
        ).float()

    _assert_close(
        "dg_scalar_finite_difference",
        analytical,
        finite_difference.view_as(analytical),
        2e-2,
    )
