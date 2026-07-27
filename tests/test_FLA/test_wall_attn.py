from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from flaggems_vllm.ops.FLA.wall_attn import parallel_wall_attn


RCP_LN2 = 1.4426950216


def _cuda_available() -> bool:
    return torch.cuda.is_available()


pytestmark = pytest.mark.skipif(
    not _cuda_available(),
    reason="Wall Attention tests require a CUDA device",
)


def _naive_wall_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Independent PyTorch correctness reference.

    This implementation is intentionally local to this test file. It does not
    import helpers or reference functions from other files under tests/test_FLA.
    """

    if g.shape != q.shape:
        raise ValueError("g must have the same shape as q")

    B, T, HQ, K = q.shape
    H = k.shape[2]

    if HQ % H != 0:
        raise ValueError("HQ must be divisible by H")

    group_size = HQ // H

    prefix = torch.cumsum(g.float(), dim=1) * RCP_LN2
    k_expanded = k.repeat_interleave(group_size, dim=2)

    prefix_diff = (
        prefix.unsqueeze(2)
        - prefix.unsqueeze(1)
    )

    scores = (
        q.unsqueeze(2).float()
        * k_expanded.unsqueeze(1).float()
        * torch.exp2(prefix_diff)
    ).sum(-1)

    scores = scores.permute(0, 3, 1, 2).contiguous()

    query_index = torch.arange(
        T,
        device=q.device,
        dtype=torch.long,
    ).view(1, T, 1)

    key_index = torch.arange(
        T,
        device=q.device,
        dtype=torch.long,
    ).view(1, 1, T)

    causal = key_index <= query_index
    causal = causal.unsqueeze(1).expand(B, HQ, T, T)

    scores = scores.masked_fill(
        ~causal,
        float("-inf"),
    )
    scores = scores * (scale * RCP_LN2)

    row_max = scores.amax(dim=-1, keepdim=True)
    stable_max = torch.where(
        torch.isfinite(row_max),
        row_max,
        torch.zeros_like(row_max),
    )

    probabilities = torch.exp2(scores - stable_max)
    denominator = probabilities.sum(dim=-1, keepdim=True)
    weights = probabilities / denominator

    v_expanded = (
        v.repeat_interleave(group_size, dim=2)
        .permute(0, 2, 1, 3)
        .float()
        .contiguous()
    )

    # The correctness reference only runs on small shapes. Use an explicit
    # weighted reduction instead of batched torch.matmul so the test does not
    # depend on the CUDA strided-batched SGEMM backend.
    output = (
        weights.unsqueeze(-1)
        * v_expanded.unsqueeze(-3)
    ).sum(dim=-2)

    return output.transpose(1, 2).contiguous()


def _relative_l2(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    actual = actual.detach().float()
    expected = expected.detach().float()

    numerator = torch.linalg.vector_norm(actual - expected)
    denominator = torch.linalg.vector_norm(expected)

    return (
        numerator
        / (denominator + 1e-8)
    ).item()


def _assert_tensor_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    relative_limit: float,
) -> None:
    assert torch.isfinite(actual).all(), f"{name}: actual contains NaN/Inf"
    assert torch.isfinite(expected).all(), f"{name}: reference contains NaN/Inf"

    max_abs = (
        actual.detach().float()
        - expected.detach().float()
    ).abs().max().item()

    rel_l2 = _relative_l2(actual, expected)

    print(
        f"{name}: max_abs={max_abs:.6f} "
        f"rel_l2={rel_l2:.6f} "
        f"limit={relative_limit:.6f}"
    )

    assert rel_l2 < relative_limit, (
        f"{name}: relative L2 {rel_l2:.6f} "
        f"exceeds {relative_limit:.6f}"
    )


def _make_inputs(
    *,
    batch: int,
    sequence: int,
    key_heads: int,
    query_heads: int,
    key_dim: int,
    value_dim: int,
    dtype: torch.dtype,
    requires_grad: bool,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(42)

    device = torch.device("cuda")

    q = (
        torch.randn(
            batch,
            sequence,
            query_heads,
            key_dim,
            device=device,
            dtype=dtype,
        )
        * 0.5
    )

    k = (
        torch.randn(
            batch,
            sequence,
            key_heads,
            key_dim,
            device=device,
            dtype=dtype,
        )
        * 0.5
    )

    v = torch.randn(
        batch,
        sequence,
        key_heads,
        value_dim,
        device=device,
        dtype=dtype,
    )

    g = (
        F.logsigmoid(
            torch.randn(
                batch,
                sequence,
                query_heads,
                key_dim,
                device=device,
                dtype=torch.float32,
            )
        )
        / 32.0
    ).to(dtype)

    if requires_grad:
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)
        g = g.detach().requires_grad_(True)

    return q, k, v, g


@pytest.mark.parametrize(
    "key_heads,query_heads",
    [
        pytest.param(2, 2, id="mha"),
        pytest.param(2, 8, id="gqa"),
    ],
)
def test_wall_attn_forward_matches_reference(
    key_heads: int,
    query_heads: int,
) -> None:
    q, k, v, g = _make_inputs(
        batch=1,
        sequence=32,
        key_heads=key_heads,
        query_heads=query_heads,
        key_dim=64,
        value_dim=64,
        dtype=torch.bfloat16,
        requires_grad=False,
    )

    scale = 1.0 / math.sqrt(q.shape[-1])

    actual = parallel_wall_attn(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=scale,
    )

    expected = _naive_wall_attn(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=scale,
    )

    _assert_tensor_close(
        "forward",
        actual,
        expected,
        relative_limit=0.015,
    )


def test_wall_attn_backward_matches_reference() -> None:
    q, k, v, g = _make_inputs(
        batch=1,
        sequence=16,
        key_heads=2,
        query_heads=2,
        key_dim=64,
        value_dim=64,
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    g_ref = g.detach().clone().requires_grad_(True)

    scale = 1.0 / math.sqrt(q.shape[-1])

    actual = parallel_wall_attn(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=scale,
    )

    expected = _naive_wall_attn(
        q=q_ref,
        k=k_ref,
        v=v_ref,
        g=g_ref,
        scale=scale,
    )

    torch.manual_seed(7)
    output_gradient = torch.randn_like(actual)

    actual_loss = (
        actual.float()
        * output_gradient.float()
    ).sum()

    expected_loss = (
        expected.float()
        * output_gradient.float()
    ).sum()

    actual_loss.backward()
    expected_loss.backward()

    gradient_pairs = [
        ("dq", q.grad, q_ref.grad, 0.02),
        ("dk", k.grad, k_ref.grad, 0.02),
        ("dv", v.grad, v_ref.grad, 0.02),
        ("dg", g.grad, g_ref.grad, 0.05),
    ]

    for name, actual_grad, expected_grad, limit in gradient_pairs:
        assert actual_grad is not None, f"{name}: actual gradient is None"
        assert expected_grad is not None, f"{name}: reference gradient is None"

        _assert_tensor_close(
            name,
            actual_grad,
            expected_grad,
            relative_limit=limit,
        )

    assert torch.count_nonzero(g.grad).item() > 0
