import inspect
import os

import pytest
import torch
import torch.nn.functional as F
from fla.ops.gla import fused_recurrent_gla

import flaggems_vllm
from flaggems_vllm.ops.FLA.chunk_gla import chunk_gla
from flaggems_vllm.ops.FLA.index import prepare_chunk_indices as _prepare_chunk_indices


def _cuda_available() -> bool:
    return torch.cuda.is_available() and flaggems_vllm.device == "cuda"


pytestmark = [
    pytest.mark.chunk_gla,
    pytest.mark.skipif(not _cuda_available(), reason="CUDA required"),
]


@pytest.fixture(scope="module", autouse=True)
def _compat_prepare_chunk_indices_kwarg():
    sig = inspect.signature(_prepare_chunk_indices)
    if "cu_seqlens_cpu" in sig.parameters:
        yield
        return
    import sys

    mod = sys.modules["flaggems_vllm.ops.FLA.chunk_gla"]
    old = mod.prepare_chunk_indices

    def _wrapped(cu_seqlens, chunk_size, cu_seqlens_cpu=None):
        del cu_seqlens_cpu
        return _prepare_chunk_indices(cu_seqlens, chunk_size)

    mod.prepare_chunk_indices = _wrapped
    try:
        yield
    finally:
        mod.prepare_chunk_indices = old


def _assert_close(name, actual, expected, ratio, err_atol=1e-6):
    """对标 FLA assert_close：error_rate（RMSE/信号RMS）判断。"""
    abs_atol = (actual.detach() - expected.detach()).flatten().abs().max().item()
    error_rate = (
        (actual.detach() - expected.detach()).flatten().square().mean().sqrt()
        / actual.detach().flatten().square().mean().sqrt().clamp_min(1e-8)
    ).item()
    print(f"[{name}] diff: {abs_atol:.6f} ratio: {error_rate:.6f}")
    if abs_atol <= err_atol:
        return
    assert not torch.isnan(actual).any(), f"{name}: NaN detected in actual"
    assert not torch.isnan(expected).any(), f"{name}: NaN detected in expected"
    assert error_rate < ratio, f"{name}: diff: {abs_atol:.6f} ratio: {error_rate:.6f}"


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "gate_logit_normalizer", "dtype"),
    [
        pytest.param(*c, id="B{}-T{}-H{}-D{}-gate_logit_normalizer{}-{}".format(*c))
        for c in [
            (1, 63, 1, 64, 1.0, torch.float16),
            (2, 1024, 4, 60, 1.0, torch.float16),
            (2, 1024, 8, 128, 0.1, torch.float16),
            (2, 1024, 8, 128, 1.0, torch.float16),
            (2, 1024, 8, 128, 10.0, torch.float16),
            (4, 2048, 8, 64, 1.0, torch.float16),
        ]
    ],
)
def test_chunk(B, T, H, D, dtype, gate_logit_normalizer):
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    device = flaggems_vllm.device

    q = torch.rand(B, T, H, D, dtype=dtype, device=device).requires_grad_()
    k = torch.rand(B, T, H, D, dtype=dtype, device=device).requires_grad_()
    v = torch.rand(B, T, H, D, dtype=dtype, device=device).requires_grad_()
    g = (
        F.logsigmoid(torch.rand(B, T, H, D, dtype=dtype, device=device))
        / gate_logit_normalizer
    ).requires_grad_()
    h0 = torch.rand(B, H, D, D, dtype=torch.float32, device=device).requires_grad_()
    do = torch.randn_like(v)
    dht = torch.randn(B, H, D, D, dtype=torch.float32, device=device)

    tri, tri_ht = chunk_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=D**-0.5,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=None,
    )
    ((tri * do).sum() + (tri_ht * dht).sum().to(do.dtype)).backward()
    tri_dq, tri_dk, tri_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
    tri_dg, tri_dh0 = g.grad.clone(), h0.grad.clone()
    q.grad = k.grad = v.grad = g.grad = h0.grad = None

    ref, ref_ht = fused_recurrent_gla(
        q=q,
        k=k,
        v=v,
        gk=g,
        initial_state=h0,
        output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dq, ref_dk, ref_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
    ref_dg, ref_dh0 = g.grad.clone(), h0.grad.clone()

    _assert_close("o", ref, tri, 0.004)
    _assert_close("ht", ref_ht, tri_ht, 0.005)
    _assert_close("dq", ref_dq, tri_dq, 0.005)
    _assert_close("dk", ref_dk, tri_dk, 0.005)
    _assert_close("dv", ref_dv, tri_dv, 0.005)
    _assert_close("dg", ref_dg, tri_dg, 0.005)
    _assert_close("dh0", ref_dh0, tri_dh0, 0.005)


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "dtype"),
    [
        pytest.param(*c, id="B{}-T{}-H{}-D{}-{}".format(*c))
        for c in [
            (2, 256, 4, 64, torch.float),
            (2, 1024, 4, 128, torch.float16),
        ]
    ],
)
def test_chunk_state_v_first(B, T, H, D, dtype):
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    device = flaggems_vllm.device

    q = torch.rand(B, T, H, D, dtype=dtype, device=device)
    k = torch.rand(B, T, H, D, dtype=dtype, device=device)
    v = torch.rand(B, T, H, D, dtype=dtype, device=device)
    g = F.logsigmoid(torch.rand(B, T, H, D, dtype=dtype, device=device))
    h0 = torch.rand(B, H, D, D, dtype=torch.float32, device=device)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    def run(state_v_first):
        q_ = q.detach().clone().requires_grad_()
        k_ = k.detach().clone().requires_grad_()
        v_ = v.detach().clone().requires_grad_()
        g_ = g.detach().clone().requires_grad_()
        h0_in = h0.transpose(-1, -2).contiguous() if state_v_first else h0.clone()
        dht_in = dht.transpose(-1, -2).contiguous() if state_v_first else dht
        h0_in = h0_in.requires_grad_()
        out, ht = chunk_gla(
            q=q_,
            k=k_,
            v=v_,
            g=g_,
            scale=D**-0.5,
            initial_state=h0_in,
            output_final_state=True,
            state_v_first=state_v_first,
        )
        ((out * do).sum() + (ht * dht_in).sum()).backward()
        return out, ht, q_.grad, k_.grad, v_.grad, g_.grad, h0_in.grad

    ref_o, ref_ht, ref_dq, ref_dk, ref_dv, ref_dg, ref_dh0 = run(False)
    tri_o, tri_ht, tri_dq, tri_dk, tri_dv, tri_dg, tri_dh0 = run(True)

    _assert_close("o", ref_o, tri_o, 0.005)
    _assert_close("ht", ref_ht, tri_ht.transpose(-1, -2), 0.005)
    _assert_close("dq", ref_dq, tri_dq, 0.005)
    _assert_close("dk", ref_dk, tri_dk, 0.005)
    _assert_close("dv", ref_dv, tri_dv, 0.005)
    _assert_close("dg", ref_dg, tri_dg, 0.005)
    _assert_close("dh0", ref_dh0, tri_dh0.transpose(-1, -2), 0.005)


@pytest.mark.parametrize(
    ("H", "D", "cu_seqlens", "dtype"),
    [
        pytest.param(*c, id="H{}-D{}-cu_seqlens{}-{}".format(*c))
        for c in [
            (4, 64, [0, 15], torch.float16),
            (4, 64, [0, 256, 500, 1000], torch.float16),
            (4, 100, [0, 15, 100, 300, 1200, 2000], torch.float16),
        ]
    ],
)
def test_chunk_varlen(H, D, cu_seqlens, dtype):
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    device = flaggems_vllm.device

    N = len(cu_seqlens) - 1
    T = cu_seqlens[-1]
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.rand(1, T, H, D, dtype=dtype, device=device).requires_grad_()
    k = torch.rand(1, T, H, D, dtype=dtype, device=device).requires_grad_()
    v = torch.rand(1, T, H, D, dtype=dtype, device=device).requires_grad_()
    g = F.logsigmoid(
        torch.rand(1, T, H, D, dtype=dtype, device=device)
    ).requires_grad_()
    h0 = torch.rand(N, H, D, D, dtype=torch.float32, device=device).requires_grad_()
    do = torch.randn_like(v)
    dht = torch.rand(N, H, D, D, dtype=torch.float32, device=device)

    ref, ref_ht = fused_recurrent_gla(
        q=q,
        k=k,
        v=v,
        gk=g,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu,
    )
    ((ref * do).sum() + (ref_ht * dht).sum().to(do.dtype)).backward()
    ref_dq, ref_dk, ref_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
    ref_dg, ref_dh0 = g.grad.clone(), h0.grad.clone()
    q.grad = k.grad = v.grad = g.grad = h0.grad = None

    tri, tri_ht = chunk_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=D**-0.5,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dq, tri_dk, tri_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
    tri_dg, tri_dh0 = g.grad.clone(), h0.grad.clone()

    _assert_close("o", ref, tri, 0.004)
    _assert_close("ht", ref_ht, tri_ht, 0.005)
    _assert_close("dq", ref_dq, tri_dq, 0.005)
    _assert_close("dk", ref_dk, tri_dk, 0.005)
    _assert_close("dv", ref_dv, tri_dv, 0.005)
    _assert_close("dg", ref_dg, tri_dg, 0.005)
    _assert_close("dh0", ref_dh0, tri_dh0, 0.005)
