# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Independent correctness tests for in-place Min-P logits filtering.

The main reference uses PyTorch FP32 logits-domain thresholds; a separate test
checks the probability definition with double-precision softmax. No TileOPs or
benchmark imports are required. Numerical checks run separately from timing.

On H100 the four path-oriented shapes cover register, partitioned, TLE cache
and general streaming execution. Other supported devices may route differently.
Full mode collects 75 cases; --quick collects 33 representative cases.
Invalid indices/thresholds and NaN logits are outside this numerical contract.
"""

import pytest
import torch

from flaggems_vllm.ops.apply_min_p import apply_min_p

from . import conftest as cfg

pytestmark = [
    pytest.mark.apply_min_p,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
# The cached case exceeds the largest register+shared prefix, exercising its tail.
PATH_SHAPES = [(7, 32768), (7, 32769), (128, 131073), (7, 262145)]
INDEX_LAYOUTS = (
    [(torch.int32, "contiguous")]
    if cfg.QUICK_MODE
    else [
        (dtype, layout)
        for dtype in (torch.int32, torch.int64)
        for layout in ("contiguous", "padded")
    ]
)
EDGE_SHAPES = (
    [(1, 1), (7, 1025)] if cfg.QUICK_MODE else [(1, 1), (3, 17), (7, 1025), (3, 262144)]
)


def _reference(logits, mapping, min_p):
    """FP32 thresholds, exact preservation of unfiltered input values."""
    values = logits.float()
    row_min_p = min_p[mapping.long()].float().unsqueeze(1)
    threshold = values.amax(dim=1, keepdim=True) + row_min_p.log()
    remove = (row_min_p > 0) & (values < threshold)
    return logits.masked_fill(remove, float("-inf"))


def _check(logits, mapping, min_p):
    ref_device = "cpu" if cfg.TO_CPU else logits.device
    expected = _reference(
        logits.detach().to(ref_device),
        mapping.to(ref_device),
        min_p.to(ref_device),
    )
    mapping_before, min_p_before = mapping.clone(), min_p.clone()
    ptr, stride = logits.data_ptr(), logits.stride()

    result = apply_min_p(logits, mapping, min_p)
    torch.cuda.synchronize()

    assert result is None
    assert logits.data_ptr() == ptr
    assert logits.stride() == stride
    # Check the exact mask and retained bits, including signed zero.
    bits = torch.int32 if logits.dtype == torch.float32 else torch.int16
    torch.testing.assert_close(
        logits.to(ref_device).view(bits), expected.view(bits), rtol=0, atol=0
    )
    torch.testing.assert_close(mapping, mapping_before, rtol=0, atol=0)
    torch.testing.assert_close(min_p, min_p_before, rtol=0, atol=0)


def _check_random_case(shape, dtype, index_dtype, layout):
    rows, vocab = shape
    generator = torch.Generator(device="cuda").manual_seed(2026)
    values = torch.randn(shape, device="cuda", dtype=dtype, generator=generator) * 3
    if layout == "padded":
        storage = torch.full((rows, vocab + 39), 123.0, device="cuda", dtype=dtype)
        logits = storage[:, 1 : vocab + 1]
        logits.copy_(values)
    else:
        storage = None
        logits = values
    # The second row maps to the disabled request; its signed zeros must survive.
    if rows > 1 and vocab > 1:
        logits[1, 0], logits[1, 1] = -0.0, 0.0
    before = storage.clone() if storage is not None else None
    min_p = torch.tensor([0.0, 0.01, 0.1, 0.5, 1.0], device="cuda", dtype=torch.float32)
    mapping = ((torch.arange(rows, device="cuda") * 3 + 2) % min_p.numel()).to(
        index_dtype
    )

    _check(logits, mapping, min_p)
    if storage is not None:
        torch.testing.assert_close(storage[:, :1], before[:, :1], rtol=0, atol=0)
        torch.testing.assert_close(
            storage[:, vocab + 1 :], before[:, vocab + 1 :], rtol=0, atol=0
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", PATH_SHAPES)
@pytest.mark.parametrize("index_dtype,layout", INDEX_LAYOUTS)
def test_apply_min_p_execution_paths(shape, dtype, index_dtype, layout):
    _check_random_case(shape, dtype, index_dtype, layout)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", EDGE_SHAPES)
def test_apply_min_p_shape_edges(shape, dtype):
    _check_random_case(shape, dtype, torch.int32, "contiguous")


@pytest.mark.parametrize("dtype", DTYPES)
def test_apply_min_p_small_threshold(dtype):
    # Threshold is about max-13.816, so both rows must retain and mask values.
    logits = torch.tensor(
        [[0, -10, -13, -14, -20], [100, 90, 87, 86, 80]], device="cuda", dtype=dtype
    )
    mapping = torch.zeros(2, device="cuda", dtype=torch.int32)
    min_p = torch.tensor([1e-6], device="cuda", dtype=torch.float32)
    _check(logits, mapping, min_p)


@pytest.mark.parametrize("dtype", DTYPES)
def test_apply_min_p_one_keeps_all_tied_maxima(dtype):
    logits = torch.tensor([[2, 2, 1, -5], [-3, -3, -4, -8]], device="cuda", dtype=dtype)
    mapping = torch.tensor([1, 0], device="cuda", dtype=torch.int64)
    min_p = torch.ones(2, device="cuda")
    _check(logits, mapping, min_p)
    expected = torch.tensor(
        [[2, 2, -float("inf"), -float("inf")], [-3, -3, -float("inf"), -float("inf")]],
        device="cuda",
        dtype=dtype,
    )
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_apply_min_p_infinities_and_disabled_rows(dtype):
    logits = torch.tensor(
        [
            [float("-inf"), -2, 0, 1],
            [float("-inf")] * 4,
            [float("inf"), 2, float("-inf"), float("inf")],
            [float("inf"), -2, 0, float("-inf")],
        ],
        device="cuda",
        dtype=dtype,
    )
    mapping = torch.tensor([1, 1, 1, 0], device="cuda", dtype=torch.int32)
    min_p = torch.tensor([0.0, 0.5], device="cuda")
    _check(logits, mapping, min_p)


def test_apply_min_p_probability_definition():
    # Independent softmax oracle, with values away from the threshold boundary.
    logits = torch.tensor(
        [[0, -0.2, -1, -4], [5, 4, 3, -10], [-2, -2, -3, -4]],
        device="cuda",
        dtype=torch.float32,
    )
    mapping = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int64)
    min_p = torch.tensor([0.25, 1.0, 0.0], device="cuda")
    probs = torch.softmax(logits.double(), dim=-1)
    cutoff = probs.amax(dim=-1, keepdim=True) * min_p[mapping, None].double()
    expected = logits.masked_fill(probs < cutoff, float("-inf"))
    apply_min_p(logits, mapping, min_p)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_apply_min_p_idempotent(dtype):
    # Reapplying the same filter must preserve the already-filtered values.
    generator = torch.Generator(device="cuda").manual_seed(31)
    logits = torch.randn((7, 32769), device="cuda", dtype=dtype, generator=generator)
    mapping = torch.arange(7, device="cuda", dtype=torch.int32) % 5
    min_p = torch.tensor([0, 0.01, 0.1, 0.5, 1], device="cuda")
    _check(logits, mapping, min_p)
    once = logits.clone()
    apply_min_p(logits, mapping, min_p)
    torch.testing.assert_close(logits, once, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(0, 17), (3, 0)])
def test_apply_min_p_empty(shape):
    logits = torch.empty(shape, device="cuda", dtype=torch.float32)
    mapping = torch.zeros(shape[0], device="cuda", dtype=torch.int32)
    min_p = torch.tensor([0.1], device="cuda", dtype=torch.float32)
    pointer, stride = logits.data_ptr(), logits.stride()
    assert apply_min_p(logits, mapping, min_p) is None
    assert logits.shape == shape
    assert logits.data_ptr() == pointer
    assert logits.stride() == stride
