# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Independent numerical tests for thinking-budget logits and cache updates."""

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.runtime import torch_device_fn

from . import conftest as cfg

pytestmark = [
    pytest.mark.skipif(
        not torch_device_fn.is_available(), reason="Accelerator backend is unavailable"
    ),
]
DTYPES = [torch.float16, torch.float32]
if flaggems_vllm.runtime.device.support_bf16:
    DTYPES.insert(1, torch.bfloat16)


def _positions(tokens, marker, lo=0, hi=None):
    hi = len(tokens) if hi is None else hi
    return [pos for pos in range(lo, hi) if tokens[pos : pos + len(marker)] == marker]


def _reference(args):
    values = [value.detach().cpu().clone() for value in args]
    (
        logits,
        req_ids,
        mapping,
        budgets,
        history,
        lengths,
        input_ids,
        local_positions,
        starts,
        ends,
        scan_positions,
        start_ids,
        natural_ids,
        forced_ids,
    ) = values
    start_ids, natural_ids, forced_ids = (
        value.tolist() for value in (start_ids, natural_ids, forced_ids)
    )
    for req in req_ids.tolist():
        if budgets[req] < 0:
            continue
        tokens = history[req, : lengths[req]].tolist()
        scan, start, end = (
            int(scan_positions[req]),
            int(starts[req]),
            int(ends[req]),
        )
        if scan > len(tokens):
            scan, start, end = 0, -1, -1
        if scan == 0 and start < 0 and end < 0:
            hi = len(tokens)
            while hi > 0 and start < 0 and end < 0:
                lo = max(0, hi - 1024)
                start = max(_positions(tokens, start_ids, lo, hi), default=-1)
                end = max(_positions(tokens, natural_ids, lo, hi), default=-1)
                hi = lo
        else:
            start = max([start] + _positions(tokens, start_ids, scan))
            end = max([end] + _positions(tokens, natural_ids, scan))
        starts[req], ends[req] = start, end
        scan_positions[req] = max(
            0, len(tokens) - max(len(start_ids), len(natural_ids)) + 1
        )
    for row, req in enumerate(mapping.tolist()):
        budget = int(budgets[req])
        if budget < 0:
            continue
        local = int(local_positions[row])
        first = row - local
        tokens = history[req, : lengths[req]].tolist()
        tokens += input_ids[first + 1 : row + 1].tolist()
        start = max(_positions(tokens, start_ids), default=-1)
        end = max(_positions(tokens, natural_ids), default=-1)
        if start < 0 or start <= end:
            continue
        if len(tokens) - start - len(start_ids) < budget:
            continue
        prefix = max(
            (
                size
                for size in range(1, len(forced_ids))
                if size <= len(tokens) and tokens[-size:] == forced_ids[:size]
            ),
            default=0,
        )
        logits[row, forced_ids[prefix]] = torch.tensor(1e9).to(logits.dtype)
    return values


def _make_inputs(
    histories,
    budgets,
    *,
    drafts=None,
    req_ids=None,
    cached=None,
    start=(90, 91),
    natural=(92, 93),
    forced=(94, 95, 96),
    dtype=torch.float32,
    padded=False,
    capacity=None,
    vocab=113,
):
    requests = len(histories)
    drafts = [[] for _ in histories] if drafts is None else drafts
    req_ids = list(range(requests)) if req_ids is None else req_ids
    mapping, local, inputs = [], [], []
    for req in reversed(req_ids):
        mapping += [req] * (len(drafts[req]) + 1)
        local += list(range(len(drafts[req]) + 1))
        inputs += [7] + drafts[req]
    rows = len(mapping)
    capacity = (
        max([1] + [len(tokens) for tokens in histories])
        if capacity is None
        else capacity
    )
    device = flaggems_vllm.device

    def integer(data):
        return torch.tensor(data, dtype=torch.int32, device=device)

    logits = torch.randn((rows, vocab), device=device, dtype=dtype)
    if rows:
        logits[:, :4] = torch.tensor(
            [-0.0, 0.0, float("nan"), float("inf")], device=device, dtype=dtype
        )
    history = torch.full((requests, capacity), 17, device=device, dtype=torch.int32)
    for req, tokens in enumerate(histories):
        if tokens:
            history[req, : len(tokens)] = integer(tokens)
    if padded:
        for name, value in (("logits", logits), ("history", history)):
            storage = torch.full(
                (value.shape[0], value.shape[1] + 7),
                19,
                device=device,
                dtype=value.dtype,
            )
            view = storage[:, 1 : value.shape[1] + 1]
            view.copy_(value)
            if name == "logits":
                logits = view
            else:
                history = view
    cached = [(-1, -1, 0)] * requests if cached is None else cached
    return [
        logits,
        integer(req_ids),
        integer(mapping),
        integer(budgets),
        history,
        integer([len(tokens) for tokens in histories]),
        integer(inputs),
        integer(local),
        integer([value[0] for value in cached]),
        integer([value[1] for value in cached]),
        integer([value[2] for value in cached]),
        integer(start),
        integer(natural),
        integer(forced),
    ]


def _check(args):
    expected = _reference(args)
    pointers = [value.data_ptr() for value in args]
    strides = [value.stride() for value in args]
    guard = args[0]._base
    before_guard = guard.clone() if guard is not None else None
    with torch_device_fn.device(args[0].device):
        assert flaggems_vllm.apply_thinking_budget(*args) is None
        torch_device_fn.synchronize()
    for index, (actual, want) in enumerate(zip(args, expected)):
        assert actual.data_ptr() == pointers[index]
        assert actual.stride() == strides[index]
        if index == 0:
            bits = torch.int16 if actual.element_size() == 2 else torch.int32
            torch.testing.assert_close(
                actual.cpu().view(bits), want.view(bits), rtol=0, atol=0
            )
        else:
            torch.testing.assert_close(actual.cpu(), want, rtol=0, atol=0)
    if guard is not None:
        torch.testing.assert_close(
            guard[:, :1].cpu(), before_guard[:, :1].cpu(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            guard[:, args[0].shape[1] + 1 :].cpu(),
            before_guard[:, args[0].shape[1] + 1 :].cpu(),
            rtol=0,
            atol=0,
        )


SCENARIOS = [
    ("no_start", [7, 8, 9], 0, [], {}),
    ("disabled_stale_cache", [90, 91, 7], -1, [], {"cached": [(20, 30, 40)]}),
    ("under_budget", [90, 91, 7], 8, [], {}),
    ("zero_budget", [90, 91], 0, [], {}),
    ("exact_budget", [90, 91, 7, 8], 2, [], {}),
    ("natural_end", [90, 91, 7, 92, 93], 0, [], {}),
    ("reopened", [90, 91, 92, 93, 90, 91, 7], 1, [], {}),
    ("end_prefix", [90, 91, 7, 94], 1, [], {}),
    ("end_prefix_long", [90, 91, 7, 94, 95], 1, [], {}),
    ("overlapping_prefix", [90, 91, 94, 94], 0, [], {"forced": (94, 94, 95)}),
    ("start_boundary", [7, 90], 0, [91, 7], {}),
    ("natural_boundary", [90, 91, 7, 92], 0, [93], {}),
    ("forced_boundary", [90, 91, 7, 94], 0, [95], {}),
    ("empty_history", [], 0, [90, 91, 7], {}),
    ("scan_block_boundary", [7] * 1023 + [90, 91, 7], 0, [], {}),
    ("older_marker_block", [90, 91] + [7] * 2050, 1, [], {}),
    ("rollback", [90, 91, 7], 0, [], {"cached": [(20, 30, 40)]}),
    ("warm", [90, 91, 7, 92, 93], 0, [], {"cached": [(0, -1, 2)]}),
    (
        "single_markers",
        [90, 7],
        0,
        [],
        {"start": (90,), "natural": (92,), "forced": (94,)},
    ),
    (
        "long_markers",
        [80, 81, 82, 7, 86, 87],
        0,
        [],
        {"start": (80, 81, 82), "natural": (83, 84, 85), "forced": (86, 87, 88, 89)},
    ),
]
DTYPE_SCENARIOS = {
    "under_budget",
    "exact_budget",
    "end_prefix_long",
    "forced_boundary",
    "single_markers",
    "long_markers",
}
QUICK_SCENARIOS = {
    "disabled_stale_cache",
    "under_budget",
    "exact_budget",
    "natural_boundary",
    "forced_boundary",
    "scan_block_boundary",
    "long_markers",
}


@pytest.mark.apply_thinking_budget
@pytest.mark.parametrize(
    "history,budget,draft,options,dtype",
    [
        pytest.param(
            history,
            budget,
            draft,
            options,
            dtype,
            id=f"{name}-{str(dtype).split('.')[-1]}",
        )
        for name, history, budget, draft, options in SCENARIOS
        if not cfg.QUICK_MODE or name in QUICK_SCENARIOS
        for dtype in DTYPES
        if dtype == torch.float32 or name in DTYPE_SCENARIOS
    ],
)
def test_apply_thinking_budget_scenarios(history, budget, draft, options, dtype):
    _check(_make_inputs([history], [budget], drafts=[draft], dtype=dtype, **options))


@pytest.mark.apply_thinking_budget
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("padded", [False] if cfg.QUICK_MODE else [False, True])
def test_apply_thinking_budget_request_mapping(dtype, padded):
    args = _make_inputs(
        [[7], [90, 91, 7], [90, 91, 92, 93], [90, 91, 94]],
        [-1, 0, 0, 1],
        drafts=[[], [8, 94], [], [95]],
        req_ids=[3, 1, 2],
        dtype=dtype,
        padded=padded,
    )
    _check(args)


@pytest.mark.apply_thinking_budget
@pytest.mark.parametrize("dtype", DTYPES)
def test_apply_thinking_budget_cache_progression(dtype):
    args = _make_inputs([[90]], [0], capacity=16, dtype=dtype)
    _check(args)
    for tokens in ([90, 91], [90, 91, 7], [90, 91, 7, 92, 93], [90, 91]):
        args[4][0, : len(tokens)] = torch.tensor(
            tokens, device=args[4].device, dtype=torch.int32
        )
        args[5][0] = len(tokens)
        _check(args)
        _check(args)


@pytest.mark.apply_thinking_budget
@pytest.mark.parametrize("mode", ["no_tokens", "no_requests", "neither"])
def test_apply_thinking_budget_empty_stages(mode):
    args = _make_inputs([[90, 91]], [0])
    if mode in ("no_tokens", "neither"):
        for index in (0, 2, 6, 7):
            args[index] = args[index][:0]
    if mode in ("no_requests", "neither"):
        args[1] = args[1][:0]
        args[8][0], args[10][0] = 0, 1
    _check(args)
