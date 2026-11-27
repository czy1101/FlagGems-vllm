# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Check thinking-budget semantics with an independent CPU reference."""

import pytest
import torch

from flaggems_vllm import apply_thinking_budget
from flaggems_vllm import device as DEVICE
from flaggems_vllm import runtime

try:
    from vllm.v1.worker.gpu.sample.thinking_budget import (
        apply_thinking_budget as vllm_reference,
    )
except (ImportError, OSError):
    vllm_reference = None

pytestmark = [
    pytest.mark.apply_thinking_budget,
    pytest.mark.skipif(
        DEVICE == "cpu" or not runtime.torch_device_fn.is_available(),
        reason="requires the active accelerator",
    ),
]


def make_inputs(
    tokens,
    input_ids,
    budget=3,
    natural_end=(91,),
    forced_end=(91,),
    logits_kind="zeros",
    start=(90,),
):
    # Upstream RequestState allocates slot 3 first from a four-request pool.
    rows = len(input_ids)
    logits = torch.zeros((rows, 128), dtype=torch.float32, device=DEVICE)
    if logits_kind == "range":
        logits.copy_(torch.arange(128, dtype=torch.float32, device=DEVICE))
    elif logits_kind == "masked":
        logits[0, 91] = -float("inf")
    history = torch.zeros(
        (4, max(64, len(tokens) + 1)), dtype=torch.int32, device=DEVICE
    )
    history[3, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32, device=DEVICE)
    budgets = torch.full((4,), -1, dtype=torch.int32, device=DEVICE)
    budgets[3] = budget
    lengths = torch.zeros(4, dtype=torch.int32, device=DEVICE)
    lengths[3] = len(tokens)
    req_ids = torch.tensor([3], dtype=torch.int32, device=DEVICE)
    # Match model_runner's shared mapping for one sampled row per request.
    mapping = (
        req_ids
        if rows == 1
        else torch.full((rows,), 3, dtype=torch.int32, device=DEVICE)
    )
    return [
        logits,
        req_ids,
        mapping,
        budgets,
        history,
        lengths,
        torch.tensor(input_ids, dtype=torch.int32, device=DEVICE),
        torch.arange(rows, dtype=torch.int32, device=DEVICE),
        torch.full((4,), -1, dtype=torch.int32, device=DEVICE),
        torch.full((4,), -1, dtype=torch.int32, device=DEVICE),
        torch.zeros(4, dtype=torch.int32, device=DEVICE),
        torch.tensor(start, dtype=torch.int32, device=DEVICE),
        torch.tensor(natural_end, dtype=torch.int32, device=DEVICE),
        torch.tensor(forced_end, dtype=torch.int32, device=DEVICE),
    ]


def _marker_positions(tokens, marker):
    return [
        pos
        for pos in range(len(tokens) - len(marker) + 1)
        if tokens[pos : pos + len(marker)] == marker
    ]


def _cpu_reference(args):
    """Derive logits from full token sequences, independently of the caches."""
    (
        logits,
        req_ids,
        mapping,
        budgets,
        history,
        lengths,
        input_ids,
        local_positions,
        last_starts,
        last_ends,
        scan_positions,
        start,
        natural_end,
        forced_end,
    ) = [value.detach().cpu().clone() for value in args]
    start, natural_end, forced_end = (
        marker.tolist() for marker in (start, natural_end, forced_end)
    )
    histories = [
        row[:length] for row, length in zip(history.tolist(), lengths.tolist())
    ]

    # Cache contents follow vLLM's public helper: on a cold scan, retain the
    # last marker of each kind from the newest occupied 1024-token window.
    # Logit expectations below do not use these derived cache values.
    for req in req_ids.tolist():
        if budgets[req].item() < 0:
            continue
        committed = histories[req]
        starts = _marker_positions(committed, start)
        ends = _marker_positions(committed, natural_end)
        scan = scan_positions[req].item()
        last_start, last_end = last_starts[req].item(), last_ends[req].item()
        if scan > len(committed):
            scan, last_start, last_end = 0, -1, -1
        if scan == 0 and last_start < 0 and last_end < 0:
            latest = max(starts + ends, default=-1)
            if latest >= 0:
                blocks_from_end = (len(committed) - 1 - latest) // 1024 + 1
                window_start = max(0, len(committed) - blocks_from_end * 1024)
                last_start = max((p for p in starts if p >= window_start), default=-1)
                last_end = max((p for p in ends if p >= window_start), default=-1)
        else:
            last_start = max((p for p in starts if p >= scan), default=last_start)
            last_end = max((p for p in ends if p >= scan), default=last_end)
        last_starts[req], last_ends[req] = last_start, last_end
        scan_positions[req] = max(
            0, len(committed) - max(len(start), len(natural_end)) + 1
        )

    draft_tokens = input_ids.tolist()
    for row, (req, local_pos) in enumerate(
        zip(mapping.tolist(), local_positions.tolist())
    ):
        budget = budgets[req].item()
        if budget < 0:
            continue
        first = row - local_pos
        # Local position zero repeats the already committed final token.
        effective = histories[req] + draft_tokens[first + 1 : row + 1]
        last_start = max(_marker_positions(effective, start), default=-1)
        last_end = max(_marker_positions(effective, natural_end), default=-1)
        if last_start < 0 or last_start <= last_end:
            continue
        if len(effective) - last_start - len(start) < budget:
            continue
        prefix = max(
            (
                size
                for size in range(1, min(len(forced_end), len(effective) + 1))
                if effective[-size:] == forced_end[:size]
            ),
            default=0,
        )
        # vLLM changes only this score, including when it was previously -inf.
        logits[row, forced_end[prefix]] = 1.0e9
    return {0: logits, 8: last_starts, 9: last_ends, 10: scan_positions}


def check(args):
    before = [value.detach().cpu().clone() for value in args]
    expected = _cpu_reference(args)
    reference_args = [value.clone() for value in args] if vllm_reference else None
    output_pointers = {index: args[index].data_ptr() for index in expected}
    assert apply_thinking_budget(*args) is None
    for index, value in enumerate(args):
        torch.testing.assert_close(
            value.cpu(), expected.get(index, before[index]), rtol=0, atol=0
        )
    assert all(args[index].data_ptr() == ptr for index, ptr in output_pointers.items())
    if reference_args is not None:
        vllm_reference(*reference_args)
        for index, value in enumerate(reference_args):
            torch.testing.assert_close(
                value.cpu(), expected.get(index, before[index]), rtol=0, atol=0
            )


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            dict(tokens=[1, 90, 10, 11, 12], input_ids=[12], logits_kind="range"),
            id="budget-reached",
        ),
        pytest.param(
            dict(tokens=[1, 90, 10, 11, 12], input_ids=[12], logits_kind="masked"),
            id="restore-masked-end",
        ),
        pytest.param(dict(tokens=[1, 90, 10, 11], input_ids=[11]), id="before-budget"),
        pytest.param(
            dict(
                tokens=[1, 90, 10, 11, 12],
                input_ids=[12, 92],
                natural_end=(92, 93),
                forced_end=(92, 93),
            ),
            id="multi-token-end",
        ),
        pytest.param(
            dict(tokens=[1, 90, 10, 11, 12], input_ids=[12, 92], forced_end=(92, 93)),
            id="distinct-forced-end",
        ),
        pytest.param(
            dict(
                tokens=[1, 90, 10, 91, 20, 21, 22],
                input_ids=[22],
                forced_end=(92, 93),
            ),
            id="natural-end",
        ),
        pytest.param(
            dict(tokens=[1, 90, 10, 11, 12], input_ids=[12], budget=-1),
            id="plain-request",
        ),
        pytest.param(
            dict(tokens=[1, 90, 10, 11, 12, 91, 13], input_ids=[13]),
            id="latest-prefill-end",
        ),
        pytest.param(
            dict(tokens=[1, 90, 10, 11, 12, 91, 13, 90, 14, 15, 16], input_ids=[16]),
            id="latest-prefill-start",
        ),
        pytest.param(
            # ThinkingBudgetState clamps the upstream 2**40 input before launch.
            dict(tokens=[1, 90, 10, 11, 12], input_ids=[12], budget=2**31 - 1),
            id="clamped-budget",
        ),
        pytest.param(
            dict(
                tokens=[1, 90, 10, 11, 92],
                input_ids=[92],
                natural_end=(92, 93),
                forced_end=(92, 93),
            ),
            id="resume-end-prefix",
        ),
    ],
)
def test_thinking_budget(case):
    check(make_inputs(**case))


def test_thinking_budget_incremental_generation():
    tokens = [1, 90, *([10] * 16382)]
    args = make_inputs(tokens, [10], budget=32768)
    check(args)
    args[4][3, len(tokens)] = 10
    args[5][3] = len(tokens) + 1
    check(args)


@pytest.mark.parametrize(
    "case, forced_tokens",
    [
        pytest.param(
            dict(tokens=[1, 90], input_ids=[90], budget=0), [91], id="zero-budget"
        ),
        pytest.param(
            dict(tokens=[1, 10], input_ids=[10], budget=0), [None], id="no-start"
        ),
        pytest.param(
            dict(tokens=[1, 89], input_ids=[89], start=(89, 90), budget=0),
            [None],
            id="incomplete-start",
        ),
        pytest.param(
            dict(tokens=[1, 89], input_ids=[89, 90], start=(89, 90), budget=0),
            [None, 91],
            id="start-across-draft-boundary",
        ),
        pytest.param(
            dict(
                tokens=[1, 90, 10, 92],
                input_ids=[92, 93],
                budget=1,
                natural_end=(92, 93),
                forced_end=(94,),
            ),
            [94, None],
            id="natural-end-across-draft-boundary",
        ),
        pytest.param(
            dict(
                tokens=[1, 90, 10, 92, 92],
                input_ids=[92],
                budget=1,
                forced_end=(92, 92, 93),
            ),
            [93],
            id="longest-forced-end-prefix",
        ),
        pytest.param(
            dict(
                tokens=[1, 90, 10],
                input_ids=[10, 92, 92],
                budget=1,
                forced_end=(92, 92, 93),
            ),
            [92, 92, 93],
            id="draft-forced-end-continuation",
        ),
    ],
)
def test_thinking_budget_known_logits(case, forced_tokens):
    args = make_inputs(**case, logits_kind="range")
    expected = args[0].cpu().clone()
    for row, token in enumerate(forced_tokens):
        if token is not None:
            expected[row, token] = 1.0e9
    check(args)
    torch.testing.assert_close(args[0].cpu(), expected, rtol=0, atol=0)


def test_thinking_budget_cache_reset():
    args = make_inputs([1, 90, 10, 11, 12], [12], logits_kind="range")
    check(args)
    args[4][3, :2] = torch.tensor([1, 10], dtype=torch.int32, device=DEVICE)
    args[5][3] = 2
    args[6][0] = 10
    args[0].copy_(torch.arange(128, dtype=torch.float32, device=DEVICE))
    check(args)
    assert [args[index][3].item() for index in (8, 9, 10)] == [-1, -1, 2]


def test_thinking_budget_incremental_marker_boundaries():
    tokens = [1, 89]
    args = make_inputs(tokens, [89], budget=0, start=(89, 90), natural_end=(91, 92))
    check(args)
    for token in (90, 91, 92):
        args[4][3, len(tokens)] = token
        tokens.append(token)
        args[5][3] = len(tokens)
        args[6][0] = token
        args[0].zero_()
        check(args)
    assert [args[index][3].item() for index in (8, 9, 10)] == [1, 3, 4]


@pytest.mark.parametrize("latest_marker", [90, 91], ids=["latest-start", "latest-end"])
def test_thinking_budget_cold_cache_window(latest_marker):
    tokens = [10] * 2050
    tokens[1] = 91 if latest_marker == 90 else 90
    tokens[-1] = latest_marker
    args = make_inputs(tokens, [latest_marker], budget=0)
    check(args)
    expected = [2049, -1, 2050] if latest_marker == 90 else [-1, 2049, 2050]
    assert [args[index][3].item() for index in (8, 9, 10)] == expected


def test_thinking_budget_mixed_requests():
    args = make_inputs([1, 90, 10, 11, 12], [12, 12, 12], logits_kind="range")
    args[1] = torch.tensor([2, 3, 0], dtype=torch.int32, device=DEVICE)
    args[2] = torch.tensor([3, 0, 2], dtype=torch.int32, device=DEVICE)
    args[3][1:3] = torch.tensor([0, 3], dtype=torch.int32, device=DEVICE)
    args[4][0, :5] = args[4][3, :5]
    args[4][2, :5] = torch.tensor([1, 90, 10, 91, 12], dtype=torch.int32, device=DEVICE)
    args[5][[0, 2]] = 5
    args[7].zero_()
    # Disabled and inactive slots must preserve even stale cache contents.
    for index in (8, 9, 10):
        args[index][:2] = 42
    expected = args[0].cpu().clone()
    expected[0, 91] = 1.0e9
    check(args)
    torch.testing.assert_close(args[0].cpu(), expected, rtol=0, atol=0)
