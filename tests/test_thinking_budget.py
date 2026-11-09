# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors
"""Independent CPU-reference checks with scenarios adapted from vLLM
tests/v1/worker/test_gpu_thinking_budget.py at commit
947dd62c0700851a79b0a8f32cb823e25a003b31.
Sampler integration and request-level oversized-budget clamping are out of scope.
"""

from types import SimpleNamespace

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.runtime import torch_device_fn

pytestmark = pytest.mark.skipif(
    not torch_device_fn.is_available(), reason="Accelerator backend is unavailable"
)

DEVICE = flaggems_vllm.device
START = 90
END = 91
END_A = 92
END_B = 93
VOCAB_SIZE = 128


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


def _make_inputs(tokens, budget, *, natural=(END,), forced=(END,)):
    capacity = max(64, len(tokens) + 1)
    history = torch.zeros((4, capacity), dtype=torch.int32, device=DEVICE)
    history[3, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32, device=DEVICE)
    lengths = torch.zeros(4, dtype=torch.int32, device=DEVICE)
    lengths[3] = len(tokens)
    budgets = torch.full((4,), -1, dtype=torch.int32, device=DEVICE)
    budgets[3] = budget
    return SimpleNamespace(
        req_ids=torch.tensor([3], dtype=torch.int32, device=DEVICE),
        thinking_token_budget=budgets,
        all_token_ids=history,
        total_len=lengths,
        cached_last_start=torch.full((4,), -1, dtype=torch.int32, device=DEVICE),
        cached_last_end=torch.full((4,), -1, dtype=torch.int32, device=DEVICE),
        cached_scan_pos=torch.zeros(4, dtype=torch.int32, device=DEVICE),
        reasoning_start_token_ids=torch.tensor(
            [START], dtype=torch.int32, device=DEVICE
        ),
        natural_reasoning_end_token_ids=torch.tensor(
            natural, dtype=torch.int32, device=DEVICE
        ),
        reasoning_end_token_ids=torch.tensor(forced, dtype=torch.int32, device=DEVICE),
    )


def _apply(state, logits, input_ids, local_pos):
    mapping = torch.tensor([3] * len(input_ids), dtype=torch.int32, device=DEVICE)
    inputs = torch.tensor(input_ids, dtype=torch.int32, device=DEVICE)
    positions = torch.tensor(local_pos, dtype=torch.int32, device=DEVICE)
    args = (
        logits,
        state.req_ids,
        mapping,
        state.thinking_token_budget,
        state.all_token_ids,
        state.total_len,
        inputs,
        positions,
        state.cached_last_start,
        state.cached_last_end,
        state.cached_scan_pos,
        state.reasoning_start_token_ids,
        state.natural_reasoning_end_token_ids,
        state.reasoning_end_token_ids,
    )
    expected = _reference(args)
    pointers = [value.data_ptr() for value in args]
    strides = [value.stride() for value in args]
    with torch_device_fn.device(logits.device):
        result = flaggems_vllm.apply_thinking_budget(*args)
        assert result is None
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
    return logits.cpu()


@pytest.mark.apply_thinking_budget
def test_v2_thinking_budget_forces_end_after_budget_reached():
    state = _make_inputs([1, START, 10, 11, 12], 3)
    logits = torch.arange(VOCAB_SIZE, dtype=torch.float32, device=DEVICE).view(1, -1)
    expected = logits.cpu().clone()
    out = _apply(state, logits, input_ids=[12], local_pos=[0])
    expected[0, END] = 1.0e9
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.apply_thinking_budget
def test_v2_thinking_budget_restores_masked_end_token():
    state = _make_inputs([1, START, 10, 11, 12], 3)
    logits = torch.zeros((1, VOCAB_SIZE), dtype=torch.float32, device=DEVICE)
    logits[0, END] = -float("inf")
    out = _apply(state, logits, input_ids=[12], local_pos=[0])
    assert out[0, END] == pytest.approx(1.0e9)


@pytest.mark.apply_thinking_budget
def test_v2_thinking_budget_allows_tokens_before_budget():
    state = _make_inputs([1, START, 10, 11], 3)
    logits = torch.zeros((1, VOCAB_SIZE), dtype=torch.float32, device=DEVICE)
    out = _apply(state, logits, input_ids=[11], local_pos=[0])
    assert torch.all(out == 0)


@pytest.mark.apply_thinking_budget
def test_v2_thinking_budget_continues_multi_token_end_marker():
    state = _make_inputs(
        [1, START, 10, 11, 12], 3, natural=(END_A, END_B), forced=(END_A, END_B)
    )
    logits = torch.zeros((2, VOCAB_SIZE), dtype=torch.float32, device=DEVICE)
    out = _apply(state, logits, input_ids=[12, END_A], local_pos=[0, 1])
    assert out[0, END_A] == pytest.approx(1.0e9)
    assert out[1, END_B] == pytest.approx(1.0e9)


@pytest.mark.apply_thinking_budget
def test_v2_thinking_budget_uses_distinct_forced_end_marker():
    state = _make_inputs([1, START, 10, 11, 12], 3, forced=(END_A, END_B))
    logits = torch.zeros((2, VOCAB_SIZE), dtype=torch.float32, device=DEVICE)
    out = _apply(state, logits, input_ids=[12, END_A], local_pos=[0, 1])
    assert out[0, END_A] == pytest.approx(1.0e9)
    assert out[1, END_B] == pytest.approx(1.0e9)


@pytest.mark.apply_thinking_budget
def test_v2_thinking_budget_stops_after_natural_end_marker():
    state = _make_inputs([1, START, 10, END, 20, 21, 22], 3, forced=(END_A, END_B))
    logits = torch.zeros((1, VOCAB_SIZE), dtype=torch.float32, device=DEVICE)
    out = _apply(state, logits, input_ids=[22], local_pos=[0])
    assert torch.all(out == 0)


@pytest.mark.apply_thinking_budget
def test_v2_thinking_budget_ignores_plain_request():
    state = _make_inputs([1, START, 10, 11, 12], -1)
    logits = torch.zeros((1, VOCAB_SIZE), dtype=torch.float32, device=DEVICE)
    out = _apply(state, logits, input_ids=[12], local_pos=[0])
    assert torch.all(out == 0)


@pytest.mark.apply_thinking_budget
def test_v2_thinking_budget_latest_prefill_end_disables_forcing():
    state = _make_inputs([1, START, 10, 11, 12, END, 13], 3)
    logits = torch.zeros((1, VOCAB_SIZE), dtype=torch.float32, device=DEVICE)
    out = _apply(state, logits, input_ids=[13], local_pos=[0])
    assert torch.all(out == 0)


@pytest.mark.apply_thinking_budget
def test_v2_thinking_budget_uses_latest_prefill_start_boundary():
    state = _make_inputs([1, START, 10, 11, 12, END, 13, START, 14, 15, 16], 3)
    logits = torch.zeros((1, VOCAB_SIZE), dtype=torch.float32, device=DEVICE)
    out = _apply(state, logits, input_ids=[16], local_pos=[0])
    assert out[0, END] == pytest.approx(1.0e9)


@pytest.mark.apply_thinking_budget
def test_v2_thinking_budget_incrementally_scans_long_generation():
    tokens = [1, START, *([10] * 16382)]
    state = _make_inputs(tokens, 32768)
    out = _apply(
        state,
        torch.zeros((1, VOCAB_SIZE), dtype=torch.float32, device=DEVICE),
        [10],
        [0],
    )
    assert torch.all(out == 0)
    assert state.cached_scan_pos[3].item() == len(tokens)
    state.all_token_ids[3, len(tokens)] = 10
    state.total_len[3] = len(tokens) + 1
    out = _apply(
        state,
        torch.zeros((1, VOCAB_SIZE), dtype=torch.float32, device=DEVICE),
        [10],
        [0],
    )
    assert torch.all(out == 0)
    assert state.cached_scan_pos[3].item() == len(tokens) + 1


@pytest.mark.apply_thinking_budget
def test_v2_thinking_budget_continues_end_prefix_from_prompt():
    state = _make_inputs(
        [1, START, 10, 11, END_A], 3, natural=(END_A, END_B), forced=(END_A, END_B)
    )
    logits = torch.zeros((1, VOCAB_SIZE), dtype=torch.float32, device=DEVICE)
    out = _apply(state, logits, input_ids=[END_A], local_pos=[0])
    assert out[0, END_B] == pytest.approx(1.0e9)
    assert out[0, END_A] == 0
