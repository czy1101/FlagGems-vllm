# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors

import statistics
from pathlib import Path

import pytest
import torch
import vllm
from vllm.v1.worker.gpu.sample.thinking_budget import (
    apply_thinking_budget as vllm_apply_thinking_budget,
)

import flaggems_vllm
from flaggems_vllm.runtime import torch_device_fn

from . import base
from .conftest import Config
from .consts import BenchMode

SCAN_WARMUP_STEPS = 10
SCAN_ITERATIONS = 100
STEP_MODES = {"incremental-decode", "cold-prefill", "cold-resume-worst-case"}


def _create_case(mode, batch, history_len, device):
    device = torch.device(device)
    extra = SCAN_WARMUP_STEPS + SCAN_ITERATIONS if mode == "incremental-decode" else 0
    request_order = list(reversed(range(batch)))
    req_ids = torch.tensor(request_order, device=device, dtype=torch.int32)
    history = torch.full(
        (batch, history_len + extra + 1), 10, device=device, dtype=torch.int32
    )
    marker_pos = history_len - 1 if mode == "cold-prefill" else 1
    history[:, marker_pos] = 90
    budgets = [history_len + extra + 1] * batch
    if mode == "forced-end":
        budgets = [max(1, history_len - 2)] * batch
    starts = [marker_pos] * batch
    scan = [history_len] * batch
    if mode == "batched-mixed":
        for req in request_order[1::2]:
            budgets[req], starts[req], scan[req] = -1, -1, 0
    return (
        torch.zeros((batch, 128), device=device, dtype=torch.float32),
        req_ids,
        req_ids.clone(),
        torch.tensor(budgets, device=device, dtype=torch.int32),
        history,
        torch.full((batch,), history_len, device=device, dtype=torch.int32),
        torch.full((batch,), 10, device=device, dtype=torch.int32),
        torch.zeros(batch, device=device, dtype=torch.int32),
        torch.tensor(starts, device=device, dtype=torch.int32),
        torch.full((batch,), -1, device=device, dtype=torch.int32),
        torch.tensor(scan, device=device, dtype=torch.int32),
        torch.tensor([90], device=device, dtype=torch.int32),
        torch.tensor([91], device=device, dtype=torch.int32),
        torch.tensor([91], device=device, dtype=torch.int32),
    )


def _prepare_step(inputs, mode, history_len, step):
    if mode == "incremental-decode":
        inputs[4][0, history_len + step] = 10
        inputs[5].fill_(history_len + step + 1)
    else:
        inputs[8].fill_(-1)
        inputs[9].fill_(-1)
        inputs[10].zero_()


def _time_call(fn):
    start = torch_device_fn.Event(enable_timing=True)
    end = torch_device_fn.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def _measure_scan_steps(fn, inputs, mode, history_len):
    fn()
    torch_device_fn.synchronize()
    samples = []
    for step in range(SCAN_WARMUP_STEPS + SCAN_ITERATIONS):
        _prepare_step(inputs, mode, history_len, step)
        # Match upstream: finish history/cache preparation before timing.
        torch_device_fn.synchronize()
        elapsed_ms = _time_call(fn)
        if step >= SCAN_WARMUP_STEPS:
            samples.append(elapsed_ms)
    return statistics.median(samples)


class ThinkingBudgetBenchmark(base.Benchmark):
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_DTYPES = [torch.float32]
    DEFAULT_SHAPE_DESC = "mode, batch, history"
    DEFAULT_SHAPES = ()
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))

    def init_user_config(self):
        # Keep the same workload and honor --shape_file on every backend.
        # The generic base replaces this file with vendor-specific defaults.
        self.mode = Config.mode
        self.set_dtypes(Config.user_desired_dtypes)
        self.set_metrics(Config.user_desired_metrics)
        self.set_shapes(Config.shape_file)

    def get_input_iter(self, dtype):
        for mode, batch, history_len in self.shapes:
            self._case = (mode, batch, history_len)
            yield _create_case(mode, batch, history_len, self.device)

    def record_shapes(self, *args, **kwargs):
        mode, batch, history_len = self._case
        return {"mode": mode, "batch": batch, "history": history_len, "vocab": 128}

    def get_latency(self, op, *args, **kwargs):
        mode, _, history_len = self._case
        # Each implementation starts from independent history, logits, and caches.
        # Input copies and preparation stay outside the timed calls.
        args = tuple(tensor.clone() for tensor in args)
        fn = lambda: op(*args, **kwargs)
        with torch_device_fn.device(args[0].device):
            if mode in STEP_MODES:
                return _measure_scan_steps(fn, args, mode, history_len)
            # Upstream initializes each case with one untimed call.
            fn()
            torch_device_fn.synchronize()
            return super().get_latency(op, *args, **kwargs)


@pytest.mark.apply_thinking_budget
@pytest.mark.skipif(
    not torch_device_fn.is_available(), reason="Accelerator backend is unavailable"
)
def test_apply_thinking_budget():
    if not Config.query and Config.mode != BenchMode.KERNEL:
        pytest.skip("Stateful scan workloads require --mode kernel")
    if not Config.query and Config.mode == BenchMode.KERNEL:
        if not hasattr(torch_device_fn, "Event"):
            pytest.skip("Backend timing events are unavailable")
    if not Config.query:
        print(
            f"Baseline (Torch Latency): vLLM {vllm.__version__}; "
            "Speedup = vLLM / FlagGems"
        )
    ThinkingBudgetBenchmark(
        op_name="apply_thinking_budget",
        torch_op=vllm_apply_thinking_budget,
        gems_op=flaggems_vllm.apply_thinking_budget,
    ).run()
