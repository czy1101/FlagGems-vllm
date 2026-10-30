# SPDX-License-Identifier: Apache-2.0 AND MIT
# Copyright 2026 FlagOS Contributors
"""Performance-only Triton latency benchmark for apply_min_p.

    python -m pytest -s benchmark/test_apply_min_p.py --level core --warmup 100 --iter 300
"""

import functools
import hashlib
import inspect
import json
import math
import statistics
import time
import warnings
from pathlib import Path

import pytest
import torch

from flaggems_vllm.ops.apply_min_p import apply_min_p

from . import base
from .conftest import Config, emit_record_logger, update_result
from .consts import BenchMode

# Cold-L2 preparation helpers follow TileOPs benchmarks/timing.py.
# MIT License
#
# Copyright (c) Tile-AI.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE


@functools.lru_cache(maxsize=8)
def _get_l2_flush_cache(device):
    l2_bytes = getattr(torch.cuda.get_device_properties(device), "L2_cache_size", 0)
    if l2_bytes <= 0:
        warnings.warn("L2 size unavailable; using the upstream 512 MB flush buffer")
        l2_bytes = int(256e6)
    return torch.empty(2 * l2_bytes, dtype=torch.int8, device=device)


def _measure_device_time(fn, reset, device, warmup_ms, repeat_ms, label):
    """Measure current-stream elapsed time, including inter-kernel gaps.

    Reset and cold-L2 preparation precede the start event on the same stream.
    Warmup and sampling budgets use wall time, including preparation and
    synchronization; each sampling phase collects at least five readings.
    Compilation and lazy event initialization are outside both budgets.
    """
    if not math.isfinite(warmup_ms) or warmup_ms < 0:
        raise ValueError("warmup_ms must be finite and >= 0")
    if not math.isfinite(repeat_ms) or repeat_ms <= 0:
        raise ValueError("repeat_ms must be finite and > 0")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Min-P CUDA Event timing requires a CUDA device")

    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        cache = _get_l2_flush_cache(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        end.record(stream)
        end.synchronize()

        def prepare():
            reset()
            cache.zero_()

        # Compile/autotune and finish first allocations before collecting samples.
        prepare()
        _ = fn()
        stream.synchronize()
        del _

        warmup_start = time.perf_counter()
        warmup_count = 0
        while (time.perf_counter() - warmup_start) * 1000 < warmup_ms:
            prepare()
            _ = fn()
            stream.synchronize()
            del _
            warmup_count += 1
        warmup_wall_ms = (time.perf_counter() - warmup_start) * 1000

        samples = []
        sample_start = time.perf_counter()
        while (
            len(samples) < 5 or (time.perf_counter() - sample_start) * 1000 < repeat_ms
        ):
            prepare()
            start.record(stream)
            _ = fn()
            end.record(stream)
            end.synchronize()
            samples.append({"latency_ms": start.elapsed_time(end)})
            # Keep any returned result alive until the GPU is finished.
            del _
        sample_wall_ms = (time.perf_counter() - sample_start) * 1000

    return samples, {
        "timing": "cuda-event",
        "device_metric": "complete-call current-stream elapsed time",
        "includes_inter_kernel_gaps": True,
        "preparation": "input reset and cold-L2 flush before the start event",
        "budget_basis": "wall time including preparation and synchronization",
        "warmup_ms": warmup_ms,
        "sample_ms": repeat_ms,
        "warmup_wall_ms": warmup_wall_ms,
        "sample_wall_ms": sample_wall_ms,
        "warmup_iterations": warmup_count,
        "sample_iterations": len(samples),
        "label": label,
    }


class ApplyMinPBenchmark(base.Benchmark):
    DEFAULT_METRICS = ["latency"]
    DEFAULT_SHAPE_DESC = "batch, vocab, dtype"
    DEFAULT_SHAPES = [
        (1, 128256, "bfloat16"),
        (1, 128256, "float32"),
        (64, 128256, "bfloat16"),
        (256, 128256, "float16"),
        (256, 128256, "bfloat16"),
        (128, 151936, "bfloat16"),
        (17, 129280, "bfloat16"),
        (100, 32000, "float16"),
    ]
    CASE_DTYPES = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    def get_input_iter(self, dtype):
        generator = torch.Generator(device=self.device).manual_seed(2026)
        for shape in self.shapes:
            if len(shape) != 3:
                raise ValueError("Expected apply_min_p shape (batch, vocab, dtype)")
            tokens, vocab, case_dtype = shape
            if tokens <= 0 or vocab <= 0 or case_dtype not in self.CASE_DTYPES:
                raise ValueError(
                    "Expected positive batch/vocab and dtype float16/bfloat16/float32"
                )
            if dtype != self.CASE_DTYPES[case_dtype]:
                continue
            logits = torch.randn(
                (tokens, vocab),
                device=self.device,
                dtype=dtype,
                generator=generator,
            )
            mapping = torch.arange(tokens, device=self.device, dtype=torch.int32)
            min_p = torch.full((tokens,), 0.1, device=self.device, dtype=torch.float32)
            yield logits, mapping, min_p

    def record_shapes(self, *args, **kwargs):
        return {
            "batch": args[0].shape[0],
            "vocab": args[0].shape[1],
            "inputs": super().record_shapes(*args, **kwargs),
        }


def _run_benchmark(bench):
    if Config.query:
        return bench.run()
    if Config.mode != BenchMode.KERNEL:
        raise ValueError("The CUDA Event latency benchmark requires --mode kernel")

    bench.init_user_config()
    triton_path = Path(inspect.getfile(bench.gems_op))
    source = {
        "triton_path": str(triton_path),
        "triton_sha256": hashlib.sha256(triton_path.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(),
    }
    protocol = {
        "timer": "cuda-event",
        "metric": "latency_ms",
        "measurement": "complete-call current-stream elapsed time, including kernel gaps",
        "warmup_ms": Config.warm_up,
        "sample_ms": Config.repetition,
        "minimum_samples": 5,
        "budget_basis": "wall time including preparation and synchronization",
        "l2": "flush before every invocation, after input reset",
        "input_reset": "private input restored outside every timed invocation",
        "min_p": 0.1,
        "seed": 2026,
        "api": "in-place, identity request mapping",
    }
    print("Implementation: Triton; timer: CUDA Event")
    print(json.dumps(source))
    for dtype in bench.to_bench_dtypes:
        rows = []
        print(f"\napply_min_p dtype={dtype}; times in ms")
        print(f"{'Batch':>8} {'Vocab':>10} {'Triton Latency':>16}")
        for logits, mapping, min_p in bench.get_input_iter(dtype):
            batch, vocab = logits.shape
            working_logits = torch.empty_like(logits)
            samples, metadata = _measure_device_time(
                lambda: bench.gems_op(working_logits, mapping, min_p),
                reset=lambda: working_logits.copy_(logits),
                device=logits.device,
                warmup_ms=Config.warm_up,
                repeat_ms=Config.repetition,
                label="triton",
            )
            latency_ms = statistics.median(sample["latency_ms"] for sample in samples)
            rows.append(
                {
                    "shape_detail": bench.record_shapes(logits, mapping, min_p),
                    "latency": latency_ms,
                    "samples": samples,
                    "timing_metadata": metadata,
                }
            )
            print(f"{batch:>8} {vocab:>10} {latency_ms:>16.6f}", flush=True)
        result = {
            "op_name": bench.op_name,
            "dtype": str(dtype),
            "level": Config.bench_level.value,
            "mode": Config.mode.value,
            "implementation": "triton",
            "timer": "cuda-event",
            "source": source,
            "protocol": protocol,
            "result": rows,
        }
        update_result(bench.op_name, result)
        emit_record_logger(json.dumps(result))


@pytest.mark.apply_min_p
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_min_p():
    if Config.warm_up < 0 or Config.repetition <= 0:
        raise ValueError("--warmup must be >= 0 and --iter must be > 0 (milliseconds)")
    print(f"apply_min_p device: {torch.cuda.get_device_name()}")
    bench = ApplyMinPBenchmark(
        op_name="apply_min_p",
        torch_op=None,
        gems_op=apply_min_p,
        dtypes=[torch.float16, torch.bfloat16, torch.float32],
    )
    _run_benchmark(bench)
