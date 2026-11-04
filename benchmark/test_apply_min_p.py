# SPDX-License-Identifier: Apache-2.0 AND MIT
# Copyright 2026 FlagOS Contributors

# Timing helpers adapted from TileOPs benchmarks/timing.py.
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

import hashlib
import inspect
import json
import statistics
import time
from pathlib import Path

import pytest
import torch

from flaggems_vllm.ops.min_p import apply_min_p

from . import base
from .conftest import Config, emit_record_logger, update_result
from .consts import BenchMode, OperationAttribute

_TILEOPS_UNAVAILABLE_REASON = None
try:
    import tilelang
    from tileops._kernels.sampling.min_p_mask import MinPMaskFwdKernel
    from tileops.backend import BUILTIN
    from tileops.sampling import MinPMaskFwdOp
except ModuleNotFoundError as exc:
    if exc.name not in ("tileops", "tilelang"):
        raise
    _TILEOPS_UNAVAILABLE_REASON = (
        f"TileOPs/TileLang baseline dependency is not installed: {exc.name}"
    )


def _measure_device_time(fn, reset, device, warmup_ms, repeat_ms, label):
    device = torch.device(device)
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        end.record(stream)
        end.synchronize()

        warmup_start = time.perf_counter()
        warmup_count = 0
        while (time.perf_counter() - warmup_start) * 1000 < warmup_ms:
            reset()
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
            reset()
            start.record(stream)
            _ = fn()
            end.record(stream)
            end.synchronize()
            samples.append({"latency_ms": start.elapsed_time(end)})
            del _
        sample_wall_ms = (time.perf_counter() - sample_start) * 1000

    return samples, {
        "timing": "cuda-event",
        "device_metric": "complete-call current-stream elapsed time",
        "includes_inter_kernel_gaps": True,
        "preparation": "input reset before the start event; no explicit L2 flush",
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
    DEFAULT_SHAPES = ()
    CASE_DTYPES = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    DEFAULT_DTYPES = list(CASE_DTYPES.values())

    def get_input_iter(self, dtype):
        generator = torch.Generator(device=self.device).manual_seed(2026)
        for tokens, vocab, case_dtype in self.shapes:
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

    def run(self):
        self.init_user_config()
        if Config.query:
            attributes = OperationAttribute(self.op_name, self.shapes, self.shape_desc)
            print(attributes)
            emit_record_logger(attributes.to_dict())
            return
        if _TILEOPS_UNAVAILABLE_REASON is not None:
            pytest.skip(_TILEOPS_UNAVAILABLE_REASON)
        tileops_op = MinPMaskFwdOp(target=BUILTIN)
        source = {
            "tilelang_version": tilelang.__version__,
            "torch_version": torch.__version__,
            "device": torch.cuda.get_device_name(),
        }
        for name, implementation in (
            ("tileops", MinPMaskFwdOp),
            ("tileops_kernel", MinPMaskFwdKernel),
            ("triton", self.gems_op),
        ):
            path = Path(inspect.getfile(implementation))
            source[f"{name}_path"] = str(path)
            source[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        protocol = {
            "timer": "cuda-event",
            "metric": "latency_ms",
            "measurement": "complete-call current-stream elapsed time, including kernel gaps",
            "phase_order": ["tileops", "triton", "triton", "tileops"],
            "warmup_ms_per_implementation": Config.warm_up,
            "sample_ms_per_implementation": Config.repetition,
            "minimum_samples_per_phase": 5,
            "budget_basis": "wall time including preparation and synchronization",
            "l2": "no explicit flush; cache state follows input reset and prior work",
            "input_reset": "private input restored outside every timed invocation",
            "min_p": 0.1,
            "seed": 2026,
            "tileops_api": "out-of-place, native output/scratch allocation",
            "triton_api": "in-place, identity request mapping",
            "speedup": "TileOPs median / Triton median",
        }
        print("Baseline: native TileOPs/TileLang API; timer: CUDA Event")
        print(json.dumps(source))
        for dtype in self.to_bench_dtypes:
            rows = []
            print(f"\napply_min_p dtype={dtype}; times in ms; speedup=TileOPs/Triton")
            print(
                f"{'Batch':>8} {'Vocab':>10} {'TileOPs':>12} {'Triton':>12} {'Speedup':>10}"
            )
            for logits, mapping, min_p in self.get_input_iter(dtype):
                batch, vocab = logits.shape
                tileops_logits = torch.empty_like(logits)
                triton_logits = torch.empty_like(logits)
                calls = {
                    "tileops": lambda: tileops_op(tileops_logits, min_p),
                    "triton": lambda: self.gems_op(triton_logits, mapping, min_p),
                }
                resets = {
                    "tileops": lambda: tileops_logits.copy_(logits),
                    "triton": lambda: triton_logits.copy_(logits),
                }
                for tag in calls:
                    resets[tag]()
                    result = calls[tag]()
                    torch.cuda.current_stream(logits.device).synchronize()
                    del result
                samples = {"tileops": [], "triton": []}
                phases = []
                for phase_index, tag in enumerate(protocol["phase_order"]):
                    measured, metadata = _measure_device_time(
                        calls[tag],
                        reset=resets[tag],
                        device=logits.device,
                        warmup_ms=Config.warm_up / 2,
                        repeat_ms=Config.repetition / 2,
                        label=f"{phase_index}-{tag}",
                    )
                    samples[tag].extend(measured)
                    phases.append({"implementation": tag, "metadata": metadata})
                tileops_ms = statistics.median(
                    sample["latency_ms"] for sample in samples["tileops"]
                )
                triton_ms = statistics.median(
                    sample["latency_ms"] for sample in samples["triton"]
                )
                speedup = tileops_ms / triton_ms
                rows.append(
                    {
                        "shape_detail": self.record_shapes(logits, mapping, min_p),
                        "latency_base": tileops_ms,
                        "latency": triton_ms,
                        "speedup": speedup,
                        "samples": samples,
                        "phases": phases,
                    }
                )
                print(
                    f"{batch:>8} {vocab:>10} {tileops_ms:>12.6f} {triton_ms:>12.6f} {speedup:>9.3f}x",
                    flush=True,
                )
            result = {
                "op_name": self.op_name,
                "dtype": str(dtype),
                "level": Config.bench_level.value,
                "mode": BenchMode.KERNEL.value,
                "baseline": "tileops-tilelang",
                "implementation": "triton",
                "timer": "cuda-event",
                "source": source,
                "protocol": protocol,
                "result": rows,
            }
            update_result(self.op_name, result)
            emit_record_logger(json.dumps(result))


@pytest.mark.apply_min_p(shape_desc=ApplyMinPBenchmark.DEFAULT_SHAPE_DESC)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_min_p():
    ApplyMinPBenchmark(op_name="apply_min_p", torch_op=None, gems_op=apply_min_p).run()
