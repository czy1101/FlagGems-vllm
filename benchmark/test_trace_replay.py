# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark trace replay against the upstream vLLM Triton implementation."""

from dataclasses import asdict

import pytest
import torch

import flaggems_vllm
from flaggems_vllm import apply_trace_tokens as flaggems_apply_trace_tokens

from . import base
from .conftest import Config, emit_record_logger, update_result
from .consts import (
    BenchmarkMetrics,
    BenchmarkResult,
    lookup_base_record_ex,
    lookup_scaling_factor,
)

VLLM_REFERENCE_UNAVAILABLE_REASON = None

try:
    from vllm.v1.worker.gpu.sample.trace_replay import (
        apply_trace_tokens as vllm_reference,
    )
except (ImportError, OSError) as exc:
    vllm_reference = None
    VLLM_REFERENCE_UNAVAILABLE_REASON = (
        "vLLM apply_trace_tokens reference unavailable: " f"{type(exc).__name__}: {exc}"
    )


class TraceReplayBenchmark(base.Benchmark):
    DEFAULT_SHAPES = [
        (n, 2048) for n in (1, 2, 3, 5, 32, 33, 127, 128, 129, 256, 257, 4096)
    ]
    DEFAULT_SHAPE_DESC = "num_reqs, max_model_len"

    def get_latency(self, op, sampled, *args, **kwargs):
        # Replaying fixed trace tokens is idempotent. Isolate implementations,
        # then let the base class handle all timing modes.
        return super().get_latency(op, sampled.clone(), *args, **kwargs)

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(self.DEFAULT_SHAPES)

    def get_input_iter(self, dtype):
        for num_reqs, max_model_len in self.shapes:
            sampled = torch.full((num_reqs,), -7, dtype=dtype, device=self.device)
            mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
            traces = torch.arange(
                num_reqs * max_model_len, dtype=torch.int32, device=self.device
            ).reshape(num_reqs, max_model_len)
            lengths = torch.full((num_reqs,), 3, dtype=torch.int32, device=self.device)
            prompt = torch.full((num_reqs,), 5, dtype=torch.int32, device=self.device)
            total = prompt + 1
            if num_reqs >= 3:
                # Exercise replayed, skipped and untraced requests.
                mapping[0] = 1
                mapping[1] = -1
                mapping[2] = 0
                lengths[0] = 0

            arguments = (mapping, traces, lengths, total, prompt)
            if vllm_reference is not None:
                candidate_sampled = sampled.clone()
                reference_sampled = sampled.clone()
                assert (
                    flaggems_apply_trace_tokens(candidate_sampled, *arguments) is None
                )
                assert vllm_reference(reference_sampled, *arguments) is None
                torch.testing.assert_close(
                    candidate_sampled, reference_sampled, rtol=0, atol=0
                )
            yield (sampled, *arguments)

    def run(self):
        if vllm_reference is not None or Config.query:
            return super().run()

        self.init_user_config()
        # Preserve measured latency when the native vLLM baseline is unavailable.
        for dtype in self.to_bench_dtypes:
            metrics = []
            for inputs in self.get_input_iter(dtype):
                args, kwargs = self.unpack_to_args_kwargs(inputs)
                metric = BenchmarkMetrics()
                metric.shape_detail = self.record_shapes(*args, **kwargs)
                if Config.model_profile and self.op_name in Config.model_profile:
                    metric.time_share = Config.model_profile[self.op_name]
                metric.latency = self.get_latency(self.gems_op, *args, **kwargs)
                if Config.base_data is not None:
                    record, reason = lookup_base_record_ex(
                        Config.base_data,
                        self.op_name,
                        dtype,
                        metric.shape_detail,
                        self.base_variant,
                    )
                    base_ms = record.get("latency_ms") if record else None
                    if base_ms is not None:
                        factor = lookup_scaling_factor(
                            Config.base_data,
                            flaggems_vllm.vendor_name,
                            record.get("bottle_neck_unit"),
                            self.scaling_dtype or dtype,
                        )
                        metric.compared_speedup = base_ms / metric.latency
                        if factor:
                            metric.compared_speedup /= factor
                    else:
                        metric.base_miss_reason = reason
                metrics.append(metric)

            result = BenchmarkResult(
                level=Config.bench_level.value,
                op_name=self.op_name,
                dtype=str(dtype),
                mode=Config.mode.value,
                result=metrics,
            )
            payload = asdict(result)
            # optest uses this metadata to handle an absent native speedup.
            payload["native_baseline_skip_reason"] = VLLM_REFERENCE_UNAVAILABLE_REASON
            print(result)
            print(VLLM_REFERENCE_UNAVAILABLE_REASON)
            update_result(self.op_name, payload)
            emit_record_logger(result.to_json())


@pytest.mark.apply_trace_tokens
def test_trace_replay_perf():
    bench = TraceReplayBenchmark(
        op_name="apply_trace_tokens",
        torch_op=vllm_reference,
        dtypes=[torch.int64],
        is_inplace=True,
    )
    bench.set_gems(flaggems_apply_trace_tokens)
    bench.run()
