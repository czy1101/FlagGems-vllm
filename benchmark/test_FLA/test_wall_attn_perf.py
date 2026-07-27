from __future__ import annotations

import math
import os
import statistics

import pytest
import torch
import torch.nn.functional as F

from flaggems_vllm.ops.FLA.wall_attn import parallel_wall_attn


CASES = [
    {
        "name": "MHA_T4096_V64",
        "B": 1,
        "T": 4096,
        "H": 8,
        "HQ": 8,
        "K": 64,
        "V": 64,
    },
    {
        "name": "GQA_T4096_V64",
        "B": 1,
        "T": 4096,
        "H": 2,
        "HQ": 8,
        "K": 64,
        "V": 64,
    },
    {
        "name": "MHA_T4096_V65",
        "B": 1,
        "T": 4096,
        "H": 8,
        "HQ": 8,
        "K": 64,
        "V": 65,
    },
    {
        "name": "MHA_T4096_V128",
        "B": 1,
        "T": 4096,
        "H": 8,
        "HQ": 8,
        "K": 64,
        "V": 128,
    },
    {
        "name": "MHA_T4096_V129",
        "B": 1,
        "T": 4096,
        "H": 8,
        "HQ": 8,
        "K": 64,
        "V": 129,
    },
    {
        "name": "MHA_T8192_V64",
        "B": 1,
        "T": 8192,
        "H": 8,
        "HQ": 8,
        "K": 64,
        "V": 64,
    },
]


def _read_positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))

    if value <= 0:
        raise ValueError(f"{name} must be positive")

    return value


def _make_inputs(case: dict[str, int]) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(42)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    q = torch.randn(
        case["B"],
        case["T"],
        case["HQ"],
        case["K"],
        device=device,
        dtype=dtype,
    )

    k = torch.randn(
        case["B"],
        case["T"],
        case["H"],
        case["K"],
        device=device,
        dtype=dtype,
    )

    v = torch.randn(
        case["B"],
        case["T"],
        case["H"],
        case["V"],
        device=device,
        dtype=dtype,
    )

    g = (
        F.logsigmoid(
            torch.randn(
                case["B"],
                case["T"],
                case["HQ"],
                case["K"],
                device=device,
                dtype=torch.float32,
            )
        )
        / 32.0
    ).to(dtype)

    return q, k, v, g


@torch.inference_mode()
def _measure_case(
    case: dict[str, int],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, float | bool | str]:
    q, k, v, g = _make_inputs(case)
    scale = 1.0 / math.sqrt(case["K"])

    def run() -> torch.Tensor:
        return parallel_wall_attn(
            q=q,
            k=k,
            v=v,
            g=g,
            scale=scale,
        )

    for _ in range(warmup):
        output = run()

    torch.cuda.synchronize()

    finite = bool(torch.isfinite(output).all().item())
    timings: list[float] = []

    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        for _ in range(iterations):
            run()

        end.record()
        end.synchronize()

        timings.append(
            start.elapsed_time(end) / iterations
        )

    return {
        "name": case["name"],
        "median_ms": statistics.median(timings),
        "mean_ms": statistics.mean(timings),
        "std_ms": (
            statistics.pstdev(timings)
            if len(timings) > 1
            else 0.0
        ),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "finite": finite,
    }


def run_benchmark() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Wall Attention benchmark requires CUDA")

    warmup = _read_positive_int(
        "WALL_ATTN_BENCH_WARMUP",
        30,
    )
    iterations = _read_positive_int(
        "WALL_ATTN_BENCH_ITERS",
        200,
    )
    repeats = _read_positive_int(
        "WALL_ATTN_BENCH_REPEATS",
        7,
    )

    case_limit = int(
        os.environ.get(
            "WALL_ATTN_BENCH_CASE_LIMIT",
            len(CASES),
        )
    )

    selected_cases = CASES[:case_limit]

    print("=" * 108)
    print("WALL ATTENTION PUBLIC API BENCHMARK")
    print(
        f"device={torch.cuda.get_device_name(0)} "
        f"dtype=BF16 "
        f"warmup={warmup} "
        f"iters={iterations} "
        f"repeats={repeats}"
    )
    print("=" * 108)

    print(
        f"{'Case':<24}"
        f"{'Median(ms)':>13}"
        f"{'Mean(ms)':>13}"
        f"{'Std(ms)':>12}"
        f"{'Min(ms)':>12}"
        f"{'Max(ms)':>12}"
        f"{'Finite':>10}"
    )
    print("-" * 108)

    for case in selected_cases:
        result = _measure_case(
            case,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )

        print(
            f"{result['name']:<24}"
            f"{result['median_ms']:>13.6f}"
            f"{result['mean_ms']:>13.6f}"
            f"{result['std_ms']:>12.6f}"
            f"{result['min_ms']:>12.6f}"
            f"{result['max_ms']:>12.6f}"
            f"{str(result['finite']):>10}"
        )

        if not result["finite"]:
            raise AssertionError(
                f"{result['name']}: output contains NaN/Inf"
            )

    print("=" * 108)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Wall Attention benchmark requires CUDA",
)
def test_wall_attn_perf() -> None:
    run_benchmark()


if __name__ == "__main__":
    run_benchmark()
