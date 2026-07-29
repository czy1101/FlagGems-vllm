from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import inspect
import json
import statistics
from pathlib import Path
from typing import Any, Callable

import torch
import triton

import fla
import flaggems_vllm
from fla.ops.wall_attn import naive_wall_attn
from fla.ops.wall_attn import (
    parallel_wall_attn as fla_parallel_wall_attn,
)
from flaggems_vllm.ops.FLA.wall_attn import (
    parallel_wall_attn as flaggems_parallel_wall_attn,
)


BASELINE_REPOSITORY = "https://github.com/fla-org/flash-linear-attention.git"
BASELINE_COMMIT = "2019e841d5a02aac7a9678a98dc20dbd9e7f71f9"
BASELINE_SOURCE_SHA256 = (
    "e0c65c289d151bf1bcf0f0913c196d44a4a18c46a585811370362c4680d04d1b"
)
BASELINE_INSTALL_COMMAND = (
    "python -m pip install --no-deps --force-reinstall "
    f'"flash-linear-attention @ git+{BASELINE_REPOSITORY}@{BASELINE_COMMIT}"'
)

Provider = Callable[..., torch.Tensor | tuple[torch.Tensor, ...]]


def _unwrap(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _make_inputs(
    args: argparse.Namespace,
    dtype: torch.dtype,
    *,
    sequence_length: int | None = None,
) -> tuple[torch.Tensor, ...]:
    """Build one input set shared by both providers."""
    device = torch.device(flaggems_vllm.device)
    T = args.T if sequence_length is None else sequence_length

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    q = torch.randn(args.B, T, args.HQ, args.K, device=device, dtype=dtype)
    k = torch.randn(args.B, T, args.H, args.K, device=device, dtype=dtype)
    v = torch.randn(args.B, T, args.H, args.V, device=device, dtype=dtype)
    g = (
        -torch.randn(
            args.B,
            T,
            args.HQ,
            args.K,
            device=device,
            dtype=torch.float32,
        ).abs()
        * args.gate_scale
    ).to(dtype)
    return q, k, v, g


def _check_provider_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve and verify the exact code behind baseline and optimized."""

    def resolve(
        name: str,
        distribution: str,
        module: Any,
        provider: Provider,
    ) -> dict[str, Any]:
        source = inspect.getsourcefile(provider)
        module_file = getattr(module, "__file__", None)
        if source is None or module_file is None:
            raise RuntimeError(f"Unable to resolve {name} provider source")

        metadata = importlib_metadata.distribution(distribution)
        direct_url_text = metadata.read_text("direct_url.json")
        return {
            "name": name,
            "distribution": distribution,
            "version": metadata.version,
            "module_file": str(Path(module_file).resolve()),
            "source_file": str(Path(source).resolve()),
            "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            "direct_url": (json.loads(direct_url_text) if direct_url_text else None),
        }

    baseline = resolve(
        "baseline",
        "flash-linear-attention",
        fla,
        fla_parallel_wall_attn,
    )
    optimized = resolve(
        "optimized",
        "flaggems_vllm",
        flaggems_vllm,
        flaggems_parallel_wall_attn,
    )

    if fla_parallel_wall_attn is flaggems_parallel_wall_attn:
        raise RuntimeError("Baseline and optimized are the same callable")
    if Path(baseline["source_file"]).samefile(optimized["source_file"]):
        raise RuntimeError("Baseline and optimized use the same source file")

    for info in (baseline, optimized):
        package_root = Path(info["module_file"]).parent
        if not Path(info["source_file"]).is_relative_to(package_root):
            raise RuntimeError(
                f"{info['name']} source is outside its imported package: "
                f"{info['source_file']}"
            )

    direct_url = baseline["direct_url"] or {}
    vcs_info = direct_url.get("vcs_info", {})
    problems = []
    if baseline["source_sha256"] != BASELINE_SOURCE_SHA256:
        problems.append("source SHA-256 does not match the original baseline")
    if direct_url.get("dir_info", {}).get("editable"):
        problems.append("flash-linear-attention is installed in editable mode")
    if vcs_info.get("vcs") != "git":
        problems.append("baseline is not a Git VCS installation")
    if vcs_info.get("commit_id") != BASELINE_COMMIT:
        problems.append("baseline is not pinned to the required commit")

    if problems:
        raise RuntimeError(
            "Invalid FLA baseline installation:\n  - "
            + "\n  - ".join(problems)
            + f"\nResolved source: {baseline['source_file']}"
            + f"\nResolved SHA-256: {baseline['source_sha256']}"
            + f"\nInstall the exact non-editable baseline with:\n"
            + f"  {BASELINE_INSTALL_COMMAND}"
        )

    for info in (baseline, optimized):
        print(f"{info['name']} provider:")
        print(f"  distribution = {info['distribution']} {info['version']}")
        print(f"  module file   = {info['module_file']}")
        print(f"  source file   = {info['source_file']}")
        print(f"  source sha256 = {info['source_sha256']}")
        print(
            "  direct_url    = "
            + (
                json.dumps(info["direct_url"], sort_keys=True)
                if info["direct_url"] is not None
                else "<none>"
            )
        )

    return baseline, optimized


def _run(
    provider: Provider,
    inputs: tuple[torch.Tensor, ...],
    scale: float,
) -> torch.Tensor:
    q, k, v, g = inputs
    return _unwrap(provider(q, k, v, g, scale=scale))


def _provider_order(index: int) -> list[tuple[str, Provider]]:
    providers = [
        ("baseline", fla_parallel_wall_attn),
        ("optimized", flaggems_parallel_wall_attn),
    ]
    return providers if index % 2 == 0 else providers[::-1]


@torch.inference_mode()
def _run_correctness(args: argparse.Namespace, dtype: torch.dtype) -> None:
    """Compare both providers, and use the original eager oracle when available."""
    check_T = min(args.T, args.check_T)
    inputs = _make_inputs(args, dtype, sequence_length=check_T)
    q, k, v, g = inputs
    scale = args.K**-0.5

    reference = None
    reference_error = ""
    try:
        reference = _unwrap(
            naive_wall_attn(
                q.float(),
                k.float(),
                v.float(),
                g.float(),
                scale=scale,
            )
        ).float()
    except RuntimeError as error:
        if "CUBLAS_STATUS_INVALID_VALUE" not in str(error):
            raise
        reference_error = str(error)

    baseline = _run(fla_parallel_wall_attn, inputs, scale).float()
    optimized = _run(flaggems_parallel_wall_attn, inputs, scale).float()
    torch.cuda.synchronize()

    def metrics(expected: torch.Tensor, actual: torch.Tensor) -> tuple[float, ...]:
        diff = (expected - actual).abs()
        return (
            diff.max().item(),
            diff.mean().item(),
            (diff / expected.abs().clamp_min(1e-6)).mean().item(),
        )

    def print_metrics(label: str, values: tuple[float, ...]) -> None:
        print(
            f"  {label}: max_abs={values[0]:.6e}, "
            f"mean_abs={values[1]:.6e}, mean_rel={values[2]:.6e}"
        )

    baseline_finite = bool(torch.isfinite(baseline).all().item())
    optimized_finite = bool(torch.isfinite(optimized).all().item())
    provider_error = metrics(baseline, optimized)

    print("correctness:")
    print(
        f"  check_T={check_T}, atol={args.check_atol:.3e}, "
        f"rtol={args.check_rtol:.3e}"
    )
    print(f"  baseline finite  = {baseline_finite}")
    print(f"  optimized finite = {optimized_finite}")
    print_metrics("optimized vs baseline", provider_error)

    if not baseline_finite or not optimized_finite:
        raise AssertionError("Provider output contains NaN or Inf")
    torch.testing.assert_close(
        optimized,
        baseline,
        atol=args.check_atol,
        rtol=args.check_rtol,
    )

    if reference is None:
        print("  reference available = False")
        print(
            "  reference error = independent FP32 batched matmul is unavailable: "
            f"{reference_error}"
        )
        return

    reference_finite = bool(torch.isfinite(reference).all().item())
    print("  reference available = True")
    print(f"  reference finite = {reference_finite}")
    print_metrics("baseline vs reference", metrics(reference, baseline))
    print_metrics("optimized vs reference", metrics(reference, optimized))
    if not reference_finite:
        raise AssertionError("Reference output contains NaN or Inf")
    torch.testing.assert_close(
        baseline,
        reference,
        atol=args.check_atol,
        rtol=args.check_rtol,
    )
    torch.testing.assert_close(
        optimized,
        reference,
        atol=args.check_atol,
        rtol=args.check_rtol,
    )


@torch.inference_mode()
def _precompile(
    args: argparse.Namespace,
    inputs: tuple[torch.Tensor, ...],
    scale: float,
    providers: list[tuple[str, Provider]],
) -> dict[str, torch.Tensor]:
    """Compile/autotune the actual shape, then warm both providers equally."""
    print("[precompile] compiling actual shape and warming both providers ...")
    outputs = {}

    for name, provider in providers:
        outputs[name] = _run(provider, inputs, scale)
        torch.cuda.synchronize()

    for warmup_index in range(args.warmup):
        ordered = providers if warmup_index % 2 == 0 else providers[::-1]
        for name, provider in ordered:
            outputs[name] = _run(provider, inputs, scale)
    torch.cuda.synchronize()
    return outputs


def _bench_ms(fn: Callable[[], torch.Tensor], measurement_ms: int) -> float:
    """Return median kernel latency using Triton's benchmark helper."""
    return float(
        triton.testing.do_bench(
            fn,
            warmup=0,
            rep=measurement_ms,
            return_mode="median",
        )
    )


@torch.inference_mode()
def _run_benchmark(args: argparse.Namespace, dtype: torch.dtype) -> None:
    scale = args.K**-0.5
    inputs = _make_inputs(args, dtype)
    providers = _provider_order(0)
    outputs = _precompile(args, inputs, scale, providers)
    timings = {"baseline": [], "optimized": []}

    for repeat_index in range(args.repeats):
        order = _provider_order(repeat_index)
        print(
            f"repeat {repeat_index + 1}/{args.repeats}: "
            + " -> ".join(name for name, _ in order)
        )
        for name, provider in order:
            timings[name].append(
                _bench_ms(
                    lambda provider=provider: _run(provider, inputs, scale),
                    args.iters,
                )
            )

    for name, provider in providers:
        outputs[name] = _run(provider, inputs, scale)
    torch.cuda.synchronize()

    def summarize(values: list[float]) -> dict[str, float]:
        return {
            "median": statistics.median(values),
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    baseline = summarize(timings["baseline"])
    optimized = summarize(timings["optimized"])
    baseline_finite = bool(torch.isfinite(outputs["baseline"]).all().item())
    optimized_finite = bool(torch.isfinite(outputs["optimized"]).all().item())

    tokens = args.B * args.T
    pairs = args.B * args.HQ * args.T * (args.T + 1) / 2
    baseline_tokens = tokens / baseline["median"]
    optimized_tokens = tokens / optimized["median"]
    baseline_pairs = pairs / baseline["median"] / 1e6
    optimized_pairs = pairs / optimized["median"] / 1e6
    speedup = baseline["median"] / optimized["median"]
    latency_reduction = (1 - optimized["median"] / baseline["median"]) * 100
    throughput_gain = (optimized_tokens / baseline_tokens - 1) * 100

    print("benchmark:")
    print(
        f"  label={args.label}, B={args.B}, T={args.T}, H={args.H}, "
        f"HQ={args.HQ}, GQA={args.HQ // args.H}, K={args.K}, V={args.V}, "
        f"dtype={dtype}"
    )
    print(f"  scale={scale:.8f}, gate_scale={args.gate_scale}, seed={args.seed}")
    print(
        f"  warmup_calls={args.warmup}, "
        f"do_bench_window={args.iters}ms, repeats={args.repeats}"
    )
    print(
        f"{'provider':<12}{'median(ms)':>12}{'mean(ms)':>12}"
        f"{'std(ms)':>12}{'min(ms)':>12}{'max(ms)':>12}"
        f"{'tokens/ms':>14}{'M pairs/ms':>14}{'finite':>10}"
    )
    for name, stats, tokens_ms, pairs_ms, finite in [
        (
            "baseline",
            baseline,
            baseline_tokens,
            baseline_pairs,
            baseline_finite,
        ),
        (
            "optimized",
            optimized,
            optimized_tokens,
            optimized_pairs,
            optimized_finite,
        ),
    ]:
        print(
            f"{name:<12}{stats['median']:>12.6f}{stats['mean']:>12.6f}"
            f"{stats['std']:>12.6f}{stats['min']:>12.6f}"
            f"{stats['max']:>12.6f}{tokens_ms:>14.3f}"
            f"{pairs_ms:>14.6f}{str(finite):>10}"
        )

    print(f"speedup (baseline/optimized) = {speedup:.6f}x")
    print(f"latency reduction = {latency_reduction:.6f}%")
    print(f"throughput gain = {throughput_gain:.6f}%")

    if not baseline_finite or not optimized_finite:
        raise AssertionError("Benchmark output contains NaN or Inf")


@torch.inference_mode()
def _run_ncu(args: argparse.Namespace, dtype: torch.dtype) -> None:
    scale = args.K**-0.5
    inputs = _make_inputs(args, dtype)
    all_providers = dict(_provider_order(0))
    names = (
        ["baseline", "optimized"]
        if args.ncu_provider == "both"
        else [args.ncu_provider]
    )
    providers = [(name, all_providers[name]) for name in names]
    outputs = _precompile(args, inputs, scale, providers)

    for name, provider in providers:
        torch.cuda.nvtx.range_push(f"wall_attn_{name}_profile_region")
        try:
            for _ in range(args.profile_iters):
                outputs[name] = _run(provider, inputs, scale)
            torch.cuda.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()

    print("ncu target:")
    print(
        f"  label={args.label}, B={args.B}, T={args.T}, H={args.H}, "
        f"HQ={args.HQ}, K={args.K}, V={args.V}, dtype={dtype}"
    )
    print(f"  provider={args.ncu_provider}, profile_iters={args.profile_iters}")
    for name in names:
        finite = bool(torch.isfinite(outputs[name]).all().item())
        print(f"  {name} output finite = {finite}")
        if not finite:
            raise AssertionError(f"{name} NCU output contains NaN or Inf")


def _parse_args() -> argparse.Namespace:
    def positive_int(value: str) -> int:
        parsed = int(value)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("value must be positive")
        return parsed

    parser = argparse.ArgumentParser(
        description=(
            "Compare the original FLA Wall Attention baseline with the "
            "FlagGems-vLLM implementation."
        )
    )
    parser.add_argument("--mode", default="bench", choices=["bench", "ncu"])
    parser.add_argument("--label", default="wall-attn")
    parser.add_argument("--B", type=positive_int, default=1)
    parser.add_argument("--T", type=positive_int, default=4096)
    parser.add_argument("--H", type=positive_int, default=8)
    parser.add_argument("--HQ", type=positive_int, default=8)
    parser.add_argument("--K", type=positive_int, default=64)
    parser.add_argument("--V", type=positive_int, default=64)
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
    )
    parser.add_argument("--gate-scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-T", dest="check_T", type=positive_int, default=128)
    parser.add_argument("--check-atol", type=float, default=5e-2)
    parser.add_argument("--check-rtol", type=float, default=5e-2)
    parser.add_argument(
        "--warmup",
        type=positive_int,
        default=30,
        help="warmup calls per provider after JIT/autotune",
    )
    parser.add_argument(
        "--iters",
        type=positive_int,
        default=200,
        help="measurement window in milliseconds passed to triton.do_bench",
    )
    parser.add_argument("--repeats", type=positive_int, default=7)
    parser.add_argument("--profile-iters", type=positive_int, default=1)
    parser.add_argument(
        "--ncu-provider",
        default="optimized",
        choices=["baseline", "optimized", "both"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.HQ % args.H != 0:
        raise ValueError("HQ must be divisible by H")
    if args.gate_scale < 0:
        raise ValueError("gate-scale must be non-negative")
    if args.check_atol < 0 or args.check_rtol < 0:
        raise ValueError("correctness tolerances must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if str(flaggems_vllm.device) != "cuda":
        raise RuntimeError(
            f"FlagGems-vLLM device must be cuda, got {flaggems_vllm.device}"
        )

    print("environment:")
    print(f"  torch={torch.__version__}, triton={triton.__version__}")
    print(f"  CUDA runtime={torch.version.cuda}")
    print(f"  GPU={torch.cuda.get_device_name(0)}, mode={args.mode}")
    print(f"  baseline repository={BASELINE_REPOSITORY}")
    print(f"  baseline commit={BASELINE_COMMIT}")
    _check_provider_sources()

    dtype = _dtype(args.dtype)
    if args.check:
        _run_correctness(args, dtype)
    if args.mode == "bench":
        _run_benchmark(args, dtype)
    else:
        _run_ncu(args, dtype)


if __name__ == "__main__":
    main()
