from __future__ import annotations
from dataclasses import dataclass
from statistics import median
from typing import Callable
import pytest
import torch
import triton

import flaggems_vllm
from benchmark.conftest import Config
from fla.ops.parallax.parallel import (
    _block_size as fla_block_size,
    parallel_parallax_bwd as fla_parallel_parallax_bwd,
    parallel_parallax_fwd as fla_parallel_parallax_fwd,
)
from fla.ops.utils import prepare_chunk_indices as fla_prepare_chunk_indices
from flaggems_vllm.ops.FLA.index import (
    prepare_chunk_indices as flaggems_prepare_chunk_indices,
)
from flaggems_vllm.ops.FLA.parallel_parallax import (
    _block_size as flaggems_block_size,
    parallel_parallax_bwd as flaggems_parallel_parallax_bwd,
    parallel_parallax_fwd as flaggems_parallel_parallax_fwd,
)


DEFAULT_DTYPES = (torch.bfloat16,)
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

PAIR_WARMUP_CYCLES = 2
BALANCED_MEASUREMENT_CYCLES = 2
TABLE_WIDTH = 113


@dataclass(frozen=True)
class ParallaxCase:
    B: int
    T: int
    H: int
    HQ: int
    D: int
    window_size: int | None = None
    cu_seqlens: tuple[int, ...] | None = None


@dataclass(frozen=True)
class PhaseBenchmarkResult:
    fla_ms: float
    flaggems_ms: float
    fla_mad_pct: float
    flaggems_mad_pct: float

    @property
    def speedup(self) -> float:
        # FLA is the baseline. A value greater than 1 means that
        # FlagGems has lower latency and is therefore faster.
        return self.fla_ms / self.flaggems_ms


DEFAULT_CASES = (
    ParallaxCase(B=1, T=15, H=2, HQ=2, D=64),
    ParallaxCase(B=1, T=63, H=1, HQ=1, D=64),
    ParallaxCase(B=1, T=111, H=2, HQ=2, D=64),
    ParallaxCase(B=2, T=200, H=2, HQ=8, D=64),
    ParallaxCase(B=2, T=256, H=2, HQ=8, D=64),
    ParallaxCase(B=2, T=512, H=2, HQ=8, D=64),
    ParallaxCase(B=2, T=1024, H=2, HQ=2, D=64),
    ParallaxCase(B=2, T=2048, H=2, HQ=8, D=64),
    ParallaxCase(B=2, T=4096, H=2, HQ=2, D=64),
    ParallaxCase(B=2, T=8192, H=2, HQ=8, D=64),
    ParallaxCase(B=3, T=111, H=2, HQ=2, D=100),
    ParallaxCase(B=4, T=16384, H=16, HQ=32, D=128),
    ParallaxCase(B=2, T=2048, H=2, HQ=8, D=128),
    ParallaxCase(B=2, T=8192, H=8, HQ=16, D=128),
)


def _build_inputs(
    case: ParallaxCase,
    dtype: torch.dtype,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    torch.Tensor | None,
    int,
]:
    if case.HQ % case.H != 0:
        raise ValueError("HQ must be divisible by H")

    device = flaggems_vllm.device

    query_shape = (
        case.B,
        case.T,
        case.HQ,
        case.D,
    )
    kv_shape = (
        case.B,
        case.T,
        case.H,
        case.D,
    )

    q = torch.randn(
        query_shape,
        dtype=dtype,
        device=device,
    )
    r = torch.randn(
        query_shape,
        dtype=dtype,
        device=device,
    )
    k = torch.randn(
        kv_shape,
        dtype=dtype,
        device=device,
    )
    v = torch.randn(
        kv_shape,
        dtype=dtype,
        device=device,
    )

    cu_seqlens = None

    if case.cu_seqlens is not None:
        if (
            case.B != 1
            or case.cu_seqlens[0] != 0
            or case.cu_seqlens[-1] != case.T
        ):
            raise ValueError(
                "A variable-length case requires B=1 and "
                "cu_seqlens spanning [0, T]"
            )

        cu_seqlens = torch.tensor(
            case.cu_seqlens,
            dtype=torch.long,
            device=device,
        )

    scale = case.D**-0.5

    window_size_left = (
        -1
        if case.window_size is None
        else case.window_size
    )

    return (
        q,
        r,
        k,
        v,
        scale,
        cu_seqlens,
        window_size_left,
    )


def _build_chunk_indices(
    case: ParallaxCase,
    q: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
]:

    if cu_seqlens is None:
        return None, None

    device_index = q.device.index

    fla_bt = fla_block_size(
        case.D,
        device_index,
    )
    flaggems_bt = flaggems_block_size(
        case.D,
        device_index,
    )

    fla_chunk_indices = fla_prepare_chunk_indices(
        cu_seqlens,
        fla_bt,
    )
    flaggems_chunk_indices = flaggems_prepare_chunk_indices(
        cu_seqlens,
        flaggems_bt,
    )

    return (
        fla_chunk_indices,
        flaggems_chunk_indices,
    )


def _bench_ms(fn: Callable[[], object]) -> float:
    torch.cuda.synchronize()

    if Config.mode.value == "kernel":
        result = float(
            triton.testing.do_bench(
                fn,
                warmup=Config.warm_up,
                rep=Config.repetition,
                return_mode="median",
            )
        )

        torch.cuda.synchronize()
        return result

    for _ in range(Config.warm_up):
        fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(Config.repetition):
        fn()

    end.record()

    torch.cuda.synchronize()

    return (
        start.elapsed_time(end)
        / Config.repetition
    )


def _measurement_order(
    cycle: int,
    fla_fn: Callable[[], object],
    flaggems_fn: Callable[[], object],
) -> tuple[Callable[[], object], ...]:


    if cycle % 2 == 0:
        return (
            fla_fn,
            flaggems_fn,
            flaggems_fn,
            fla_fn,
        )

    return (
        flaggems_fn,
        fla_fn,
        fla_fn,
        flaggems_fn,
    )


def _median_and_mad_pct(
    samples: list[float],
) -> tuple[float, float]:

    center = float(median(samples))

    if center == 0.0:
        return center, 0.0

    absolute_deviations = [
        abs(sample - center)
        for sample in samples
    ]
    mad = float(median(absolute_deviations))

    return center, mad / center * 100.0


def _bench_balanced_pair(
    fla_fn: Callable[[], object],
    flaggems_fn: Callable[[], object],
) -> PhaseBenchmarkResult:

    for cycle in range(PAIR_WARMUP_CYCLES):
        for fn in _measurement_order(
            cycle,
            fla_fn,
            flaggems_fn,
        ):
            fn()

        torch.cuda.synchronize()

    fla_samples: list[float] = []
    flaggems_samples: list[float] = []

    for cycle in range(BALANCED_MEASUREMENT_CYCLES):
        for fn in _measurement_order(
            cycle,
            fla_fn,
            flaggems_fn,
        ):
            sample_ms = _bench_ms(fn)

            if fn is fla_fn:
                fla_samples.append(sample_ms)
            else:
                flaggems_samples.append(sample_ms)

    fla_ms, fla_mad_pct = _median_and_mad_pct(
        fla_samples
    )
    flaggems_ms, flaggems_mad_pct = (
        _median_and_mad_pct(
            flaggems_samples
        )
    )

    return PhaseBenchmarkResult(
        fla_ms=fla_ms,
        flaggems_ms=flaggems_ms,
        fla_mad_pct=fla_mad_pct,
        flaggems_mad_pct=flaggems_mad_pct,
    )


def _benchmark_case(
    case: ParallaxCase,
    dtype: torch.dtype,
    phase: str,
) -> PhaseBenchmarkResult:
    (
        q,
        r,
        k,
        v,
        scale,
        cu_seqlens,
        window_size_left,
    ) = _build_inputs(
        case,
        dtype,
    )

    (
        fla_chunk_indices,
        flaggems_chunk_indices,
    ) = _build_chunk_indices(
        case,
        q,
        cu_seqlens,
    )

    grad_output = torch.randn_like(q)

    def run_fla_fwd():
        return fla_parallel_parallax_fwd(
            q,
            r,
            k,
            v,
            scale,
            cu_seqlens,
            fla_chunk_indices,
            window_size_left,
        )

    def run_fla_fwd_bwd():
        o, barv, d1, bart, m = run_fla_fwd()

        return fla_parallel_parallax_bwd(
            q,
            r,
            k,
            v,
            o,
            barv,
            d1,
            bart,
            m,
            grad_output,
            scale,
            cu_seqlens,
            fla_chunk_indices,
            window_size_left,
        )

    def run_flaggems_fwd():
        return flaggems_parallel_parallax_fwd(
            q,
            r,
            k,
            v,
            scale,
            cu_seqlens,
            flaggems_chunk_indices,
            window_size_left,
        )

    def run_flaggems_fwd_bwd():
        o, barv, d1, bart, m = run_flaggems_fwd()

        return flaggems_parallel_parallax_bwd(
            q,
            r,
            k,
            v,
            o,
            barv,
            d1,
            bart,
            m,
            grad_output,
            scale,
            cu_seqlens,
            flaggems_chunk_indices,
            window_size_left,
        )

    if phase == "fwd":
        fla_fn = run_fla_fwd
        flaggems_fn = run_flaggems_fwd
    elif phase == "fwd_bwd":
        fla_fn = run_fla_fwd_bwd
        flaggems_fn = run_flaggems_fwd_bwd
    else:
        raise ValueError(
            f"Unsupported benchmark phase: {phase}"
        )

    return _bench_balanced_pair(
        fla_fn=fla_fn,
        flaggems_fn=flaggems_fn,
    )


def _selected_dtypes() -> list[torch.dtype]:
    dtypes = (
        Config.user_desired_dtypes
        or DEFAULT_DTYPES
    )

    unsupported = [
        dtype
        for dtype in dtypes
        if dtype not in SUPPORTED_DTYPES
    ]

    if unsupported:
        names = ", ".join(
            str(dtype)
            for dtype in unsupported
        )
        raise ValueError(
            f"parallel_parallax does not support: {names}"
        )

    return list(dtypes)


def _print_header(
    title: str,
    fla_column: str,
    flaggems_column: str,
) -> None:
    print()
    print("=" * TABLE_WIDTH)
    print(title)
    print("=" * TABLE_WIDTH)
    print(
        f"{'B':>3} "
        f"{'T':>7} "
        f"{'H':>4} "
        f"{'HQ':>4} "
        f"{'D':>4} "
        f"{'dtype':>9} "
        f"{fla_column:>18} "
        f"{flaggems_column:>23} "
        f"{'speedup':>11} "

    )

    print("-" * TABLE_WIDTH)


def _print_result(
    case: ParallaxCase,
    dtype: torch.dtype,
    result: PhaseBenchmarkResult,
) -> None:
    dtype_name = str(dtype).removeprefix(
        "torch."
    )

    print(
        f"{case.B:>3} "
        f"{case.T:>7} "
        f"{case.H:>4} "
        f"{case.HQ:>4} "
        f"{case.D:>4} "
        f"{dtype_name:>9} "
        f"{result.fla_ms:>18.3f} "
        f"{result.flaggems_ms:>23.3f} "
        f"{result.speedup:>10.3f}x "


    )


def _run_phase_table(
    phase: str,
    title: str,
    fla_column: str,
    flaggems_column: str,
) -> None:
    _print_header(
        title=title,
        fla_column=fla_column,
        flaggems_column=flaggems_column,
    )

    for dtype in _selected_dtypes():
        for case in DEFAULT_CASES:
            result = _benchmark_case(
                case=case,
                dtype=dtype,
                phase=phase,
            )
            _print_result(
                case=case,
                dtype=dtype,
                result=result,
            )

    print("-" * TABLE_WIDTH)


@pytest.mark.skipif(
    flaggems_vllm.device != "cuda",
    reason="parallel_parallax benchmark requires CUDA",
)
@pytest.mark.parallax
def test_perf_parallel_parallax() -> None:
    torch.manual_seed(42)

    print(
        "\n"
        "[parallel_parallax: "
        "FLA baseline vs FlagGems]"
    )

    print(
        f"device={torch.cuda.get_device_name()} "
        f"mode={Config.mode.value} "
        f"warmup={Config.warm_up} "
        f"iter={Config.repetition}"
    )

    print(
        "measurement_order = mirrored ABBA/BAAB; "
        f"samples/provider = "
        f"{BALANCED_MEASUREMENT_CYCLES * 2}"
    )
    print(
        "baseline = FLA; "
        "speedup = FLA latency / FlagGems latency; "
        ">1 means FlagGems is faster"
    )
    print(
        "MAD% = relative median absolute deviation; "
        "lower is more stable"
    )
    print("fwd+bwd = forward + backward")

    _run_phase_table(
        phase="fwd",
        title="[Forward]",
        fla_column="FLA-fwd(ms)",
        flaggems_column="FlagGems-fwd(ms)",
    )

    _run_phase_table(
        phase="fwd_bwd",
        title="[Forward + Backward]",
        fla_column="FLA-fwdbwd(ms)",
        flaggems_column="FlagGems-fwdbwd(ms)",
    )