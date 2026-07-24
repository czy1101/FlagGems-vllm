# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Standalone benchmark for the Parallel and Improve Parallax kernels.

Examples::

    python benchmarks/ops/benchmark.py
    python benchmarks/ops/benchmark.py --case B2_T2048_H2_HQ8_D64 --mode fwd
    python benchmarks/ops/benchmark.py --provider improve --dtype float16 --csv parallax.csv

The script loads the two implementation files directly and owns its shape
cases, input generation, and timing. Forward-plus-backward timings call the
low-level operator functions directly so autograd graph construction is excluded.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import sys
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
import torch 
import torch.cuda.nvtx as nvtx
import triton 
_SCRIPT_DIR = Path(__file__).resolve().parent

for _candidate in (_SCRIPT_DIR, *_SCRIPT_DIR.parents):
    if (
        (_candidate / 'src' / 'flaggems_vllm' / 'ops' / 'FLA' / 'parallel_parallax.py').is_file()
    ):
        _REPO_ROOT = _candidate
        break
else:
    raise RuntimeError(
        'Could not locate FlagGems-vllm repository root'
    )

sys.path.insert(
    0,
    str(_REPO_ROOT / 'src')
)


def _load_module(path: Path, module_name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

_FLA_DIR = (
    _REPO_ROOT
    / 'src'
    / 'flaggems_vllm'
    / 'ops'
    / 'FLA'
)


_IMPROVE_PATH = (
    _FLA_DIR
    / 'improve_parallax.py'
)


_PARALLEL_PATH = (
    _FLA_DIR
    / 'parallel_parallax.py'
)


if not _IMPROVE_PATH.exists():
    raise FileNotFoundError(_IMPROVE_PATH)


if not _PARALLEL_PATH.exists():
    raise FileNotFoundError(_PARALLEL_PATH)
_fla_package = types.ModuleType(
    'flaggems_vllm.ops.FLA'
)

_fla_package.__path__ = [
    str(_FLA_DIR)
]

sys.modules[
    'flaggems_vllm.ops.FLA'
] = _fla_package

_decode_module = types.ModuleType('fla.ops.parallax.decode')
_decode_module.parallax_decode = None
_decode_module.parallax_decode_one_step = None
sys.modules['fla.ops.parallax.decode'] = _decode_module
_parallel_impl = _load_module(
    _PARALLEL_PATH,
    'flag_gems_parallel_parallax_impl',
)


_improve_impl = _load_module(
    _IMPROVE_PATH,
    'flag_gems_improve_parallax_impl',
)
parallel_parallax = (
    _parallel_impl.parallel_parallax
)

improve_parallax = (
    _improve_impl.improve_parallax
)

DEVICE = torch.device('cuda')

@dataclass(frozen=True)
class BenchmarkCase:
    B: int
    T: int
    H: int
    HQ: int
    D: int
    window_size: int | None = None
    cu_seqlens: tuple[int, ...] | None = None

@dataclass(frozen=True)
class BenchmarkProvider:
    name: str
    op: Callable

# low-level operator benchmark
    fwd: Callable
    bwd: Callable


    block_size: Callable

CASES = {
    # 'B1_T15_H2_HQ2_D64': BenchmarkCase(B=1, T=15, H=2, HQ=2, D=64),
    # 'B1_T63_H1_HQ1_D64': BenchmarkCase(B=1, T=63, H=1, HQ=1, D=64),
    # 'B1_T111_H2_HQ2_D64': BenchmarkCase(B=1, T=111, H=2, HQ=2, D=64),
    # 'B2_T200_H2_HQ8_D64': BenchmarkCase(B=2, T=200, H=2, HQ=8, D=64),
    # 'B2_T256_H2_HQ8_D64': BenchmarkCase(B=2, T=256, H=2, HQ=8, D=64),
    # 'B2_T512_H2_HQ8_D64': BenchmarkCase(B=2, T=512, H=2, HQ=8, D=64),   
    # 'B2_T1024_H2_HQ2_D64': BenchmarkCase(B=2, T=1024, H=2, HQ=2, D=64),
    # 'B2_T2048_H2_HQ8_D64': BenchmarkCase(B=2, T=2048, H=2, HQ=8, D=64),
    'B2_T4096_H2_HQ2_D64': BenchmarkCase(B=2, T=4096, H=2, HQ=2, D=64),
    # 'B2_T8192_H2_HQ8_D64': BenchmarkCase(B=2, T=8192, H=2, HQ=8, D=64),
    # 'B4_T16K_H16_HQ32_D128': BenchmarkCase(B=4, T=16384, H=16, HQ=32, D=128),
}

def _default_warmup(case: BenchmarkCase) -> int:
    if case.T >= 16384:
        return 200
    if case.T >= 1024:
        return 100
    return 50

PROVIDERS = {
    'parallel': BenchmarkProvider(
        name='parallel',
        op=parallel_parallax,
        fwd=_parallel_impl.parallel_parallax_fwd,
        bwd=_parallel_impl.parallel_parallax_bwd,
        block_size=_parallel_impl._block_size,
    ),

    'improve': BenchmarkProvider(
        name='improve',
        op=improve_parallax,
        fwd=_improve_impl.improve_parallax_fwd,
        bwd=_improve_impl.improve_parallax_bwd,
        block_size=_improve_impl._block_size,
    ),
}
def _make_inputs(case: BenchmarkCase, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    query_shape = (case.B, case.T, case.HQ, case.D)
    kv_shape = (case.B, case.T, case.H, case.D)
    return {
        'q': torch.randn(query_shape, dtype=dtype, device=DEVICE),
        'r': torch.randn(query_shape, dtype=dtype, device=DEVICE),
        'k': torch.randn(kv_shape, dtype=dtype, device=DEVICE),
        'v': torch.randn(kv_shape, dtype=dtype, device=DEVICE),
    }

def _op_kwargs(case: BenchmarkCase) -> dict:
    kwargs = {'window_size': case.window_size}
    if case.cu_seqlens is not None:
        if case.B != 1 or case.cu_seqlens[0] != 0 or case.cu_seqlens[-1] != case.T:
            raise ValueError('A varlen case must use B=1 and cu_seqlens spanning [0, T]')
        kwargs['cu_seqlens'] = torch.tensor(case.cu_seqlens, dtype=torch.int32, device=DEVICE)
    return kwargs

def _autotune_warmup(
    provider: BenchmarkProvider,
    case: BenchmarkCase,
    dtype: torch.dtype,
):
    """
    Run once to trigger:
    - Triton JIT compilation
    - autotune config search
    - CUDA kernel cache
    """

    inputs = _make_inputs(case, dtype)

    kwargs = _op_kwargs(case)

    scale = case.D ** -0.5

    window_size_left = (
        -1
        if case.window_size is None
        else case.window_size
    )

    cu_seqlens = kwargs.get(
        'cu_seqlens'
    )

    chunk_indices = None

    if cu_seqlens is not None:
        chunk_indices = (
            _parallel_impl.prepare_chunk_indices(
                cu_seqlens,
                provider.block_size(
                    case.D,
                    torch.cuda.current_device()
                )
            )
        )
    # ==========================
    # warmup forward
    # ==========================

    fwd_cache = provider.fwd(
        inputs['q'],
        inputs['r'],
        inputs['k'],
        inputs['v'],
        scale,
        cu_seqlens,
        chunk_indices,
        window_size_left,
    )


    torch.cuda.synchronize()

    o, barv, d1, bart, m = fwd_cache

    grad_output = torch.randn_like(o)


    provider.bwd(
        inputs['q'],
        inputs['r'],
        inputs['k'],
        inputs['v'],
        o,
        barv,
        d1,
        bart,
        m,
        grad_output,
        scale,
        cu_seqlens,
        chunk_indices,
        window_size_left,
    )


    torch.cuda.synchronize()


def _benchmark_provider(
    provider: BenchmarkProvider,
    case: BenchmarkCase,
    dtype: torch.dtype,
    mode: str,
    warmup: int,
    rep: int,
):

    inputs = _make_inputs(case, dtype)

    kwargs = _op_kwargs(case)


    scale = case.D ** -0.5

    window_size_left = (
        -1
        if case.window_size is None
        else case.window_size
    )


    cu_seqlens = kwargs.get(
        'cu_seqlens'
    )


    chunk_indices = None

    if cu_seqlens is not None:

        chunk_indices = (
            _parallel_impl.prepare_chunk_indices(
                cu_seqlens,
                provider.block_size(
                    case.D,
                    inputs['q'].device.index
                )
            )
        )



    # ============================
    # forward
    # ============================

    def run_fwd():


        nvtx.range_push(
            f'{provider.name}_forward'
        )


        result = provider.fwd(
            inputs['q'],
            inputs['r'],
            inputs['k'],
            inputs['v'],
            scale,
            cu_seqlens,
            chunk_indices,
            window_size_left,
        )


        nvtx.range_pop()


        return result



    # ============================
    # forward + backward
    # ============================

    grad_output = torch.randn(
        (
            case.B,
            case.T,
            case.HQ,
            case.D
        ),
        dtype=dtype,
        device=DEVICE
    )

    def run_bwd(fwd_cache):

        o, barv, d1, bart, m = fwd_cache


        nvtx.range_push(
            f'{provider.name}_backward'
        )


        out = provider.bwd(
            inputs['q'],
            inputs['r'],
            inputs['k'],
            inputs['v'],

            o,
            barv,
            d1,
            bart,
            m,

            grad_output,

            scale,

            cu_seqlens,
            chunk_indices,

            window_size_left,
        )


        nvtx.range_pop()


        return out

    # ============================
    # select benchmark target
    # ============================


    if mode == 'fwd':

        fn = run_fwd


    else:

        def fn():

            fwd_cache = run_fwd()

            run_bwd(
                fwd_cache
            )



    torch.cuda.synchronize()


    result = triton.testing.do_bench(
        fn,
        warmup=warmup,
        rep=rep,
        quantiles=[
            0.5,
            0.2,
            0.8
        ],
    )


    torch.cuda.synchronize()


    return tuple(
        float(x)
        for x in result
    )
def _write_csv(path: str, rows: Sequence[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def _print_results(rows: Sequence[dict], modes: Sequence[str]) -> None:
    width = 96
    provider_names = {
        'improve': 'improve_parallax',
        'parallel': 'parallel_parallax',
    }
    provider_order = {'improve': 0, 'parallel': 1}

    print('=' * width)
    print(
        f'  Machine: {torch.cuda.get_device_name()} | CUDA {torch.version.cuda} | '
        f'PyTorch {torch.__version__}',
    )
    for mode in modes:
        mode_rows = [row for row in rows if row['mode'] == mode]
        if not mode_rows:
            continue

        mode_name = 'fwdbwd' if mode == 'fwd_bwd' else mode
        print('=' * width)
        print(
            f'  {mode_name:<10}{"B":>5}{"T":>8}{"H":>5}{"HQ":>5}{"D":>6}  '
            f'{"op":<24}{"latency(ms)":>14}{"speedup":>12}',
        )
        print(f'  {"":10}{"-" * 82}')

        case_names = dict.fromkeys(row['case'] for row in mode_rows)
        for case_name in case_names:
            case = CASES[case_name]
            case_rows = sorted(
                (row for row in mode_rows if row['case'] == case_name),
                key=lambda row: provider_order[row['provider']],
            )
            for index, row in enumerate(case_rows):
                B = case.B if index == 0 else ''
                T = case.T if index == 0 else ''
                H = case.H if index == 0 else ''
                HQ = case.HQ if index == 0 else ''
                D = case.D if index == 0 else ''
                op = provider_names[row['provider']]
                speedup = '-' if row['speedup'] is None else f'{row["speedup"]:.3f}x'
                print(
                    f'  {"":10}{B:>5}{T:>8}{H:>5}{HQ:>5}{D:>6}  '
                    f'{op:<24}{row["p50_ms"]:>14.3f}{speedup:>12}',
                )
            print(f'  {"":10}{"-" * 82}')
    print('=' * width)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark the Parallel and Improve Parallax kernels.')
    parser.add_argument('--case', choices=['all', *CASES], default='all')
    parser.add_argument(
        '--mode',
        choices=[
            'all',
            'fwd',
            'fwd_bwd'
        ],
        default='all'
    )
    parser.add_argument('--provider', choices=['all', *PROVIDERS], default='all')
    parser.add_argument('--dtype', choices=['float16', 'bfloat16'], default='bfloat16')
    parser.add_argument(
        '--warmup',
        type=int,
        help='Number of warmup iterations'
    )
    parser.add_argument('--rep', type=int, default=100, help='Number of benchmark iterations')
    parser.add_argument('--csv', help='Optional path for the raw benchmark results.')
    return parser.parse_args()

def main() -> None:
    args = _parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            'The Parallax benchmark requires a CUDA GPU.'
        )

    dtype = getattr(torch, args.dtype)

    selected_cases = (
        CASES
        if args.case == 'all'
        else {args.case: CASES[args.case]}
    )

    selected_modes = (
        ('fwd', 'fwd_bwd')
        if args.mode == 'all'
        else (args.mode,)
    )

    selected_providers = (
        PROVIDERS
        if args.provider == 'all'
        else {args.provider: PROVIDERS[args.provider]}
    )


    # =====================================================
    # Phase 1:
    # Triton JIT compile + autotune warmup
    #
    # 目的:
    # 1. 编译 Triton kernel
    # 2. 搜索 autotune config
    # 3. 建立 CUDA kernel cache
    #
    # 不计入最终 benchmark 时间
    # =====================================================

    print('=' * 96)
    print('Running Triton autotune warmup...')


    for case_name, case in selected_cases.items():

        for provider_name, provider in selected_providers.items():

            print(
                f'  warmup {provider_name:<10} '
                f'{case_name}'
            )

            _autotune_warmup(
                provider,
                case,
                dtype,
            )


    torch.cuda.synchronize()

    gc.collect()
    torch.cuda.empty_cache()


    print('Triton autotune warmup finished')
    print('=' * 96)



    # =====================================================
    # Phase 2:
    # Real benchmark
    # =====================================================

    rows = []


    for case_name, case in selected_cases.items():

        warmup = (
            args.warmup
            if args.warmup is not None
            else _default_warmup(case)
        )


        for mode in selected_modes:

            case_mode_rows = []


            for provider_name, provider in selected_providers.items():


                p50_ms, p20_ms, p80_ms = _benchmark_provider(
                    provider,
                    case,
                    dtype,
                    mode,
                    warmup,
                    args.rep,
                )


                tokens_per_second = (
                    case.B *
                    case.T *
                    1000 /
                    p50_ms
                )


                row = {
                    'case': case_name,
                    'mode': mode,
                    'provider': provider_name,

                    'B': case.B,
                    'T': case.T,
                    'H': case.H,
                    'HQ': case.HQ,
                    'D': case.D,


                    # 保存当前实际block_size
                    'block_size': provider.block_size(
                        case.D,
                        torch.cuda.current_device()
                    ),


                    'p50_ms': p50_ms,
                    'p20_ms': p20_ms,
                    'p80_ms': p80_ms,

                    'tokens_per_second':
                        tokens_per_second,

                    'speedup': None,
                }


                rows.append(row)

                case_mode_rows.append(row)



            # =============================================
            # speedup计算
            # parallel作为baseline
            # =============================================

            timings = {
                row['provider']: row['p50_ms']
                for row in case_mode_rows
            }


            if (
                'parallel' in timings
                and 'improve' in timings
            ):

                for row in case_mode_rows:

                    row['speedup'] = (
                        timings['parallel']
                        /
                        row['p50_ms']
                    )



        # 当前case结束，释放显存

        gc.collect()
        torch.cuda.empty_cache()


    _print_results(
        rows,
        selected_modes,
    )


    if args.csv and rows:

        _write_csv(
            args.csv,
            rows,
        )

        print(
            f'wrote {args.csv}'
        )


if __name__ == '__main__':
    main()