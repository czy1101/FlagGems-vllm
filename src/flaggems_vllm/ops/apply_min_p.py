# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-place min-p filtering with optional TLE shared-memory caching."""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

if has_triton_tle():
    import triton.experimental.tle.language as tle

    _HAS_TLE_CACHE = all(
        callable(getattr(tle.gpu, name, None)) for name in ("alloc", "local_ptr")
    )
else:
    tle = None
    _HAS_TLE_CACHE = False

# Bound the full-row SSA footprint before selecting a wider-row path.
_MAX_ROW_BLOCK = 32768
_MIN_PARTIAL_TILE = 512
_MAX_PARTIALS = 512
_MAX_PARTITIONED_VOCAB = _MIN_PARTIAL_TILE * _MAX_PARTIALS
# Keep only launch metadata; scratch and caller tensors remain per invocation.
_MAX_PARTITIONED_PLANS = 128
_PARTITIONED_LAUNCH_PLANS = {}
# Match the cache candidates' byte budgets; amortize staging over useful data.
_MIN_CACHED_ROW_BYTES = 131072
_CACHE_BYTES_PER_SM = 196608
# Prune wasteful tails, but always keep a general candidate for tiny rows.
_STREAM_UTIL_NUM = 3
_STREAM_UTIL_DEN = 4


@lru_cache(maxsize=None)
def _device_info(device_index):
    properties = torch.cuda.get_device_properties(device_index)
    return properties.major, properties.multi_processor_count


def _select_min_p_kernel(logits):
    """Choose one implementation from metadata, with a general fallback."""
    batch, vocab = logits.shape
    major, sms = _device_info(logits.device.index)
    if major >= 9:
        if vocab <= _MAX_ROW_BLOCK:
            return _min_p_row_kernel
        if vocab <= _MAX_PARTITIONED_VOCAB:
            if batch * 2 < sms:
                return _partitioned_min_p
            row_bytes = vocab * logits.element_size()
            if (
                _HAS_TLE_CACHE
                and row_bytes >= _MIN_CACHED_ROW_BYTES
                and batch * row_bytes >= sms * _CACHE_BYTES_PER_SM
            ):
                return _min_p_cached_word_kernel
    return _min_p_general_kernel


def _with_meta(config, **meta):
    """Derive shape-dependent metadata without dropping launch attributes."""
    return triton.Config(
        {**config.kwargs, **meta},
        num_warps=config.num_warps,
        num_ctas=config.num_ctas,
        num_stages=config.num_stages,
        maxnreg=config.maxnreg,
    )


@triton.jit
def _min_p_store_filtered(output_ptr, values, threshold, mask):
    tl.store(
        output_ptr,
        tl.where(values.to(tl.float32) < threshold, float("-inf"), values),
        mask,
    )


@triton.jit
def _min_p_cache_encode(values, SIZE: tl.constexpr):
    """Preserve input bits in shared-memory uint32 words."""
    if values.dtype == tl.float32:
        return values.to(tl.uint32, bitcast=True)
    else:
        pair = tl.reshape(values.to(tl.uint16, bitcast=True), (SIZE // 2, 2))
        low, high = tl.split(pair)
        return low.to(tl.uint32) | (high.to(tl.uint32) << 16)


@triton.jit
def _min_p_cache_decode(words, SIZE: tl.constexpr, TYPE: tl.constexpr):
    """Restore the original floating-point representation from cache words."""
    if TYPE == tl.float32:
        return words.to(TYPE, bitcast=True)
    else:
        low = (words & 65535).to(tl.uint16)
        high = (words >> 16).to(tl.uint16)
        return tl.reshape(tl.join(low, high), (SIZE,)).to(TYPE, bitcast=True)


def _prune_cache_configs(configs, named_args, **kwargs):
    dtype_bytes = {**named_args, **kwargs}["logits_ptr"].element_size()
    return [
        _with_meta(config, WORD_BLOCK=config.kwargs["TILE"] * dtype_bytes // 4)
        for config in configs
    ]


@libentry()
@libtuner(
    configs=[
        triton.Config(
            {"TILE": 8192, "REG": reg, "CACHE_BYTES": cache_bytes, "WORD_BLOCK": 0},
            num_warps=warps,
            num_stages=1,
        )
        for reg, cache_bytes, warps in (
            (8192, 196608, 8),
            (16384, 196608, 8),
            (8192, 196608, 16),
            (16384, 196608, 16),
            (32768, 196608, 16),
            (16384, 131072, 8),
        )
    ],
    key=["N_ROWS", "VOCAB_SIZE"],
    prune_configs_by={"early_config_prune": _prune_cache_configs},
    warmup=10,
    rep=25,
    use_cuda_graph=True,
)
@triton.jit
def _min_p_cached_word_kernel(
    logits_ptr,
    logits_stride,
    mapping_ptr,
    min_p_ptr,
    N_ROWS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    TILE: tl.constexpr,
    REG: tl.constexpr,
    CACHE_BYTES: tl.constexpr,
    WORD_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * logits_stride
    request = tl.load(mapping_ptr + row)
    min_p = tl.load(min_p_ptr + request)
    if min_p != 0.0:
        CACHE: tl.constexpr = CACHE_BYTES // (
            logits_ptr.dtype.element_ty.primitive_bitwidth // 8
        )
        PACK: tl.constexpr = 32 // logits_ptr.dtype.element_ty.primitive_bitwidth
        MAIN: tl.constexpr = triton.next_power_of_2(CACHE) // 2
        EXTRA: tl.constexpr = CACHE - MAIN
        lane = tl.arange(0, TILE)
        held_columns = tl.arange(0, REG)
        held = tl.load(
            row_ptr + held_columns,
            held_columns < VOCAB_SIZE,
            other=float("-inf"),
        )
        maxima = tl.full((TILE,), float("-inf"), tl.float32)
        primary = tle.gpu.alloc(
            [MAIN // PACK],
            dtype=tl.uint32,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        secondary = tle.gpu.alloc(
            [EXTRA // PACK],
            dtype=tl.uint32,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        for start in range(0, MAIN, TILE):
            columns = REG + start + lane
            values = tl.load(
                row_ptr + columns,
                columns < VOCAB_SIZE,
                other=float("-inf"),
            )
            pointers = tle.gpu.local_ptr(
                primary, (start // PACK + tl.arange(0, WORD_BLOCK),)
            )
            tl.store(pointers, _min_p_cache_encode(values, TILE))
            maxima = tl.maximum(maxima, values.to(tl.float32))
        for start in range(0, EXTRA, TILE):
            columns = REG + MAIN + start + lane
            values = tl.load(
                row_ptr + columns,
                columns < VOCAB_SIZE,
                other=float("-inf"),
            )
            pointers = tle.gpu.local_ptr(
                secondary, (start // PACK + tl.arange(0, WORD_BLOCK),)
            )
            tl.store(pointers, _min_p_cache_encode(values, TILE))
            maxima = tl.maximum(maxima, values.to(tl.float32))
        for start in range(REG + CACHE, VOCAB_SIZE, TILE):
            columns = start + lane
            values = tl.load(
                row_ptr + columns,
                columns < VOCAB_SIZE,
                other=float("-inf"),
            )
            maxima = tl.maximum(maxima, values.to(tl.float32))
        maximum = tl.maximum(tl.max(maxima, 0), tl.max(held.to(tl.float32), 0))
        threshold = maximum + tl.log(min_p)
        # Streamed tiles are reused first; the other values remain on chip.
        for start in range(REG + CACHE, VOCAB_SIZE, TILE):
            columns = start + lane
            values = tl.load(
                row_ptr + columns,
                columns < VOCAB_SIZE,
                other=float("-inf"),
            )
            _min_p_store_filtered(
                row_ptr + columns, values, threshold, columns < VOCAB_SIZE
            )
        for start in range(0, MAIN, TILE):
            columns = REG + start + lane
            values = _min_p_cache_decode(
                tl.load(
                    tle.gpu.local_ptr(
                        primary, (start // PACK + tl.arange(0, WORD_BLOCK),)
                    )
                ),
                TILE,
                logits_ptr.dtype.element_ty,
            )
            _min_p_store_filtered(
                row_ptr + columns, values, threshold, columns < VOCAB_SIZE
            )
        for start in range(0, EXTRA, TILE):
            columns = REG + MAIN + start + lane
            values = _min_p_cache_decode(
                tl.load(
                    tle.gpu.local_ptr(
                        secondary, (start // PACK + tl.arange(0, WORD_BLOCK),)
                    )
                ),
                TILE,
                logits_ptr.dtype.element_ty,
            )
            _min_p_store_filtered(
                row_ptr + columns, values, threshold, columns < VOCAB_SIZE
            )
        _min_p_store_filtered(
            row_ptr + held_columns, held, threshold, held_columns < VOCAB_SIZE
        )


def _prune_row_configs(configs, named_args, **kwargs):
    block = triton.next_power_of_2({**named_args, **kwargs}["VOCAB_SIZE"])
    # Bound per-thread arrays while preserving each candidate's register cap.
    return [
        _with_meta(config, ROW_BLOCK=block)
        for config in configs
        if block <= 512 * 32 * config.num_warps
    ]


@libentry()
@libtuner(
    configs=[
        triton.Config(
            {"ROW_BLOCK": 0},
            num_warps=warps,
            num_ctas=1,
            num_stages=1,
            maxnreg=maxnreg,
        )
        for warps, maxnreg in ((4, 128), (8, 128), (16, 128), (32, 64))
    ],
    key=["N_ROWS", "VOCAB_SIZE"],
    prune_configs_by={"early_config_prune": _prune_row_configs},
    warmup=10,
    rep=25,
    use_cuda_graph=True,
)
@triton.jit
def _min_p_row_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    min_p_ptr,
    N_ROWS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * logits_stride
    request = tl.load(expanded_idx_mapping_ptr + row)
    min_p = tl.load(min_p_ptr + request).to(tl.float32)
    if min_p != 0.0:
        columns = tl.arange(0, ROW_BLOCK)[None, :]
        valid = columns < VOCAB_SIZE
        values = tl.load(
            row_ptr + columns,
            mask=valid,
            other=float("-inf"),
        )
        maximum = tl.max(values, 1).to(tl.float32)
        threshold = maximum + tl.log(min_p)
        _min_p_store_filtered(row_ptr + columns, values, threshold[:, None], valid)


def _prune_stream_configs(configs, named_args, **kwargs):
    vocab = {**named_args, **kwargs}["vocab_size"]
    smallest_tile = min(config.kwargs["TILE"] for config in configs)
    return [
        config
        for config in configs
        if config.kwargs["TILE"] == smallest_tile
        or _STREAM_UTIL_DEN * vocab
        >= _STREAM_UTIL_NUM
        * triton.cdiv(vocab, config.kwargs["TILE"])
        * config.kwargs["TILE"]
    ]


@libentry()
@libtuner(
    configs=[
        triton.Config({"TILE": tile}, num_warps=warps, num_stages=1)
        for tile in (4096, 8192, 16384)
        for warps in (4, 8, 16)
    ],
    key=["N_ROWS", "vocab_size"],
    prune_configs_by={"early_config_prune": _prune_stream_configs},
    warmup=10,
    rep=25,
    use_cuda_graph=False,
)
@triton.jit
def _min_p_general_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    min_p_ptr,
    N_ROWS,
    vocab_size,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * logits_stride
    request = tl.load(expanded_idx_mapping_ptr + row)
    min_p = tl.load(min_p_ptr + request).to(tl.float32)
    if min_p != 0.0:
        lane = tl.arange(0, TILE)
        maxima = tl.full((TILE,), float("-inf"), tl.float32)
        for start in range(0, vocab_size, TILE):
            columns = start + lane
            values = tl.load(
                row_ptr + columns,
                mask=columns < vocab_size,
                other=float("-inf"),
            ).to(tl.float32)
            maxima = tl.maximum(maxima, values)
        maximum = tl.max(maxima, 0)
        threshold = maximum + tl.log(min_p)
        for start in range(0, vocab_size, TILE):
            columns = start + lane
            values = tl.load(
                row_ptr + columns,
                mask=columns < vocab_size,
                other=float("-inf"),
            )
            _min_p_store_filtered(
                row_ptr + columns, values, threshold, columns < vocab_size
            )


@libentry()
@libtuner(
    configs=[
        triton.Config({"PART_BLOCK": block}, num_warps=warps, num_stages=1)
        for block, warps in ((512, 4), (1024, 4), (2048, 4), (4096, 4), (8192, 8))
    ],
    key=["N_ROWS", "vocab_size"],
    warmup=10,
    rep=25,
    use_cuda_graph=True,
)
@triton.jit
def _min_p_partial_kernel(
    logits_ptr,
    logits_stride,
    mapping_ptr,
    min_p_ptr,
    partial_ptr,
    partial_stride,
    N_ROWS,
    vocab_size,
    PART_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * logits_stride
    part = tl.program_id(1)
    request = tl.load(mapping_ptr + row)
    min_p = tl.load(min_p_ptr + request)
    if min_p != 0.0:
        columns = part.to(tl.int64) * PART_BLOCK + tl.arange(0, PART_BLOCK)
        values = tl.load(
            row_ptr + columns,
            mask=columns < vocab_size,
            other=float("-inf"),
        ).to(tl.float32)
        maximum = tl.max(values, 0)
        tl.store(partial_ptr + row * partial_stride + part, maximum)


def _prune_mask_configs(configs, named_args, **kwargs):
    partial_block = triton.next_power_of_2({**named_args, **kwargs}["N_PARTS"])
    return [_with_meta(config, PARTIAL_BLOCK=partial_block) for config in configs]


@libentry()
@libtuner(
    configs=[
        triton.Config(
            {"MASK_BLOCK": block, "PARTIAL_BLOCK": 0}, num_warps=warps, num_stages=1
        )
        for block, warps in (
            (512, 4),
            (1024, 1),
            (1024, 4),
            (2048, 4),
            (4096, 4),
            (4096, 8),
        )
    ],
    key=["N_ROWS", "vocab_size", "N_PARTS"],
    prune_configs_by={"early_config_prune": _prune_mask_configs},
    warmup=10,
    rep=25,
    use_cuda_graph=True,
)
@triton.jit
def _min_p_mask_kernel(
    logits_ptr,
    logits_stride,
    mapping_ptr,
    min_p_ptr,
    partial_ptr,
    partial_stride,
    N_ROWS,
    vocab_size,
    N_PARTS: tl.constexpr,
    MASK_BLOCK: tl.constexpr,
    PARTIAL_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * logits_stride
    tile = tl.program_id(1).to(tl.int64)
    request = tl.load(mapping_ptr + row)
    min_p = tl.load(min_p_ptr + request).to(tl.float32)
    if min_p != 0.0:
        parts = tl.arange(0, PARTIAL_BLOCK)
        maxima = tl.load(
            partial_ptr + row * partial_stride + parts,
            mask=parts < N_PARTS,
            other=float("-inf"),
        )
        maximum = tl.max(maxima, 0)
        threshold = maximum + tl.log(min_p)
        columns = tile * MASK_BLOCK + tl.arange(0, MASK_BLOCK)
        values = tl.load(
            row_ptr + columns,
            mask=columns < vocab_size,
            other=float("-inf"),
        )
        _min_p_store_filtered(
            row_ptr + columns, values, threshold, columns < vocab_size
        )


# Match LibEntry's positional constexpr ABI; other versions keep normal entries.
_PARTITIONED_FULL_ARGS_ABI = tuple(triton.__version__.split(".")[:2]) in (
    ("3", "3"),
    ("3", "4"),
    ("3", "5"),
    ("3", "6"),
)


def _partitioned_launch_key(logits, mapping, min_p, partial, batch, vocab, stride):
    partial_entry = _min_p_partial_kernel
    mask_entry = _min_p_mask_kernel
    if (
        not _PARTITIONED_FULL_ARGS_ABI
        or getattr(partial_entry, "_has_flagtune_tuner", True)
        or getattr(mask_entry, "_has_flagtune_tuner", True)
        or partial_entry.divisibility != mask_entry.divisibility
    ):
        return None
    alignment = partial_entry.divisibility
    # Match LibEntry tensor specialization, including offset views and scratch.
    pointers = tuple(
        (value.dtype, value.data_ptr() % alignment == 0)
        for value in (logits, mapping, min_p, partial)
    )
    return (
        partial_entry,
        mask_entry,
        logits.device.index,
        batch,
        vocab,
        stride,
        alignment,
        pointers,
    )


def _partitioned_min_p(logits, mapping, min_p):
    batch, vocab = logits.shape
    stride = logits.stride(0)
    capacity = triton.cdiv(vocab, _MIN_PARTIAL_TILE)
    partial = torch.empty((batch, capacity), device=logits.device, dtype=torch.float32)
    key = _partitioned_launch_key(logits, mapping, min_p, partial, batch, vocab, stride)
    plan = _PARTITIONED_LAUNCH_PLANS.get(key) if key is not None else None
    if plan is not None:
        partial_launch, mask_launch, part_block, parts, mask_block, partial_block = plan
        partial_launch(
            logits, stride, mapping, min_p, partial, capacity, batch, vocab, part_block
        )
        mask_launch(
            logits,
            stride,
            mapping,
            min_p,
            partial,
            capacity,
            batch,
            vocab,
            parts,
            mask_block,
            partial_block,
        )
        return

    # Cold calls retain the normal entry/tuning policy for both passes.
    partial_kernel, meta = _min_p_partial_kernel[
        lambda config: (batch, triton.cdiv(vocab, config["PART_BLOCK"]))
    ](logits, stride, mapping, min_p, partial, capacity, batch, vocab)
    parts = triton.cdiv(vocab, meta["PART_BLOCK"])
    mask_kernel, mask_meta = _min_p_mask_kernel[
        lambda config: (batch, triton.cdiv(vocab, config["MASK_BLOCK"]))
    ](
        logits,
        stride,
        mapping,
        min_p,
        partial,
        capacity,
        batch,
        vocab,
        parts,
    )
    # Config pre-hooks belong to the normal entry path, not a bare launcher.
    if key is not None and not any(
        record[-1]
        for entry in (_min_p_partial_kernel, _min_p_mask_kernel)
        for record in entry.kernel_cache[logits.device.index].values()
    ):
        if len(_PARTITIONED_LAUNCH_PLANS) >= _MAX_PARTITIONED_PLANS:
            _PARTITIONED_LAUNCH_PLANS.clear()
        _PARTITIONED_LAUNCH_PLANS[key] = (
            partial_kernel[(batch, parts, 1)],
            mask_kernel[(batch, triton.cdiv(vocab, mask_meta["MASK_BLOCK"]), 1)],
            meta["PART_BLOCK"],
            parts,
            mask_meta["MASK_BLOCK"],
            mask_meta["PARTIAL_BLOCK"],
        )


def apply_min_p(
    logits: torch.Tensor, expanded_idx_mapping: torch.Tensor, min_p: torch.Tensor
) -> None:
    """Filter logits[B, V] in place using per-request FP32 min-p thresholds.

    Columns must be contiguous and rows disjoint; row padding is supported.
    Empty inputs are a no-op. Mapping indices must be valid and thresholds
    must lie in [0, 1]. This operation is inference-only and returns None.
    """
    if logits.ndim != 2:
        raise ValueError("min_p expects rank-2 logits")
    num_tokens, vocab_size = logits.shape
    if num_tokens == 0 or vocab_size == 0:
        return
    if logits.device.type != "cuda":
        raise NotImplementedError("min_p requires CUDA logits")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError("min_p supports float16, bfloat16 and float32 logits")
    if logits.stride(1) != 1 or (num_tokens > 1 and logits.stride(0) < vocab_size):
        raise NotImplementedError("min_p requires contiguous columns and disjoint rows")
    if expanded_idx_mapping.dtype not in (torch.int32, torch.int64):
        raise NotImplementedError("min_p requires int32 or int64 request indices")
    if (
        expanded_idx_mapping.ndim != 1
        or expanded_idx_mapping.shape[0] != num_tokens
        or expanded_idx_mapping.stride(0) != 1
        or min_p.ndim != 1
        or min_p.stride(0) != 1
        or min_p.dtype != torch.float32
    ):
        raise NotImplementedError("min_p requires dense mapping[B] and FP32 thresholds")
    if expanded_idx_mapping.device != logits.device or min_p.device != logits.device:
        raise ValueError("min_p inputs must be on the same CUDA device")
    if logits.requires_grad or min_p.requires_grad:
        raise NotImplementedError("min_p is an inference-only in-place operation")

    kernel = _select_min_p_kernel(logits)
    if kernel is _partitioned_min_p:
        kernel(logits, expanded_idx_mapping, min_p)
        return
    kernel[(num_tokens,)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        min_p,
        num_tokens,
        vocab_size,
    )
