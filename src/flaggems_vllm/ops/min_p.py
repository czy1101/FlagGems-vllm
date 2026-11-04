# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-place min-p filtering with streaming and optional TLE cluster reduction."""

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

if has_triton_tle():
    import triton.experimental.tle.language as tle

    _HAS_TLE_SHARED = all(
        callable(getattr(tle.gpu, name, None)) for name in ("alloc", "local_ptr")
    )
else:
    tle = None
    _HAS_TLE_SHARED = False

_TLE_CLUSTER_SIZE = 8
_MAX_TLE_CLUSTER_BLOCK = 16384
_HAS_TLE_CLUSTER = _HAS_TLE_SHARED and all(
    callable(getattr(tle, name, None))
    for name in ("device_mesh", "shard_id", "remote", "distributed_barrier")
)
_TLE_CLUSTER_MESH = (
    tle.device_mesh({"block_cluster": [("cluster_x", _TLE_CLUSTER_SIZE)]})
    if _HAS_TLE_CLUSTER
    else None
)

_MIN_WIDE_VOCAB = 32768
# Prune wasteful tails, but always keep a general candidate for tiny rows.
_STREAM_UTIL_NUM = 3
_STREAM_UTIL_DEN = 4


def _device_sm_count(device_index):
    properties = torch.cuda.get_device_properties(device_index)
    return properties.multi_processor_count


def _select_min_p_kernel(logits):
    batch, vocab = logits.shape
    sms = _device_sm_count(logits.device.index)
    if (
        _HAS_TLE_CLUSTER
        and _MIN_WIDE_VOCAB < vocab <= _TLE_CLUSTER_SIZE * _MAX_TLE_CLUSTER_BLOCK
        and batch * 2 < sms
    ):
        return _min_p_tle_cluster_kernel
    if (
        _HAS_TLE_CLUSTER
        and _MIN_WIDE_VOCAB < vocab <= _TLE_CLUSTER_SIZE * _MAX_TLE_CLUSTER_BLOCK
        and _STREAM_UTIL_DEN * vocab
        >= _STREAM_UTIL_NUM * (1 << (vocab - 1).bit_length())
    ):
        # Reuse actual input shards when padding waste is small.
        return _min_p_tle_staged_cluster_kernel
    return _min_p_general_kernel


def _with_meta(config, **meta):
    """Derive shape-dependent metadata without dropping launch attributes."""
    return triton.Config(
        {**config.kwargs, **meta},
        num_warps=config.num_warps,
        num_ctas=config.num_ctas,
        num_stages=config.num_stages,
    )


@triton.jit
def _min_p_store_filtered(output_ptr, values, threshold, mask):
    tl.store(
        output_ptr,
        tl.where(values.to(tl.float32) < threshold, float("-inf"), values),
        mask,
    )


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


def _prune_tle_cluster_configs(configs, named_args, **kwargs):
    block = (
        triton.next_power_of_2({**named_args, **kwargs}["VOCAB_SIZE"])
        // _TLE_CLUSTER_SIZE
    )
    return [_with_meta(config, BLOCK=block) for config in configs]


if _HAS_TLE_CLUSTER:

    @libentry()
    @libtuner(
        configs=[
            triton.Config(
                {"BLOCK": 0},
                num_warps=warps,
                num_ctas=1,
                num_stages=1,
            )
            for warps in (8, 16)
        ],
        key=["N_ROWS", "VOCAB_SIZE"],
        prune_configs_by={"early_config_prune": _prune_tle_cluster_configs},
        warmup=10,
        rep=25,
        use_cuda_graph=True,
    )
    @triton.jit
    def _min_p_tle_cluster_kernel(
        logits_ptr,
        logits_stride,
        mapping_ptr,
        min_p_ptr,
        N_ROWS: tl.constexpr,
        VOCAB_SIZE: tl.constexpr,
        PARTS: tl.constexpr,
        MESH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64) // PARTS
        rank = tle.shard_id(MESH, "cluster_x")
        request = tl.load(mapping_ptr + row)
        min_p = tl.load(min_p_ptr + request).to(tl.float32)
        if min_p != 0.0:
            columns = rank.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
            values = tl.load(
                logits_ptr + row * logits_stride + columns,
                columns < VOCAB_SIZE,
                other=-float("inf"),
            )
            local_maximum = tl.max(values.to(tl.float32), 0)
            scratch = tle.gpu.alloc(
                [1],
                dtype=tl.float32,
                scope=tle.gpu.smem,
                init_value=local_maximum[None],
                nv_mma_shared_layout=False,
            )
            tle.distributed_barrier(MESH)
            maximum = local_maximum
            for peer in tl.static_range(0, PARTS):
                remote_scratch = tle.remote(scratch, peer, scope=MESH)
                peer_pointer = tle.gpu.local_ptr(remote_scratch, (tl.arange(0, 1),))
                peer_maximum = tl.max(tl.load(peer_pointer), 0)
                maximum = tl.maximum(maximum, peer_maximum)
            tle.distributed_barrier(MESH)
            threshold = maximum + tl.log(min_p)
            _min_p_store_filtered(
                logits_ptr + row * logits_stride + columns,
                values,
                threshold,
                columns < VOCAB_SIZE,
            )


    @libentry()
    @libtuner(
        configs=[
            triton.Config(
                {"BLOCK": 0},
                num_warps=warps,
                num_ctas=1,
                num_stages=1,
            )
            for warps in (8, 16)
        ],
        key=["N_ROWS", "VOCAB_SIZE"],
        prune_configs_by={"early_config_prune": _prune_tle_cluster_configs},
        warmup=10,
        rep=25,
        use_cuda_graph=True,
    )
    @triton.jit
    def _min_p_tle_staged_cluster_kernel(
        logits_ptr,
        logits_stride,
        mapping_ptr,
        min_p_ptr,
        N_ROWS: tl.constexpr,
        VOCAB_SIZE: tl.constexpr,
        PARTS: tl.constexpr,
        MESH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64) // PARTS
        rank = tle.shard_id(MESH, "cluster_x")
        request = tl.load(mapping_ptr + row)
        min_p = tl.load(min_p_ptr + request).to(tl.float32)
        if min_p != 0.0:
            columns = rank.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
            values = tl.load(
                logits_ptr + row * logits_stride + columns,
                columns < VOCAB_SIZE,
                other=-float("inf"),
            )
            staged_values = tle.gpu.alloc(
                values.shape,
                dtype=values.dtype,
                scope=tle.gpu.smem,
                init_value=values,
                nv_mma_shared_layout=False,
            )
            local_maximum = tl.max(values.to(tl.float32), 0)
            scratch = tle.gpu.alloc(
                [1],
                dtype=tl.float32,
                scope=tle.gpu.smem,
                init_value=local_maximum[None],
                nv_mma_shared_layout=False,
            )
            tle.distributed_barrier(MESH)
            maximum = local_maximum
            for peer in tl.static_range(0, PARTS):
                remote_scratch = tle.remote(scratch, peer, scope=MESH)
                peer_pointer = tle.gpu.local_ptr(remote_scratch, (tl.arange(0, 1),))
                peer_maximum = tl.max(tl.load(peer_pointer), 0)
                maximum = tl.maximum(maximum, peer_maximum)
            tle.distributed_barrier(MESH)
            threshold = maximum + tl.log(min_p)
            values = tl.load(tle.gpu.local_ptr(staged_values))
            _min_p_store_filtered(
                logits_ptr + row * logits_stride + columns,
                values,
                threshold,
                columns < VOCAB_SIZE,
            )


def apply_min_p(
    logits: torch.Tensor, expanded_idx_mapping: torch.Tensor, min_p: torch.Tensor
) -> None:
    """Filter logits in place; require valid mappings and FP32 min_p in [0, 1]."""
    num_tokens, vocab_size = logits.shape
    if num_tokens == 0 or vocab_size == 0:
        return
    kernel = _select_min_p_kernel(logits)
    if _HAS_TLE_CLUSTER and kernel in (
        _min_p_tle_cluster_kernel,
        _min_p_tle_staged_cluster_kernel,
    ):
        kernel[(num_tokens,)](
            logits,
            logits.stride(0),
            expanded_idx_mapping,
            min_p,
            num_tokens,
            vocab_size,
            _TLE_CLUSTER_SIZE,
            _TLE_CLUSTER_MESH,
        )
        return
    kernel[(num_tokens,)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        min_p,
        num_tokens,
        vocab_size,
    )
