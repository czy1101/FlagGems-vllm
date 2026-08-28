# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""TLE-semantic fallback for Ascend Triton 3.5.x.

The upstream KDA TLE kernels use shared-memory allocation, local pointers,
asynchronous copies, pipes, and warp-specialization.  Triton 3.5.1's Ascend
compiler accepts the Python module but aborts while lowering those dialect
operations.  This module keeps the *phase boundaries* of that implementation
without emitting any TLE IR:

1. the intra-chunk producer computes ``w/u/qg/kg/Aqk/Akk``;
2. the state/output consumer propagates the recurrent state and writes ``o``.

The two phases are executed sequentially with ordinary Triton kernels.  This
is a correctness-compatible replacement, not a claim of TLE performance.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch


# These names are intentionally public to make the compatibility mapping easy
# to inspect from a debug shell and to keep the documentation tied to code.
TLE_PRIMITIVE_REPLACEMENTS = {
    "tle.gpu.alloc": "torch.empty/torch.zeros intermediate tensors",
    "tle.gpu.local_ptr": "tl.make_block_ptr with explicit offsets",
    "tle.load(is_async=True)": "ordinary tl.load in the producer phase",
    "tle.gpu.copy": "ordinary tl.load/tl.store through global-memory tensors",
    "tle.pipe": "sequential producer -> consumer phase boundary",
    "tle.gpu.smem": "compiler-managed registers and global-memory workspaces",
    "tle.gpu.warp_specialize": "ordinary Triton program instances",
}


def chunk_kda_fwd_infer_tle_emulated(
    *,
    intra_forward: Callable,
    state_output_forward: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 16,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Execute KDA using the two phases represented by the TLE path.

    ``intra_forward`` and ``state_output_forward`` are injected by the
    Ascend adapter so its conservative autotune restrictions and state-layout
    workaround remain active.  No ``triton.experimental.tle`` symbol is
    imported or compiled here.
    """

    BT = chunk_size
    if scale is None:
        scale = q.shape[-1] ** -0.5

    # TLE producer phase: materialize the same intermediates that the
    # async-copy/pipe implementation would publish to its consumer.
    w, u, qg, kg, Aqk, Akk, g_cumsum = intra_forward(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=BT,
        lower_bound=lower_bound,
        A_log=A_log if use_gate_in_kernel else None,
        dt_bias=dt_bias,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        apply_beta_sigmoid=use_beta_sigmoid_in_kernel,
    )

    if os.environ.get("FLAGGEMS_ASCEND_KDA_DEBUG_FINITE", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        for name, tensor in (
            ("w", w),
            ("u", u),
            ("qg", qg),
            ("kg", kg),
            ("Aqk", Aqk),
            ("Akk", Akk),
            ("g_cumsum", g_cumsum),
        ):
            finite = bool(torch.isfinite(tensor).all().item())
            max_abs = float(torch.nan_to_num(tensor.float()).abs().max().item())
            print(
                f"[ascend chunk_kda debug] {name}: "
                f"finite={finite}, max_abs={max_abs:.6g}",
                flush=True,
            )

    # TLE consumer phase: run the recurrent state update and output write.
    # Ascend 910B needs K/V state layout for the STATE_V_FIRST case; the
    # adapter's ordinary Triton kernel is correct in that layout, so transpose
    # only at this compatibility boundary.
    use_npu_state_layout_workaround = state_v_first and q.device.type == "npu"
    triton_state_v_first = False if use_npu_state_layout_workaround else state_v_first
    triton_initial_state = initial_state
    if use_npu_state_layout_workaround and initial_state is not None:
        triton_initial_state = initial_state.transpose(-1, -2).contiguous()

    o, final_state = state_output_forward(
        kg=kg,
        w=w,
        u=u,
        gk=g_cumsum,
        qg=qg,
        Aqk=Aqk,
        scale=scale,
        initial_state=triton_initial_state,
        output_final_state=output_final_state,
        state_v_first=triton_state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=BT,
    )

    if use_npu_state_layout_workaround and final_state is not None:
        final_state = final_state.transpose(-1, -2).contiguous()
    return o, final_state
