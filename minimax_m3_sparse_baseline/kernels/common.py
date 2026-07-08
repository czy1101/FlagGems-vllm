# -*- coding: utf-8 -*-
"""Shared constants and utilities for MiniMax M3 sparse attention kernels."""

import torch

# One sparse block == one KV page.
SPARSE_BLOCK_SIZE = 128

_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
)


def round_up(x: int, y: int) -> int:
    """Round up x to the nearest multiple of y."""
    return ((x + y - 1) // y) * y


def is_pdl_supported() -> bool:
    """Check if the current GPU supports Programmatic Dependent Launch (SM90+)."""
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 9
    except Exception:
        return False
