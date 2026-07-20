"""Utility functions extracted from vLLM framework dependencies."""

class _CurrentPlatform:
    """Mock platform info for H100 (SM90)."""
    def is_arch_support_pdl(self) -> bool:
        return True

current_platform = _CurrentPlatform()


def round_up(n: int, d: int) -> int:
    """Round n up to the nearest multiple of d."""
    return (n + d - 1) // d * d
