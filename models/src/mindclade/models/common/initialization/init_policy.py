"""Explicit initialization policy values."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class InitializationPolicy:
    standard_deviation: float = 0.02
    zero_bias: bool = True
    unit_norm_scale: bool = True


__all__ = ["InitializationPolicy"]
