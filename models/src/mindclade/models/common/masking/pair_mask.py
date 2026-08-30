"""Pair-mask construction."""

from __future__ import annotations

from torch import Tensor


def make_pair_mask(sequence_mask: Tensor) -> Tensor:
    if sequence_mask.ndim != 2:
        raise ValueError("sequence_mask must have shape [B, T]")
    return sequence_mask.unsqueeze(2) & sequence_mask.unsqueeze(1)


def apply_pair_mask(values: Tensor, mask: Tensor) -> Tensor:
    if values.shape[:3] != mask.shape:
        raise ValueError("pair values must begin with [B, T, T] matching mask")
    return values * mask.to(dtype=values.dtype).unsqueeze(-1)


__all__ = ["apply_pair_mask", "make_pair_mask"]
