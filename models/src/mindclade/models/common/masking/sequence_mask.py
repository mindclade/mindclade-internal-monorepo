"""Sequence masking helpers."""

from __future__ import annotations

from torch import Tensor


def apply_sequence_mask(values: Tensor, mask: Tensor) -> Tensor:
    if values.shape[:2] != mask.shape:
        raise ValueError("sequence values must begin with [B, T] matching mask")
    return values * mask.to(dtype=values.dtype).unsqueeze(-1)


__all__ = ["apply_sequence_mask"]
