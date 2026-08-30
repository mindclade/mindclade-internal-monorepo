"""Atom-coordinate masking helpers."""

from __future__ import annotations

from torch import Tensor


def apply_coordinate_mask(values: Tensor, atom_mask: Tensor) -> Tensor:
    if values.shape[:2] != atom_mask.shape or values.shape[-1] != 3:
        raise ValueError("coordinate values must have shape [B, A, 3]")
    return values * atom_mask.to(dtype=values.dtype).unsqueeze(-1)


def center_coordinates(values: Tensor, atom_mask: Tensor) -> Tensor:
    masked = apply_coordinate_mask(values, atom_mask)
    weight = atom_mask.to(dtype=values.dtype).unsqueeze(-1)
    center = masked.float().sum(dim=1, keepdim=True) / weight.float().sum(
        dim=1, keepdim=True
    ).clamp_min(1.0)
    return apply_coordinate_mask(values - center.to(dtype=values.dtype), atom_mask)


__all__ = ["apply_coordinate_mask", "center_coordinates"]
