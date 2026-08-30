"""Proper rigid-transform helpers used by equivariance tests."""

from __future__ import annotations

import torch
from torch import Tensor


def validate_proper_rotation(rotation: Tensor, atol: float = 1e-5) -> None:
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape [3, 3]")
    identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
    if not torch.allclose(rotation.transpose(0, 1) @ rotation, identity, atol=atol, rtol=0.0):
        raise ValueError("rotation must be orthonormal")
    if not torch.allclose(torch.det(rotation), rotation.new_tensor(1.0), atol=atol, rtol=0.0):
        raise ValueError("rotation must be proper (determinant +1)")


def apply_rigid_transform(coordinates: Tensor, rotation: Tensor, translation: Tensor) -> Tensor:
    validate_proper_rotation(rotation)
    if translation.shape != (3,):
        raise ValueError("translation must have shape [3]")
    return coordinates @ rotation.transpose(0, 1) + translation


__all__ = ["apply_rigid_transform", "validate_proper_rotation"]
