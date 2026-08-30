"""Mask-preserving coordinate projections for sampled structures."""

from __future__ import annotations

import torch


def center_coordinates(coordinates: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape [B, A, 3]")
    if atom_mask.shape != coordinates.shape[:2] or atom_mask.dtype is not torch.bool:
        raise TypeError("atom_mask must be bool with shape [B, A]")
    if not atom_mask.any(dim=1).all():
        raise ValueError("each batch item needs at least one valid atom")
    weight = atom_mask.to(coordinates.dtype).unsqueeze(-1)
    center = (coordinates * weight).sum(dim=1, keepdim=True) / weight.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0)
    return (coordinates - center) * weight


def project_bond_lengths(
    coordinates: torch.Tensor,
    bond_indices: torch.Tensor,
    bond_mask: torch.Tensor,
    target_lengths: torch.Tensor,
    *,
    iterations: int = 1,
) -> torch.Tensor:
    """Apply symmetric pair corrections while keeping padded atoms untouched."""

    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape [B, A, 3]")
    if bond_indices.ndim != 3 or bond_indices.shape[-1] != 2:
        raise ValueError("bond_indices must have shape [B, E, 2]")
    if bond_mask.shape != bond_indices.shape[:2] or bond_mask.dtype is not torch.bool:
        raise TypeError("bond_mask must be bool with shape [B, E]")
    if target_lengths.shape != bond_mask.shape:
        raise ValueError("target_lengths must have shape [B, E]")
    if iterations < 0:
        raise ValueError("iterations cannot be negative")
    if bond_mask.any():
        valid_indices = bond_indices[bond_mask]
        if valid_indices.min() < 0 or valid_indices.max() >= coordinates.shape[1]:
            raise ValueError("valid bond index is out of range")

    result = coordinates
    for _ in range(iterations):
        next_rows: list[torch.Tensor] = []
        for batch_index in range(coordinates.shape[0]):
            row = result[batch_index]
            correction = torch.zeros_like(row)
            degree = torch.zeros((row.shape[0], 1), dtype=row.dtype, device=row.device)
            valid = bond_mask[batch_index]
            if valid.any():
                edges = bond_indices[batch_index, valid]
                left = edges[:, 0]
                right = edges[:, 1]
                delta = row[right] - row[left]
                distance = (
                    torch.linalg.vector_norm(delta.float(), dim=-1).to(row.dtype).clamp_min(1e-6)
                )
                target = target_lengths[batch_index, valid].to(row.dtype)
                shift = 0.5 * ((distance - target) / distance).unsqueeze(-1) * delta
                correction.index_add_(0, left, shift)
                correction.index_add_(0, right, -shift)
                ones = torch.ones((edges.shape[0], 1), dtype=row.dtype, device=row.device)
                degree.index_add_(0, left, ones)
                degree.index_add_(0, right, ones)
            next_rows.append(row + correction / degree.clamp_min(1.0))
        result = torch.stack(next_rows, dim=0)
    return result
