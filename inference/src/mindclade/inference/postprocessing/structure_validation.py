"""Fail-fast numerical and topology validation for generated coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class StructureValidationReport:
    batch_size: int
    valid_atoms: int
    max_absolute_coordinate: float
    minimum_nonbonded_distance: float | None


def validate_structure(
    coordinates: torch.Tensor,
    *,
    atom_mask: torch.Tensor,
    bond_indices: torch.Tensor | None = None,
    bond_mask: torch.Tensor | None = None,
    max_absolute_coordinate: float = 10000.0,
    reject_collisions_below: float = 0.0,
    report_minimum_nonbonded_distance: bool = False,
) -> StructureValidationReport:
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape [B, A, 3]")
    if type(report_minimum_nonbonded_distance) is not bool:
        raise TypeError("report_minimum_nonbonded_distance must be boolean")
    if atom_mask.shape != coordinates.shape[:2] or atom_mask.dtype is not torch.bool:
        raise TypeError("atom_mask must be bool with shape [B, A]")
    if not torch.isfinite(coordinates).all():
        raise FloatingPointError("coordinates contain NaN or infinity")
    if not atom_mask.any(dim=1).all():
        raise ValueError("every batch item must contain a valid atom")
    if coordinates[~atom_mask].numel() and not torch.equal(
        coordinates[~atom_mask], torch.zeros_like(coordinates[~atom_mask])
    ):
        raise ValueError("padded atom coordinates must be exactly zero")
    maximum = float(coordinates.abs().max()) if coordinates.numel() else 0.0
    if maximum > max_absolute_coordinate:
        raise ValueError("coordinates exceed the configured numerical envelope")

    bonded: set[tuple[int, int, int]] = set()
    if (bond_indices is None) != (bond_mask is None):
        raise ValueError("bond_indices and bond_mask must be provided together")
    if bond_indices is not None and bond_mask is not None:
        if bond_indices.shape[:2] != bond_mask.shape or bond_indices.shape[-1] != 2:
            raise ValueError("bond tensors have incompatible shapes")
        batch_indices = torch.nonzero(bond_mask, as_tuple=False)[:, 0].tolist()
        edge_values = bond_indices[bond_mask].tolist()
        for batch_index, edge in zip(batch_indices, edge_values, strict=True):
            left, right = sorted((int(edge[0]), int(edge[1])))
            if left < 0 or right >= coordinates.shape[1] or left == right:
                raise ValueError("valid bond indices are invalid")
            bonded.add((batch_index, left, right))

    minimum: float | None = None
    if report_minimum_nonbonded_distance or reject_collisions_below > 0:
        for batch_index in range(coordinates.shape[0]):
            valid_indices = torch.nonzero(atom_mask[batch_index], as_tuple=False).flatten()
            if valid_indices.numel() < 2:
                continue
            points = coordinates[batch_index, valid_indices].float()
            distances = torch.cdist(points, points)
            for left_local in range(valid_indices.numel()):
                for right_local in range(left_local + 1, valid_indices.numel()):
                    left = int(valid_indices[left_local])
                    right = int(valid_indices[right_local])
                    if (batch_index, left, right) in bonded:
                        continue
                    value = float(distances[left_local, right_local])
                    minimum = value if minimum is None else min(minimum, value)
    if minimum is not None and minimum < reject_collisions_below:
        raise ValueError("structure contains a non-bonded atom collision")
    return StructureValidationReport(
        batch_size=coordinates.shape[0],
        valid_atoms=int(atom_mask.sum()),
        max_absolute_coordinate=maximum,
        minimum_nonbonded_distance=minimum,
    )
