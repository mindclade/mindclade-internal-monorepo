"""Equivariant scaling of atom-vector updates."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mindclade.models.common.masking.coordinate_mask import center_coordinates


class StructureHead(nn.Module):
    def __init__(self, atom_dim: int) -> None:
        super().__init__()
        self.scale = nn.Linear(atom_dim, 1)

    def forward(self, atoms: Tensor, vector_update: Tensor, atom_mask: Tensor) -> Tensor:
        return center_coordinates(torch.tanh(self.scale(atoms)) * vector_update, atom_mask)


__all__ = ["StructureHead"]
