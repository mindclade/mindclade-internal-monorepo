"""SE(3)-equivariant atom denoiser using invariant scalar attention."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from mindclade.models.common.masking.coordinate_mask import (
    apply_coordinate_mask,
    center_coordinates,
)


def _gather_pair_features(pair: Tensor, atom_to_token: Tensor) -> Tensor:
    batch, atoms = atom_to_token.shape
    batch_index = torch.arange(batch, device=pair.device).view(batch, 1, 1)
    left = atom_to_token.clamp_min(0).unsqueeze(2).expand(batch, atoms, atoms)
    right = atom_to_token.clamp_min(0).unsqueeze(1).expand(batch, atoms, atoms)
    return pair[batch_index, left, right]


class _EquivariantDenoiserBlock(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        pair_dim: int,
        time_dim: int,
        heads: int,
        radial_basis_functions: int,
        atom_knn: int,
        bond_type_vocab_size: int,
        bond_stereo_vocab_size: int,
        dropout: float,
        epsilon: float,
        sigma_max: float,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = atom_dim // heads
        self.atom_knn = atom_knn
        self.norm = nn.LayerNorm(atom_dim, eps=epsilon)
        self.query = nn.Linear(atom_dim, atom_dim, bias=False)
        self.key = nn.Linear(atom_dim, atom_dim, bias=False)
        self.value = nn.Linear(atom_dim, atom_dim, bias=False)
        self.pair_bias = nn.Linear(pair_dim, heads, bias=False)
        self.rbf_bias = nn.Linear(radial_basis_functions, heads, bias=False)
        self.rbf_gate = nn.Linear(radial_basis_functions + pair_dim, 2)
        self.bond_type_bias = nn.Embedding(bond_type_vocab_size, heads)
        self.bond_stereo_bias = nn.Embedding(bond_stereo_vocab_size, heads)
        self.time_projection = nn.Linear(time_dim, atom_dim)
        self.scalar_output = nn.Linear(atom_dim, atom_dim)
        self.vector_invariants = nn.Linear(2, atom_dim)
        self.coordinate_scale = nn.Linear(atom_dim, 2)
        self.transition = nn.Sequential(
            nn.LayerNorm(atom_dim, eps=epsilon),
            nn.Linear(atom_dim, atom_dim * 2),
            nn.SiLU(),
            nn.Linear(atom_dim * 2, atom_dim),
        )
        self.dropout = nn.Dropout(dropout)
        centers = torch.linspace(0.0, sigma_max * 2.0, radial_basis_functions)
        self.register_buffer("rbf_centers", centers, persistent=False)
        spacing = float(centers[1] - centers[0]) if radial_basis_functions > 1 else sigma_max
        self.rbf_width = max(spacing, 1e-3)

    def _edge_mask(
        self,
        distance: Tensor,
        atom_mask: Tensor,
        bond_indices: Tensor,
        bond_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, atoms, _ = distance.shape
        valid_pair = atom_mask.unsqueeze(2) & atom_mask.unsqueeze(1)
        diagonal = torch.eye(atoms, device=distance.device, dtype=torch.bool).unsqueeze(0)
        candidates = valid_pair & ~diagonal
        k = min(self.atom_knn, max(atoms - 1, 1))
        ranking = distance.masked_fill(~candidates, float("inf"))
        neighbor_index = ranking.topk(k, dim=-1, largest=False, sorted=True).indices
        nearest = torch.zeros_like(candidates)
        nearest.scatter_(2, neighbor_index, True)
        nearest &= candidates
        bond_edges = torch.zeros_like(candidates)
        for batch_index in range(batch):
            valid = bond_mask[batch_index]
            if bool(valid.any()):
                edges = bond_indices[batch_index, valid]
                left, right = edges[:, 0], edges[:, 1]
                bond_edges[batch_index, left, right] = True
                bond_edges[batch_index, right, left] = True
        # Self edges make attention well-defined for a single valid atom. Their
        # displacement is zero, so they cannot introduce a coordinate frame.
        self_edges = diagonal & valid_pair
        return (nearest | bond_edges | self_edges), bond_edges

    def _bond_bias(
        self,
        atom_count: int,
        bond_indices: Tensor,
        bond_type: Tensor,
        bond_stereo: Tensor,
        bond_mask: Tensor,
    ) -> Tensor:
        batch = bond_indices.shape[0]
        result = self.bond_type_bias.weight.new_zeros((batch, atom_count, atom_count, self.heads))
        for batch_index in range(batch):
            valid = bond_mask[batch_index]
            if bool(valid.any()):
                edges = bond_indices[batch_index, valid]
                values = self.bond_type_bias(bond_type[batch_index, valid])
                values = values + self.bond_stereo_bias(bond_stereo[batch_index, valid])
                left, right = edges[:, 0], edges[:, 1]
                result[batch_index, left, right] = values
                result[batch_index, right, left] = values
        return result

    def forward(
        self,
        atoms: Tensor,
        coordinates: Tensor,
        pair: Tensor,
        atom_to_token: Tensor,
        atom_mask: Tensor,
        bond_indices: Tensor,
        bond_type: Tensor,
        bond_stereo: Tensor,
        bond_mask: Tensor,
        time_embedding: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, atom_count, _ = atoms.shape
        normalized = self.norm(atoms) + self.time_projection(time_embedding).unsqueeze(1)
        pair_features = _gather_pair_features(pair, atom_to_token)
        # displacement[i,j] points from atom i to atom j.
        displacement = coordinates.unsqueeze(1) - coordinates.unsqueeze(2)
        distance = torch.linalg.vector_norm(displacement.float(), dim=-1).to(dtype=atoms.dtype)
        edge_mask, _ = self._edge_mask(distance, atom_mask, bond_indices, bond_mask)
        rbf = torch.exp(
            -(
                (
                    (distance.unsqueeze(-1) - self.rbf_centers.to(dtype=distance.dtype))
                    / self.rbf_width
                )
                ** 2
            )
        )
        query = self.query(normalized).reshape(batch, atom_count, self.heads, self.head_dim)
        key = self.key(normalized).reshape(batch, atom_count, self.heads, self.head_dim)
        value = self.value(normalized).reshape(batch, atom_count, self.heads, self.head_dim)
        score = torch.einsum("bihd,bjhd->bijh", query, key) / math.sqrt(self.head_dim)
        score = score + self.pair_bias(pair_features) + self.rbf_bias(rbf)
        score = score + self._bond_bias(atom_count, bond_indices, bond_type, bond_stereo, bond_mask)
        score = score.masked_fill(~edge_mask.unsqueeze(-1), torch.finfo(score.dtype).min)
        attention = torch.softmax(score.float(), dim=2).to(dtype=atoms.dtype)
        scalar = torch.einsum("bijh,bjhd->bihd", attention, value).reshape(batch, atom_count, -1)

        vector_gates = torch.tanh(self.rbf_gate(torch.cat((rbf, pair_features), dim=-1)))
        scalar_attention = attention.mean(dim=-1)
        first_vector = (scalar_attention.unsqueeze(-1) * vector_gates[..., :1] * displacement).sum(
            dim=2
        )
        second_vector = (scalar_attention.unsqueeze(-1) * vector_gates[..., 1:] * displacement).sum(
            dim=2
        )
        axial_vector = torch.linalg.cross(first_vector, second_vector, dim=-1)
        vector_norms = torch.stack(
            (
                torch.linalg.vector_norm(first_vector.float(), dim=-1),
                torch.linalg.vector_norm(axial_vector.float(), dim=-1),
            ),
            dim=-1,
        ).to(dtype=atoms.dtype)
        atom_update = self.scalar_output(scalar) + self.vector_invariants(vector_norms)
        atoms = atoms + self.dropout(atom_update)
        atoms = atoms + self.dropout(self.transition(atoms))
        atoms = atoms * atom_mask.to(dtype=atoms.dtype).unsqueeze(-1)
        scales = torch.tanh(self.coordinate_scale(atoms))
        coordinate_update = (
            scales[..., :1] * first_vector + scales[..., 1:] * axial_vector / self.rbf_width
        )
        coordinate_update = center_coordinates(coordinate_update, atom_mask)
        coordinates = apply_coordinate_mask(coordinates + coordinate_update, atom_mask)
        return atoms, coordinates


class CoordinateDenoiser(nn.Module):
    """Stacked scalar/vector blocks returning centered predicted noise."""

    def __init__(
        self,
        *,
        atom_dim: int,
        pair_dim: int,
        time_dim: int,
        blocks: int,
        heads: int,
        radial_basis_functions: int,
        atom_knn: int,
        bond_type_vocab_size: int,
        bond_stereo_vocab_size: int,
        dropout: float,
        epsilon: float,
        sigma_max: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _EquivariantDenoiserBlock(
                    atom_dim,
                    pair_dim,
                    time_dim,
                    heads,
                    radial_basis_functions,
                    atom_knn,
                    bond_type_vocab_size,
                    bond_stereo_vocab_size,
                    dropout,
                    epsilon,
                    sigma_max,
                )
                for _ in range(blocks)
            ]
        )

    def forward(
        self,
        atoms: Tensor,
        coordinates: Tensor,
        pair: Tensor,
        atom_to_token: Tensor,
        atom_mask: Tensor,
        bond_indices: Tensor,
        bond_type: Tensor,
        bond_stereo: Tensor,
        bond_mask: Tensor,
        time_embedding: Tensor,
    ) -> tuple[Tensor, Tensor]:
        original = coordinates
        working = coordinates
        for block in self.blocks:
            atoms, working = block(
                atoms,
                working,
                pair,
                atom_to_token,
                atom_mask,
                bond_indices,
                bond_type,
                bond_stereo,
                bond_mask,
                time_embedding,
            )
        predicted_noise = center_coordinates(original - working, atom_mask)
        return predicted_noise, atoms


__all__ = ["CoordinateDenoiser"]
