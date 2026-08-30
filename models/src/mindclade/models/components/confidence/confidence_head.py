"""Atom and token confidence distributions."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConfidenceHead(nn.Module):
    def __init__(self, atom_dim: int, token_dim: int, bins: int) -> None:
        super().__init__()
        self.atom_logits = nn.Linear(atom_dim, bins)
        self.token_logits = nn.Linear(token_dim, bins)
        self.register_buffer("bin_centers", torch.linspace(0.0, 1.0, bins), persistent=False)

    def forward(
        self, atoms: Tensor, tokens: Tensor, atom_mask: Tensor, token_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        atom_logits = self.atom_logits(atoms)
        token_logits = self.token_logits(tokens)
        centers = self.get_buffer("bin_centers").to(dtype=atom_logits.dtype)
        atom_confidence = (
            torch.softmax(atom_logits.float(), dim=-1).to(atom_logits.dtype) * centers
        ).sum(-1)
        token_confidence = (
            torch.softmax(token_logits.float(), dim=-1).to(token_logits.dtype) * centers
        ).sum(-1)
        atom_confidence = atom_confidence * atom_mask.to(atom_confidence.dtype)
        token_confidence = token_confidence * token_mask.to(token_confidence.dtype)
        atom_logits = atom_logits * atom_mask.unsqueeze(-1).to(atom_logits.dtype)
        token_logits = token_logits * token_mask.unsqueeze(-1).to(token_logits.dtype)
        return atom_logits, token_logits, atom_confidence, token_confidence


__all__ = ["ConfidenceHead"]
