"""Masked incoming/outgoing triangle multiplication."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mindclade.models.common.masking.pair_mask import apply_pair_mask


class TriangleMultiplication(nn.Module):
    def __init__(self, pair_dim: int, hidden_dim: int, *, outgoing: bool, epsilon: float) -> None:
        super().__init__()
        self.outgoing = outgoing
        self.norm = nn.LayerNorm(pair_dim, eps=epsilon)
        self.left = nn.Linear(pair_dim, hidden_dim)
        self.right = nn.Linear(pair_dim, hidden_dim)
        self.gate = nn.Linear(pair_dim, pair_dim)
        self.output = nn.Linear(hidden_dim, pair_dim)

    def forward(self, pair: Tensor, pair_mask: Tensor) -> Tensor:
        values = self.norm(pair)
        left = self.left(values) * pair_mask.unsqueeze(-1).to(dtype=values.dtype)
        right = self.right(values) * pair_mask.unsqueeze(-1).to(dtype=values.dtype)
        if self.outgoing:
            update = torch.einsum("bikh,bkjh->bijh", left, right)
            count = torch.einsum(
                "bik,bkj->bij", pair_mask.to(dtype=values.dtype), pair_mask.to(dtype=values.dtype)
            )
        else:
            update = torch.einsum("bkih,bkjh->bijh", left, right)
            count = torch.einsum(
                "bki,bkj->bij", pair_mask.to(dtype=values.dtype), pair_mask.to(dtype=values.dtype)
            )
        update = update / count.clamp_min(1.0).unsqueeze(-1)
        update = self.output(update) * torch.sigmoid(self.gate(values))
        return apply_pair_mask(update, pair_mask)


__all__ = ["TriangleMultiplication"]
