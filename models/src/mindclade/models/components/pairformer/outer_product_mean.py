"""Symmetric outer-product token-to-pair update."""

from __future__ import annotations

from torch import Tensor, nn

from mindclade.models.common.masking.pair_mask import apply_pair_mask


class OuterProductMean(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int, pair_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(token_dim)
        self.left = nn.Linear(token_dim, hidden_dim)
        self.right = nn.Linear(token_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, pair_dim)

    def forward(self, tokens: Tensor, pair_mask: Tensor) -> Tensor:
        values = self.norm(tokens)
        left, right = self.left(values), self.right(values)
        outer = left.unsqueeze(2) * right.unsqueeze(1)
        outer = 0.5 * (outer + outer.transpose(1, 2))
        return apply_pair_mask(self.output(outer), pair_mask)


__all__ = ["OuterProductMean"]
