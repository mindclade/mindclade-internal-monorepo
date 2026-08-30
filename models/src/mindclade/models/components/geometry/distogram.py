"""Symmetric token-pair distance logits."""

from __future__ import annotations

from torch import Tensor, nn

from mindclade.models.common.masking.pair_mask import apply_pair_mask


class DistogramHead(nn.Module):
    def __init__(self, pair_dim: int, bins: int) -> None:
        super().__init__()
        self.projection = nn.Linear(pair_dim, bins)

    def forward(self, pair: Tensor, pair_mask: Tensor) -> Tensor:
        logits = self.projection(pair)
        logits = 0.5 * (logits + logits.transpose(1, 2))
        return apply_pair_mask(logits, pair_mask)


__all__ = ["DistogramHead"]
