"""Pre-norm gated token transition."""

from __future__ import annotations

from torch import Tensor, nn

from mindclade.models.common.masking.sequence_mask import apply_sequence_mask


class SequenceTransition(nn.Module):
    def __init__(self, dimension: int, multiplier: int, dropout: float, epsilon: float) -> None:
        super().__init__()
        hidden = dimension * multiplier
        self.norm = nn.LayerNorm(dimension, eps=epsilon)
        self.up = nn.Linear(dimension, hidden * 2)
        self.down = nn.Linear(hidden, dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        gate, content = self.up(self.norm(values)).chunk(2, dim=-1)
        update = self.down(nn.functional.silu(gate) * content)
        return apply_sequence_mask(values + self.dropout(update), mask)


__all__ = ["SequenceTransition"]
