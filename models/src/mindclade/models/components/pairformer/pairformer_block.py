"""Q0 pairformer block with token, outer-product, and triangle updates."""

from __future__ import annotations

from torch import Tensor, nn

from mindclade.models.common.masking.pair_mask import apply_pair_mask
from mindclade.models.components.sequence.sequence_encoder import SequenceEncoder

from .outer_product_mean import OuterProductMean
from .triangle_attention import TriangleAttention
from .triangle_multiplication import TriangleMultiplication


class _PairTransition(nn.Module):
    def __init__(self, pair_dim: int, multiplier: int, dropout: float, epsilon: float) -> None:
        super().__init__()
        hidden = pair_dim * multiplier
        self.norm = nn.LayerNorm(pair_dim, eps=epsilon)
        self.layers = nn.Sequential(
            nn.Linear(pair_dim, hidden * 2),
            nn.GLU(dim=-1),
            nn.SiLU(),
            nn.Linear(hidden, pair_dim),
            nn.Dropout(dropout),
        )

    def forward(self, pair: Tensor, pair_mask: Tensor) -> Tensor:
        return apply_pair_mask(self.layers(self.norm(pair)), pair_mask)


class PairformerBlock(nn.Module):
    def __init__(
        self,
        token_dim: int,
        pair_dim: int,
        token_heads: int,
        triangle_heads: int,
        transition_multiplier: int,
        outer_product_dim: int,
        dropout: float,
        epsilon: float,
    ) -> None:
        super().__init__()
        triangle_hidden = max(outer_product_dim, pair_dim // 2)
        self.sequence = SequenceEncoder(
            token_dim, pair_dim, token_heads, transition_multiplier, dropout, epsilon
        )
        self.outer_product = OuterProductMean(token_dim, outer_product_dim, pair_dim)
        self.outgoing = TriangleMultiplication(
            pair_dim, triangle_hidden, outgoing=True, epsilon=epsilon
        )
        self.incoming = TriangleMultiplication(
            pair_dim, triangle_hidden, outgoing=False, epsilon=epsilon
        )
        self.starting_attention = TriangleAttention(
            pair_dim, triangle_heads, starting=True, dropout=dropout, epsilon=epsilon
        )
        self.ending_attention = TriangleAttention(
            pair_dim, triangle_heads, starting=False, dropout=dropout, epsilon=epsilon
        )
        self.transition = _PairTransition(pair_dim, transition_multiplier, dropout, epsilon)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, tokens: Tensor, pair: Tensor, token_mask: Tensor, pair_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        tokens = self.sequence(tokens, pair, token_mask, pair_mask)
        pair = apply_pair_mask(
            pair + self.dropout(self.outer_product(tokens, pair_mask)), pair_mask
        )
        pair = apply_pair_mask(pair + self.dropout(self.outgoing(pair, pair_mask)), pair_mask)
        pair = apply_pair_mask(pair + self.dropout(self.incoming(pair, pair_mask)), pair_mask)
        pair = apply_pair_mask(
            pair + self.dropout(self.starting_attention(pair, pair_mask)), pair_mask
        )
        pair = apply_pair_mask(
            pair + self.dropout(self.ending_attention(pair, pair_mask)), pair_mask
        )
        pair = apply_pair_mask(pair + self.transition(pair, pair_mask), pair_mask)
        return tokens, pair


__all__ = ["PairformerBlock"]
