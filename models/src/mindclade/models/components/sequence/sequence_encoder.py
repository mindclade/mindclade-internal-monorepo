"""Pair-conditioned token self-attention."""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn

from mindclade.models.common.masking.sequence_mask import apply_sequence_mask

from .sequence_transition import SequenceTransition


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        token_dim: int,
        pair_dim: int,
        heads: int,
        transition_multiplier: int,
        dropout: float,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(token_dim, eps=epsilon)
        self.pair_context = nn.Linear(pair_dim, token_dim, bias=False)
        self.attention = nn.MultiheadAttention(token_dim, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.transition = SequenceTransition(token_dim, transition_multiplier, dropout, epsilon)

    def forward(
        self, tokens: Tensor, pair: Tensor, token_mask: Tensor, pair_mask: Tensor
    ) -> Tensor:
        pair_weights = pair_mask.to(dtype=pair.dtype).unsqueeze(-1)
        context = (pair * pair_weights).sum(dim=2) / pair_weights.sum(dim=2).clamp_min(1.0)
        normalized = self.norm(tokens) + self.pair_context(context)
        update, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~token_mask,
            need_weights=False,
        )
        tokens = apply_sequence_mask(tokens + self.dropout(update), token_mask)
        return cast(Tensor, self.transition(tokens, token_mask))


__all__ = ["SequenceEncoder"]
