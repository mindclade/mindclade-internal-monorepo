"""Symmetric pair initialization from token representations."""

from __future__ import annotations

from torch import Tensor, nn

from mindclade.models.common.masking.pair_mask import apply_pair_mask


class PairEmbedding(nn.Module):
    def __init__(self, token_dim: int, pair_dim: int, max_relative_position: int = 64) -> None:
        super().__init__()
        self.max_relative_position = max_relative_position
        self.token_projection = nn.Linear(token_dim, pair_dim, bias=False)
        self.relative_position = nn.Embedding(max_relative_position + 1, pair_dim)
        self.same_chain = nn.Embedding(2, pair_dim)
        self.norm = nn.LayerNorm(pair_dim)

    def forward(
        self,
        tokens: Tensor,
        position_id: Tensor,
        chain_id: Tensor,
        pair_mask: Tensor,
    ) -> Tensor:
        projected = self.token_projection(tokens)
        relative = (position_id.unsqueeze(2) - position_id.unsqueeze(1)).abs()
        relative = relative.clamp_max(self.max_relative_position)
        same_chain = (chain_id.unsqueeze(2) == chain_id.unsqueeze(1)).long()
        pair = (
            projected.unsqueeze(2)
            + projected.unsqueeze(1)
            + self.relative_position(relative)
            + self.same_chain(same_chain)
        )
        return apply_pair_mask(self.norm(pair), pair_mask)


__all__ = ["PairEmbedding"]
