"""Registered Q0 pairformer stack."""

from __future__ import annotations

from torch import Tensor, nn

from mindclade.models.components.pairformer.pairformer_block import PairformerBlock
from mindclade.models.families.clade.cladefold.configuration.cladefold_q0 import CladeFoldConfig


class PairformerStack(nn.Module):
    def __init__(self, config: CladeFoldConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                PairformerBlock(
                    config.token_dim,
                    config.pair_dim,
                    config.token_heads,
                    config.triangle_heads,
                    config.transition_multiplier,
                    config.outer_product_dim,
                    config.dropout,
                    config.layer_norm_epsilon,
                )
                for _ in range(config.pairformer_blocks)
            ]
        )

    def forward(
        self, tokens: Tensor, pair: Tensor, token_mask: Tensor, pair_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        for block in self.blocks:
            tokens, pair = block(tokens, pair, token_mask, pair_mask)
        return tokens, pair


__all__ = ["PairformerStack"]
