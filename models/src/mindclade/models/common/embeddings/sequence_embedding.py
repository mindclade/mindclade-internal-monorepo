"""Token metadata embedding."""

from __future__ import annotations

from torch import Tensor, nn

from mindclade.models.common.masking.sequence_mask import apply_sequence_mask


class SequenceEmbedding(nn.Module):
    def __init__(
        self,
        *,
        token_vocab_size: int,
        molecule_vocab_size: int,
        max_chain_id: int,
        max_position_id: int,
        dimension: int,
    ) -> None:
        super().__init__()
        self.token = nn.Embedding(token_vocab_size, dimension, padding_idx=0)
        self.molecule = nn.Embedding(molecule_vocab_size, dimension)
        self.chain = nn.Embedding(max_chain_id, dimension)
        self.position = nn.Embedding(max_position_id, dimension)
        self.norm = nn.LayerNorm(dimension)

    def forward(
        self,
        token_type: Tensor,
        molecule_type: Tensor,
        chain_id: Tensor,
        position_id: Tensor,
        mask: Tensor,
    ) -> Tensor:
        values = (
            self.token(token_type)
            + self.molecule(molecule_type)
            + self.chain(chain_id)
            + self.position(position_id)
        )
        return apply_sequence_mask(self.norm(values), mask)


__all__ = ["SequenceEmbedding"]
