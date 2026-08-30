"""Row/column triangle attention over pair representations."""

from __future__ import annotations

from torch import Tensor, nn

from mindclade.models.common.masking.pair_mask import apply_pair_mask


class TriangleAttention(nn.Module):
    def __init__(
        self, pair_dim: int, heads: int, *, starting: bool, dropout: float, epsilon: float
    ) -> None:
        super().__init__()
        self.starting = starting
        self.norm = nn.LayerNorm(pair_dim, eps=epsilon)
        self.attention = nn.MultiheadAttention(pair_dim, heads, dropout=dropout, batch_first=True)

    def forward(self, pair: Tensor, pair_mask: Tensor) -> Tensor:
        values = pair if self.starting else pair.transpose(1, 2)
        normalized = self.norm(values)
        batch, tokens, _, dimension = normalized.shape
        rows = normalized.reshape(batch * tokens, tokens, dimension)
        token_mask = pair_mask.diagonal(dim1=1, dim2=2)
        key_padding = (
            (~token_mask).unsqueeze(1).expand(batch, tokens, tokens).reshape(batch * tokens, tokens)
        )
        update, _ = self.attention(
            rows, rows, rows, key_padding_mask=key_padding, need_weights=False
        )
        update = update.reshape(batch, tokens, tokens, dimension)
        if not self.starting:
            update = update.transpose(1, 2)
        return apply_pair_mask(update, pair_mask)


__all__ = ["TriangleAttention"]
