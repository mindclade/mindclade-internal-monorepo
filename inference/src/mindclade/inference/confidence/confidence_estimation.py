"""Masked confidence reductions in fp32."""

from __future__ import annotations

import torch


def estimate_confidence(values: torch.Tensor, mask: torch.Tensor) -> float:
    """Return the mean valid confidence, accepting probabilities or logits."""

    if values.shape != mask.shape or mask.dtype is not torch.bool:
        raise TypeError("confidence values and boolean mask must have identical shapes")
    if not mask.any():
        raise ValueError("confidence mask cannot be empty")
    valid = values.float()[mask]
    if not torch.isfinite(valid).all():
        raise FloatingPointError("confidence contains non-finite values")
    if float(valid.min()) < 0.0 or float(valid.max()) > 1.0:
        valid = torch.sigmoid(valid)
    return float(valid.mean().clamp(0.0, 1.0))


def token_confidence_from_atoms(
    atom_confidence: torch.Tensor,
    atom_to_token: torch.Tensor,
    atom_mask: torch.Tensor,
    *,
    token_count: int,
) -> torch.Tensor:
    if atom_confidence.shape != atom_to_token.shape or atom_mask.shape != atom_to_token.shape:
        raise ValueError("atom confidence, mapping, and mask must have shape [B, A]")
    if atom_mask.dtype is not torch.bool or atom_to_token.dtype not in (torch.int32, torch.int64):
        raise TypeError("atom mask must be bool and atom_to_token must be integral")
    if token_count < 1:
        raise ValueError("token_count must be positive")
    if atom_mask.any():
        valid_mapping = atom_to_token[atom_mask]
        if valid_mapping.min() < 0 or valid_mapping.max() >= token_count:
            raise ValueError("valid atom_to_token index is out of range")
    output = atom_confidence.new_zeros((atom_confidence.shape[0], token_count), dtype=torch.float32)
    counts = output.new_zeros(output.shape)
    for batch_index in range(atom_confidence.shape[0]):
        valid = atom_mask[batch_index]
        indices = atom_to_token[batch_index, valid].long()
        output[batch_index].index_add_(0, indices, atom_confidence[batch_index, valid].float())
        counts[batch_index].index_add_(0, indices, torch.ones_like(indices, dtype=torch.float32))
    return output / counts.clamp_min(1.0)
