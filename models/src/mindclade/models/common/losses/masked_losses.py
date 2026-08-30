"""Masked regression and classification losses."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F

from .loss_reduction import differentiable_zero, masked_mean


def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    return masked_mean((prediction.float() - target.float()).square(), mask, reference=prediction)


def masked_huber(prediction: Tensor, target: Tensor, mask: Tensor, delta: float = 1.0) -> Tensor:
    values = F.huber_loss(prediction.float(), target.float(), delta=delta, reduction="none")
    return masked_mean(values, mask, reference=prediction)


def masked_cross_entropy(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if not bool(mask.any()):
        return differentiable_zero(logits)
    selected_logits = logits[mask].float()
    selected_target = target[mask].long()
    return F.cross_entropy(selected_logits, selected_target)


__all__ = ["masked_cross_entropy", "masked_huber", "masked_mse"]
