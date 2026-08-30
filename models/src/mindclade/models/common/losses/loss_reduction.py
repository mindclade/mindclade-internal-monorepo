"""Numerically stable masked reductions."""

from __future__ import annotations

import torch
from torch import Tensor


def differentiable_zero(reference: Tensor) -> Tensor:
    return reference.float().sum() * 0.0


def masked_mean(values: Tensor, mask: Tensor, *, reference: Tensor | None = None) -> Tensor:
    if values.shape[: mask.ndim] != mask.shape:
        raise ValueError("mask must match the leading value dimensions")
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    weights = expanded.to(dtype=torch.float32)
    numerator = (values.float() * weights).sum()
    denominator = weights.expand_as(values).sum()
    if not bool(mask.any()):
        return differentiable_zero(reference if reference is not None else values)
    return numerator / denominator.clamp_min(1.0)


__all__ = ["differentiable_zero", "masked_mean"]
