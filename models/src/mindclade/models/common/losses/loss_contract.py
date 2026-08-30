"""Stable model loss names."""

from __future__ import annotations

import dataclasses

from torch import Tensor


@dataclasses.dataclass(frozen=True)
class LossBreakdown:
    loss: Tensor
    noise_loss: Tensor
    distogram_loss: Tensor
    confidence_loss: Tensor
    calibration_loss: Tensor
    geometry_loss: Tensor


__all__ = ["LossBreakdown"]
