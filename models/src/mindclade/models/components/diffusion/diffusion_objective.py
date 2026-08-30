"""Centered VE noise-prediction objective."""

from __future__ import annotations

from torch import Tensor

from mindclade.models.common.losses.masked_losses import masked_mse
from mindclade.models.common.masking.coordinate_mask import center_coordinates


def noise_prediction_loss(
    predicted_noise: Tensor,
    noisy_coordinates: Tensor,
    target_coordinates: Tensor,
    sigma: Tensor,
    target_mask: Tensor,
) -> Tensor:
    target_noise = (noisy_coordinates.float() - target_coordinates.float()) / sigma.float().view(
        -1, 1, 1
    )
    target_noise = center_coordinates(target_noise, target_mask)
    return masked_mse(predicted_noise, target_noise, target_mask)


__all__ = ["noise_prediction_loss"]
