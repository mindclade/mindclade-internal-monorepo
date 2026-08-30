"""Coordinate denoising head wrapper."""

from __future__ import annotations

from torch import Tensor, nn

from mindclade.models.common.masking.coordinate_mask import apply_coordinate_mask


class CoordinateDiffusionHead(nn.Module):
    def forward(
        self, noisy_coordinates: Tensor, predicted_noise: Tensor, sigma: Tensor, atom_mask: Tensor
    ) -> Tensor:
        denoised = (
            noisy_coordinates
            - sigma.to(dtype=noisy_coordinates.dtype).view(-1, 1, 1) * predicted_noise
        )
        return apply_coordinate_mask(denoised, atom_mask)


__all__ = ["CoordinateDiffusionHead"]
