"""Continuous sinusoidal diffusion-time embedding."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn


class TimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.projection = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
        )

    def forward(self, time: Tensor) -> Tensor:
        if time.ndim != 1:
            raise ValueError("time must have shape [B]")
        half = self.dimension // 2
        frequencies = torch.exp(
            torch.arange(half, device=time.device, dtype=torch.float32)
            * (-math.log(10_000.0) / max(half - 1, 1))
        )
        phases = time.float().unsqueeze(1) * frequencies.unsqueeze(0)
        values = torch.cat((torch.sin(phases), torch.cos(phases)), dim=1)
        if values.shape[1] < self.dimension:
            values = torch.nn.functional.pad(values, (0, self.dimension - values.shape[1]))
        projection_dtype = next(self.projection.parameters()).dtype
        return cast(Tensor, self.projection(values.to(dtype=projection_dtype)))


__all__ = ["TimeEmbedding"]
