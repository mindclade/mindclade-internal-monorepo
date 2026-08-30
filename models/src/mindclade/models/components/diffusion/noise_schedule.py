"""Variance-exploding diffusion schedule."""

from __future__ import annotations

import math

import torch
from torch import Tensor


class VESchedule:
    def __init__(self, sigma_min: float, sigma_max: float) -> None:
        if sigma_min <= 0 or sigma_max <= sigma_min:
            raise ValueError("VE schedule requires 0 < sigma_min < sigma_max")
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    def sigma(self, time: Tensor) -> Tensor:
        return self.sigma_min * torch.exp(time.float() * math.log(self.sigma_max / self.sigma_min))

    def normalized_time(self, sigma: Tensor) -> Tensor:
        return torch.log(sigma.float() / self.sigma_min) / math.log(self.sigma_max / self.sigma_min)

    def levels(self, steps: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if steps < 2:
            raise ValueError("sampling requires at least two diffusion steps")
        levels = torch.logspace(
            math.log10(self.sigma_max),
            math.log10(self.sigma_min),
            steps,
            device=device,
            dtype=torch.float32,
        )
        return levels.to(dtype=dtype)


__all__ = ["VESchedule"]
