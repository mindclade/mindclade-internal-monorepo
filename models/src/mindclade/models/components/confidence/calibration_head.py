"""Positive confidence-temperature calibration."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CalibrationHead(nn.Module):
    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.temperature = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 1))

    def forward(
        self, tokens: Tensor, token_mask: Tensor, confidence: Tensor
    ) -> tuple[Tensor, Tensor]:
        weights = token_mask.to(dtype=tokens.dtype).unsqueeze(-1)
        pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        temperature = torch.nn.functional.softplus(self.temperature(pooled).squeeze(-1)) + 1e-4
        probability = confidence.clamp(1e-6, 1.0 - 1e-6)
        logits = torch.logit(probability.float()) / temperature.float().unsqueeze(-1)
        return temperature, torch.sigmoid(logits).to(dtype=confidence.dtype)


__all__ = ["CalibrationHead"]
