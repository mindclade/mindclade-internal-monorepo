"""Typed contracts shared by inference samplers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from ..adaptive_compute.resume_frontier import ResumeFrontier

DenoiseFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
ConfidenceFunction = Callable[[torch.Tensor], float]


@dataclass(frozen=True, slots=True)
class SamplingOutcome:
    coordinates: torch.Tensor
    seed: int
    completed_steps: int
    stop_reason: str
    confidence: float | None = None
    trajectory: tuple[torch.Tensor, ...] = ()
    resume_frontier: ResumeFrontier | None = None

    def __post_init__(self) -> None:
        if self.coordinates.ndim != 3 or self.coordinates.shape[-1] != 3:
            raise ValueError("sampled coordinates must have shape [B, A, 3]")
        if self.completed_steps < 1:
            raise ValueError("completed_steps must be positive")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be within [0, 1]")


class Sampler(Protocol):
    @property
    def digest(self) -> str: ...
