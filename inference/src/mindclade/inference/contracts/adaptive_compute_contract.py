"""Wire-independent adaptive-compute request contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AdaptiveComputeRequest:
    """Bounded adaptive sampling controls.

    Defaults require at least sixteen diffusion steps and three consecutive
    convergence observations before early stopping.
    """

    enabled: bool = False
    min_steps: int = 16
    max_steps: int = 32
    evaluation_interval: int = 4
    patience: int = 3
    confidence_gain_threshold: float = 0.002
    displacement_threshold_angstrom: float = 0.05
    max_candidates: int = 1

    def __post_init__(self) -> None:
        if self.min_steps < 2:
            raise ValueError("min_steps must be at least 2")
        if self.max_steps < self.min_steps or self.max_steps > 128:
            raise ValueError("max_steps must be within [min_steps, 128]")
        if self.evaluation_interval < 1:
            raise ValueError("evaluation_interval must be positive")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if not 0.0 <= self.confidence_gain_threshold <= 1.0:
            raise ValueError("confidence_gain_threshold must be within [0, 1]")
        if self.displacement_threshold_angstrom < 0.0:
            raise ValueError("displacement threshold cannot be negative")
        if not 1 <= self.max_candidates <= 16:
            raise ValueError("max_candidates must be within [1, 16]")
