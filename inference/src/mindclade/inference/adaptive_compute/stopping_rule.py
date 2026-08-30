"""Confidence-and-geometry adaptive stopping rule."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .compute_policy import ComputePolicy


@dataclass(frozen=True, slots=True)
class Observation:
    completed_steps: int
    calibrated_confidence: float
    rms_displacement_angstrom: float

    def __post_init__(self) -> None:
        if self.completed_steps < 1:
            raise ValueError("completed_steps must be positive")
        if (
            not math.isfinite(self.calibrated_confidence)
            or not 0 <= self.calibrated_confidence <= 1
        ):
            raise ValueError("calibrated_confidence must be finite and within [0, 1]")
        if not math.isfinite(self.rms_displacement_angstrom) or self.rms_displacement_angstrom < 0:
            raise ValueError("rms displacement must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class StopDecision:
    stop: bool
    reason: str
    consecutive_converged: int
    confidence_gain: float | None


class StoppingRule:
    def __init__(self, policy: ComputePolicy) -> None:
        self.policy = policy
        self._previous: Observation | None = None
        self._consecutive = 0

    def observe(self, observation: Observation) -> StopDecision:
        if observation.completed_steps > self.policy.max_steps:
            raise ValueError("observation exceeds max_steps")
        if self._previous and observation.completed_steps <= self._previous.completed_steps:
            raise ValueError("observations must have monotonically increasing step counts")
        if not self.policy.should_evaluate(observation.completed_steps):
            return StopDecision(False, "evaluation-not-due", self._consecutive, None)

        gain = None
        converged = False
        if self._previous is not None:
            gain = observation.calibrated_confidence - self._previous.calibrated_confidence
            converged = (
                gain < self.policy.confidence_gain_threshold
                and observation.rms_displacement_angstrom
                < self.policy.displacement_threshold_angstrom
            )
        self._consecutive = self._consecutive + 1 if converged else 0
        self._previous = observation
        if self._consecutive >= self.policy.patience:
            return StopDecision(True, "converged", self._consecutive, gain)
        if observation.completed_steps >= self.policy.max_steps:
            return StopDecision(True, "budget-exhausted", self._consecutive, gain)
        return StopDecision(False, "continue", self._consecutive, gain)
