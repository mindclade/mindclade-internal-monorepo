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
        if type(self.completed_steps) is not int or self.completed_steps < 1:
            raise ValueError("completed_steps must be a positive integer")
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


@dataclass(frozen=True, slots=True)
class StoppingState:
    """Serializable convergence state required for exact adaptive resume."""

    previous_observation: Observation | None = None
    consecutive_converged: int = 0

    def __post_init__(self) -> None:
        if type(self.consecutive_converged) is not int or self.consecutive_converged < 0:
            raise ValueError("consecutive_converged must be a non-negative integer")
        if self.previous_observation is None and self.consecutive_converged != 0:
            raise ValueError("convergence cannot precede the first observation")


class StoppingRule:
    def __init__(self, policy: ComputePolicy, *, state: StoppingState | None = None) -> None:
        self.policy = policy
        restored = state or StoppingState()
        if restored.previous_observation is not None:
            if restored.previous_observation.completed_steps > policy.max_steps:
                raise ValueError("stopping state observation exceeds max_steps")
            if not policy.should_evaluate(restored.previous_observation.completed_steps):
                raise ValueError("stopping state observation was not due under the policy")
            first_evaluation = (
                (policy.min_steps + policy.evaluation_interval - 1) // policy.evaluation_interval
            ) * policy.evaluation_interval
            evaluation_count = (
                (restored.previous_observation.completed_steps - first_evaluation)
                // policy.evaluation_interval
            ) + 1
            if restored.consecutive_converged > max(0, evaluation_count - 1):
                raise ValueError("stopping state convergence streak is not reachable")
            if restored.consecutive_converged >= policy.patience:
                raise ValueError("terminal stopping state cannot be resumed")
        self._previous = restored.previous_observation
        self._consecutive = restored.consecutive_converged

    @property
    def state(self) -> StoppingState:
        return StoppingState(
            previous_observation=self._previous,
            consecutive_converged=self._consecutive,
        )

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
