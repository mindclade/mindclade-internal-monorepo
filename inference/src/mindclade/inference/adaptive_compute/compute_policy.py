"""Resolved and digestible adaptive-compute policy."""

from __future__ import annotations

from dataclasses import dataclass

from .._identity import content_digest
from ..contracts.adaptive_compute_contract import AdaptiveComputeRequest


@dataclass(frozen=True, slots=True)
class ComputePolicy:
    min_steps: int = 16
    max_steps: int = 32
    evaluation_interval: int = 4
    patience: int = 3
    confidence_gain_threshold: float = 0.002
    displacement_threshold_angstrom: float = 0.05
    max_candidates: int = 1
    policy_version: str = "adaptive-v1alpha1"

    def __post_init__(self) -> None:
        # Reuse the public contract as the single validation source.
        AdaptiveComputeRequest(
            enabled=True,
            min_steps=self.min_steps,
            max_steps=self.max_steps,
            evaluation_interval=self.evaluation_interval,
            patience=self.patience,
            confidence_gain_threshold=self.confidence_gain_threshold,
            displacement_threshold_angstrom=self.displacement_threshold_angstrom,
            max_candidates=self.max_candidates,
        )
        if self.policy_version != "adaptive-v1alpha1":
            raise ValueError("unsupported adaptive policy version")

    @classmethod
    def from_request(cls, request: AdaptiveComputeRequest) -> ComputePolicy:
        return cls(
            min_steps=request.min_steps,
            max_steps=request.max_steps,
            evaluation_interval=request.evaluation_interval,
            patience=request.patience,
            confidence_gain_threshold=request.confidence_gain_threshold,
            displacement_threshold_angstrom=request.displacement_threshold_angstrom,
            max_candidates=request.max_candidates,
        )

    @property
    def digest(self) -> str:
        return content_digest(self)

    def should_evaluate(self, completed_steps: int) -> bool:
        return completed_steps >= self.min_steps and completed_steps % self.evaluation_interval == 0
