"""Temperature/bias calibration bound to immutable evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .._identity import content_digest, require_sha256_digest


@dataclass(frozen=True, slots=True)
class CalibrationParameters:
    temperature: float = 1.0
    bias: float = 0.0
    evidence_digest: str | None = None
    calibration_version: str = "confidence-calibration.v1alpha1"

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("calibration temperature must be finite and positive")
        if not math.isfinite(self.bias):
            raise ValueError("calibration bias must be finite")
        if self.evidence_digest is not None:
            object.__setattr__(
                self,
                "evidence_digest",
                require_sha256_digest(self.evidence_digest, field="evidence_digest"),
            )

    @property
    def digest(self) -> str:
        return content_digest(self)


class ConfidenceCalibrator:
    def __init__(self, parameters: CalibrationParameters) -> None:
        self.parameters = parameters

    @classmethod
    def identity(cls) -> ConfidenceCalibrator:
        return cls(CalibrationParameters())

    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.float().div(self.parameters.temperature).add(self.parameters.bias)

    def calibrate_scalar(self, probability: float) -> float:
        if not 0 <= probability <= 1 or not math.isfinite(probability):
            raise ValueError("probability must be finite and within [0, 1]")
        if self.parameters.temperature == 1.0 and self.parameters.bias == 0.0:
            return probability
        epsilon = 1e-7
        clipped = min(max(probability, epsilon), 1.0 - epsilon)
        logit = math.log(clipped / (1.0 - clipped))
        calibrated = 1.0 / (
            1.0 + math.exp(-(logit / self.parameters.temperature + self.parameters.bias))
        )
        return min(max(calibrated, 0.0), 1.0)
