"""Content-addressed evidence for an emitted sampling candidate."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .._identity import content_digest, require_sha256_digest, tensor_digest


@dataclass(frozen=True, slots=True)
class CandidateReceipt:
    candidate_id: str
    request_fingerprint: str
    model_digest: str
    sampler_digest: str
    coordinates_digest: str
    seed: int
    completed_steps: int
    raw_confidence: float
    calibrated_confidence: float
    stop_reason: str
    schema_version: str = "candidate-receipt.v1alpha1"

    def __post_init__(self) -> None:
        for name in (
            "request_fingerprint",
            "model_digest",
            "sampler_digest",
            "coordinates_digest",
        ):
            object.__setattr__(self, name, require_sha256_digest(getattr(self, name), field=name))
        if not self.candidate_id:
            raise ValueError("candidate_id cannot be empty")
        if self.completed_steps < 1:
            raise ValueError("completed_steps must be positive")
        for name in ("raw_confidence", "calibrated_confidence"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be within [0, 1]")

    @classmethod
    def create(
        cls,
        *,
        candidate_id: str,
        request_fingerprint: str,
        model_digest: str,
        sampler_digest: str,
        coordinates: torch.Tensor,
        seed: int,
        completed_steps: int,
        raw_confidence: float,
        calibrated_confidence: float,
        stop_reason: str,
    ) -> CandidateReceipt:
        return cls(
            candidate_id=candidate_id,
            request_fingerprint=request_fingerprint,
            model_digest=model_digest,
            sampler_digest=sampler_digest,
            coordinates_digest=tensor_digest(coordinates),
            seed=seed,
            completed_steps=completed_steps,
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            stop_reason=stop_reason,
        )

    @property
    def digest(self) -> str:
        return content_digest(self)
