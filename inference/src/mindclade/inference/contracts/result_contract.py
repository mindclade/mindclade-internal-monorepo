"""Immutable inference result and candidate contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch

from .._identity import require_sha256_digest


@dataclass(frozen=True, slots=True)
class InferenceCandidate:
    candidate_id: str
    coordinates: torch.Tensor
    confidence: float
    calibrated_confidence: float
    batch_seeds: tuple[int, ...]
    steps: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id cannot be empty")
        if self.coordinates.ndim != 3 or self.coordinates.shape[-1] != 3:
            raise ValueError("coordinates must have shape [B, A, 3]")
        if not torch.isfinite(self.coordinates).all():
            raise ValueError("coordinates must be finite")
        if (
            type(self.batch_seeds) is not tuple
            or len(self.batch_seeds) != self.coordinates.shape[0]
            or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in self.batch_seeds)
        ):
            raise ValueError(
                "batch_seeds must contain one signed-int64-compatible seed per batch row"
            )
        for name in ("confidence", "calibrated_confidence"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.steps < 1:
            raise ValueError("steps must be positive")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def seed(self) -> int:
        """Return the scalar seed for a single-batch candidate."""

        if len(self.batch_seeds) != 1:
            raise ValueError(
                "seed is only scalar for batch size 1; use batch_seeds for batched results"
            )
        return self.batch_seeds[0]


@dataclass(frozen=True, slots=True)
class InferenceResult:
    request_id: str
    request_fingerprint: str
    model_digest: str
    serving_revision_digest: str
    candidates: tuple[InferenceCandidate, ...]
    selected_candidate_id: str
    execution_mode: str
    sampler_digest: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "v1alpha1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "model_digest", require_sha256_digest(self.model_digest, field="model_digest")
        )
        object.__setattr__(
            self,
            "serving_revision_digest",
            require_sha256_digest(self.serving_revision_digest, field="serving_revision_digest"),
        )
        object.__setattr__(
            self,
            "request_fingerprint",
            require_sha256_digest(self.request_fingerprint, field="request_fingerprint"),
        )
        object.__setattr__(
            self,
            "sampler_digest",
            require_sha256_digest(self.sampler_digest, field="sampler_digest"),
        )
        if not self.candidates:
            raise ValueError("at least one candidate is required")
        if len({len(candidate.batch_seeds) for candidate in self.candidates}) != 1:
            raise ValueError("all candidates must preserve the same batch seed dimension")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate identifiers must be unique")
        if self.selected_candidate_id not in ids:
            raise ValueError("selected_candidate_id does not identify a candidate")
        if self.schema_version != "v1alpha1":
            raise ValueError("only v1alpha1 results are supported")
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @property
    def selected(self) -> InferenceCandidate:
        return next(c for c in self.candidates if c.candidate_id == self.selected_candidate_id)

    @property
    def sample_seeds(self) -> tuple[tuple[int, ...], ...]:
        """Return canonical seed provenance in ``[B, S]`` tuple layout."""

        batch_size = len(self.candidates[0].batch_seeds)
        return tuple(
            tuple(candidate.batch_seeds[batch_index] for candidate in self.candidates)
            for batch_index in range(batch_size)
        )
