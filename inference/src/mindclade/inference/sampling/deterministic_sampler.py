"""Seed-stable adapter for models exposing the CladeFold ``fold`` contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mindclade.models.api.sampling import derive_sample_seed as derive_sample_seed

from .._identity import content_digest
from ..pipeline.model_execution import ModelExecutor
from ..pipeline.preprocessing import PreparedInference


@dataclass(frozen=True, slots=True)
class DeterministicModelSampler:
    executor: ModelExecutor[Any]
    sampler_version: str = "model-fold-v1alpha1"

    @property
    def digest(self) -> str:
        return content_digest({"sampler_version": self.sampler_version})

    def sample(self, prepared: PreparedInference, *, return_trajectory: bool = False) -> Any:
        """Run a model fold with an explicit seed and no global RNG mutation."""

        return self.executor.fold(
            prepared,
            seed=prepared.request.seed,
            num_samples=prepared.request.num_samples,
            num_steps=prepared.request.num_steps,
            return_trajectory=return_trajectory,
        )
