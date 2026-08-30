"""Local, side-effect-free CladeFold inference composition."""

from __future__ import annotations

from mindclade.models.api.batch import CladeFoldBatch
from mindclade.models.families.clade.cladefold.architecture.cladefold import (
    CladeFoldFoldOutput,
    CladeFoldModel,
)


class CladeFoldInferencePipeline:
    def __init__(self, model: CladeFoldModel) -> None:
        if model.training:
            model.eval()
        self.model = model

    def __call__(
        self, batch: CladeFoldBatch, *, seed: int, samples: int = 1, steps: int | None = None
    ) -> CladeFoldFoldOutput:
        return self.model.fold(batch, seed=seed, num_samples=samples, num_steps=steps)


__all__ = ["CladeFoldInferencePipeline"]
