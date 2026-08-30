"""Default deterministic sampling entrypoint."""

from __future__ import annotations

from mindclade.models.api.batch import CladeFoldBatch
from mindclade.models.families.clade.cladefold.architecture.cladefold import (
    CladeFoldFoldOutput,
    CladeFoldModel,
)


def fold(
    model: CladeFoldModel,
    batch: CladeFoldBatch,
    *,
    seed: int,
    num_samples: int = 1,
    num_steps: int | None = None,
    return_trajectory: bool = False,
) -> CladeFoldFoldOutput:
    return model.fold(
        batch,
        seed=seed,
        num_samples=num_samples,
        num_steps=num_steps,
        return_trajectory=return_trajectory,
    )


__all__ = ["fold"]
