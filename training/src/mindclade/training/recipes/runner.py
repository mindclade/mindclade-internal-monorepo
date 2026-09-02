"""Runnable local reference recipe and qualification helpers."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mindclade.training.api import ParallelismMode, seed_everything
from mindclade.training.checkpointing import DCPCheckpointManager
from mindclade.training.core import Trainer, TrainingResult, build_synthetic_loader
from mindclade.training.tasks import MultitaskDiffusionTask

from .schema import ResolvedRecipe


@dataclass(frozen=True)
class OverfitQualification:
    passed: bool
    initial_mean: float
    final_mean: float
    required_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "initial_mean": self.initial_mean,
            "final_mean": self.final_mean,
            "required_ratio": self.required_ratio,
        }


def qualify_overfit(
    losses: Sequence[float], *, window: int = 10, required_ratio: float = 0.90
) -> OverfitQualification:
    if window <= 0 or len(losses) < 2 * window:
        raise ValueError("qualification requires at least two complete positive windows")
    if not 0.0 < required_ratio < 1.0:
        raise ValueError("required_ratio must be in (0, 1)")
    initial = sum(float(value) for value in losses[:window]) / window
    final = sum(float(value) for value in losses[-window:]) / window
    return OverfitQualification(
        passed=final <= required_ratio * initial,
        initial_mean=initial,
        final_mean=final,
        required_ratio=required_ratio,
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def run_reference_recipe(
    recipe: ResolvedRecipe,
    *,
    output: Path,
    resume_from: Path | None = None,
    allow_reshard: bool = False,
) -> TrainingResult:
    if recipe.program.parallelism.mode is not ParallelismMode.SINGLE_PROCESS:
        raise RuntimeError(
            "the local reference runner supports single_process only; launch FSDP2 through "
            "torchrun and apply the provider adapter explicitly"
        )
    seed_everything(recipe.program.reproducibility)
    try:
        from mindclade.models import CladeFoldConfig, CladeFoldModel
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("reference recipes require the mindclade-models distribution") from exc
    model = CladeFoldModel(CladeFoldConfig.tiny())
    loader = build_synthetic_loader(
        recipe.dataset.split,
        batch_size=recipe.dataset.batch_size,
        seed=recipe.program.reproducibility.seed,
        token_count=recipe.dataset.token_count,
        atom_count=recipe.dataset.atom_count,
        sigma_min=model.config.sigma_min,
        sigma_max=model.config.sigma_max,
        shuffle=recipe.dataset.shuffle,
    )
    checkpoint_manager = DCPCheckpointManager(output / "checkpoints")
    trainer = Trainer(
        model,
        MultitaskDiffusionTask(),
        recipe.program,
        run_id=recipe.program.name,
        checkpoint_manager=checkpoint_manager,
        checkpoint_identity=recipe.to_dict(),
    )
    if resume_from is not None:
        trainer.resume(resume_from, allow_reshard=allow_reshard)
    result = trainer.run(loader)
    _write_json_atomic(
        output / "training-history.json",
        {
            "recipe_sha256": recipe.sha256,
            "state": result.state.to_dict(),
            "steps": [
                {
                    "global_step": record.global_step,
                    "metrics": dict(record.metrics),
                    "sample_ids": list(record.sample_ids),
                }
                for record in result.history
            ],
        },
    )
    return result
