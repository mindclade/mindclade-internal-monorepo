"""Strict v1alpha1 recipe schema represented as typed dataclasses."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from mindclade.training.api import (
    OptimizerConfig,
    ParallelismConfig,
    ParallelismMode,
    PrecisionConfig,
    PrecisionMode,
    ReproducibilityConfig,
    SchedulerConfig,
    TrainingProgram,
)
from mindclade.training.core.data import SyntheticSplit


@dataclass(frozen=True)
class ModelRecipe:
    family: str = "cladefold_q0"
    preset: str = "tiny"

    def __post_init__(self) -> None:
        if self.family != "cladefold_q0" or self.preset != "tiny":
            raise ValueError(
                "reference training supports only the random-initialized cladefold_q0 tiny preset"
            )


@dataclass(frozen=True)
class DatasetRecipe:
    name: str = "synthetic_cladefold_v1"
    split: SyntheticSplit = SyntheticSplit.TRAIN
    batch_size: int = 1
    token_count: int = 8
    atom_count: int = 16
    shuffle: bool = False

    def __post_init__(self) -> None:
        if self.name != "synthetic_cladefold_v1":
            raise ValueError("reference training supports only synthetic_cladefold_v1")
        if self.batch_size <= 0:
            raise ValueError("dataset batch_size must be positive")
        if self.token_count <= 0 or self.atom_count < self.token_count:
            raise ValueError("dataset requires atom_count >= token_count > 0")


@dataclass(frozen=True)
class ResolvedRecipe:
    schema_version: str
    model: ModelRecipe
    dataset: DatasetRecipe
    program: TrainingProgram

    def __post_init__(self) -> None:
        if self.schema_version != "v1alpha1":
            raise ValueError(f"unsupported recipe schema {self.schema_version!r}")

    def to_dict(self) -> dict[str, Any]:
        dataset = asdict(self.dataset)
        dataset["split"] = self.dataset.split.value
        return {
            "schema_version": self.schema_version,
            "model": asdict(self.model),
            "dataset": dataset,
            "program": self.program.to_dict(),
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _require_keys(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    location: str,
) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown or missing:
        raise ValueError(
            f"invalid {location} fields; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def recipe_from_mapping(value: Mapping[str, Any]) -> ResolvedRecipe:
    _require_keys(
        value,
        allowed={"schema_version", "model", "dataset", "program"},
        required={"schema_version", "model", "dataset", "program"},
        location="recipe",
    )
    model_value = dict(value["model"])
    _require_keys(
        model_value,
        allowed={"family", "preset"},
        required={"family", "preset"},
        location="model",
    )
    dataset_value = dict(value["dataset"])
    _require_keys(
        dataset_value,
        allowed={"name", "split", "batch_size", "token_count", "atom_count", "shuffle"},
        required={"name", "split", "batch_size", "token_count", "atom_count"},
        location="dataset",
    )
    dataset_value["split"] = SyntheticSplit(str(dataset_value["split"]))

    program_value = dict(value["program"])
    _require_keys(
        program_value,
        allowed={
            "name",
            "max_steps",
            "gradient_accumulation_steps",
            "optimizer",
            "scheduler",
            "parallelism",
            "precision",
            "reproducibility",
            "checkpoint_every_steps",
        },
        required={"name", "max_steps"},
        location="program",
    )
    optimizer = OptimizerConfig(**dict(program_value.pop("optimizer", {})))
    scheduler = SchedulerConfig(**dict(program_value.pop("scheduler", {})))
    parallelism_value = dict(program_value.pop("parallelism", {}))
    if "mode" in parallelism_value:
        parallelism_value["mode"] = ParallelismMode(str(parallelism_value["mode"]))
    parallelism = ParallelismConfig(**parallelism_value)
    precision_value = dict(program_value.pop("precision", {}))
    if "mode" in precision_value:
        precision_value["mode"] = PrecisionMode(str(precision_value["mode"]))
    if "reduction_dtype" in precision_value:
        precision_value["reduction_dtype"] = PrecisionMode(str(precision_value["reduction_dtype"]))
    precision = PrecisionConfig(**precision_value)
    reproducibility = ReproducibilityConfig(**dict(program_value.pop("reproducibility", {})))
    program = TrainingProgram(
        optimizer=optimizer,
        scheduler=scheduler,
        parallelism=parallelism,
        precision=precision,
        reproducibility=reproducibility,
        **program_value,
    )
    return ResolvedRecipe(
        schema_version=str(value["schema_version"]),
        model=ModelRecipe(**model_value),
        dataset=DatasetRecipe(**dataset_value),
        program=program,
    )
