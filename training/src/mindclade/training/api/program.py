"""Resolved, serializable training program contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .optimization import OptimizerConfig, SchedulerConfig
from .parallelism import ParallelismConfig
from .precision import PrecisionConfig
from .reproducibility import ReproducibilityConfig


@dataclass(frozen=True)
class TrainingProgram:
    name: str
    max_steps: int
    gradient_accumulation_steps: int = 1
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
    checkpoint_every_steps: int = 100

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("training program name must be non-empty")
        for name in (
            "max_steps",
            "gradient_accumulation_steps",
            "checkpoint_every_steps",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_steps": self.max_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "optimizer": self.optimizer.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "parallelism": self.parallelism.to_dict(),
            "precision": self.precision.to_dict(),
            "reproducibility": self.reproducibility.to_dict(),
            "checkpoint_every_steps": self.checkpoint_every_steps,
        }
