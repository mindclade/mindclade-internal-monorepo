"""Model-agnostic task contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from torch.nn import Module

from .loss import LossReport


@dataclass(frozen=True)
class BatchEnvelope[BatchT]:
    """Associates immutable sample identities with a model-native batch."""

    payload: BatchT
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.sample_ids:
            raise ValueError("batch envelope must contain at least one sample id")
        if any(not sample_id for sample_id in self.sample_ids):
            raise ValueError("sample ids must be non-empty")


@runtime_checkable
class TrainingTask[BatchT, OutputT](Protocol):
    """Defines the model invocation and objective for one kind of batch."""

    @property
    def name(self) -> str: ...

    def validate_batch(self, batch: BatchT) -> None: ...

    def forward(self, model: Module, batch: BatchT) -> OutputT: ...

    def compute_loss(self, output: OutputT, batch: BatchT) -> LossReport: ...


class UnsupportedTaskError(RuntimeError):
    """Raised for an intentionally unsupported task family."""
