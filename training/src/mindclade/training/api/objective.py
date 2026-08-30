"""Objective interfaces and output-field objective implementation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from torch import Tensor

from .loss import LossTerm


@runtime_checkable
class Objective[BatchT, OutputT](Protocol):
    """Maps a model output and its batch to one named scalar loss."""

    @property
    def name(self) -> str: ...

    def compute(self, output: OutputT, batch: BatchT) -> LossTerm: ...


def read_output_field(output: Any, field: str) -> Any:
    if isinstance(output, Mapping):
        if field not in output:
            raise KeyError(f"model output has no {field!r} field")
        return output[field]
    try:
        return getattr(output, field)
    except AttributeError as exc:
        raise AttributeError(f"model output has no {field!r} attribute") from exc


@dataclass(frozen=True)
class OutputFieldObjective[BatchT, OutputT]:
    """Reads a scalar tensor from a mapping-like or attribute-based output."""

    name: str
    field: str
    weight: float = 1.0

    def compute(self, output: OutputT, batch: BatchT) -> LossTerm:
        del batch
        value = read_output_field(output, self.field)
        if not isinstance(value, Tensor):
            raise TypeError(f"model output field {self.field!r} must be a torch.Tensor")
        return LossTerm(name=self.name, value=value, weight=self.weight)
