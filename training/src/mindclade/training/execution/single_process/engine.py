"""Small eager reference engine used as the correctness oracle."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from mindclade.training.api.precision import PrecisionConfig
from mindclade.training.precision import ResolvedPrecision, resolve_precision
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer


def move_to_device(value: Any, device: torch.device) -> Any:
    """Recursively move tensors while preserving integral and boolean dtypes."""

    if isinstance(value, Tensor):
        return value.to(device=device)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: move_to_device(getattr(value, field.name), device)
            for field in dataclasses.fields(value)
        }
        return type(value)(**updates)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return type(value)(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


@dataclass
class SingleProcessEngine:
    device: torch.device
    precision: ResolvedPrecision
    scaler: Any | None

    @classmethod
    def create(
        cls,
        precision_config: PrecisionConfig,
        *,
        device: torch.device | None = None,
    ) -> SingleProcessEngine:
        selected = device if device is not None else torch.device("cpu")
        if selected.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"requested device {selected}, but CUDA is unavailable")
        precision = resolve_precision(precision_config, selected)
        return cls(device=selected, precision=precision, scaler=precision.create_scaler())

    def prepare_model(self, model: Module) -> Module:
        return model.to(device=self.device, dtype=self.precision.parameter_dtype)

    def prepare_batch(self, batch: Any) -> Any:
        return move_to_device(batch, self.device)

    def backward(self, loss: Tensor) -> None:
        if self.scaler is None:
            torch.autograd.backward(loss)
        else:
            self.scaler.scale(loss).backward()

    def optimizer_step(self, optimizer: Optimizer) -> None:
        if self.scaler is None:
            optimizer.step()
        else:
            self.scaler.step(optimizer)
            self.scaler.update()

    def unscale_(self, optimizer: Optimizer) -> None:
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
