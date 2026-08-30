"""Aggregate-only tensor health diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class NumericalSummary:
    name: str
    shape: tuple[int, ...]
    dtype: str
    device_type: str
    element_count: int
    finite_count: int
    nan_count: int
    infinity_count: int
    absolute_max: float | None
    mean: float | None

    @property
    def valid(self) -> bool:
        return self.finite_count == self.element_count


def summarize_tensor(name: str, tensor: torch.Tensor) -> NumericalSummary:
    if not name or len(name) > 64:
        raise ValueError("diagnostic name must contain 1..64 characters")
    values = tensor.detach().float()
    finite = torch.isfinite(values)
    finite_values = values[finite]
    return NumericalSummary(
        name=name,
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        device_type=tensor.device.type,
        element_count=tensor.numel(),
        finite_count=int(finite.sum()),
        nan_count=int(torch.isnan(values).sum()),
        infinity_count=int(torch.isinf(values).sum()),
        absolute_max=(float(finite_values.abs().max()) if finite_values.numel() else None),
        mean=(float(finite_values.mean()) if finite_values.numel() else None),
    )
