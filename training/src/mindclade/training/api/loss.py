"""Differentiable loss contracts used by training tasks and engines."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor


@dataclass(frozen=True)
class LossTerm:
    """One named scalar loss and its composition weight.

    ``value`` remains attached to autograd. Callers should only use
    :meth:`LossReport.detached_metrics` when emitting telemetry.
    """

    name: str
    value: Tensor
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("loss term name must be non-empty")
        if self.value.numel() != 1:
            raise ValueError(
                f"loss term {self.name!r} must be scalar; got shape {tuple(self.value.shape)}"
            )
        if not math.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError(f"loss term {self.name!r} weight must be finite and non-negative")

    @property
    def weighted(self) -> Tensor:
        return self.value.to(dtype=torch.float32) * self.weight


@dataclass(frozen=True)
class LossReport:
    """Stable loss output consumed by every execution engine.

    The reference trainer interprets these scalar values as per-sample means
    when it combines differently sized microbatches.
    """

    total: Tensor
    terms: Mapping[str, LossTerm]

    def __post_init__(self) -> None:
        if self.total.numel() != 1:
            raise ValueError(f"total loss must be scalar; got shape {tuple(self.total.shape)}")
        normalized: dict[str, LossTerm] = dict(self.terms)
        if not normalized:
            raise ValueError("a loss report must contain at least one term")
        if set(normalized) != {term.name for term in normalized.values()}:
            raise ValueError("loss report keys must exactly match LossTerm.name values")
        object.__setattr__(self, "terms", MappingProxyType(normalized))

    @classmethod
    def compose(cls, terms: Iterable[LossTerm]) -> LossReport:
        by_name: dict[str, LossTerm] = {}
        total: Tensor | None = None
        for term in terms:
            if term.name in by_name:
                raise ValueError(f"duplicate loss term {term.name!r}")
            by_name[term.name] = term
            weighted = term.weighted
            total = weighted if total is None else total + weighted
        if total is None:
            raise ValueError("cannot compose an empty loss collection")
        return cls(total=total.to(dtype=torch.float32), terms=by_name)

    def detached_metrics(self) -> Mapping[str, float]:
        metrics = {"loss": float(self.total.detach().to(dtype=torch.float32).cpu())}
        metrics.update(
            {
                f"loss/{name}": float(term.value.detach().to(dtype=torch.float32).cpu())
                for name, term in self.terms.items()
            }
        )
        return MappingProxyType(metrics)


class NonFiniteLossError(FloatingPointError):
    """Raised before backward when a task returns NaN or infinite loss."""


def require_finite_loss(report: LossReport) -> None:
    if not bool(torch.isfinite(report.total.detach()).all()):
        values = ", ".join(
            f"{name}={float(term.value.detach().to(dtype=torch.float32).cpu())}"
            for name, term in report.terms.items()
        )
        raise NonFiniteLossError(f"non-finite total loss; components: {values}")
