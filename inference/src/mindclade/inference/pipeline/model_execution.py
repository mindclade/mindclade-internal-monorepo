"""Immutable model resolution and no-grad execution boundary."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .._identity import require_sha256_digest
from .preprocessing import PreparedInference


@dataclass(frozen=True, slots=True)
class ResolvedModel[TModel]:
    model_digest: str
    model: TModel
    batch_factory: Callable[..., Any]
    capabilities: frozenset[str] = frozenset({"fold"})

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_digest", require_sha256_digest(self.model_digest))


class ModelResolver[TModel]:
    """Process-local resolver that accepts immutable digests, never aliases."""

    def __init__(self) -> None:
        self._models: dict[str, ResolvedModel[TModel]] = {}
        self._lock = threading.RLock()

    def register(self, resolved: ResolvedModel[TModel]) -> None:
        with self._lock:
            previous = self._models.get(resolved.model_digest)
            if previous is not None and previous.model is not resolved.model:
                raise ValueError("a model digest cannot be rebound to a different model")
            self._models[resolved.model_digest] = resolved

    def resolve(self, model_digest: str) -> ResolvedModel[TModel]:
        digest = require_sha256_digest(model_digest, field="model_digest")
        with self._lock:
            try:
                return self._models[digest]
            except KeyError as exc:
                raise LookupError(f"model digest is not loaded: {digest}") from exc


class ModelExecutor[TModel]:
    def __init__(self, resolver: ModelResolver[TModel]) -> None:
        self.resolver = resolver

    def forward(self, prepared: PreparedInference) -> Any:
        prepared.verify_integrity()
        resolved = self.resolver.resolve(prepared.request.model_digest)
        batch = prepared.as_model_batch(resolved.batch_factory, validation_mode="forward")
        model = resolved.model
        if not callable(model):
            raise TypeError("resolved model is not callable")
        with torch.no_grad():
            return model(batch)

    def fold(
        self,
        prepared: PreparedInference,
        *,
        seed: int | None = None,
        num_samples: int | None = None,
        num_steps: int | None = None,
        return_trajectory: bool = False,
    ) -> Any:
        prepared.verify_integrity()
        resolved = self.resolver.resolve(prepared.request.model_digest)
        if "fold" not in resolved.capabilities:
            raise ValueError("resolved model is not qualified for fold inference")
        batch = prepared.as_model_batch(resolved.batch_factory, validation_mode="static")
        fold = getattr(resolved.model, "fold", None)
        if not callable(fold):
            raise TypeError("resolved model does not implement fold(...)")
        with torch.no_grad():
            return fold(
                batch,
                seed=prepared.request.seed if seed is None else seed,
                num_samples=prepared.request.num_samples if num_samples is None else num_samples,
                num_steps=prepared.request.num_steps if num_steps is None else num_steps,
                return_trajectory=return_trajectory,
            )
