"""Bounded execution trace that never records raw request tensors or identifiers."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

_SENSITIVE_FRAGMENTS = (
    "tenant",
    "project",
    "request",
    "token",
    "sequence",
    "coordinate",
    "credential",
    "authorization",
    "secret",
)


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": value.device.type,
            "numel": value.numel(),
        }
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:128]
    if isinstance(value, (tuple, list, set, frozenset)):
        return {"kind": "collection", "length": len(value)}
    return {"kind": type(value).__name__}


@dataclass(frozen=True, slots=True)
class TraceEvent:
    name: str
    elapsed_ms: float
    attributes: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))


class ExecutionTrace:
    def __init__(
        self,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        max_events: int = 256,
    ) -> None:
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self._clock_ns = clock_ns
        self._started = clock_ns()
        self._max_events = max_events
        self._events: list[TraceEvent] = []

    @staticmethod
    def pseudonym(identifier: str) -> str:
        return "id:" + hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:16]

    def record(self, name: str, **attributes: Any) -> TraceEvent:
        if len(self._events) >= self._max_events:
            raise ValueError("execution trace event budget exhausted")
        if not name or len(name) > 64:
            raise ValueError("trace event name must contain 1..64 characters")
        sanitized = {}
        for key, value in attributes.items():
            lowered = key.lower()
            if any(fragment in lowered for fragment in _SENSITIVE_FRAGMENTS):
                sanitized[key] = "<redacted>"
            else:
                sanitized[key] = _sanitize_value(value)
        event = TraceEvent(
            name=name,
            elapsed_ms=(self._clock_ns() - self._started) / 1_000_000.0,
            attributes=sanitized,
        )
        self._events.append(event)
        return event

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)
