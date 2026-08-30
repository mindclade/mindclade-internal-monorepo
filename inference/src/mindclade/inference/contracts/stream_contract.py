"""Monotonic inference stream events."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class StreamEventKind(StrEnum):
    ACCEPTED = "accepted"
    PROGRESS = "progress"
    CANDIDATE = "candidate"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL = {
    StreamEventKind.COMPLETED,
    StreamEventKind.FAILED,
    StreamEventKind.CANCELLED,
}


@dataclass(frozen=True, slots=True)
class InferenceStreamEvent:
    request_id: str
    sequence: int
    kind: StreamEventKind
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("stream sequence cannot be negative")
        if not isinstance(self.kind, StreamEventKind):
            object.__setattr__(self, "kind", StreamEventKind(self.kind))
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    @property
    def terminal(self) -> bool:
        return self.kind in _TERMINAL


class StreamSequence:
    """Validate ordered events and prohibit events after a terminal state."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._next = 0
        self._terminal = False

    def accept(self, event: InferenceStreamEvent) -> None:
        if self._terminal:
            raise ValueError("cannot append after a terminal stream event")
        if event.request_id != self.request_id:
            raise ValueError("stream event request_id mismatch")
        if event.sequence != self._next:
            raise ValueError(f"expected stream sequence {self._next}, got {event.sequence}")
        self._next += 1
        self._terminal = event.terminal
