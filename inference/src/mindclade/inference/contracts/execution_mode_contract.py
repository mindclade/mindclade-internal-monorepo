"""Execution-mode request and selection values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ExecutionMode(StrEnum):
    AUTO = "auto"
    EAGER = "eager"
    COMPILED = "compiled"


@dataclass(frozen=True, slots=True)
class ExecutionModeRequest:
    requested: ExecutionMode = ExecutionMode.AUTO
    allow_fallback: bool = True
    require_qualified: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.requested, ExecutionMode):
            object.__setattr__(self, "requested", ExecutionMode(self.requested))
        if self.requested is ExecutionMode.AUTO and not self.allow_fallback:
            raise ValueError("AUTO mode requires fallback to be allowed")
