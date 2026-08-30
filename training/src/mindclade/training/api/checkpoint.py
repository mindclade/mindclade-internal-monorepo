"""Public checkpoint policy and immutable references."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckpointPolicy:
    root: Path
    every_steps: int = 100
    keep_last: int | None = None

    def __post_init__(self) -> None:
        if self.every_steps <= 0:
            raise ValueError("checkpoint every_steps must be positive")
        if self.keep_last is not None and self.keep_last <= 0:
            raise ValueError("checkpoint keep_last must be positive when set")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["root"] = str(self.root)
        return value


@dataclass(frozen=True)
class CheckpointRef:
    checkpoint_id: str
    path: Path
    global_step: int
    manifest_sha256: str


class CheckpointError(RuntimeError):
    """Base class for integrity, compatibility, and persistence failures."""


class IncompleteCheckpointError(CheckpointError):
    """Raised when no commit manifest exists."""


class CheckpointIntegrityError(CheckpointError):
    """Raised when a committed file does not match the manifest."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised before restore when logical schemas are incompatible."""
