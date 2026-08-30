"""Versioned serving capability values."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any


@dataclasses.dataclass(frozen=True)
class ModelCapabilities:
    schema_version: str
    model_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    supports_training: bool
    supports_sampling: bool
    claim_level: str = "systems-reference-only"

    def to_dict(self) -> Mapping[str, Any]:
        return dataclasses.asdict(self)


__all__ = ["ModelCapabilities"]
