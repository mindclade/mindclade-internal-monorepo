"""Digest-bound model conversion receipt."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any


@dataclasses.dataclass(frozen=True)
class ConversionReceipt:
    schema_version: int
    source_digest: str
    destination_digest: str
    converter: str
    key_mapping_digest: str

    def __post_init__(self) -> None:
        for name in ("source_digest", "destination_digest", "key_mapping_digest"):
            if not getattr(self, name).startswith("sha256:"):
                raise ValueError(f"{name} must be sha256-addressed")

    def to_dict(self) -> Mapping[str, Any]:
        return dataclasses.asdict(self)


__all__ = ["ConversionReceipt"]
