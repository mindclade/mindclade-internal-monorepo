"""Digest-bound capability index."""

from __future__ import annotations

import re
from collections.abc import Mapping

from mindclade.models.api.capabilities import ModelCapabilities

_SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class CapabilityIndex:
    def __init__(self) -> None:
        self._entries: dict[str, ModelCapabilities] = {}

    def add(self, model_digest: str, capabilities: ModelCapabilities) -> None:
        if type(model_digest) is not str or _SHA256_DIGEST.fullmatch(model_digest) is None:
            raise ValueError("capabilities must bind an immutable sha256 model digest")
        existing = self._entries.get(model_digest)
        if existing is not None and existing != capabilities:
            raise ValueError("a model digest cannot be rebound to different capabilities")
        self._entries[model_digest] = capabilities

    def snapshot(self) -> Mapping[str, ModelCapabilities]:
        return dict(self._entries)


__all__ = ["CapabilityIndex"]
