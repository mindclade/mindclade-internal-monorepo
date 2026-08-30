"""Bounded LRU cache of exact-key compiled callables."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .._identity import require_sha256_digest
from .compile_key import CompileKey


@dataclass(frozen=True, slots=True)
class CompiledVariant:
    key: CompileKey
    execute: Callable[..., Any]
    qualification_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "qualification_digest",
            require_sha256_digest(self.qualification_digest, field="qualification_digest"),
        )


class CompiledVariantCache:
    def __init__(self, capacity: int = 16) -> None:
        if capacity < 1:
            raise ValueError("cache capacity must be positive")
        self.capacity = capacity
        self._entries: OrderedDict[CompileKey, CompiledVariant] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: CompileKey) -> CompiledVariant | None:
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
            return value

    def put(self, variant: CompiledVariant) -> tuple[CompileKey, ...]:
        evicted: list[CompileKey] = []
        with self._lock:
            self._entries[variant.key] = variant
            self._entries.move_to_end(variant.key)
            while len(self._entries) > self.capacity:
                key, _ = self._entries.popitem(last=False)
                evicted.append(key)
        return tuple(evicted)

    def invalidate_model(self, model_digest: str) -> int:
        digest = require_sha256_digest(model_digest, field="model_digest")
        with self._lock:
            doomed = [key for key in self._entries if key.model_digest == digest]
            for key in doomed:
                del self._entries[key]
            return len(doomed)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
