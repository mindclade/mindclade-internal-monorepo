"""Hard inference admission and queue limits."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import torch


class BatchProfile(StrEnum):
    SYNC = "sync"
    ASYNC = "async"


@dataclass(frozen=True, slots=True)
class BatchLimits:
    max_batch_size: int
    max_tokens: int
    max_atoms: int
    max_bonds: int
    max_samples: int
    max_steps: int
    max_tensor_bytes: int
    max_queue_depth: int = 256
    max_queue_delay_ms: int = 10

    def __post_init__(self) -> None:
        if any(
            value < 1
            for value in (
                self.max_batch_size,
                self.max_tokens,
                self.max_atoms,
                self.max_bonds,
                self.max_samples,
                self.max_steps,
                self.max_tensor_bytes,
                self.max_queue_depth,
            )
        ):
            raise ValueError("all batch limits except delay must be positive")
        if self.max_queue_delay_ms < 0:
            raise ValueError("max_queue_delay_ms cannot be negative")

    @classmethod
    def sync(cls) -> BatchLimits:
        return cls(
            max_batch_size=1,
            max_tokens=256,
            max_atoms=2048,
            max_bonds=4096,
            max_samples=1,
            max_steps=32,
            max_tensor_bytes=8 * 1024 * 1024,
        )

    @classmethod
    def async_(cls) -> BatchLimits:
        return cls(
            max_batch_size=8,
            max_tokens=2048,
            max_atoms=16384,
            max_bonds=32768,
            max_samples=16,
            max_steps=128,
            max_tensor_bytes=512 * 1024 * 1024,
            max_queue_depth=1024,
            max_queue_delay_ms=50,
        )

    def validate(
        self,
        *,
        batch_size: int,
        tokens: int,
        atoms: int,
        bonds: int,
        samples: int,
        steps: int,
        tensors: Mapping[str, torch.Tensor],
    ) -> int:
        dimensions = {
            "batch_size": (batch_size, self.max_batch_size),
            "tokens": (tokens, self.max_tokens),
            "atoms": (atoms, self.max_atoms),
            "bonds": (bonds, self.max_bonds),
            "samples": (samples, self.max_samples),
            "steps": (steps, self.max_steps),
        }
        for name, (actual, limit) in dimensions.items():
            if actual < 0 or actual > limit:
                raise ValueError(f"{name}={actual} exceeds limit {limit}")
        tensor_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
        if tensor_bytes > self.max_tensor_bytes:
            raise ValueError(f"tensor_bytes={tensor_bytes} exceeds limit {self.max_tensor_bytes}")
        return tensor_bytes
