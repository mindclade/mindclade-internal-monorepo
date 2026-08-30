"""Canonical identity helpers shared by inference subsystems."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch


class CanonicalizationError(ValueError):
    """Raised when a value cannot safely participate in an immutable identity."""


def require_sha256_digest(value: str, *, field: str = "digest") -> str:
    """Validate and normalize a ``sha256:<hex>`` content digest."""

    normalized = value.lower()
    if not normalized.startswith("sha256:") or len(normalized) != 71:
        raise ValueError(f"{field} must be a sha256:<64 lowercase hex> digest")
    try:
        int(normalized[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{field} contains non-hexadecimal characters") from exc
    return normalized


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError("non-finite floats cannot be canonicalized")
        return value
    if isinstance(value, enum.Enum):
        return _canonical(value.value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonical(dataclasses.asdict(cast(Any, value)))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise CanonicalizationError("canonical mapping keys must be strings")
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical(item) for item in value]
    raise CanonicalizationError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable UTF-8 JSON suitable for hashing and receipts."""

    return json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def content_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def tensor_digest(tensor: torch.Tensor) -> str:
    """Hash tensor metadata and exact contiguous CPU bytes without pickle."""

    if tensor.layout is not torch.strided:
        raise CanonicalizationError("only strided tensors have a supported tensor identity")
    detached = tensor.detach().to(device="cpu").contiguous().clone()
    metadata = canonical_json_bytes({"dtype": str(detached.dtype), "shape": list(detached.shape)})
    payload = bytes(detached.untyped_storage())
    hasher = hashlib.sha256()
    hasher.update(metadata)
    hasher.update(b"\0")
    hasher.update(payload)
    return "sha256:" + hasher.hexdigest()
