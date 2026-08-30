"""Configuration validation primitives."""

from __future__ import annotations

import math
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from typing import Any


class ConfigurationError(ValueError):
    """Raised when a model configuration violates its public contract."""


def reject_unknown_fields(value: Mapping[str, Any], allowed: AbstractSet[str]) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ConfigurationError(f"unknown configuration fields: {', '.join(unknown)}")


def _require_finite_real(name: str, value: float) -> None:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ConfigurationError(f"{name} must be a finite real number, got {value!r}")


def require_positive(name: str, value: float) -> None:
    _require_finite_real(name, value)
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive, got {value}")


def require_positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer, got {value!r}")


def require_nonnegative(name: str, value: float) -> None:
    _require_finite_real(name, value)
    if value < 0:
        raise ConfigurationError(f"{name} cannot be negative, got {value}")


def require_probability(name: str, value: float, *, allow_one: bool = False) -> None:
    _require_finite_real(name, value)
    upper_ok = value <= 1.0 if allow_one else value < 1.0
    if value < 0.0 or not upper_ok:
        bound = "[0, 1]" if allow_one else "[0, 1)"
        raise ConfigurationError(f"{name} must be in {bound}, got {value}")


__all__ = [
    "ConfigurationError",
    "reject_unknown_fields",
    "require_nonnegative",
    "require_positive",
    "require_positive_integer",
    "require_probability",
]
