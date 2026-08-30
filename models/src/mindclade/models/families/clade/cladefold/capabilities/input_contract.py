"""Machine-readable Q0 input tensor contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def input_contract() -> Mapping[str, Any]:
    return {
        "schema_version": "v1alpha1",
        "dimensions": {"batch": "B", "tokens": "T", "atoms": "A", "bonds": "E"},
        "units": {"coordinates": "angstrom", "diffusion_time": "normalized_[0,1]"},
        "padding": {"mask": True, "token_id": 0, "atom_id": 0, "index": -1},
        "remote_payloads": False,
    }


__all__ = ["input_contract"]
