"""Machine-readable Q0 output tensor contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def output_contract() -> Mapping[str, Any]:
    return {
        "schema_version": "v1alpha1",
        "forward": {
            "predicted_noise": "[B,A,3]",
            "denoised_coordinates": "[B,A,3]",
            "distogram_logits": "[B,T,T,D]",
            "atom_confidence": "[B,A]",
            "token_confidence": "[B,T]",
        },
        "fold": {
            "atom_coordinates": "[B,S,A,3]",
            "sample_confidence": "[B,S]",
            "trajectories": "optional_[B,S,K+1,A,3]",
        },
        "padded_values": 0,
    }


__all__ = ["output_contract"]
