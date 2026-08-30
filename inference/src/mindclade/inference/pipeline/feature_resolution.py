"""Deterministic tensor-feature identity and reuse receipts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from .._identity import content_digest
from .preprocessing import PreparedInference


@dataclass(frozen=True, slots=True)
class FeatureReceipt:
    derivation_digest: str
    input_digests: Mapping[str, str]
    feature_schema: str = "mindclade.tensor-features.v1alpha1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_digests", MappingProxyType(dict(self.input_digests)))


def resolve_features(
    prepared: PreparedInference,
) -> tuple[Mapping[str, torch.Tensor], FeatureReceipt]:
    """Resolve already-materialized tensor features without implicit featurization."""

    prepared.request.verify_integrity()
    digests = dict(prepared.request.input_digests)
    receipt = FeatureReceipt(
        derivation_digest=content_digest(
            {"schema": "mindclade.tensor-features.v1alpha1", "inputs": digests}
        ),
        input_digests=digests,
    )
    return prepared.inputs, receipt
