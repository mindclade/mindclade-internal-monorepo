"""Bounded scientific qualification profile."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ScientificProfile:
    profile_id: str
    model_digest: str
    allowed_claims: tuple[str, ...]
    evidence_digests: tuple[str, ...]
    status: str

    def __post_init__(self) -> None:
        if self.status not in {"REFERENCE_ONLY", "QUALIFIED", "REJECTED", "REVOKED"}:
            raise ValueError("unknown scientific profile status")
        if not self.model_digest.startswith("sha256:"):
            raise ValueError("scientific profiles bind immutable model digests")
        if self.status == "QUALIFIED" and not self.evidence_digests:
            raise ValueError("qualified profiles require evidence")


__all__ = ["ScientificProfile"]
