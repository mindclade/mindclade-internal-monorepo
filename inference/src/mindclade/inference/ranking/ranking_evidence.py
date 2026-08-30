"""Candidate ranking evidence receipt."""

from __future__ import annotations

from dataclasses import dataclass

from .._identity import content_digest, require_sha256_digest


@dataclass(frozen=True, slots=True)
class RankingEvidence:
    request_fingerprint: str
    ranking_policy: str
    ordered_candidate_ids: tuple[str, ...]
    scores: tuple[float, ...]
    tie_breakers: tuple[str, ...]
    schema_version: str = "ranking-evidence.v1alpha1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_fingerprint",
            require_sha256_digest(self.request_fingerprint, field="request_fingerprint"),
        )
        if len(self.ordered_candidate_ids) != len(self.scores):
            raise ValueError("candidate IDs and ranking scores must align")
        if len(set(self.ordered_candidate_ids)) != len(self.ordered_candidate_ids):
            raise ValueError("ranking candidate IDs must be unique")

    @property
    def digest(self) -> str:
        return content_digest(self)
