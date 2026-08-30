"""Stable candidate ranking using descriptive calibrated confidence."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts.result_contract import InferenceCandidate
from .ranking_evidence import RankingEvidence


@dataclass(frozen=True, slots=True)
class RankedCandidates:
    candidates: tuple[InferenceCandidate, ...]
    evidence: RankingEvidence

    @property
    def selected(self) -> InferenceCandidate:
        return self.candidates[0]


class CandidateRanker:
    """Rank descriptively; this score is not a scientific capability claim."""

    policy_name = "calibrated-confidence-desc.v1alpha1"

    def rank(
        self, candidates: tuple[InferenceCandidate, ...], *, request_fingerprint: str
    ) -> RankedCandidates:
        if not candidates:
            raise ValueError("cannot rank an empty candidate set")
        ordered = tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    -candidate.calibrated_confidence,
                    -candidate.confidence,
                    candidate.steps,
                    candidate.batch_seeds,
                    candidate.candidate_id,
                ),
            )
        )
        evidence = RankingEvidence(
            request_fingerprint=request_fingerprint,
            ranking_policy=self.policy_name,
            ordered_candidate_ids=tuple(candidate.candidate_id for candidate in ordered),
            scores=tuple(candidate.calibrated_confidence for candidate in ordered),
            tie_breakers=(
                "raw_confidence_desc",
                "steps_asc",
                "batch_seeds_lexicographic_asc",
                "candidate_id_asc",
            ),
        )
        return RankedCandidates(ordered, evidence)
