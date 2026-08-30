"""Fail-closed capability-claim compatibility."""

from __future__ import annotations

from .scientific_profile import ScientificProfile


class ClaimCompatibilityError(ValueError):
    pass


def require_claim(profile: ScientificProfile, claim: str) -> None:
    if profile.status != "QUALIFIED" or claim not in profile.allowed_claims:
        raise ClaimCompatibilityError(
            f"claim {claim!r} is not permitted by qualified profile {profile.profile_id!r}"
        )


def reject_q0_scientific_claim(claim: str) -> None:
    if claim not in {"systems-reference-only", "random-initialization"}:
        raise ClaimCompatibilityError("CladeFold Q0 has no scientific capability qualification")


__all__ = ["ClaimCompatibilityError", "reject_q0_scientific_claim", "require_claim"]
