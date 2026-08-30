"""Profile-to-artifact binding validation."""

from __future__ import annotations

from .scientific_profile import ScientificProfile


def require_profile_binding(profile: ScientificProfile, model_digest: str) -> None:
    if profile.model_digest != model_digest:
        raise ValueError("scientific profile subject does not match model digest")
    if profile.status in {"REJECTED", "REVOKED"}:
        raise ValueError(f"scientific profile is {profile.status.lower()}")


__all__ = ["require_profile_binding"]
