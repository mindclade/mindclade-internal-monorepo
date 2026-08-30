"""Capability-claim reference validation."""

from __future__ import annotations

import re

_REFERENCE = re.compile(r"^claims/[a-z0-9][a-z0-9._-]*/sha256:[0-9a-f]{64}$")


def validate_capability_claim_ref(reference: str) -> str:
    if not _REFERENCE.fullmatch(reference):
        raise ValueError("capability claim reference must be name- and sha256-addressed")
    return reference


__all__ = ["validate_capability_claim_ref"]
