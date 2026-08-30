"""Development-only model alias policy."""

from __future__ import annotations

import re


class AliasPolicyError(ValueError):
    pass


RANDOM_INIT_ALIAS = "cladefold-q0-random-init"
_FORBIDDEN = {"latest", "production", "prod", "stable", "scientific", "qualified"}


def validate_alias(alias: str, *, claim_level: str = "systems-reference-only") -> str:
    normalized = alias.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,62}", normalized):
        raise AliasPolicyError("alias must be a lowercase DNS-style label")
    if normalized in _FORBIDDEN or normalized != RANDOM_INIT_ALIAS:
        raise AliasPolicyError(
            "Q0 permits only the explicit cladefold-q0-random-init development alias"
        )
    if "systems-reference" not in claim_level:
        raise AliasPolicyError("random-initialized Q0 cannot carry scientific capability claims")
    return normalized


__all__ = ["RANDOM_INIT_ALIAS", "AliasPolicyError", "validate_alias"]
