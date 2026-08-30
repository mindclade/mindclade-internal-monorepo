"""Deterministic execution-mode selection."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts.execution_mode_contract import ExecutionMode, ExecutionModeRequest
from .mode_qualification import QualificationKey, QualificationRegistry


@dataclass(frozen=True, slots=True)
class ModeSelection:
    selected: ExecutionMode
    qualification_digest: str | None
    reason: str


def select_execution_mode(
    request: ExecutionModeRequest,
    *,
    key: QualificationKey,
    registry: QualificationRegistry,
) -> ModeSelection:
    """Prefer compiled for AUTO, but never invent qualification evidence."""

    candidates = (
        (ExecutionMode.COMPILED, ExecutionMode.EAGER)
        if request.requested is ExecutionMode.AUTO
        else (request.requested,)
    )
    for index, mode in enumerate(candidates):
        candidate_key = key.with_mode(mode)
        qualification = registry.lookup(candidate_key)
        if qualification is not None and qualification.passed:
            reason = "requested-qualified" if index == 0 else "qualified-fallback"
            return ModeSelection(mode, qualification.evidence_digest, reason)
        if not request.require_qualified and index == 0:
            return ModeSelection(mode, None, "qualification-not-required")
        if not request.allow_fallback:
            break
    requested = request.requested.value
    raise LookupError(f"no qualified execution mode satisfies {requested!r}")
