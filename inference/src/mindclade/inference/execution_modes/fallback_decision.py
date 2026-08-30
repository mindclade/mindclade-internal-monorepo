"""Same-digest runtime fallback decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..contracts.execution_mode_contract import ExecutionMode
from .mode_qualification import QualificationKey, QualificationRegistry


class FailureClass(StrEnum):
    COMPILE_UNAVAILABLE = "compile-unavailable"
    COMPILE_ERROR = "compile-error"
    RUNTIME_ERROR = "runtime-error"
    RESOURCE_EXHAUSTED = "resource-exhausted"
    NUMERICAL_INVALID = "numerical-invalid"
    INTEGRITY_FAILURE = "integrity-failure"
    CANCELLED = "cancelled"


_FALLBACK_ELIGIBLE = {
    FailureClass.COMPILE_UNAVAILABLE,
    FailureClass.COMPILE_ERROR,
    FailureClass.RUNTIME_ERROR,
}


@dataclass(frozen=True, slots=True)
class FallbackDecision:
    allowed: bool
    target_mode: ExecutionMode | None
    reason: str
    qualification_digest: str | None = None


def decide_fallback(
    *,
    failure: FailureClass,
    source_key: QualificationKey,
    target_key: QualificationKey,
    registry: QualificationRegistry,
    fallback_enabled: bool,
) -> FallbackDecision:
    if not fallback_enabled:
        return FallbackDecision(False, None, "fallback-disabled")
    if failure not in _FALLBACK_ELIGIBLE:
        return FallbackDecision(False, None, f"failure-class-{failure.value}-fails-closed")
    if source_key.model_digest != target_key.model_digest:
        return FallbackDecision(False, None, "model-digest-mismatch")
    if source_key.serving_revision_digest != target_key.serving_revision_digest:
        return FallbackDecision(False, None, "serving-revision-mismatch")
    if source_key.mode is not ExecutionMode.COMPILED or target_key.mode is not ExecutionMode.EAGER:
        return FallbackDecision(False, None, "only-compiled-to-eager-fallback-is-supported")
    record = registry.lookup(target_key)
    if record is None or not record.passed:
        return FallbackDecision(False, None, "target-mode-is-not-qualified")
    return FallbackDecision(
        True,
        ExecutionMode.EAGER,
        "same-digest-qualified-eager",
        record.evidence_digest,
    )
