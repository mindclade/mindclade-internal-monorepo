from __future__ import annotations

from dataclasses import replace

from mindclade.inference.contracts.execution_mode_contract import ExecutionMode
from mindclade.inference.execution_modes.fallback_decision import FailureClass, decide_fallback
from mindclade.inference.execution_modes.mode_qualification import (
    ModeQualification,
    QualificationRegistry,
)

from .conftest import sha
from .test_execution_mode_selection import qualification_key


def test_fallback_requires_same_digest_and_qualified_eager() -> None:
    registry = QualificationRegistry()
    compiled = qualification_key(ExecutionMode.COMPILED)
    eager = compiled.with_mode(ExecutionMode.EAGER)
    registry.record(ModeQualification.passed_record(eager, evidence_digest=sha("e")))

    decision = decide_fallback(
        failure=FailureClass.COMPILE_ERROR,
        source_key=compiled,
        target_key=eager,
        registry=registry,
        fallback_enabled=True,
    )
    assert decision.allowed
    assert decision.target_mode is ExecutionMode.EAGER

    changed_model = replace(eager, model_digest=sha("9"))
    decision = decide_fallback(
        failure=FailureClass.COMPILE_ERROR,
        source_key=compiled,
        target_key=changed_model,
        registry=registry,
        fallback_enabled=True,
    )
    assert not decision.allowed
    assert decision.reason == "model-digest-mismatch"


def test_integrity_and_numerical_failures_never_fallback() -> None:
    registry = QualificationRegistry()
    compiled = qualification_key(ExecutionMode.COMPILED)
    eager = compiled.with_mode(ExecutionMode.EAGER)
    registry.record(ModeQualification.passed_record(eager, evidence_digest=sha("e")))
    for failure in (FailureClass.INTEGRITY_FAILURE, FailureClass.NUMERICAL_INVALID):
        decision = decide_fallback(
            failure=failure,
            source_key=compiled,
            target_key=eager,
            registry=registry,
            fallback_enabled=True,
        )
        assert not decision.allowed
        assert "fails-closed" in decision.reason
