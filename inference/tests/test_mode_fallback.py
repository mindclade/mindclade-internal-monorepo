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


def test_fallback_rejects_any_non_mode_identity_change() -> None:
    compiled = qualification_key(ExecutionMode.COMPILED)
    eager = compiled.with_mode(ExecutionMode.EAGER)
    changed_targets = (
        replace(eager, device_type="cuda"),
        replace(eager, device_capability="sm_90"),
        replace(eager, dtype="float16"),
        replace(eager, shape_signature=(1, 4, 3, 2)),
        replace(eager, sampler_digest=sha("1")),
        replace(eager, runtime_config_digest=sha("2")),
        replace(eager, output_contract_digest=sha("3")),
        replace(eager, torch_version="2.14.0"),
        replace(eager, compiler_version="inductor-v2"),
    )

    for target in changed_targets:
        registry = QualificationRegistry()
        registry.record(ModeQualification.passed_record(target, evidence_digest=sha("e")))
        decision = decide_fallback(
            failure=FailureClass.RUNTIME_ERROR,
            source_key=compiled,
            target_key=target,
            registry=registry,
            fallback_enabled=True,
        )
        assert not decision.allowed
        assert decision.reason == "execution-identity-mismatch"
