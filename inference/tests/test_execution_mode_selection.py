from __future__ import annotations

from dataclasses import replace

from mindclade.inference.compilation.compile_key import CompileKey
from mindclade.inference.compilation.compiled_variant_cache import (
    CompiledVariant,
    CompiledVariantCache,
)
from mindclade.inference.contracts.execution_mode_contract import (
    ExecutionMode,
    ExecutionModeRequest,
)
from mindclade.inference.execution_modes.mode_qualification import (
    ModeQualification,
    QualificationKey,
    QualificationRegistry,
)
from mindclade.inference.execution_modes.mode_selection import select_execution_mode

from .conftest import sha


def qualification_key(mode: ExecutionMode = ExecutionMode.COMPILED) -> QualificationKey:
    return QualificationKey(
        model_digest=sha("a"),
        serving_revision_digest=sha("b"),
        mode=mode,
        device_type="cpu",
        dtype="float32",
        shape_signature=(1, 2, 3, 2),
        sampler_digest=sha("c"),
        runtime_config_digest=sha("d"),
    )


def test_auto_prefers_exact_qualified_compiled_variant() -> None:
    registry = QualificationRegistry()
    compiled = qualification_key()
    eager = compiled.with_mode(ExecutionMode.EAGER)
    registry.record(ModeQualification.passed_record(eager, evidence_digest=sha("e")))
    registry.record(ModeQualification.passed_record(compiled, evidence_digest=sha("f")))

    selection = select_execution_mode(ExecutionModeRequest(), key=compiled, registry=registry)
    assert selection.selected is ExecutionMode.COMPILED
    assert selection.qualification_digest == sha("f")


def test_auto_uses_eager_only_when_compiled_is_not_qualified() -> None:
    registry = QualificationRegistry()
    key = qualification_key()
    registry.record(
        ModeQualification.passed_record(
            key.with_mode(ExecutionMode.EAGER), evidence_digest=sha("e")
        )
    )
    selection = select_execution_mode(ExecutionModeRequest(), key=key, registry=registry)
    assert selection.selected is ExecutionMode.EAGER
    assert selection.reason == "qualified-fallback"


def test_compiled_cache_is_exact_keyed_and_bounded() -> None:
    cache = CompiledVariantCache(capacity=1)
    first_key = CompileKey(
        model_digest=sha("a"),
        serving_revision_digest=sha("b"),
        runtime_config_digest=sha("c"),
        sampler_digest=sha("d"),
        output_contract_digest=sha("e"),
        device_type="cpu",
        device_capability="generic",
        dtype="float32",
        shape_signature=(1, 2, 3, 2),
        torch_version="2.13.0",
        compiler_version="inductor-v1",
    )
    second_key = replace(first_key, shape_signature=(1, 4, 8, 4))
    first = CompiledVariant(first_key, lambda value: value, sha("f"))
    second = CompiledVariant(second_key, lambda value: value, sha("f"))
    assert cache.put(first) == ()
    assert cache.get(first_key) is first
    assert cache.put(second) == (first_key,)
    assert cache.get(first_key) is None
