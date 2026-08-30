"""Configuration wrapper around fail-closed execution fallback."""

from __future__ import annotations

from dataclasses import dataclass

from ..execution_modes.fallback_decision import FailureClass, FallbackDecision, decide_fallback
from ..execution_modes.mode_qualification import QualificationKey, QualificationRegistry


@dataclass(frozen=True, slots=True)
class CompiledFallbackPolicy:
    enabled: bool = True

    def decide(
        self,
        *,
        failure: FailureClass,
        compiled_key: QualificationKey,
        eager_key: QualificationKey,
        qualifications: QualificationRegistry,
    ) -> FallbackDecision:
        return decide_fallback(
            failure=failure,
            source_key=compiled_key,
            target_key=eager_key,
            registry=qualifications,
            fallback_enabled=self.enabled,
        )
