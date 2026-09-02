"""Exact-key qualification records for eager and compiled execution."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from .._identity import content_digest, require_sha256_digest
from ..contracts.execution_mode_contract import ExecutionMode

if TYPE_CHECKING:
    from ..compilation.compile_key import CompileKey


@dataclass(frozen=True, slots=True)
class QualificationKey:
    model_digest: str
    serving_revision_digest: str
    mode: ExecutionMode
    device_type: str
    device_capability: str
    dtype: str
    shape_signature: tuple[int, ...]
    sampler_digest: str
    runtime_config_digest: str
    output_contract_digest: str
    torch_version: str
    compiler_version: str

    def __post_init__(self) -> None:
        for name in (
            "model_digest",
            "serving_revision_digest",
            "sampler_digest",
            "runtime_config_digest",
            "output_contract_digest",
        ):
            object.__setattr__(self, name, require_sha256_digest(getattr(self, name), field=name))
        if self.mode is ExecutionMode.AUTO:
            raise ValueError("AUTO is a selection request, not a qualifiable execution mode")
        for name in (
            "device_type",
            "device_capability",
            "dtype",
            "torch_version",
            "compiler_version",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} cannot be empty")
        if any(dimension < 0 for dimension in self.shape_signature):
            raise ValueError("shape signature dimensions cannot be negative")

    @classmethod
    def from_compile_key(
        cls,
        compile_key: CompileKey,
        *,
        mode: ExecutionMode,
    ) -> QualificationKey:
        return cls(
            model_digest=compile_key.model_digest,
            serving_revision_digest=compile_key.serving_revision_digest,
            mode=mode,
            device_type=compile_key.device_type,
            device_capability=compile_key.device_capability,
            dtype=compile_key.dtype,
            shape_signature=compile_key.shape_signature,
            sampler_digest=compile_key.sampler_digest,
            runtime_config_digest=compile_key.runtime_config_digest,
            output_contract_digest=compile_key.output_contract_digest,
            torch_version=compile_key.torch_version,
            compiler_version=compile_key.compiler_version,
        )

    def with_mode(self, mode: ExecutionMode) -> QualificationKey:
        return QualificationKey(
            model_digest=self.model_digest,
            serving_revision_digest=self.serving_revision_digest,
            mode=mode,
            device_type=self.device_type,
            device_capability=self.device_capability,
            dtype=self.dtype,
            shape_signature=self.shape_signature,
            sampler_digest=self.sampler_digest,
            runtime_config_digest=self.runtime_config_digest,
            output_contract_digest=self.output_contract_digest,
            torch_version=self.torch_version,
            compiler_version=self.compiler_version,
        )

    @property
    def digest(self) -> str:
        return content_digest(self)


@dataclass(frozen=True, slots=True)
class ModeQualification:
    key: QualificationKey
    passed: bool
    evidence_digest: str
    qualified_at: str
    max_absolute_error: float
    max_relative_error: float
    qualifier_version: str = "mode-qualification.v1alpha2"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_digest",
            require_sha256_digest(self.evidence_digest, field="evidence_digest"),
        )
        try:
            parsed = datetime.fromisoformat(self.qualified_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("qualified_at must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None:
            raise ValueError("qualified_at must include a timezone")
        if (
            not math.isfinite(self.max_absolute_error)
            or not math.isfinite(self.max_relative_error)
            or self.max_absolute_error < 0
            or self.max_relative_error < 0
        ):
            raise ValueError("qualification errors must be finite and non-negative")

    @classmethod
    def passed_record(
        cls,
        key: QualificationKey,
        *,
        evidence_digest: str,
        max_absolute_error: float = 0.0,
        max_relative_error: float = 0.0,
    ) -> ModeQualification:
        return cls(
            key=key,
            passed=True,
            evidence_digest=evidence_digest,
            qualified_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            max_absolute_error=max_absolute_error,
            max_relative_error=max_relative_error,
        )


class QualificationRegistry:
    """Append/replace exact qualification evidence, keyed by full runtime identity."""

    def __init__(self) -> None:
        self._records: dict[QualificationKey, ModeQualification] = {}
        self._lock = threading.RLock()

    def record(self, qualification: ModeQualification) -> None:
        with self._lock:
            self._records[qualification.key] = qualification

    def lookup(self, key: QualificationKey) -> ModeQualification | None:
        with self._lock:
            return self._records.get(key)

    def is_qualified(self, key: QualificationKey) -> bool:
        record = self.lookup(key)
        return bool(record is not None and record.passed)
