"""Precision policy declared independently from a concrete engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class PrecisionMode(StrEnum):
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"


@dataclass(frozen=True)
class PrecisionConfig:
    mode: PrecisionMode = PrecisionMode.FP32
    reduction_dtype: PrecisionMode = PrecisionMode.FP32

    def __post_init__(self) -> None:
        if self.reduction_dtype is not PrecisionMode.FP32:
            raise ValueError("reference training requires fp32 reductions")

    @classmethod
    def cpu_reference(cls) -> PrecisionConfig:
        return cls(mode=PrecisionMode.FP32)

    @classmethod
    def cuda_bf16(cls) -> PrecisionConfig:
        return cls(mode=PrecisionMode.BF16)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["mode"] = self.mode.value
        value["reduction_dtype"] = self.reduction_dtype.value
        return value
