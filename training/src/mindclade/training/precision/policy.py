"""Runtime realization of the public precision contract."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from mindclade.training.api.precision import PrecisionConfig, PrecisionMode


class PrecisionUnavailableError(RuntimeError):
    """Raised when a requested precision is unsafe on the selected device."""


@dataclass(frozen=True)
class ResolvedPrecision:
    device: torch.device
    parameter_dtype: torch.dtype
    compute_dtype: torch.dtype
    reduction_dtype: torch.dtype = torch.float32

    def autocast(self) -> AbstractContextManager[Any]:
        if self.compute_dtype is torch.float32:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.compute_dtype)

    def create_scaler(self) -> Any | None:
        if self.device.type == "cuda" and self.compute_dtype is torch.float16:
            return torch.amp.GradScaler("cuda")
        return None


def resolve_precision(config: PrecisionConfig, device: torch.device) -> ResolvedPrecision:
    if config.mode is PrecisionMode.FP32:
        return ResolvedPrecision(device, torch.float32, torch.float32)
    if device.type != "cuda":
        raise PrecisionUnavailableError(
            f"{config.mode.value} reference training requires a CUDA device; got {device}"
        )
    if config.mode is PrecisionMode.BF16:
        if not torch.cuda.is_bf16_supported():
            raise PrecisionUnavailableError("CUDA device does not support bfloat16")
        return ResolvedPrecision(device, torch.bfloat16, torch.bfloat16)
    if config.mode is PrecisionMode.FP16:
        # CUDA AMP requires FP32 master parameters. Casting the module itself to
        # FP16 also produces FP16 gradients, which GradScaler deliberately
        # refuses to unscale.
        return ResolvedPrecision(device, torch.float32, torch.float16)
    raise PrecisionUnavailableError(f"unsupported precision mode: {config.mode}")
