"""Immutable key that prevents semantically unsafe request co-batching."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .._identity import content_digest, require_sha256_digest
from ..contracts.request_contract import InferenceRequest


def _bucket(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _shape_bucket(
    inputs: dict[str, torch.Tensor],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        (name, tuple(_bucket(int(value)) for value in tensor.shape[1:]))
        for name, tensor in sorted(inputs.items())
    )


@dataclass(frozen=True, slots=True, order=True)
class BatchKey:
    tenant_id: str
    model_digest: str
    dtype_signature: tuple[str, ...]
    device_type: str
    shape_bucket: tuple[tuple[str, tuple[int, ...]], ...]
    sampler_digest: str
    runtime_config_digest: str
    output_contract_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_digest", require_sha256_digest(self.model_digest))
        object.__setattr__(
            self,
            "sampler_digest",
            require_sha256_digest(self.sampler_digest, field="sampler_digest"),
        )
        object.__setattr__(
            self,
            "runtime_config_digest",
            require_sha256_digest(self.runtime_config_digest, field="runtime_config_digest"),
        )
        object.__setattr__(
            self,
            "output_contract_digest",
            require_sha256_digest(self.output_contract_digest, field="output_contract_digest"),
        )
        if not self.tenant_id:
            raise ValueError("tenant_id cannot be empty")

    @classmethod
    def from_request(cls, request: InferenceRequest) -> BatchKey:
        inputs = dict(request.inputs)
        devices = {tensor.device.type for tensor in inputs.values()}
        if len(devices) != 1:
            raise ValueError("batch key requires inputs on one device type")
        return cls(
            tenant_id=request.tenant_id,
            model_digest=request.model_digest,
            dtype_signature=tuple(sorted({str(t.dtype) for t in inputs.values()})),
            device_type=next(iter(devices)),
            shape_bucket=_shape_bucket(inputs),
            sampler_digest=content_digest(
                {
                    "sampler": request.sampler,
                    "steps": request.num_steps,
                    "adaptive": request.adaptive,
                }
            ),
            runtime_config_digest=request.runtime_config_digest,
            output_contract_digest=content_digest(request.output_fields),
        )
