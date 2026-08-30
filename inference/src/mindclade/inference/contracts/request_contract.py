"""Validated, immutable inference request contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch

from .._identity import content_digest, require_sha256_digest, tensor_digest
from .adaptive_compute_contract import AdaptiveComputeRequest
from .execution_mode_contract import ExecutionModeRequest

_MAX_SEED = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """A tensor-only request pinned to an immutable model digest."""

    request_id: str
    tenant_id: str
    project_id: str
    model_digest: str
    inputs: Mapping[str, torch.Tensor]
    seed: int
    num_samples: int = 1
    num_steps: int = 32
    output_fields: tuple[str, ...] = ("coordinates", "confidence")
    execution: ExecutionModeRequest = field(default_factory=ExecutionModeRequest)
    adaptive: AdaptiveComputeRequest = field(default_factory=AdaptiveComputeRequest)
    sampler: str = "diffusion-v1"
    runtime_config: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "v1alpha1"
    _input_digests: Mapping[str, str] = field(init=False, repr=False, compare=False)
    _runtime_config_digest: str = field(init=False, repr=False, compare=False)
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("request_id", "tenant_id", "project_id"):
            value = getattr(self, name)
            if type(value) is not str or not value or len(value) > 128:
                raise ValueError(f"{name} must contain 1..128 characters")
        object.__setattr__(
            self, "model_digest", require_sha256_digest(self.model_digest, field="model_digest")
        )
        if self.schema_version != "v1alpha1":
            raise ValueError("only the v1alpha1 request schema is supported")
        if type(self.seed) is not int or not 0 <= self.seed <= _MAX_SEED:
            raise ValueError(f"seed must be within [0, {_MAX_SEED}]")
        if type(self.num_samples) is not int or not 1 <= self.num_samples <= 16:
            raise ValueError("num_samples must be within [1, 16]")
        if type(self.num_steps) is not int or not 2 <= self.num_steps <= 128:
            raise ValueError("num_steps must be within [2, 128]")
        if self.adaptive.enabled:
            if self.num_steps != self.adaptive.max_steps:
                raise ValueError("num_steps must equal adaptive.max_steps when adaptive is enabled")
            if self.num_samples > self.adaptive.max_candidates:
                raise ValueError("num_samples exceeds the adaptive candidate budget")
        if type(self.sampler) is not str or not self.sampler or len(self.sampler) > 64:
            raise ValueError("sampler must contain 1..64 characters")
        if not isinstance(self.inputs, Mapping) or not self.inputs:
            raise ValueError("inputs cannot be empty")
        copied_inputs: dict[str, torch.Tensor] = {}
        for name, tensor in self.inputs.items():
            if not isinstance(name, str) or not name or len(name) > 64:
                raise ValueError("input names must contain 1..64 characters")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"input {name!r} must be a torch.Tensor")
            if tensor.layout is not torch.strided:
                raise TypeError(f"input {name!r} must use the strided tensor layout")
            # A frozen dataclass does not freeze mutable Tensor storage. Own
            # the admitted bytes so caller mutation cannot invalidate the
            # fingerprint after validation.
            copied_inputs[name] = tensor.detach().clone()
        if not isinstance(self.output_fields, tuple) or any(
            type(field_name) is not str for field_name in self.output_fields
        ):
            raise TypeError("output_fields must be a tuple of strings")
        fields = tuple(dict.fromkeys(self.output_fields))
        if not fields or any(not field_name or len(field_name) > 64 for field_name in fields):
            raise ValueError("output_fields cannot be empty")
        if not isinstance(self.runtime_config, Mapping):
            raise TypeError("runtime_config must be a mapping")
        object.__setattr__(self, "inputs", MappingProxyType(copied_inputs))
        object.__setattr__(self, "output_fields", fields)
        object.__setattr__(self, "runtime_config", MappingProxyType(dict(self.runtime_config)))
        input_digests = {name: tensor_digest(value) for name, value in copied_inputs.items()}
        runtime_config_digest = content_digest(self.runtime_config)
        object.__setattr__(self, "_input_digests", MappingProxyType(input_digests))
        object.__setattr__(self, "_runtime_config_digest", runtime_config_digest)
        object.__setattr__(
            self,
            "_fingerprint",
            content_digest(
                {
                    "schema_version": self.schema_version,
                    "model_digest": self.model_digest,
                    "inputs": input_digests,
                    "seed": self.seed,
                    "num_samples": self.num_samples,
                    "num_steps": self.num_steps,
                    "output_fields": self.output_fields,
                    "execution": self.execution,
                    "adaptive": self.adaptive,
                    "sampler": self.sampler,
                    "runtime_config_digest": runtime_config_digest,
                }
            ),
        )

    @property
    def fingerprint(self) -> str:
        """Content identity excluding tenant/project/request identifiers."""

        return self._fingerprint

    @property
    def input_digests(self) -> Mapping[str, str]:
        return self._input_digests

    @property
    def runtime_config_digest(self) -> str:
        return self._runtime_config_digest

    def verify_integrity(self) -> None:
        current = {name: tensor_digest(value) for name, value in self.inputs.items()}
        if current != dict(self._input_digests):
            raise ValueError("request input tensor changed after admission")
        if content_digest(self.runtime_config) != self._runtime_config_digest:
            raise ValueError("request runtime config changed after admission")
