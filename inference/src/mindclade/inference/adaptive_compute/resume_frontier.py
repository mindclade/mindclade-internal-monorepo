"""Portable, integrity-checked adaptive sampling resume frontier."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .._identity import content_digest, require_sha256_digest, tensor_digest
from .budget_accounting import BudgetSnapshot

_DTYPES: dict[str, torch.dtype] = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
}


@dataclass(frozen=True, slots=True)
class ResumeFrontier:
    request_fingerprint: str
    model_digest: str
    sampler_digest: str
    policy_digest: str
    completed_steps: int
    consumed_candidates: int
    seed: int
    coordinates: torch.Tensor
    schema_version: str = "resume-frontier.v1alpha1"

    def __post_init__(self) -> None:
        for name in ("request_fingerprint", "model_digest", "sampler_digest", "policy_digest"):
            object.__setattr__(self, name, require_sha256_digest(getattr(self, name), field=name))
        if self.completed_steps < 0 or self.consumed_candidates < 0:
            raise ValueError("resume progress cannot be negative")
        if self.coordinates.ndim != 3 or self.coordinates.shape[-1] != 3:
            raise ValueError("resume coordinates must have shape [B, A, 3]")
        if self.coordinates.dtype not in _DTYPES.values():
            raise TypeError("resume coordinates use an unsupported dtype")
        if not torch.isfinite(self.coordinates).all():
            raise ValueError("resume coordinates must be finite")

    @classmethod
    def capture(
        cls,
        *,
        request_fingerprint: str,
        model_digest: str,
        sampler_digest: str,
        policy_digest: str,
        budget: BudgetSnapshot,
        seed: int,
        coordinates: torch.Tensor,
    ) -> ResumeFrontier:
        return cls(
            request_fingerprint=request_fingerprint,
            model_digest=model_digest,
            sampler_digest=sampler_digest,
            policy_digest=policy_digest,
            completed_steps=budget.consumed_steps,
            consumed_candidates=budget.consumed_candidates,
            seed=seed,
            coordinates=coordinates.detach().contiguous().cpu().clone(),
        )

    def assert_compatible(
        self,
        *,
        request_fingerprint: str,
        model_digest: str,
        sampler_digest: str,
        policy_digest: str,
    ) -> None:
        expected = {
            "request_fingerprint": request_fingerprint,
            "model_digest": model_digest,
            "sampler_digest": sampler_digest,
            "policy_digest": policy_digest,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"resume {name} mismatch")

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "schema_version": self.schema_version,
                "request_fingerprint": self.request_fingerprint,
                "model_digest": self.model_digest,
                "sampler_digest": self.sampler_digest,
                "policy_digest": self.policy_digest,
                "completed_steps": self.completed_steps,
                "consumed_candidates": self.consumed_candidates,
                "seed": self.seed,
                "coordinates_digest": tensor_digest(self.coordinates),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        coordinates = self.coordinates.contiguous().clone()
        raw = bytes(coordinates.untyped_storage())
        return {
            "schema_version": self.schema_version,
            "request_fingerprint": self.request_fingerprint,
            "model_digest": self.model_digest,
            "sampler_digest": self.sampler_digest,
            "policy_digest": self.policy_digest,
            "completed_steps": self.completed_steps,
            "consumed_candidates": self.consumed_candidates,
            "seed": self.seed,
            "coordinates": {
                "dtype": str(self.coordinates.dtype),
                "shape": list(self.coordinates.shape),
                "data_base64": base64.b64encode(raw).decode("ascii"),
                "digest": tensor_digest(self.coordinates),
            },
            "frontier_digest": self.digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResumeFrontier:
        encoded = value.get("coordinates")
        if not isinstance(encoded, Mapping):
            raise ValueError("resume frontier is missing coordinates")
        dtype_name = encoded.get("dtype")
        if dtype_name not in _DTYPES:
            raise ValueError("resume frontier has an unsupported coordinate dtype")
        shape_value = encoded.get("shape")
        if not isinstance(shape_value, list) or not all(isinstance(v, int) for v in shape_value):
            raise ValueError("resume coordinate shape is invalid")
        try:
            raw = base64.b64decode(str(encoded["data_base64"]), validate=True)
        except Exception as exc:
            raise ValueError("resume coordinate payload is not valid base64") from exc
        byte_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
        coordinates = byte_tensor.view(_DTYPES[str(dtype_name)]).reshape(tuple(shape_value))
        frontier = cls(
            request_fingerprint=str(value["request_fingerprint"]),
            model_digest=str(value["model_digest"]),
            sampler_digest=str(value["sampler_digest"]),
            policy_digest=str(value["policy_digest"]),
            completed_steps=int(value["completed_steps"]),
            consumed_candidates=int(value["consumed_candidates"]),
            seed=int(value["seed"]),
            coordinates=coordinates,
            schema_version=str(value["schema_version"]),
        )
        if encoded.get("digest") != tensor_digest(frontier.coordinates):
            raise ValueError("resume coordinate digest mismatch")
        if value.get("frontier_digest") != frontier.digest:
            raise ValueError("resume frontier digest mismatch")
        return frontier
