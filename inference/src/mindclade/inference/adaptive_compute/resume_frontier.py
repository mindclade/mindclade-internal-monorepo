"""Portable, integrity-checked adaptive sampling resume frontier."""

from __future__ import annotations

import base64
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .._identity import content_digest, require_sha256_digest, tensor_digest
from .budget_accounting import BudgetSnapshot
from .stopping_rule import Observation, StoppingState

_DTYPES: dict[str, torch.dtype] = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
}


def _required_string(value: Mapping[str, Any], name: str) -> str:
    field = value.get(name)
    if type(field) is not str:
        raise ValueError(f"resume frontier {name} must be a string")
    return field


def _required_integer(value: Mapping[str, Any], name: str) -> int:
    field = value.get(name)
    if type(field) is not int:
        raise ValueError(f"resume frontier {name} must be an integer")
    return field


def _required_number(value: Mapping[str, Any], name: str) -> float:
    field = value.get(name)
    if type(field) not in (int, float):
        raise ValueError(f"resume frontier {name} must be numeric")
    assert isinstance(field, (int, float)) and not isinstance(field, bool)
    return float(field)


def _encode_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    contiguous = tensor.contiguous().clone()
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "data_base64": base64.b64encode(bytes(contiguous.untyped_storage())).decode("ascii"),
        "digest": tensor_digest(contiguous),
    }


def _decode_tensor(value: object, *, field: str) -> tuple[torch.Tensor, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"resume frontier is missing {field}")
    dtype_name = value.get("dtype")
    if type(dtype_name) is not str or dtype_name not in _DTYPES:
        raise ValueError(f"resume frontier has an unsupported {field} dtype")
    shape_value = value.get("shape")
    if not isinstance(shape_value, list) or not all(
        type(dimension) is int and dimension >= 0 for dimension in shape_value
    ):
        raise ValueError(f"resume {field} shape is invalid")
    encoded_payload = value.get("data_base64")
    if type(encoded_payload) is not str:
        raise ValueError(f"resume {field} payload must be a string")
    try:
        raw = base64.b64decode(encoded_payload, validate=True)
    except Exception as exc:
        raise ValueError(f"resume {field} payload is not valid base64") from exc

    dtype = _DTYPES[str(dtype_name)]
    expected_bytes = math.prod(shape_value) * torch.empty((), dtype=dtype).element_size()
    if len(raw) != expected_bytes:
        raise ValueError(f"resume {field} payload length does not match its shape")
    if raw:
        byte_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
        tensor = byte_tensor.view(dtype).reshape(tuple(shape_value))
    else:
        tensor = torch.empty(tuple(shape_value), dtype=dtype)
    return tensor, _required_string(value, "digest")


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
    last_evaluation_coordinates: torch.Tensor
    stopping_state: StoppingState
    schema_version: str = "resume-frontier.v1alpha2"

    def __post_init__(self) -> None:
        for name in ("request_fingerprint", "model_digest", "sampler_digest", "policy_digest"):
            object.__setattr__(self, name, require_sha256_digest(getattr(self, name), field=name))
        if type(self.completed_steps) is not int or self.completed_steps < 0:
            raise ValueError("completed_steps must be a non-negative integer")
        if type(self.consumed_candidates) is not int or self.consumed_candidates < 0:
            raise ValueError("consumed_candidates must be a non-negative integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if self.schema_version != "resume-frontier.v1alpha2":
            raise ValueError("only resume-frontier.v1alpha2 is supported")
        if self.coordinates.ndim != 3 or self.coordinates.shape[-1] != 3:
            raise ValueError("resume coordinates must have shape [B, A, 3]")
        if self.coordinates.dtype not in _DTYPES.values():
            raise TypeError("resume coordinates use an unsupported dtype")
        if not torch.isfinite(self.coordinates).all():
            raise ValueError("resume coordinates must be finite")
        if self.last_evaluation_coordinates.shape != self.coordinates.shape:
            raise ValueError("last evaluation coordinates must match resume coordinates")
        if self.last_evaluation_coordinates.dtype != self.coordinates.dtype:
            raise TypeError("last evaluation coordinates must match the resume dtype")
        if not torch.isfinite(self.last_evaluation_coordinates).all():
            raise ValueError("last evaluation coordinates must be finite")
        if not isinstance(self.stopping_state, StoppingState):
            raise TypeError("stopping_state must be a StoppingState")
        previous = self.stopping_state.previous_observation
        if previous is not None and previous.completed_steps > self.completed_steps:
            raise ValueError("stopping observation exceeds resume progress")

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
        last_evaluation_coordinates: torch.Tensor,
        stopping_state: StoppingState,
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
            last_evaluation_coordinates=(
                last_evaluation_coordinates.detach().contiguous().cpu().clone()
            ),
            stopping_state=stopping_state,
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
                "last_evaluation_coordinates_digest": tensor_digest(
                    self.last_evaluation_coordinates
                ),
                "stopping_state": self.stopping_state,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        previous = self.stopping_state.previous_observation
        return {
            "schema_version": self.schema_version,
            "request_fingerprint": self.request_fingerprint,
            "model_digest": self.model_digest,
            "sampler_digest": self.sampler_digest,
            "policy_digest": self.policy_digest,
            "completed_steps": self.completed_steps,
            "consumed_candidates": self.consumed_candidates,
            "seed": self.seed,
            "coordinates": _encode_tensor(self.coordinates),
            "last_evaluation_coordinates": _encode_tensor(self.last_evaluation_coordinates),
            "stopping_state": {
                "previous_observation": (
                    None
                    if previous is None
                    else {
                        "completed_steps": previous.completed_steps,
                        "calibrated_confidence": previous.calibrated_confidence,
                        "rms_displacement_angstrom": previous.rms_displacement_angstrom,
                    }
                ),
                "consecutive_converged": self.stopping_state.consecutive_converged,
            },
            "frontier_digest": self.digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResumeFrontier:
        schema_version = _required_string(value, "schema_version")
        if schema_version != "resume-frontier.v1alpha2":
            raise ValueError("only resume-frontier.v1alpha2 is supported")
        coordinates, coordinates_digest = _decode_tensor(
            value.get("coordinates"), field="coordinates"
        )
        last_evaluation, last_evaluation_digest = _decode_tensor(
            value.get("last_evaluation_coordinates"),
            field="last evaluation coordinates",
        )
        encoded_state = value.get("stopping_state")
        if not isinstance(encoded_state, Mapping):
            raise ValueError("resume frontier is missing stopping state")
        encoded_previous = encoded_state.get("previous_observation")
        if encoded_previous is None:
            previous = None
        elif isinstance(encoded_previous, Mapping):
            try:
                previous = Observation(
                    completed_steps=_required_integer(encoded_previous, "completed_steps"),
                    calibrated_confidence=_required_number(
                        encoded_previous, "calibrated_confidence"
                    ),
                    rms_displacement_angstrom=_required_number(
                        encoded_previous, "rms_displacement_angstrom"
                    ),
                )
            except ValueError as exc:
                raise ValueError("resume stopping observation is invalid") from exc
        else:
            raise ValueError("resume stopping observation is invalid")
        try:
            stopping_state = StoppingState(
                previous_observation=previous,
                consecutive_converged=_required_integer(encoded_state, "consecutive_converged"),
            )
        except ValueError as exc:
            raise ValueError("resume stopping state is invalid") from exc
        frontier = cls(
            request_fingerprint=_required_string(value, "request_fingerprint"),
            model_digest=_required_string(value, "model_digest"),
            sampler_digest=_required_string(value, "sampler_digest"),
            policy_digest=_required_string(value, "policy_digest"),
            completed_steps=_required_integer(value, "completed_steps"),
            consumed_candidates=_required_integer(value, "consumed_candidates"),
            seed=_required_integer(value, "seed"),
            coordinates=coordinates,
            last_evaluation_coordinates=last_evaluation,
            stopping_state=stopping_state,
            schema_version=schema_version,
        )
        if coordinates_digest != tensor_digest(frontier.coordinates):
            raise ValueError("resume coordinate digest mismatch")
        if last_evaluation_digest != tensor_digest(frontier.last_evaluation_coordinates):
            raise ValueError("resume last evaluation coordinate digest mismatch")
        if value.get("frontier_digest") != frontier.digest:
            raise ValueError("resume frontier digest mismatch")
        return frontier
