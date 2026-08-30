"""CladeFold multitask diffusion adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
from mindclade.training.api.loss import LossReport, LossTerm
from mindclade.training.api.objective import read_output_field
from mindclade.training.api.task import BatchEnvelope
from torch import Tensor
from torch.nn import Module

DEFAULT_LOSS_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "noise": 1.0,
        "distogram": 0.30,
        "confidence": 0.10,
        "calibration": 0.05,
        "geometry": 0.10,
    }
)

_LOSS_FIELDS = {
    "noise": "noise_loss",
    "distogram": "distogram_loss",
    "confidence": "confidence_loss",
    "calibration": "calibration_loss",
    "geometry": "geometry_loss",
}


@dataclass(frozen=True)
class MultitaskDiffusionTask:
    """Uses the model's authoritative total and exposes its five components."""

    loss_weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_LOSS_WEIGHTS))
    verify_model_total: bool = True
    name: str = "cladefold_multitask_diffusion"

    def __post_init__(self) -> None:
        weights = dict(self.loss_weights)
        if set(weights) != set(_LOSS_FIELDS):
            raise ValueError(f"loss weights must have exactly these keys: {sorted(_LOSS_FIELDS)}")
        if any(value < 0.0 for value in weights.values()):
            raise ValueError("loss weights must be non-negative")
        object.__setattr__(self, "loss_weights", MappingProxyType(weights))

    @staticmethod
    def _payload(batch: Any) -> Any:
        return batch.payload if isinstance(batch, BatchEnvelope) else batch

    def validate_batch(self, batch: Any) -> None:
        payload = self._payload(batch)
        required = (
            "token_type",
            "token_mask",
            "atomic_number",
            "atom_mask",
            "noisy_coordinates",
            "diffusion_time",
            "target_coordinates",
            "target_mask",
        )
        missing = [name for name in required if not hasattr(payload, name)]
        if missing:
            raise TypeError(f"CladeFold training batch is missing fields: {missing}")
        if payload.target_coordinates is None or payload.target_mask is None:
            raise ValueError(
                "multitask diffusion training requires coordinate labels and target mask"
            )
        validate = getattr(payload, "validate", None)
        if callable(validate):
            validate()

    def forward(self, model: Module, batch: Any) -> Any:
        return model(self._payload(batch))

    def compute_loss(self, output: Any, batch: Any) -> LossReport:
        del batch
        terms: dict[str, LossTerm] = {}
        for name, output_field in _LOSS_FIELDS.items():
            value = read_output_field(output, output_field)
            if not isinstance(value, Tensor):
                raise TypeError(f"CladeFold output {output_field!r} must be a torch.Tensor")
            terms[name] = LossTerm(name=name, value=value, weight=self.loss_weights[name])
        composed = LossReport.compose(terms.values())
        try:
            total = read_output_field(output, "loss")
        except (AttributeError, KeyError):
            return composed
        if not isinstance(total, Tensor):
            raise TypeError("CladeFold output 'loss' must be a torch.Tensor")
        total = total.to(dtype=torch.float32)
        if self.verify_model_total and not bool(
            torch.isclose(total.detach(), composed.total.detach(), rtol=1.0e-5, atol=1.0e-6).all()
        ):
            raise ValueError("CladeFold total loss does not match the declared weighted components")
        return LossReport(total=total, terms=terms)
