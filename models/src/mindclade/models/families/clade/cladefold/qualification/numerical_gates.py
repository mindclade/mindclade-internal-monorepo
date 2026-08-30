"""Q0 numerical qualification checks."""

from __future__ import annotations

import dataclasses

import torch

from mindclade.models.api.batch import CladeFoldBatch
from mindclade.models.families.clade.cladefold.architecture.cladefold import CladeFoldModel


@dataclasses.dataclass(frozen=True)
class NumericalGateResult:
    finite_outputs: bool
    finite_gradients: bool
    parameter_count: int
    passed: bool


def evaluate_numerical_gates(model: CladeFoldModel, batch: CladeFoldBatch) -> NumericalGateResult:
    model.train()
    output = model(batch)
    finite_outputs = all(
        bool(torch.isfinite(value).all())
        for value in (
            output.loss,
            output.predicted_noise,
            output.denoised_coordinates,
            output.distogram_logits,
        )
    )
    output.loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    finite_gradients = bool(gradients) and all(
        gradient is None or bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    count = model.parameter_count
    upper = 2_000_000 if model.config.token_dim <= 64 else 150_000_000
    lower = 1 if model.config.token_dim <= 64 else 50_000_000
    passed = finite_outputs and finite_gradients and lower <= count <= upper
    return NumericalGateResult(finite_outputs, finite_gradients, count, passed)


__all__ = ["NumericalGateResult", "evaluate_numerical_gates"]
