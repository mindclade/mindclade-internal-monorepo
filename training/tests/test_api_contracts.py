from __future__ import annotations

import pytest
import torch
from mindclade.training import (
    LossReport,
    LossTerm,
    MultitaskDiffusionTask,
    OptimizerConfig,
    ParallelismConfig,
    ParallelismMode,
    PrecisionConfig,
    PrecisionMode,
    TrainingProgram,
)
from mindclade.training.precision import resolve_precision


def test_loss_composition_preserves_autograd_and_reports_fp32() -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    report = LossReport.compose(
        [
            LossTerm("first", parameter.square(), weight=0.5),
            LossTerm("second", parameter, weight=2.0),
        ]
    )
    report.total.backward()
    assert parameter.grad is not None
    torch.testing.assert_close(parameter.grad, torch.tensor(4.0))
    assert report.detached_metrics() == {
        "loss": 6.0,
        "loss/first": 4.0,
        "loss/second": 2.0,
    }


def test_multitask_adapter_uses_exact_declared_components() -> None:
    values = {
        "noise_loss": torch.tensor(1.0, requires_grad=True),
        "distogram_loss": torch.tensor(2.0, requires_grad=True),
        "confidence_loss": torch.tensor(3.0, requires_grad=True),
        "calibration_loss": torch.tensor(4.0, requires_grad=True),
        "geometry_loss": torch.tensor(5.0, requires_grad=True),
    }
    values["loss"] = (
        values["noise_loss"]
        + 0.30 * values["distogram_loss"]
        + 0.10 * values["confidence_loss"]
        + 0.05 * values["calibration_loss"]
        + 0.10 * values["geometry_loss"]
    )
    report = MultitaskDiffusionTask().compute_loss(values, None)
    torch.testing.assert_close(report.total, torch.tensor(2.6))
    report.total.backward()
    assert values["noise_loss"].grad is not None
    assert values["geometry_loss"].grad is not None


def test_multitask_adapter_rejects_model_total_drift() -> None:
    output = {
        "loss": torch.tensor(0.0),
        "noise_loss": torch.tensor(1.0),
        "distogram_loss": torch.tensor(0.0),
        "confidence_loss": torch.tensor(0.0),
        "calibration_loss": torch.tensor(0.0),
        "geometry_loss": torch.tensor(0.0),
    }
    with pytest.raises(ValueError, match="does not match"):
        MultitaskDiffusionTask().compute_loss(output, None)


def test_program_rejects_unsafe_or_unsupported_defaults() -> None:
    with pytest.raises(ValueError, match="unsupported optimizer"):
        OptimizerConfig(name="sgd")
    with pytest.raises(ValueError, match="world_size=1"):
        ParallelismConfig(mode=ParallelismMode.SINGLE_PROCESS, world_size=2)
    with pytest.raises(ValueError, match="max_steps"):
        TrainingProgram(name="bad", max_steps=0)
    assert PrecisionConfig.cpu_reference().to_dict() == {
        "mode": "fp32",
        "reduction_dtype": "fp32",
    }


@pytest.mark.parametrize(
    "field",
    ("max_steps", "gradient_accumulation_steps", "checkpoint_every_steps"),
)
@pytest.mark.parametrize("value", (True, 1.5))
def test_program_requires_exact_integer_step_counts(field: str, value: object) -> None:
    arguments = {"name": "invalid-steps", "max_steps": 1, field: value}
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        TrainingProgram(**arguments)


@pytest.mark.parametrize(
    "field",
    (
        "learning_rate",
        "beta1",
        "beta2",
        "epsilon",
        "weight_decay",
        "max_gradient_norm",
    ),
)
@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_optimizer_rejects_non_finite_numeric_controls(field: str, value: float) -> None:
    arguments = {field: value}
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        OptimizerConfig(**arguments)


def test_fp16_amp_keeps_fp32_master_parameters() -> None:
    resolved = resolve_precision(
        PrecisionConfig(mode=PrecisionMode.FP16),
        torch.device("cuda"),
    )
    assert resolved.parameter_dtype is torch.float32
    assert resolved.compute_dtype is torch.float16
