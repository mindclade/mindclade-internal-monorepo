from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from mindclade.models.common.configuration import ModelConfig
from mindclade.training import (
    BatchEnvelope,
    CheckpointCompatibilityError,
    LossReport,
    LossTerm,
    OptimizerConfig,
    ReproducibilityConfig,
    SchedulerConfig,
    Trainer,
    TrainingProgram,
)
from mindclade.training.api import RunStatus
from mindclade.training.checkpointing import DCPCheckpointManager


@dataclass(frozen=True)
class RegressionTask:
    name: str = "regression"

    def validate_batch(self, batch: BatchEnvelope[Any]) -> None:
        x, y = batch.payload
        assert x.ndim == 2 and y.ndim == 2

    def forward(self, model: torch.nn.Module, batch: BatchEnvelope[Any]) -> torch.Tensor:
        return model(batch.payload[0])

    def compute_loss(self, output: torch.Tensor, batch: BatchEnvelope[Any]) -> LossReport:
        value = torch.nn.functional.mse_loss(output, batch.payload[1])
        return LossReport.compose([LossTerm("regression", value)])


@dataclass(frozen=True)
class RegressionModelConfig(ModelConfig):
    objective_revision: int = 1


class ConfiguredLinear(torch.nn.Linear):
    def __init__(self, config: RegressionModelConfig) -> None:
        super().__init__(1, 1)
        self.config = config


class ModeRecordingLinear(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(1, 1)
        self.training_modes: list[bool] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.training_modes.append(self.training)
        return super().forward(value)


def _program(max_steps: int = 25, accumulation: int = 1) -> TrainingProgram:
    return TrainingProgram(
        name="linear-regression-smoke",
        max_steps=max_steps,
        gradient_accumulation_steps=accumulation,
        optimizer=OptimizerConfig(learning_rate=0.1, weight_decay=0.0),
        scheduler=SchedulerConfig(name="constant", minimum_learning_rate_ratio=1.0),
        reproducibility=ReproducibilityConfig(seed=123),
        checkpoint_every_steps=100,
    )


def _data() -> list[BatchEnvelope[Any]]:
    return [
        BatchEnvelope(
            payload=(torch.tensor([[x]], dtype=torch.float32), torch.tensor([[2.0 * x + 1.0]])),
            sample_ids=(f"sample-{index}",),
        )
        for index, x in enumerate((-2.0, -1.0, 0.0, 1.0, 2.0))
    ]


def test_reference_trainer_reduces_loss_and_tracks_progress() -> None:
    torch.manual_seed(123)
    trainer = Trainer(torch.nn.Linear(1, 1), RegressionTask(), _program(), run_id="run-1")
    result = trainer.run(_data())
    assert result.state.global_step == 25
    assert result.state.optimizer_steps == 25
    assert result.state.data.samples_seen == 25
    # Epoch is the zero-based epoch containing the next unread batch.
    assert result.state.data.epoch == 5
    assert result.history[-1].metrics["loss"] < result.history[0].metrics["loss"]
    assert all(parameter.grad is not None for parameter in trainer.model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainer.model.parameters())


def test_gradient_accumulation_commits_one_step_for_two_microbatches() -> None:
    torch.manual_seed(9)
    trainer = Trainer(
        torch.nn.Linear(1, 1),
        RegressionTask(),
        _program(max_steps=3, accumulation=2),
        run_id="run-accumulate",
    )
    result = trainer.run(_data())
    assert result.state.global_step == 3
    assert result.state.data.batches_seen == 6
    assert result.state.data.samples_seen == 6
    assert len(result.history[0].sample_ids) == 2


def test_gradient_accumulation_weights_partial_microbatches_by_sample_count() -> None:
    program_accumulated = TrainingProgram(
        name="weighted-accumulation",
        max_steps=1,
        gradient_accumulation_steps=2,
        optimizer=OptimizerConfig(max_gradient_norm=1000.0, weight_decay=0.0),
        scheduler=SchedulerConfig(name="constant", minimum_learning_rate_ratio=1.0),
    )
    program_combined = TrainingProgram(
        name="weighted-accumulation",
        max_steps=1,
        gradient_accumulation_steps=1,
        optimizer=OptimizerConfig(max_gradient_norm=1000.0, weight_decay=0.0),
        scheduler=SchedulerConfig(name="constant", minimum_learning_rate_ratio=1.0),
    )
    accumulated_model = torch.nn.Linear(1, 1, bias=False)
    combined_model = torch.nn.Linear(1, 1, bias=False)
    combined_model.load_state_dict(accumulated_model.state_dict())
    accumulated_optimizer = torch.optim.SGD(accumulated_model.parameters(), lr=0.01)
    combined_optimizer = torch.optim.SGD(combined_model.parameters(), lr=0.01)
    first = BatchEnvelope(
        payload=(torch.tensor([[1.0], [2.0]]), torch.tensor([[2.0], [4.0]])),
        sample_ids=("first", "second"),
    )
    second = BatchEnvelope(
        payload=(torch.tensor([[3.0]]), torch.tensor([[6.0]])),
        sample_ids=("third",),
    )
    combined = BatchEnvelope(
        payload=(torch.tensor([[1.0], [2.0], [3.0]]), torch.tensor([[2.0], [4.0], [6.0]])),
        sample_ids=("first", "second", "third"),
    )

    accumulated = Trainer(
        accumulated_model,
        RegressionTask(),
        program_accumulated,
        run_id="accumulated",
        optimizer=accumulated_optimizer,
    ).run([first, second])
    combined_result = Trainer(
        combined_model,
        RegressionTask(),
        program_combined,
        run_id="combined",
        optimizer=combined_optimizer,
    ).run([combined])

    torch.testing.assert_close(accumulated_model.weight, combined_model.weight)
    assert accumulated.history[0].metrics["loss"] == pytest.approx(
        combined_result.history[0].metrics["loss"]
    )
    assert accumulated.state.data.samples_seen == 3


def test_training_is_repeatable_on_the_same_cpu_backend() -> None:
    losses = []
    states = []
    for _ in range(2):
        torch.manual_seed(44)
        trainer = Trainer(
            torch.nn.Linear(1, 1),
            RegressionTask(),
            _program(max_steps=5),
            run_id="repeatable",
        )
        result = trainer.run(_data())
        losses.append([record.metrics["loss"] for record in result.history])
        states.append(
            {name: value.detach().clone() for name, value in trainer.model.state_dict().items()}
        )
    assert losses[0] == losses[1]
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=0.0, atol=0.0)


def test_run_puts_an_eval_mode_model_back_into_training_mode() -> None:
    model = ModeRecordingLinear().eval()
    trainer = Trainer(model, RegressionTask(), _program(max_steps=1), run_id="training-mode")

    trainer.run(_data())

    assert model.training
    assert model.training_modes == [True]


def test_pre_loop_data_failure_marks_the_run_failed() -> None:
    trainer = Trainer(
        torch.nn.Linear(1, 1),
        RegressionTask(),
        _program(max_steps=1),
        run_id="empty-data",
    )

    with pytest.raises(ValueError, match="training data iterable is empty"):
        trainer.run([])

    assert trainer.state.status is RunStatus.FAILED


def test_uninterrupted_and_resumed_steps_have_identical_losses_and_samples(tmp_path) -> None:
    program = TrainingProgram(
        name="resume-parity",
        max_steps=6,
        optimizer=OptimizerConfig(learning_rate=0.03, weight_decay=0.0),
        scheduler=SchedulerConfig(name="constant", minimum_learning_rate_ratio=1.0),
        reproducibility=ReproducibilityConfig(seed=333),
        checkpoint_every_steps=3,
    )
    torch.manual_seed(55)
    uninterrupted = Trainer(
        torch.nn.Sequential(torch.nn.Linear(1, 4), torch.nn.Dropout(0.25), torch.nn.Linear(4, 1)),
        RegressionTask(),
        program,
        run_id="resume-parity",
        checkpoint_manager=DCPCheckpointManager(tmp_path / "full"),
        checkpoint_identity={"model": "regression-v1"},
    )
    full_result = uninterrupted.run(_data())

    torch.manual_seed(999)
    resumed = Trainer(
        torch.nn.Sequential(torch.nn.Linear(1, 4), torch.nn.Dropout(0.25), torch.nn.Linear(4, 1)),
        RegressionTask(),
        program,
        run_id="resume-parity",
        checkpoint_manager=DCPCheckpointManager(tmp_path / "resumed"),
        checkpoint_identity={"model": "regression-v1"},
    )
    resumed.resume(tmp_path / "full" / "step-00000003")
    resumed_result = resumed.run(_data())

    assert [record.sample_ids for record in resumed_result.history] == [
        record.sample_ids for record in full_result.history[3:]
    ]
    assert [record.metrics["loss"] for record in resumed_result.history] == [
        record.metrics["loss"] for record in full_result.history[3:]
    ]
    assert resumed_result.state.data.to_dict() == full_result.state.data.to_dict()
    for name, value in uninterrupted.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0.0, atol=0.0)


def test_checkpoint_resume_rejects_same_schema_with_different_model_config(tmp_path) -> None:
    program = _program(max_steps=1)
    manager = DCPCheckpointManager(tmp_path / "configured")
    source = Trainer(
        ConfiguredLinear(RegressionModelConfig(objective_revision=1)),
        RegressionTask(),
        program,
        run_id="configured-source",
        checkpoint_manager=manager,
    )
    checkpoint = source.save_checkpoint("configured")
    target = Trainer(
        ConfiguredLinear(RegressionModelConfig(objective_revision=2)),
        RegressionTask(),
        program,
        run_id="configured-target",
        checkpoint_manager=manager,
    )

    with pytest.raises(CheckpointCompatibilityError, match="program digest"):
        target.resume(checkpoint.path)


def test_checkpointing_generic_model_requires_explicit_identity(tmp_path) -> None:
    with pytest.raises(ValueError, match="checkpoint_identity is required"):
        Trainer(
            torch.nn.Linear(1, 1),
            RegressionTask(),
            _program(max_steps=1),
            run_id="missing-identity",
            checkpoint_manager=DCPCheckpointManager(tmp_path),
        )
