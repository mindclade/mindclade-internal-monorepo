"""Reference trainer shared by eager and provider-backed execution."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Sized
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
)

import torch
from mindclade.models.common.configuration import ModelConfig
from mindclade.training.api import (
    RunStatus,
    TrainerState,
    TrainingProgram,
    TrainingTask,
    build_optimizer,
    build_scheduler,
    require_finite_loss,
    seed_everything,
)
from mindclade.training.checkpointing import DCPCheckpointManager, RestoredCheckpoint
from mindclade.training.execution import SingleProcessEngine
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class EmptyTrainingDataError(ValueError):
    """Raised when an iterable produces no microbatches."""


@dataclass(frozen=True)
class StepRecord:
    global_step: int
    metrics: Mapping[str, float]
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


@dataclass(frozen=True)
class TrainingResult:
    state: TrainerState
    history: tuple[StepRecord, ...]


class Trainer:
    """Correctness-first eager trainer with deterministic progress commits."""

    def __init__(
        self,
        model: Module,
        task: TrainingTask[Any, Any],
        program: TrainingProgram,
        *,
        run_id: str,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        engine: SingleProcessEngine | None = None,
        checkpoint_manager: DCPCheckpointManager | None = None,
        checkpoint_identity: Mapping[str, Any] | None = None,
        on_step: Callable[[StepRecord], None] | None = None,
    ) -> None:
        self.program = program
        self.task = task
        self.engine = engine or SingleProcessEngine.create(program.precision)
        self.model = self.engine.prepare_model(model)
        self.optimizer = optimizer or build_optimizer(self.model.parameters(), program.optimizer)
        self.scheduler = scheduler or build_scheduler(
            self.optimizer,
            program.scheduler,
            total_steps=program.max_steps,
        )
        self.checkpoint_manager = checkpoint_manager
        model_config = getattr(self.model, "config", None)
        if (
            checkpoint_manager is not None
            and checkpoint_identity is None
            and not isinstance(model_config, ModelConfig)
        ):
            raise ValueError(
                "checkpoint_identity is required for models without a Mindclade ModelConfig"
            )
        effective_checkpoint_identity: dict[str, Any] = {"program": program.to_dict()}
        if isinstance(model_config, ModelConfig):
            model_config.validate()
            effective_checkpoint_identity["model_config"] = model_config.to_dict()
        if checkpoint_identity is not None:
            effective_checkpoint_identity["caller"] = dict(checkpoint_identity)
        self.checkpoint_identity = effective_checkpoint_identity
        self.on_step = on_step
        self.state = TrainerState(run_id=run_id)
        self._history: list[StepRecord] = []
        self._data_length: int | None = None

    @property
    def scaler(self) -> Any | None:
        return self.engine.scaler

    def _next_batch(
        self, data: Iterable[Any], iterator: Iterator[Any]
    ) -> tuple[Any, Iterator[Any]]:
        try:
            return next(iterator), iterator
        except StopIteration:
            if self._data_length is None:
                self.state.data.epoch += 1
            else:
                self.state.data.epoch = self.state.data.batches_seen // self._data_length
            set_epoch = getattr(data, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(self.state.data.epoch)
            replacement = iter(data)
            try:
                return next(replacement), replacement
            except StopIteration as exc:
                raise EmptyTrainingDataError(
                    "training data iterable is empty or not reiterable"
                ) from exc

    @staticmethod
    def _sample_ids(batch: Any) -> tuple[str, ...]:
        values = getattr(batch, "sample_ids", ())
        return tuple(str(value) for value in values)

    def _train_update(self, batches: Sequence[Any]) -> StepRecord:
        if len(batches) != self.program.gradient_accumulation_steps:
            raise ValueError("microbatch count must equal gradient_accumulation_steps")
        self.optimizer.zero_grad(set_to_none=True)
        aggregate: dict[str, float] = {}
        sample_ids: list[str] = []
        microbatch_ids = [self._sample_ids(batch) for batch in batches]
        sample_counts = [len(ids) if ids else 1 for ids in microbatch_ids]
        total_samples = float(sum(sample_counts))

        for raw_batch, ids, sample_count in zip(
            batches, microbatch_ids, sample_counts, strict=True
        ):
            contribution = float(sample_count) / total_samples
            batch = self.engine.prepare_batch(raw_batch)
            self.task.validate_batch(batch)
            with self.engine.precision.autocast():
                output = self.task.forward(self.model, batch)
                report = self.task.compute_loss(output, batch)
            require_finite_loss(report)
            self.engine.backward(report.total * contribution)
            for name, value in report.detached_metrics().items():
                aggregate[name] = aggregate.get(name, 0.0) + value * contribution
            sample_ids.extend(ids)
            self.state.data.record(ids)
            if (
                self._data_length is not None
                and self.state.data.batches_seen % self._data_length == 0
            ):
                self.state.data.epoch = self.state.data.batches_seen // self._data_length

        self.engine.unscale_(self.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.program.optimizer.max_gradient_norm,
            error_if_nonfinite=True,
        )
        self.engine.optimizer_step(self.optimizer)
        self.scheduler.step()
        self.state.global_step += 1
        self.state.optimizer_steps += 1
        self.state.last_loss = aggregate["loss"]
        aggregate["gradient_norm"] = float(gradient_norm.detach().to(dtype=torch.float32).cpu())
        aggregate["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        record = StepRecord(
            global_step=self.state.global_step,
            metrics=aggregate,
            sample_ids=tuple(sample_ids),
        )
        self._history.append(record)
        if self.on_step is not None:
            self.on_step(record)
        return record

    def save_checkpoint(self, checkpoint_id: str | None = None) -> Any:
        if self.checkpoint_manager is None:
            raise RuntimeError("no checkpoint manager was configured")
        resolved_id = checkpoint_id or f"step-{self.state.global_step:08d}"
        return self.checkpoint_manager.save(
            resolved_id,
            model=self.model,
            optimizer=self.optimizer,
            trainer_state=self.state,
            program=self.checkpoint_identity,
            scheduler=self.scheduler,
            scaler=self.scaler,
        )

    def resume(self, checkpoint: Path, *, allow_reshard: bool = False) -> RestoredCheckpoint:
        if self.checkpoint_manager is None:
            raise RuntimeError("no checkpoint manager was configured")
        restored = self.checkpoint_manager.restore(
            checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            program=self.checkpoint_identity,
            scheduler=self.scheduler,
            scaler=self.scaler,
            allow_reshard=allow_reshard,
        )
        self.state = restored.trainer_state
        self.state.attempt += 1
        self.state.status = RunStatus.RUNNING
        return restored

    def run(self, data: Iterable[Any]) -> TrainingResult:
        created = self.state.status is RunStatus.CREATED
        self.state.status = RunStatus.RUNNING
        try:
            if created:
                seed_everything(self.program.reproducibility)
            self.model.train()
            if isinstance(data, Sized):
                self._data_length = len(data)
                if self._data_length <= 0:
                    raise EmptyTrainingDataError("training data iterable is empty")
                self.state.data.epoch = self.state.data.batches_seen // self._data_length
                offset = self.state.data.batches_seen % self._data_length
            else:
                self._data_length = None
                if self.state.data.batches_seen:
                    raise ValueError("resuming requires a sized, reiterable training data source")
                offset = 0
            set_epoch = getattr(data, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(self.state.data.epoch)
            iterator = iter(data)
            for _ in range(offset):
                try:
                    next(iterator)
                except StopIteration as exc:
                    raise ValueError(
                        "saved data progress exceeds the current epoch length"
                    ) from exc
            while self.state.global_step < self.program.max_steps:
                microbatches = []
                for _ in range(self.program.gradient_accumulation_steps):
                    batch, iterator = self._next_batch(data, iterator)
                    microbatches.append(batch)
                self._train_update(microbatches)
                if (
                    self.checkpoint_manager is not None
                    and self.state.global_step % self.program.checkpoint_every_steps == 0
                ):
                    self.save_checkpoint()
            self.state.status = RunStatus.COMPLETED
        except BaseException:
            self.state.status = RunStatus.FAILED
            raise
        return TrainingResult(state=self.state, history=tuple(self._history))
