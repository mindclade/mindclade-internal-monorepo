"""Serializable logical trainer and data progress state."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DataProgress:
    epoch: int = 0
    batches_seen: int = 0
    samples_seen: int = 0
    sample_ids: tuple[str, ...] = field(default_factory=tuple)

    def record(self, sample_ids: Iterable[str]) -> None:
        ids = tuple(str(value) for value in sample_ids)
        self.batches_seen += 1
        self.samples_seen += len(ids)
        self.sample_ids = ids

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["sample_ids"] = list(self.sample_ids)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DataProgress:
        return cls(
            epoch=int(value["epoch"]),
            batches_seen=int(value["batches_seen"]),
            samples_seen=int(value["samples_seen"]),
            sample_ids=tuple(str(item) for item in value.get("sample_ids", [])),
        )


@dataclass
class TrainerState:
    run_id: str
    attempt: int = 0
    global_step: int = 0
    optimizer_steps: int = 0
    status: RunStatus = RunStatus.CREATED
    last_loss: float | None = None
    data: DataProgress = field(default_factory=DataProgress)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if min(self.attempt, self.global_step, self.optimizer_steps) < 0:
            raise ValueError("attempt and step counts must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "attempt": self.attempt,
            "global_step": self.global_step,
            "optimizer_steps": self.optimizer_steps,
            "status": self.status.value,
            "last_loss": self.last_loss,
            "data": self.data.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrainerState:
        return cls(
            run_id=str(value["run_id"]),
            attempt=int(value.get("attempt", 0)),
            global_step=int(value.get("global_step", 0)),
            optimizer_steps=int(value.get("optimizer_steps", 0)),
            status=RunStatus(str(value.get("status", RunStatus.CREATED.value))),
            last_loss=(None if value.get("last_loss") is None else float(value["last_loss"])),
            data=DataProgress.from_dict(dict(value.get("data", {}))),
        )
