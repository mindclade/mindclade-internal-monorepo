"""Explicit execution-topology contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class ParallelismMode(StrEnum):
    SINGLE_PROCESS = "single_process"
    FSDP2 = "fsdp2"


@dataclass(frozen=True)
class ParallelismConfig:
    mode: ParallelismMode = ParallelismMode.SINGLE_PROCESS
    world_size: int = 1
    reshard_after_forward: bool = True

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.mode is ParallelismMode.SINGLE_PROCESS and self.world_size != 1:
            raise ValueError("single_process parallelism requires world_size=1")
        if self.mode is ParallelismMode.FSDP2 and self.world_size < 2:
            raise ValueError("fsdp2 parallelism requires world_size>=2")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["mode"] = self.mode.value
        return value
