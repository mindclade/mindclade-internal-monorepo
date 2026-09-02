"""Optimizer and learning-rate schedule configuration."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 1.0e-4
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1.0e-8
    weight_decay: float = 0.1
    max_gradient_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.name != "adamw":
            raise ValueError(f"unsupported optimizer {self.name!r}; supported: adamw")
        for name in (
            "learning_rate",
            "beta1",
            "beta2",
            "epsilon",
            "weight_decay",
            "max_gradient_norm",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not (0.0 <= self.beta1 < 1.0 and 0.0 <= self.beta2 < 1.0):
            raise ValueError("AdamW betas must be in [0, 1)")
        if self.epsilon <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("epsilon must be positive and weight_decay non-negative")
        if self.max_gradient_norm <= 0.0:
            raise ValueError("max_gradient_norm must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SchedulerConfig:
    name: str = "warmup_cosine"
    warmup_ratio: float = 0.02
    minimum_learning_rate_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.name not in {"warmup_cosine", "constant"}:
            raise ValueError(
                f"unsupported scheduler {self.name!r}; supported: constant, warmup_cosine"
            )
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not 0.0 <= self.minimum_learning_rate_ratio <= 1.0:
            raise ValueError("minimum_learning_rate_ratio must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_optimizer(parameters: Iterable[Tensor], config: OptimizerConfig) -> Optimizer:
    return AdamW(
        parameters,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )


def build_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
    *,
    total_steps: int,
) -> LRScheduler:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if config.name == "constant":
        return LambdaLR(optimizer, lambda _: 1.0)

    warmup_steps = int(total_steps * config.warmup_ratio)

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / remaining, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = config.minimum_learning_rate_ratio
        return floor + (1.0 - floor) * cosine

    return LambdaLR(optimizer, multiplier)


def optimizer_config_from_mapping(value: Mapping[str, Any]) -> OptimizerConfig:
    return OptimizerConfig(**dict(value))


def scheduler_config_from_mapping(value: Mapping[str, Any]) -> SchedulerConfig:
    return SchedulerConfig(**dict(value))
