"""Reproducibility configuration and serializable random-number state."""

from __future__ import annotations

import base64
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import torch


def _lists_to_tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_lists_to_tuples(item) for item in value)
    return value


def _encode_rng_tensor(value: torch.Tensor) -> str:
    # RNG state is only a few KiB; using ``bytes`` keeps NumPy out of the core
    # dependency contract and preserves every uint8 value exactly.
    raw = bytes(value.detach().cpu().contiguous().tolist())
    return base64.b64encode(raw).decode("ascii")


def _decode_rng_tensor(value: str) -> torch.Tensor:
    raw = base64.b64decode(value.encode("ascii"), validate=True)
    return torch.tensor(list(raw), dtype=torch.uint8)


@dataclass(frozen=True)
class ReproducibilityConfig:
    seed: int = 17
    deterministic_algorithms: bool = True
    warn_only: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.seed < 2**63:
            raise ValueError("seed must be in [0, 2**63)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RNGState:
    python: tuple[Any, ...]
    torch_cpu: str
    torch_cuda: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "python": list(self.python),
            "torch_cpu": self.torch_cpu,
            "torch_cuda": list(self.torch_cuda),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RNGState:
        return cls(
            python=_lists_to_tuples(value["python"]),
            torch_cpu=str(value["torch_cpu"]),
            torch_cuda=tuple(str(item) for item in value.get("torch_cuda", [])),
        )


def seed_everything(config: ReproducibilityConfig) -> None:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(
        config.deterministic_algorithms,
        warn_only=config.warn_only,
    )
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = config.deterministic_algorithms


def capture_rng_state() -> RNGState:
    cuda_states: Iterable[torch.Tensor]
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else ()
    return RNGState(
        python=random.getstate(),
        torch_cpu=_encode_rng_tensor(torch.get_rng_state()),
        torch_cuda=tuple(_encode_rng_tensor(state) for state in cuda_states),
    )


def restore_rng_state(state: RNGState) -> None:
    random.setstate(state.python)
    torch.set_rng_state(_decode_rng_tensor(state.torch_cpu))
    if state.torch_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(state.torch_cuda) != torch.cuda.device_count():
            raise RuntimeError(
                "checkpoint CUDA RNG state count does not match visible CUDA device count"
            )
        torch.cuda.set_rng_state_all([_decode_rng_tensor(value) for value in state.torch_cuda])
