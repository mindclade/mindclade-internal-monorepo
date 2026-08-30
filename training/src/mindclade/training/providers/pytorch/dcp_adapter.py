"""Thin, version-checked PyTorch Distributed Checkpoint adapter."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import Module
from torch.optim import Optimizer


class DCPUnavailableError(RuntimeError):
    """Raised when the installed PyTorch lacks the required public DCP API."""


@dataclass(frozen=True)
class DCPCapability:
    available: bool
    torch_version: str
    reason: str


def dcp_capability() -> DCPCapability:
    try:
        importlib.import_module("torch.distributed.checkpoint")
        state_dict_module = importlib.import_module("torch.distributed.checkpoint.state_dict")
        stateful_module = importlib.import_module("torch.distributed.checkpoint.stateful")
        required = (
            state_dict_module.get_state_dict,
            state_dict_module.set_state_dict,
            stateful_module.Stateful,
        )
        if not all(callable(value) for value in required):
            raise AttributeError("DCP stateful API objects are not callable")
    except (ImportError, AttributeError) as exc:
        return DCPCapability(False, torch.__version__, str(exc))
    return DCPCapability(True, torch.__version__, "public DCP stateful API available")


class _ModelOptimizerState:
    def __init__(self, model: Module, optimizer: Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> dict[str, Any]:
        from torch.distributed.checkpoint.state_dict import get_state_dict

        model_state, optimizer_state = get_state_dict(self.model, self.optimizer)
        return {"model": model_state, "optimizer": optimizer_state}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        from torch.distributed.checkpoint.state_dict import set_state_dict

        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
        )


def _require_dcp() -> Any:
    capability = dcp_capability()
    if not capability.available:
        raise DCPUnavailableError(
            f"PyTorch DCP is unavailable in torch {capability.torch_version}: {capability.reason}"
        )
    import torch.distributed.checkpoint as dcp

    return dcp


def save_model_optimizer(path: Path, model: Module, optimizer: Optimizer) -> None:
    dcp = _require_dcp()
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    dcp.save(
        {"model_optimizer": _ModelOptimizerState(model, optimizer)},
        checkpoint_id=path,
        no_dist=not distributed,
    )


def load_model_optimizer(path: Path, model: Module, optimizer: Optimizer) -> None:
    dcp = _require_dcp()
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    dcp.load(
        {"model_optimizer": _ModelOptimizerState(model, optimizer)},
        checkpoint_id=path,
        no_dist=not distributed,
    )
