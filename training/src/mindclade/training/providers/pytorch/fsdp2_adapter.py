"""Bottom-up FSDP2 application with explicit capability failures."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import Module


class FSDP2UnavailableError(RuntimeError):
    """Raised before mutation when FSDP2 prerequisites are not satisfied."""


@dataclass(frozen=True)
class FSDP2Capability:
    api_available: bool
    distributed_initialized: bool
    cuda_available: bool
    torch_version: str
    reason: str

    @property
    def ready(self) -> bool:
        return self.api_available and self.distributed_initialized and self.cuda_available


def fsdp2_capability() -> FSDP2Capability:
    try:
        fsdp_module = importlib.import_module("torch.distributed.fsdp")
        if not callable(fsdp_module.fully_shard):
            raise AttributeError("FSDP2 fully_shard is not callable")
    except (ImportError, AttributeError) as exc:
        return FSDP2Capability(
            False,
            torch.distributed.is_available() and torch.distributed.is_initialized(),
            torch.cuda.is_available(),
            torch.__version__,
            f"public torch.distributed.fsdp.fully_shard API unavailable: {exc}",
        )
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    cuda = torch.cuda.is_available()
    reasons = []
    if not distributed:
        reasons.append("torch.distributed process group is not initialized")
    if not cuda:
        reasons.append("CUDA is unavailable")
    return FSDP2Capability(
        True, distributed, cuda, torch.__version__, "; ".join(reasons) or "ready"
    )


def apply_fsdp2(
    model: Module,
    *,
    block_types: Sequence[type[Module]] = (),
    block_predicate: Callable[[str, Module], bool] | None = None,
    mesh: Any | None = None,
    reshard_after_forward: bool = True,
) -> Module:
    """Shard selected blocks bottom-up, then shard the model root.

    The caller supplies CladeFold block types or a predicate so this provider
    layer does not import architecture internals. At least one block must match;
    silently root-sharding an unintended architecture is rejected.
    """

    capability = fsdp2_capability()
    if not capability.ready:
        raise FSDP2UnavailableError(
            f"FSDP2 unavailable with torch {capability.torch_version}: {capability.reason}"
        )
    if not block_types and block_predicate is None:
        raise ValueError("block_types or block_predicate is required for bottom-up FSDP2")

    from torch.distributed.fsdp import fully_shard

    selected = []
    for name, module in model.named_modules():
        if module is model:
            continue
        matches_type = bool(block_types) and isinstance(module, tuple(block_types))
        matches_predicate = block_predicate is not None and block_predicate(name, module)
        if matches_type or matches_predicate:
            selected.append((name.count("."), name, module))
    if not selected:
        raise ValueError("FSDP2 block selector matched no model submodules")
    for _, _, module in sorted(selected, key=lambda item: (-item[0], item[1])):
        fully_shard(module, mesh=mesh, reshard_after_forward=reshard_after_forward)
    fully_shard(model, mesh=mesh, reshard_after_forward=reshard_after_forward)
    return model
