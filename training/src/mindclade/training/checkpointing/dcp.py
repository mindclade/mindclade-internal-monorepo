"""Integrity-checked DCP save and resume coordinator."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from mindclade.training.api.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointRef,
)
from mindclade.training.api.reproducibility import (
    RNGState,
    capture_rng_state,
    restore_rng_state,
)
from mindclade.training.api.state import TrainerState
from mindclade.training.providers.pytorch.dcp_adapter import (
    load_model_optimizer,
    save_model_optimizer,
)
from torch.nn import Module
from torch.optim import Optimizer

from .atomic_commit import AtomicCheckpointWriter
from .manifest import (
    CheckpointManifest,
    digest_mapping,
    digest_model_schema,
    inventory_files,
    verify_checkpoint,
)

RANK_STATE_DIRECTORY = "rank-state"
DCP_DIRECTORY = "dcp"


def _distributed_rank_world() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"state value of type {type(value).__name__} is not JSON serializable")


def _model_schema_digest(model: Module) -> str:
    state = model.state_dict()
    return digest_model_schema(
        (name, tuple(value.shape), str(value.dtype)) for name, value in state.items()
    )


def _rank_state_path(checkpoint: Path, rank: int) -> Path:
    return checkpoint / RANK_STATE_DIRECTORY / f"rank-{rank:05d}.json"


@dataclass(frozen=True)
class RestoredCheckpoint:
    ref: CheckpointRef
    trainer_state: TrainerState
    saved_world_size: int
    current_world_size: int

    @property
    def resharded(self) -> bool:
        return self.saved_world_size != self.current_world_size


class DCPCheckpointManager:
    """Publishes DCP shards first and a canonical commit manifest last.

    A shared POSIX-like filesystem is required for multi-rank execution. DCP
    owns tensor resharding. Rank-local RNG state is restored exactly when the
    world size is unchanged; a deterministic saved-rank mapping is used during
    an explicitly allowed reshard.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def save(
        self,
        checkpoint_id: str,
        *,
        model: Module,
        optimizer: Optimizer,
        trainer_state: TrainerState,
        program: Mapping[str, Any],
        scheduler: Any | None = None,
        scaler: Any | None = None,
    ) -> CheckpointRef:
        rank, world_size = _distributed_rank_world()
        writer: AtomicCheckpointWriter | None = None
        staging: Path | None = None
        if rank == 0:
            writer = AtomicCheckpointWriter(self.root, checkpoint_id)
            writer.__enter__()
            staging = writer.staging
        if world_size > 1:
            values = [str(staging) if staging is not None else ""]
            torch.distributed.broadcast_object_list(values, src=0)
            staging = Path(values[0])
        if staging is None:
            raise RuntimeError("rank zero did not publish a checkpoint staging path")

        try:
            save_model_optimizer(staging / DCP_DIRECTORY, model, optimizer)
            rank_state_dir = staging / RANK_STATE_DIRECTORY
            rank_state_dir.mkdir(parents=True, exist_ok=True)
            state_payload = {
                "rng": capture_rng_state().to_dict(),
                "scaler": None if scaler is None else _json_safe(scaler.state_dict()),
                "scheduler": None if scheduler is None else _json_safe(scheduler.state_dict()),
                "trainer": trainer_state.to_dict(),
            }
            state_path = _rank_state_path(staging, rank)
            with state_path.open("x", encoding="utf-8") as handle:
                json.dump(
                    state_payload, handle, allow_nan=False, sort_keys=True, separators=(",", ":")
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            _barrier()

            manifest: CheckpointManifest | None = None
            target: Path | None = None
            if rank == 0:
                manifest = CheckpointManifest(
                    checkpoint_id=checkpoint_id,
                    created_at=datetime.now(UTC).isoformat(),
                    global_step=trainer_state.global_step,
                    world_size=world_size,
                    torch_version=torch.__version__,
                    model_schema_sha256=_model_schema_digest(model),
                    program_sha256=digest_mapping(dict(program)),
                    files=inventory_files(staging),
                )
                if writer is None:
                    raise RuntimeError("rank zero checkpoint writer is unavailable")
                target = writer.commit(manifest)
            if world_size > 1:
                values = [
                    str(target) if target is not None else "",
                    manifest.sha256 if manifest is not None else "",
                ]
                torch.distributed.broadcast_object_list(values, src=0)
                target = Path(values[0])
                manifest_sha = values[1]
            else:
                if target is None or manifest is None:
                    raise RuntimeError("single-process checkpoint commit did not complete")
                manifest_sha = manifest.sha256
            _barrier()
            return CheckpointRef(
                checkpoint_id=checkpoint_id,
                path=target,
                global_step=trainer_state.global_step,
                manifest_sha256=manifest_sha,
            )
        except BaseException:
            if writer is not None:
                writer.__exit__(None, None, None)
            raise
        finally:
            if writer is not None and writer.staging is not None:
                writer.__exit__(None, None, None)

    def restore(
        self,
        checkpoint: Path,
        *,
        model: Module,
        optimizer: Optimizer,
        program: Mapping[str, Any],
        scheduler: Any | None = None,
        scaler: Any | None = None,
        allow_reshard: bool = False,
    ) -> RestoredCheckpoint:
        manifest = verify_checkpoint(checkpoint)
        rank, world_size = _distributed_rank_world()
        actual_program_digest = digest_mapping(dict(program))
        if manifest.program_sha256 != actual_program_digest:
            raise CheckpointCompatibilityError(
                "resolved training program digest does not match checkpoint"
            )
        actual_model_digest = _model_schema_digest(model)
        if manifest.model_schema_sha256 != actual_model_digest:
            raise CheckpointCompatibilityError(
                "model parameter/buffer schema does not match checkpoint"
            )
        if manifest.world_size != world_size and not allow_reshard:
            raise CheckpointCompatibilityError(
                f"checkpoint world_size={manifest.world_size}, current world_size={world_size}; "
                "pass allow_reshard=True to authorize DCP resharding"
            )

        load_model_optimizer(checkpoint / DCP_DIRECTORY, model, optimizer)
        saved_rank = rank if rank < manifest.world_size else rank % manifest.world_size
        state_path = _rank_state_path(checkpoint, saved_rank)
        try:
            state_payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointCompatibilityError(
                f"cannot read rank state {state_path}: {exc}"
            ) from exc
        trainer_state = TrainerState.from_dict(dict(state_payload["trainer"]))
        if scheduler is not None:
            scheduler_state = state_payload.get("scheduler")
            if scheduler_state is None:
                raise CheckpointCompatibilityError("checkpoint has no scheduler state")
            scheduler.load_state_dict(scheduler_state)
        if scaler is not None:
            scaler_state = state_payload.get("scaler")
            if scaler_state is None:
                raise CheckpointCompatibilityError("checkpoint has no gradient scaler state")
            scaler.load_state_dict(scaler_state)
        restore_rng_state(RNGState.from_dict(dict(state_payload["rng"])))
        return RestoredCheckpoint(
            ref=CheckpointRef(
                checkpoint_id=manifest.checkpoint_id,
                path=checkpoint,
                global_step=manifest.global_step,
                manifest_sha256=manifest.sha256,
            ),
            trainer_state=trainer_state,
            saved_world_size=manifest.world_size,
            current_world_size=world_size,
        )

    def list_committed(self) -> tuple[CheckpointRef, ...]:
        if not self.root.exists():
            return ()
        refs = []
        for path in sorted(self.root.iterdir()):
            if not path.is_dir() or path.name.startswith("."):
                continue
            try:
                manifest = verify_checkpoint(path)
            except Exception:
                continue
            refs.append(
                CheckpointRef(
                    checkpoint_id=manifest.checkpoint_id,
                    path=path,
                    global_step=manifest.global_step,
                    manifest_sha256=manifest.sha256,
                )
            )
        return tuple(refs)
