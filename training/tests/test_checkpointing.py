from __future__ import annotations

import random
import sys
import time
from datetime import timedelta
from multiprocessing.process import BaseProcess
from pathlib import Path

import pytest
import torch
from mindclade.training.api import (
    CheckpointError,
    CheckpointIntegrityError,
    RunStatus,
    TrainerState,
)
from mindclade.training.checkpointing import DCPCheckpointManager, verify_checkpoint


def _duplicate_checkpoint_rank(
    rank: int,
    world_size: int,
    checkpoint_root: str,
    result_root: str,
    rendezvous_path: str,
) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=10),
    )
    try:
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        try:
            DCPCheckpointManager(Path(checkpoint_root)).save(
                "duplicate",
                model=model,
                optimizer=optimizer,
                trainer_state=TrainerState(run_id="distributed-duplicate"),
                program={"name": "distributed-duplicate"},
            )
        except CheckpointError as exc:
            result = f"{type(exc).__name__}: {exc}"
        else:
            result = "unexpected success"
        Path(result_root, f"rank-{rank}.txt").write_text(result, encoding="utf-8")
    finally:
        torch.distributed.destroy_process_group()


def _successful_checkpoint_rank(
    rank: int,
    world_size: int,
    checkpoint_root: str,
    result_root: str,
    rendezvous_path: str,
) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    try:
        torch.manual_seed(41)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        ref = DCPCheckpointManager(Path(checkpoint_root)).save(
            "successful",
            model=model,
            optimizer=optimizer,
            trainer_state=TrainerState(run_id="distributed-success", global_step=2),
            program={"name": "distributed-success"},
        )
        Path(result_root, f"rank-{rank}.txt").write_text(
            f"{ref.path.name}\n{ref.global_step}\n{ref.manifest_sha256}\n",
            encoding="utf-8",
        )
    finally:
        torch.distributed.destroy_process_group()


def _run_spawned(processes: list[BaseProcess], *, timeout_seconds: float) -> None:
    # Pytest's importlib mode does not place the repository root on sys.path.
    # The spawn start method must be able to re-import targets from this module
    # in every child interpreter.
    original_sys_path = sys.path[:]
    repository_root = str(Path(__file__).resolve().parents[2])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    try:
        for process in processes:
            process.start()
    finally:
        sys.path[:] = original_sys_path

    deadline = time.monotonic() + timeout_seconds
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()
        process.join(timeout=2.0)
    assert not alive, "distributed checkpoint operation did not terminate on every rank"


def _initialized_model_optimizer() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.SiLU(), torch.nn.Linear(4, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    output = model(torch.ones(2, 3)).square().mean()
    output.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def test_dcp_checkpoint_roundtrip_restores_model_optimizer_rng_and_state(tmp_path: Path) -> None:
    torch.manual_seed(123)
    random.seed(123)
    model, optimizer = _initialized_model_optimizer()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    manager = DCPCheckpointManager(tmp_path)
    state = TrainerState(
        run_id="roundtrip",
        global_step=3,
        optimizer_steps=3,
        status=RunStatus.RUNNING,
    )
    expected_output = model(torch.tensor([[1.0, 2.0, 3.0]])).detach().clone()
    ref = manager.save(
        "step-00000003",
        model=model,
        optimizer=optimizer,
        trainer_state=state,
        program={"name": "roundtrip", "version": "v1alpha1"},
        scheduler=scheduler,
    )
    expected_torch_random = torch.rand(3)
    expected_python_random = random.random()

    for parameter in model.parameters():
        parameter.data.zero_()
    torch.manual_seed(999)
    random.seed(999)
    restored = manager.restore(
        ref.path,
        model=model,
        optimizer=optimizer,
        program={"name": "roundtrip", "version": "v1alpha1"},
        scheduler=scheduler,
    )
    torch.testing.assert_close(
        model(torch.tensor([[1.0, 2.0, 3.0]])).detach(), expected_output, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(torch.rand(3), expected_torch_random, rtol=0.0, atol=0.0)
    assert random.random() == expected_python_random
    assert restored.trainer_state.global_step == 3
    assert restored.ref.manifest_sha256 == ref.manifest_sha256
    assert verify_checkpoint(ref.path).global_step == 3


def test_checkpoint_is_immutable_and_tamper_evident(tmp_path: Path) -> None:
    model, optimizer = _initialized_model_optimizer()
    manager = DCPCheckpointManager(tmp_path)
    state = TrainerState(run_id="integrity", global_step=1)
    ref = manager.save(
        "step-00000001",
        model=model,
        optimizer=optimizer,
        trainer_state=state,
        program={"name": "integrity"},
    )
    with pytest.raises(CheckpointError, match="will not be overwritten"):
        manager.save(
            "step-00000001",
            model=model,
            optimizer=optimizer,
            trainer_state=state,
            program={"name": "integrity"},
        )
    payload = next((ref.path / "dcp").glob("*.distcp"))
    with payload.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(CheckpointIntegrityError, match="integrity mismatch"):
        verify_checkpoint(ref.path)


def test_incomplete_staging_directories_are_not_resumable(tmp_path: Path) -> None:
    incomplete = tmp_path / "step-incomplete"
    incomplete.mkdir()
    manager = DCPCheckpointManager(tmp_path)
    assert manager.list_committed() == ()


def test_duplicate_checkpoint_setup_failure_reaches_every_rank(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    result_root = tmp_path / "results"
    checkpoint_root.mkdir()
    result_root.mkdir()
    (checkpoint_root / "duplicate").mkdir()
    rendezvous_path = tmp_path / "gloo-rendezvous"
    world_size = 2
    context = torch.multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_duplicate_checkpoint_rank,
            args=(
                rank,
                world_size,
                str(checkpoint_root),
                str(result_root),
                str(rendezvous_path),
            ),
        )
        for rank in range(world_size)
    ]
    _run_spawned(processes, timeout_seconds=15.0)
    assert [process.exitcode for process in processes] == [0, 0]
    results = [
        (result_root / f"rank-{rank}.txt").read_text(encoding="utf-8") for rank in range(world_size)
    ]
    assert results[0] == results[1]
    assert "checkpoint setup failed on rank 0" in results[0]
    assert "will not be overwritten" in results[0]


def test_distributed_checkpoint_successfully_commits_every_rank(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    result_root = tmp_path / "results"
    checkpoint_root.mkdir()
    result_root.mkdir()
    rendezvous_path = tmp_path / "gloo-rendezvous"
    world_size = 2
    context = torch.multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_successful_checkpoint_rank,
            args=(
                rank,
                world_size,
                str(checkpoint_root),
                str(result_root),
                str(rendezvous_path),
            ),
        )
        for rank in range(world_size)
    ]

    _run_spawned(processes, timeout_seconds=30.0)

    assert [process.exitcode for process in processes] == [0, 0]
    results = [
        (result_root / f"rank-{rank}.txt").read_text(encoding="utf-8") for rank in range(world_size)
    ]
    assert results[0] == results[1]
    name, step, manifest_digest = results[0].splitlines()
    assert name == "successful"
    assert step == "2"
    assert len(manifest_digest) == 64
    int(manifest_digest, 16)
    manifest = verify_checkpoint(checkpoint_root / "successful")
    assert manifest.world_size == world_size
    assert manifest.global_step == 2
    assert manifest.sha256 == manifest_digest
    assert sorted((checkpoint_root / "successful" / "rank-state").glob("rank-*.json")) == [
        checkpoint_root / "successful" / "rank-state" / "rank-00000.json",
        checkpoint_root / "successful" / "rank-state" / "rank-00001.json",
    ]
