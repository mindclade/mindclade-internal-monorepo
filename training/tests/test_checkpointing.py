from __future__ import annotations

import random
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
