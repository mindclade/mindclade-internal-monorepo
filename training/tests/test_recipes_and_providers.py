from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from mindclade.training.cli.main import main
from mindclade.training.providers import (
    FSDP2UnavailableError,
    apply_fsdp2,
    dcp_capability,
    fsdp2_capability,
)
from mindclade.training.recipes import qualify_overfit, resolve_recipe

RECIPE = (
    Path(__file__).parents[1]
    / "src"
    / "mindclade"
    / "training"
    / "recipes"
    / "smoke"
    / "cpu_contract.yaml"
)


def test_recipe_resolution_is_canonical_and_strict() -> None:
    first = resolve_recipe(RECIPE)
    second = resolve_recipe(RECIPE)
    assert first.sha256 == second.sha256
    assert first.resolved.program.max_steps == 4
    assert first.resolved.dataset.batch_size == 2
    assert len(first.sha256) == 64


def test_cli_inspect_emits_resolved_receipt(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["inspect", str(RECIPE)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["recipe"]["schema_version"] == "v1alpha1"
    assert len(output["recipe_sha256"]) == 64


def test_provider_capabilities_are_explicit() -> None:
    assert dcp_capability().available
    capability = fsdp2_capability()
    assert capability.api_available
    if not capability.ready:
        with pytest.raises(FSDP2UnavailableError, match="FSDP2 unavailable"):
            apply_fsdp2(torch.nn.Sequential(torch.nn.Linear(1, 1)), block_types=(torch.nn.Linear,))


def test_overfit_gate_uses_initial_and_final_ten_step_means() -> None:
    result = qualify_overfit([10.0] * 10 + [8.9] * 10)
    assert result.passed
    assert result.initial_mean == 10.0
    assert result.final_mean == pytest.approx(8.9)
    assert not qualify_overfit([10.0] * 10 + [9.1] * 10).passed
