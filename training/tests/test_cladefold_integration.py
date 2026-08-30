from __future__ import annotations

import pytest
import torch
from mindclade.training.core.data import (
    DeterministicSyntheticDataset,
    SyntheticSplit,
    collate_cladefold_examples,
)
from mindclade.training.tasks import MultitaskDiffusionTask


def test_tiny_cladefold_forward_backward_contract() -> None:
    try:
        from mindclade.models import CladeFoldConfig, CladeFoldModel
    except (ImportError, SyntaxError) as exc:
        pytest.skip(f"mindclade-models is not importable in this source-only environment: {exc}")
    example = DeterministicSyntheticDataset(
        SyntheticSplit.TRAIN,
        seed=17,
        token_count=4,
        atom_count=8,
    )[0]
    batch = collate_cladefold_examples([example])
    batch.payload.validate()
    model = CladeFoldModel(CladeFoldConfig.tiny(dropout=0.0))
    output = MultitaskDiffusionTask().forward(model, batch)
    report = MultitaskDiffusionTask().compute_loss(output, batch)
    assert output.predicted_noise.shape == (1, 8, 3)
    assert report.total.dtype is torch.float32
    report.total.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
