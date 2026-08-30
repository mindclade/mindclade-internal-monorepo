"""Deterministic CPU fixtures for model contract tests."""

from __future__ import annotations

import pytest
import torch

from mindclade.models import CladeFoldBatch, CladeFoldConfig, CladeFoldModel


def make_batch(*, labels: bool = True, batch_size: int = 2) -> CladeFoldBatch:
    if batch_size not in {1, 2}:
        raise ValueError("fixture supports batch_size one or two")
    b, a, e = 2, 6, 4
    generator = torch.Generator().manual_seed(1234)
    target = torch.randn((b, a, 3), generator=generator)
    noisy = target + 0.5 * torch.randn((b, a, 3), generator=generator)
    atom_mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    value = CladeFoldBatch(
        token_type=torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]]),
        molecule_type=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        chain_id=torch.tensor([[0, 0, 1, 0], [0, 0, 0, 0]]),
        position_id=torch.tensor([[0, 1, 0, 0], [0, 1, 0, 0]]),
        token_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool),
        anchor_atom_index=torch.tensor([[0, 2, 4, -1], [0, 2, -1, -1]]),
        atomic_number=torch.tensor([[6, 7, 6, 8, 16, 0], [6, 7, 8, 0, 0, 0]]),
        formal_charge=torch.zeros((b, a), dtype=torch.long),
        chirality=torch.zeros((b, a), dtype=torch.long),
        aromatic_mask=torch.zeros((b, a), dtype=torch.bool),
        atom_to_token=torch.tensor([[0, 0, 1, 1, 2, -1], [0, 0, 1, -1, -1, -1]]),
        atom_mask=atom_mask,
        bond_indices=torch.tensor(
            [[[0, 1], [1, 2], [2, 3], [-1, -1]], [[0, 1], [1, 2], [-1, -1], [-1, -1]]]
        ),
        bond_type=torch.tensor([[1, 1, 2, 0], [1, 1, 0, 0]]),
        bond_stereo=torch.zeros((b, e), dtype=torch.long),
        bond_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool),
        noisy_coordinates=noisy,
        diffusion_time=torch.tensor([0.4, 0.7]),
        target_coordinates=target if labels else None,
        target_mask=atom_mask if labels else None,
    )
    if batch_size == 2:
        return value
    values = {}
    for name, tensor in value.tensor_items():
        values[name] = tensor[:1]
    for name in ("target_coordinates", "target_mask"):
        if getattr(value, name) is None:
            values[name] = None
    return CladeFoldBatch(**values)


def micro_config(**overrides):
    values = {
        "token_dim": 32,
        "pair_dim": 16,
        "atom_dim": 32,
        "time_dim": 16,
        "pairformer_blocks": 1,
        "denoiser_blocks": 1,
        "token_heads": 4,
        "triangle_heads": 4,
        "atom_heads": 4,
        "transition_multiplier": 2,
        "outer_product_dim": 8,
        "radial_basis_functions": 8,
        "atom_knn": 4,
        "distogram_bins": 16,
        "confidence_bins": 10,
        "default_sampling_steps": 2,
        "dropout": 0.0,
    }
    values.update(overrides)
    return CladeFoldConfig.tiny(**values)


@pytest.fixture
def cladefold_batch() -> CladeFoldBatch:
    return make_batch()


@pytest.fixture
def cladefold_model() -> CladeFoldModel:
    torch.manual_seed(11)
    return CladeFoldModel(micro_config())
