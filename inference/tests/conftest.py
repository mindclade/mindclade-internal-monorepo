from __future__ import annotations

import pytest
import torch
from mindclade.inference.contracts.request_contract import InferenceRequest


def sha(character: str = "a") -> str:
    return "sha256:" + character * 64


def cladefold_inputs(*, forward: bool = False) -> dict[str, torch.Tensor]:
    inputs = {
        "token_type": torch.tensor([[2, 3]], dtype=torch.int64),
        "molecule_type": torch.tensor([[0, 0]], dtype=torch.int64),
        "chain_id": torch.tensor([[0, 0]], dtype=torch.int64),
        "position_id": torch.tensor([[0, 1]], dtype=torch.int64),
        "token_mask": torch.tensor([[True, True]]),
        "anchor_atom_index": torch.tensor([[0, 1]], dtype=torch.int64),
        "atomic_number": torch.tensor([[6, 6, 8]], dtype=torch.int64),
        "formal_charge": torch.tensor([[0, 0, 0]], dtype=torch.int64),
        "chirality": torch.tensor([[0, 0, 0]], dtype=torch.int64),
        "aromatic_mask": torch.tensor([[False, False, False]]),
        "atom_to_token": torch.tensor([[0, 1, 1]], dtype=torch.int64),
        "atom_mask": torch.tensor([[True, True, True]]),
        "bond_indices": torch.tensor([[[0, 1], [1, 2]]], dtype=torch.int64),
        "bond_type": torch.tensor([[1, 1]], dtype=torch.int64),
        "bond_stereo": torch.tensor([[0, 0]], dtype=torch.int64),
        "bond_mask": torch.tensor([[True, True]]),
    }
    if forward:
        inputs["noisy_coordinates"] = torch.zeros((1, 3, 3), dtype=torch.float32)
        inputs["diffusion_time"] = torch.full((1,), 0.5, dtype=torch.float32)
    return inputs


@pytest.fixture
def request_factory():
    def build(**changes: object) -> InferenceRequest:
        values: dict[str, object] = {
            "request_id": "request-1",
            "tenant_id": "tenant-a",
            "project_id": "project-a",
            "model_digest": sha("a"),
            "inputs": cladefold_inputs(),
            "seed": 7,
            "num_steps": 8,
        }
        values.update(changes)
        return InferenceRequest(**values)  # type: ignore[arg-type]

    return build
