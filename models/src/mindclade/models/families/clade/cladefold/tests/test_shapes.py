from __future__ import annotations

import torch

from mindclade.models import CladeFoldConfig, CladeFoldModel
from mindclade.models.components.geometry.rigid_frames import apply_rigid_transform


def test_tiny_parameter_budget_and_eval_determinism(cladefold_model, cladefold_batch) -> None:
    assert CladeFoldModel(CladeFoldConfig.tiny()).parameter_count < 2_000_000
    cladefold_model.eval()
    first = cladefold_model(cladefold_batch)
    second = cladefold_model(cladefold_batch)
    torch.testing.assert_close(first.predicted_noise, second.predicted_noise, atol=0, rtol=0)


def test_proper_rigid_transform_equivariance(cladefold_model, cladefold_batch) -> None:
    cladefold_model.eval()
    angle = torch.tensor(0.71)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = torch.tensor([2.0, -1.0, 0.5])
    transformed = cladefold_batch.replace(
        noisy_coordinates=apply_rigid_transform(
            cladefold_batch.noisy_coordinates, rotation, translation
        ),
        target_coordinates=apply_rigid_transform(
            cladefold_batch.target_coordinates, rotation, translation
        ),
    )
    reference = cladefold_model(cladefold_batch)
    candidate = cladefold_model(transformed)
    expected_noise = reference.predicted_noise @ rotation.transpose(0, 1)
    torch.testing.assert_close(candidate.predicted_noise, expected_noise, atol=2e-4, rtol=2e-4)
    expected_coordinates = apply_rigid_transform(
        reference.denoised_coordinates, rotation, translation
    )
    expected_coordinates = expected_coordinates * cladefold_batch.atom_mask.unsqueeze(-1)
    torch.testing.assert_close(
        candidate.denoised_coordinates, expected_coordinates, atol=2e-4, rtol=2e-4
    )
