from __future__ import annotations

import torch
from mindclade.models.components.diffusion import VESchedule
from mindclade.training.core.data import DeterministicSyntheticDataset, SyntheticSplit


def test_synthetic_split_sizes_and_ids_are_disjoint() -> None:
    train = DeterministicSyntheticDataset(SyntheticSplit.TRAIN)
    validation = DeterministicSyntheticDataset(SyntheticSplit.VALIDATION)
    test = DeterministicSyntheticDataset(SyntheticSplit.TEST)
    assert (len(train), len(validation), len(test)) == (128, 16, 16)
    assert set(train.sample_ids).isdisjoint(validation.sample_ids)
    assert set(train.sample_ids).isdisjoint(test.sample_ids)
    assert set(validation.sample_ids).isdisjoint(test.sample_ids)


def test_synthetic_example_is_stateless_and_shape_safe() -> None:
    dataset = DeterministicSyntheticDataset(
        SyntheticSplit.TRAIN,
        seed=99,
        token_count=4,
        atom_count=9,
    )
    first = dataset[7]
    repeated = dataset[7]
    assert first.sample_id == repeated.sample_id
    assert first.fields.keys() == repeated.fields.keys()
    for name in first.fields:
        torch.testing.assert_close(first.fields[name], repeated.fields[name], rtol=0.0, atol=0.0)
    assert first.fields["token_type"].shape == (4,)
    assert first.fields["noisy_coordinates"].shape == (9, 3)
    assert first.fields["bond_indices"].shape == (8, 2)
    assert first.fields["token_type"].dtype is torch.int64
    assert first.fields["token_mask"].dtype is torch.bool
    assert first.fields["target_coordinates"].dtype is torch.float32


def test_different_seed_changes_values_but_not_identity() -> None:
    left = DeterministicSyntheticDataset(SyntheticSplit.TEST, seed=1)[0]
    right = DeterministicSyntheticDataset(SyntheticSplit.TEST, seed=2)[0]
    assert left.sample_id == right.sample_id
    assert not torch.equal(left.fields["noisy_coordinates"], right.fields["noisy_coordinates"])


def test_synthetic_perturbation_uses_ve_schedule_sigma() -> None:
    first_schedule = VESchedule(0.01, 20.0)
    second_schedule = VESchedule(0.10, 2.0)
    first = DeterministicSyntheticDataset(
        SyntheticSplit.TRAIN,
        seed=41,
        sigma_min=first_schedule.sigma_min,
        sigma_max=first_schedule.sigma_max,
    )[3]
    second = DeterministicSyntheticDataset(
        SyntheticSplit.TRAIN,
        seed=41,
        sigma_min=second_schedule.sigma_min,
        sigma_max=second_schedule.sigma_max,
    )[3]

    torch.testing.assert_close(
        first.fields["target_coordinates"],
        second.fields["target_coordinates"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        first.fields["diffusion_time"],
        second.fields["diffusion_time"],
        rtol=0.0,
        atol=0.0,
    )
    time = first.fields["diffusion_time"]
    standardized_first = (
        first.fields["noisy_coordinates"] - first.fields["target_coordinates"]
    ) / first_schedule.sigma(time)
    standardized_second = (
        second.fields["noisy_coordinates"] - second.fields["target_coordinates"]
    ) / second_schedule.sigma(time)
    torch.testing.assert_close(standardized_first, standardized_second)
