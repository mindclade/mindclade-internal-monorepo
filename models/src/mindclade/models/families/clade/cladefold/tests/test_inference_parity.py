from __future__ import annotations

import torch

from mindclade.models.api.sampling import derive_sample_seed


def test_sample_seed_derivation_is_stable_bounded_and_preserves_candidate_zero() -> None:
    assert [derive_sample_seed(991, index) for index in range(4)] == [
        991,
        1451367260419495761,
        5943201774551241742,
        4121449356715345869,
    ]


def test_seeded_fold_is_bitwise_reproducible_on_same_backend(
    cladefold_model, cladefold_batch
) -> None:
    cladefold_model.eval()
    static = cladefold_batch.static()
    first = cladefold_model.fold(
        static, seed=991, num_samples=2, num_steps=2, return_trajectory=True
    )
    second = cladefold_model.fold(
        static, seed=991, num_samples=2, num_steps=2, return_trajectory=True
    )
    assert first.sample_seeds.tolist() == [
        [derive_sample_seed(991, 0), derive_sample_seed(991, 1)],
        [derive_sample_seed(991, 2), derive_sample_seed(991, 3)],
    ]
    for name in first:
        torch.testing.assert_close(first[name], second[name], atol=0, rtol=0)
    assert first.trajectories.shape == (2, 2, 3, 6, 3)


def test_maximum_seed_supports_deterministic_multisample_fold(
    cladefold_model, cladefold_batch
) -> None:
    cladefold_model.eval()
    static = cladefold_batch.static()
    maximum_seed = (1 << 63) - 1
    first = cladefold_model.fold(static, seed=maximum_seed, num_samples=2, num_steps=2)
    replay = cladefold_model.fold(static, seed=maximum_seed, num_samples=2, num_steps=2)

    assert first.sample_seeds.tolist() == [
        [9223372036854775807, 4815434108505568456],
        [7670987391547194168, 7634793827020480964],
    ]
    assert all(0 <= seed <= maximum_seed for seeds in first.sample_seeds.tolist() for seed in seeds)
    for name in first:
        torch.testing.assert_close(first[name], replay[name], atol=0, rtol=0)
