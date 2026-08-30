from __future__ import annotations

import torch
from mindclade.inference.adaptive_compute.compute_policy import ComputePolicy
from mindclade.inference.adaptive_compute.resume_frontier import ResumeFrontier
from mindclade.inference.sampling.diffusion_sampler import DiffusionSampler

from .conftest import sha


def test_paused_and_resumed_sampling_matches_uninterrupted_sampling() -> None:
    sampler = DiffusionSampler(sigma_min=0.05, sigma_max=2.0)
    policy = ComputePolicy(
        min_steps=8,
        max_steps=8,
        evaluation_interval=8,
        patience=1,
    )
    mask = torch.tensor([[True, True, False]])

    def denoise(coordinates: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return coordinates / (1.0 + sigma[:, None, None])

    def confidence(_: torch.Tensor) -> float:
        return 0.5

    common = {
        "denoise": denoise,
        "atom_mask": mask,
        "seed": 42,
        "steps": 8,
        "confidence": confidence,
        "policy": policy,
        "request_fingerprint": sha("a"),
        "model_digest": sha("b"),
    }
    uninterrupted = sampler.sample(**common)
    paused = sampler.sample(**common, pause_after_steps=4)
    assert paused.stop_reason == "paused"
    assert paused.resume_frontier is not None

    encoded = paused.resume_frontier.to_dict()
    decoded = ResumeFrontier.from_dict(encoded)
    resumed = sampler.sample(**common, resume=decoded)
    torch.testing.assert_close(resumed.coordinates, uninterrupted.coordinates, rtol=0, atol=0)
    assert resumed.completed_steps == uninterrupted.completed_steps == 8
    assert torch.equal(resumed.coordinates[:, 2], torch.zeros((1, 3)))


def test_resume_rejects_identity_mismatch() -> None:
    sampler = DiffusionSampler()
    policy = ComputePolicy(min_steps=4, max_steps=4, evaluation_interval=4, patience=1)
    mask = torch.tensor([[True]])

    def denoise(coordinates: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(coordinates)

    paused = sampler.sample(
        denoise=denoise,
        atom_mask=mask,
        seed=1,
        steps=4,
        confidence=lambda _: 0.5,
        policy=policy,
        request_fingerprint=sha("a"),
        model_digest=sha("b"),
        pause_after_steps=2,
    )
    assert paused.resume_frontier is not None
    try:
        sampler.sample(
            denoise=denoise,
            atom_mask=mask,
            seed=1,
            steps=4,
            confidence=lambda _: 0.5,
            policy=policy,
            request_fingerprint=sha("9"),
            model_digest=sha("b"),
            resume=paused.resume_frontier,
        )
    except ValueError as error:
        assert "request_fingerprint mismatch" in str(error)
    else:
        raise AssertionError("identity mismatch was accepted")
