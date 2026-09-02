from __future__ import annotations

import copy

import pytest
import torch
from mindclade.inference.adaptive_compute.budget_accounting import BudgetSnapshot
from mindclade.inference.adaptive_compute.compute_policy import ComputePolicy
from mindclade.inference.adaptive_compute.resume_frontier import ResumeFrontier
from mindclade.inference.adaptive_compute.stopping_rule import Observation, StoppingState
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
    assert decoded.schema_version == "resume-frontier.v1alpha2"
    assert decoded.stopping_state.previous_observation is None
    torch.testing.assert_close(
        decoded.last_evaluation_coordinates,
        paused.resume_frontier.last_evaluation_coordinates,
        rtol=0,
        atol=0,
    )
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


@pytest.mark.parametrize("pause_after_steps", [4, 5])
def test_resume_preserves_adaptive_convergence_state(
    pause_after_steps: int,
) -> None:
    sampler = DiffusionSampler(sigma_min=0.05, sigma_max=2.0)
    policy = ComputePolicy(
        min_steps=2,
        max_steps=10,
        evaluation_interval=2,
        patience=2,
        confidence_gain_threshold=0.01,
        displacement_threshold_angstrom=0.01,
    )
    mask = torch.tensor([[True, True, False]])

    def denoise(coordinates: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(coordinates)

    common = {
        "denoise": denoise,
        "atom_mask": mask,
        "seed": 7,
        "steps": 10,
        "confidence": lambda _: 0.5,
        "policy": policy,
        "request_fingerprint": sha("a"),
        "model_digest": sha("b"),
    }
    uninterrupted = sampler.sample(**common)
    paused = sampler.sample(**common, pause_after_steps=pause_after_steps)
    assert paused.stop_reason == "paused"
    assert paused.resume_frontier is not None
    previous = paused.resume_frontier.stopping_state.previous_observation
    assert previous is not None
    assert previous.completed_steps == 4
    assert paused.resume_frontier.stopping_state.consecutive_converged == 1

    resumed = sampler.sample(
        **common,
        resume=ResumeFrontier.from_dict(paused.resume_frontier.to_dict()),
    )
    assert resumed.stop_reason == uninterrupted.stop_reason == "converged"
    assert resumed.completed_steps == uninterrupted.completed_steps == 6
    assert resumed.confidence == uninterrupted.confidence == 0.5
    torch.testing.assert_close(resumed.coordinates, uninterrupted.coordinates, rtol=0, atol=0)


def test_due_convergence_evaluation_takes_precedence_over_pause() -> None:
    sampler = DiffusionSampler(sigma_min=0.05, sigma_max=2.0)
    policy = ComputePolicy(
        min_steps=2,
        max_steps=8,
        evaluation_interval=2,
        patience=1,
        confidence_gain_threshold=0.01,
        displacement_threshold_angstrom=0.01,
    )
    outcome = sampler.sample(
        denoise=lambda coordinates, _sigma: torch.zeros_like(coordinates),
        atom_mask=torch.tensor([[True, True]]),
        seed=11,
        steps=8,
        confidence=lambda _: 0.5,
        policy=policy,
        request_fingerprint=sha("a"),
        model_digest=sha("b"),
        pause_after_steps=4,
    )

    assert outcome.stop_reason == "converged"
    assert outcome.completed_steps == 4
    assert outcome.resume_frontier is None


def test_terminal_convergence_frontier_cannot_resume() -> None:
    sampler = DiffusionSampler(sigma_min=0.05, sigma_max=2.0)
    policy = ComputePolicy(
        min_steps=2,
        max_steps=8,
        evaluation_interval=2,
        patience=1,
        confidence_gain_threshold=0.01,
        displacement_threshold_angstrom=0.01,
    )
    mask = torch.tensor([[True, True]])
    coordinates = torch.zeros((1, 2, 3))
    frontier = ResumeFrontier.capture(
        request_fingerprint=sha("a"),
        model_digest=sha("b"),
        sampler_digest=sampler.digest,
        policy_digest=policy.digest,
        budget=BudgetSnapshot(
            max_steps=policy.max_steps,
            consumed_steps=4,
            max_candidates=policy.max_candidates,
            consumed_candidates=1,
        ),
        seed=11,
        coordinates=coordinates,
        last_evaluation_coordinates=coordinates,
        stopping_state=StoppingState(
            previous_observation=Observation(4, 0.5, 0.0),
            consecutive_converged=1,
        ),
    )
    denoise_called = False

    def unexpected_denoise(current: torch.Tensor, _sigma: torch.Tensor) -> torch.Tensor:
        nonlocal denoise_called
        denoise_called = True
        return torch.zeros_like(current)

    with pytest.raises(ValueError, match="terminal stopping state"):
        sampler.sample(
            denoise=unexpected_denoise,
            atom_mask=mask,
            seed=11,
            steps=8,
            confidence=lambda _: 0.5,
            policy=policy,
            request_fingerprint=sha("a"),
            model_digest=sha("b"),
            resume=frontier,
        )
    assert not denoise_called


def test_displacement_rms_excludes_padded_atoms() -> None:
    previous = torch.zeros((1, 5, 3))
    coordinates = previous.clone()
    coordinates[0, 0, 0] = 3.0
    coordinates[0, 1, 0] = -3.0
    mask = torch.tensor([[True, True, False, False, False]])

    displacement = DiffusionSampler._masked_rms_displacement(coordinates, previous, mask)

    assert displacement == pytest.approx(3.0**0.5)


def test_resume_frontier_rejects_old_or_tampered_stopping_state() -> None:
    sampler = DiffusionSampler(sigma_min=0.05, sigma_max=2.0)
    policy = ComputePolicy(min_steps=2, max_steps=6, evaluation_interval=2, patience=2)
    paused = sampler.sample(
        denoise=lambda coordinates, _sigma: torch.zeros_like(coordinates),
        atom_mask=torch.tensor([[True, True]]),
        seed=13,
        steps=6,
        confidence=lambda _: 0.5,
        policy=policy,
        request_fingerprint=sha("a"),
        model_digest=sha("b"),
        pause_after_steps=2,
    )
    assert paused.resume_frontier is not None
    encoded = paused.resume_frontier.to_dict()

    old = copy.deepcopy(encoded)
    old["schema_version"] = "resume-frontier.v1alpha1"
    with pytest.raises(ValueError, match="v1alpha2"):
        ResumeFrontier.from_dict(old)

    tampered = copy.deepcopy(encoded)
    tampered["stopping_state"]["consecutive_converged"] = 1
    with pytest.raises(ValueError):
        ResumeFrontier.from_dict(tampered)


@pytest.mark.parametrize("invalid", [True, 2.5, "2"])
@pytest.mark.parametrize(
    "field",
    ["completed_steps", "consumed_candidates", "seed"],
)
def test_resume_frontier_decoder_requires_exact_integer_counters(
    field: str,
    invalid: object,
) -> None:
    sampler = DiffusionSampler(sigma_min=0.05, sigma_max=2.0)
    paused = sampler.sample(
        denoise=lambda coordinates, _sigma: torch.zeros_like(coordinates),
        atom_mask=torch.tensor([[True, True]]),
        seed=13,
        steps=6,
        request_fingerprint=sha("a"),
        model_digest=sha("b"),
        pause_after_steps=2,
    )
    assert paused.resume_frontier is not None
    encoded = paused.resume_frontier.to_dict()
    encoded[field] = invalid

    with pytest.raises(ValueError, match=f"{field} must be an integer"):
        ResumeFrontier.from_dict(encoded)
