"""Reference VE diffusion sampler with fixed and adaptive execution."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .._identity import content_digest
from ..adaptive_compute.budget_accounting import BudgetLedger
from ..adaptive_compute.compute_policy import ComputePolicy
from ..adaptive_compute.resume_frontier import ResumeFrontier
from ..adaptive_compute.stopping_rule import Observation, StoppingRule
from .sampler_contract import ConfidenceFunction, DenoiseFunction, SamplingOutcome


@dataclass(frozen=True, slots=True)
class DiffusionSampler:
    sigma_min: float = 0.01
    sigma_max: float = 20.0
    sampler_version: str = "ve-heun-v1alpha1"

    def __post_init__(self) -> None:
        if not 0 < self.sigma_min < self.sigma_max:
            raise ValueError("sigma bounds must satisfy 0 < sigma_min < sigma_max")

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "sampler_version": self.sampler_version,
                "sigma_min": self.sigma_min,
                "sigma_max": self.sigma_max,
            }
        )

    def sigma_schedule(
        self, steps: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if steps < 2:
            raise ValueError("diffusion sampling requires at least two steps")
        nonzero = torch.logspace(
            math.log10(self.sigma_max),
            math.log10(self.sigma_min),
            steps,
            device=device,
            dtype=dtype,
        )
        return torch.cat((nonzero, nonzero.new_zeros((1,))), dim=0)

    @staticmethod
    def _masked_center(coordinates: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        weights = atom_mask.to(dtype=coordinates.dtype).unsqueeze(-1)
        count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        center = (coordinates * weights).sum(dim=1, keepdim=True) / count
        return (coordinates - center) * weights

    def sample(
        self,
        *,
        denoise: DenoiseFunction,
        atom_mask: torch.Tensor,
        seed: int,
        steps: int,
        dtype: torch.dtype = torch.float32,
        confidence: ConfidenceFunction | None = None,
        policy: ComputePolicy | None = None,
        request_fingerprint: str | None = None,
        model_digest: str | None = None,
        resume: ResumeFrontier | None = None,
        return_trajectory: bool = False,
        pause_after_steps: int | None = None,
    ) -> SamplingOutcome:
        if atom_mask.ndim != 2 or atom_mask.dtype is not torch.bool:
            raise TypeError("atom_mask must be bool with shape [B, A]")
        if not atom_mask.any(dim=1).all():
            raise ValueError("every batch item must contain at least one valid atom")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be in [0, 2**63)")
        if type(steps) is not int or steps < 2:
            raise ValueError("steps must be an integer of at least two")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("sampling dtype must be floating point")
        if policy is not None and steps != policy.max_steps:
            raise ValueError("steps must equal adaptive policy max_steps")
        if policy is not None and confidence is None:
            raise ValueError("adaptive sampling requires a confidence callback")
        if pause_after_steps is not None and not 1 <= pause_after_steps < steps:
            raise ValueError("pause_after_steps must be within [1, steps)")

        device = atom_mask.device
        schedule = self.sigma_schedule(steps, device=device, dtype=dtype)
        start_step = 0
        active_policy = policy or ComputePolicy(
            min_steps=steps,
            max_steps=steps,
            evaluation_interval=steps,
            patience=1,
        )
        ledger = BudgetLedger(active_policy)
        if resume is not None:
            if None in (request_fingerprint, model_digest):
                raise ValueError("resume requires request and model digests")
            resume.assert_compatible(
                request_fingerprint=str(request_fingerprint),
                model_digest=str(model_digest),
                sampler_digest=self.digest,
                policy_digest=active_policy.digest,
            )
            if resume.seed != seed:
                raise ValueError("resume seed mismatch")
            start_step = resume.completed_steps
            if start_step >= steps:
                raise ValueError("resume frontier is already complete")
            coordinates = resume.coordinates.to(device=device, dtype=dtype)
            if coordinates.shape[:2] != atom_mask.shape:
                raise ValueError("resume coordinates do not match atom_mask")
            ledger = BudgetLedger(
                active_policy,
                steps=resume.completed_steps,
                candidates=resume.consumed_candidates,
            )
        else:
            ledger.consume(candidates=1)
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            coordinates = (
                torch.randn((*atom_mask.shape, 3), generator=generator, device=device, dtype=dtype)
                * self.sigma_max
            )
            coordinates = self._masked_center(coordinates, atom_mask)

        trajectory: list[torch.Tensor] = []
        if return_trajectory:
            trajectory.append(coordinates.detach().clone())
        stopping = StoppingRule(active_policy)
        previous_evaluation = coordinates.detach().clone()
        final_confidence: float | None = None
        stop_reason = "budget-exhausted"

        for step_index in range(start_step, steps):
            sigma = schedule[step_index]
            next_sigma = schedule[step_index + 1]
            sigma_batch = sigma.expand(atom_mask.shape[0])
            predicted_noise = denoise(coordinates, sigma_batch)
            if predicted_noise.shape != coordinates.shape:
                raise ValueError("denoise callback must preserve coordinate shape")
            if not torch.isfinite(predicted_noise).all():
                raise FloatingPointError("denoise callback produced non-finite values")
            delta = next_sigma - sigma
            proposal = coordinates + delta * predicted_noise
            if float(next_sigma) > 0.0:
                next_batch = next_sigma.expand(atom_mask.shape[0])
                next_noise = denoise(proposal, next_batch)
                if next_noise.shape != coordinates.shape or not torch.isfinite(next_noise).all():
                    raise FloatingPointError("Heun correction produced invalid noise")
                coordinates = coordinates + delta * 0.5 * (predicted_noise + next_noise)
            else:
                coordinates = proposal
            coordinates = self._masked_center(coordinates, atom_mask)
            budget = ledger.consume(steps=1)
            completed = budget.consumed_steps
            if return_trajectory:
                trajectory.append(coordinates.detach().clone())

            if pause_after_steps is not None and completed >= pause_after_steps:
                stop_reason = "paused"
                break

            if policy is not None and policy.should_evaluate(completed):
                final_confidence = float(confidence(coordinates))  # type: ignore[misc]
                displacement = torch.sqrt(
                    torch.mean((coordinates - previous_evaluation).float().square())
                ).item()
                decision = stopping.observe(Observation(completed, final_confidence, displacement))
                previous_evaluation = coordinates.detach().clone()
                if decision.stop:
                    stop_reason = decision.reason
                    break

        completed_steps = ledger.snapshot().consumed_steps
        frontier = None
        if request_fingerprint is not None and model_digest is not None:
            frontier = ResumeFrontier.capture(
                request_fingerprint=request_fingerprint,
                model_digest=model_digest,
                sampler_digest=self.digest,
                policy_digest=active_policy.digest,
                budget=ledger.snapshot(),
                seed=seed,
                coordinates=coordinates,
            )
        return SamplingOutcome(
            coordinates=coordinates,
            seed=seed,
            completed_steps=completed_steps,
            stop_reason=stop_reason,
            confidence=final_confidence,
            trajectory=tuple(trajectory),
            resume_frontier=frontier,
        )
