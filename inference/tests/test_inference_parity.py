from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from mindclade.inference.confidence.calibration import (
    CalibrationParameters,
    ConfidenceCalibrator,
)
from mindclade.inference.confidence.confidence_estimation import (
    ConfidenceRepresentation,
    estimate_confidence,
)
from mindclade.inference.contracts.result_contract import InferenceCandidate, InferenceResult
from mindclade.inference.diagnostics.execution_trace import ExecutionTrace
from mindclade.inference.diagnostics.numerical_diagnostics import summarize_tensor
from mindclade.inference.pipeline.model_execution import ModelExecutor, ModelResolver, ResolvedModel
from mindclade.inference.pipeline.postprocessing import build_candidates
from mindclade.inference.pipeline.preprocessing import preprocess_request
from mindclade.inference.postprocessing.structure_validation import validate_structure
from mindclade.inference.ranking.candidate_ranker import CandidateRanker
from mindclade.inference.sampling import derive_sample_seed
from mindclade.inference.sampling.diffusion_sampler import DiffusionSampler
from mindclade.models.api.batch import CladeFoldBatch
from mindclade.models.families.clade.cladefold.architecture.cladefold import CladeFoldModel
from mindclade.models.families.clade.cladefold.configuration.cladefold_q0 import CladeFoldConfig

from .conftest import sha


def test_fixed_sampler_is_seed_reproducible_and_mask_preserving() -> None:
    sampler = DiffusionSampler(sigma_min=0.1, sigma_max=1.0)
    mask = torch.tensor([[True, True, False]])

    def denoise(coordinates: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return coordinates * torch.sigmoid(sigma[:, None, None])

    first = sampler.sample(denoise=denoise, atom_mask=mask, seed=3, steps=5)
    replay = sampler.sample(denoise=denoise, atom_mask=mask, seed=3, steps=5)
    changed = sampler.sample(denoise=denoise, atom_mask=mask, seed=4, steps=5)
    torch.testing.assert_close(first.coordinates, replay.coordinates, rtol=0, atol=0)
    assert not torch.equal(first.coordinates, changed.coordinates)
    assert torch.equal(first.coordinates[:, 2], torch.zeros((1, 3)))


@pytest.mark.parametrize(
    ("seed", "steps", "dtype", "match"),
    [
        (True, 5, torch.float32, "seed"),
        (3, 5.0, torch.float32, "steps"),
        (3, 5, torch.int64, "dtype"),
    ],
)
def test_sampler_rejects_ambiguous_controls(
    seed: object, steps: object, dtype: torch.dtype, match: str
) -> None:
    sampler = DiffusionSampler(sigma_min=0.1, sigma_max=1.0)
    mask = torch.tensor([[True]])

    with pytest.raises((TypeError, ValueError), match=match):
        sampler.sample(
            denoise=lambda coordinates, _sigma: coordinates,
            atom_mask=mask,
            seed=seed,
            steps=steps,
            dtype=dtype,
        )


def test_model_executor_uses_validated_static_batch(request_factory) -> None:
    class FakeModel:
        def fold(self, batch: CladeFoldBatch, **options: object) -> SimpleNamespace:
            assert batch.noisy_coordinates is None
            coordinates = torch.zeros((1, int(options["num_samples"]), 3, 3))
            confidence = torch.full((1, int(options["num_samples"]), 3), 0.5)
            return SimpleNamespace(coordinates=coordinates, confidence=confidence)

    request = request_factory(num_samples=2)
    prepared = preprocess_request(request)
    resolver: ModelResolver[FakeModel] = ModelResolver()
    resolver.register(ResolvedModel(request.model_digest, FakeModel(), CladeFoldBatch))
    output = ModelExecutor(resolver).fold(prepared)
    assert output.coordinates.shape == (1, 2, 3, 3)


def test_model_executor_rejects_mutated_prepared_tensor(request_factory) -> None:
    class FakeModel:
        def fold(self, batch: CladeFoldBatch, **options: object) -> SimpleNamespace:
            return SimpleNamespace()

    request = request_factory()
    prepared = preprocess_request(request)
    prepared.inputs["token_type"].fill_(7)
    resolver: ModelResolver[FakeModel] = ModelResolver()
    resolver.register(ResolvedModel(request.model_digest, FakeModel(), CladeFoldBatch))

    with pytest.raises(ValueError, match="prepared input tensor changed"):
        ModelExecutor(resolver).fold(prepared)


def test_cladefold_tiny_executor_replays_and_postprocesses(request_factory) -> None:
    torch.manual_seed(17)
    request = request_factory(num_steps=2)
    prepared = preprocess_request(request)
    model = CladeFoldModel(CladeFoldConfig.tiny()).eval()
    resolver: ModelResolver[CladeFoldModel] = ModelResolver()
    resolver.register(ResolvedModel(request.model_digest, model, CladeFoldBatch))
    executor = ModelExecutor(resolver)

    first = executor.fold(prepared)
    replay = executor.fold(prepared)
    torch.testing.assert_close(first.atom_coordinates, replay.atom_coordinates, rtol=0, atol=0)
    candidates = build_candidates(
        first,
        atom_mask=request.inputs["atom_mask"],
        seeds=first.sample_seeds,
        steps=request.num_steps,
    )
    assert len(candidates) == 1
    assert candidates[0].coordinates.shape == (1, 3, 3)
    assert candidates[0].batch_seeds == (request.seed,)
    assert candidates[0].seed == request.seed


def test_cladefold_executor_replays_maximum_seed_multisample_request(request_factory) -> None:
    maximum_seed = (1 << 63) - 1
    single = request_factory()
    batched_inputs = {
        name: tensor.repeat((2,) + (1,) * (tensor.ndim - 1))
        for name, tensor in single.inputs.items()
    }
    request = request_factory(
        inputs=batched_inputs,
        seed=maximum_seed,
        num_samples=2,
        num_steps=2,
    )
    prepared = preprocess_request(request)
    model = CladeFoldModel(CladeFoldConfig.tiny()).eval()
    resolver: ModelResolver[CladeFoldModel] = ModelResolver()
    resolver.register(ResolvedModel(request.model_digest, model, CladeFoldBatch))
    executor = ModelExecutor(resolver)

    first = executor.fold(prepared)
    replay = executor.fold(prepared)

    assert first.sample_seeds.tolist() == [
        [derive_sample_seed(maximum_seed, 0), derive_sample_seed(maximum_seed, 1)],
        [derive_sample_seed(maximum_seed, 2), derive_sample_seed(maximum_seed, 3)],
    ]
    assert first.sample_seeds[0, 0].item() == maximum_seed
    torch.testing.assert_close(first.atom_coordinates, replay.atom_coordinates, rtol=0, atol=0)
    candidates = build_candidates(
        first,
        atom_mask=request.inputs["atom_mask"],
        seeds=first.sample_seeds,
        steps=request.num_steps,
    )
    assert candidates[0].batch_seeds == (
        derive_sample_seed(maximum_seed, 0),
        derive_sample_seed(maximum_seed, 2),
    )
    assert candidates[1].batch_seeds == (
        derive_sample_seed(maximum_seed, 1),
        derive_sample_seed(maximum_seed, 3),
    )
    with pytest.raises(ValueError, match="only scalar for batch size 1"):
        _ = candidates[0].seed


@pytest.mark.parametrize("batch_seeds", [(True,), (-1,), (2**63,), (), (1, 2)])
def test_candidate_rejects_invalid_or_misaligned_batch_seeds(batch_seeds: object) -> None:
    with pytest.raises(ValueError, match="batch_seeds"):
        InferenceCandidate(
            candidate_id="candidate-0000",
            coordinates=torch.zeros((1, 3, 3)),
            confidence=0.5,
            calibrated_confidence=0.5,
            batch_seeds=batch_seeds,  # type: ignore[arg-type]
            steps=2,
        )


def test_ranking_tie_breaks_on_the_full_batch_seed_tuple() -> None:
    shared = {
        "coordinates": torch.zeros((2, 3, 3)),
        "confidence": 0.5,
        "calibrated_confidence": 0.5,
        "steps": 2,
    }
    candidates = (
        InferenceCandidate(candidate_id="candidate-0000", batch_seeds=(4, 9), **shared),
        InferenceCandidate(candidate_id="candidate-0001", batch_seeds=(4, 8), **shared),
    )

    ranked = CandidateRanker().rank(candidates, request_fingerprint=sha("a"))

    assert ranked.selected.candidate_id == "candidate-0001"
    assert "batch_seeds_lexicographic_asc" in ranked.evidence.tie_breakers

    result = InferenceResult(
        request_id="request-1",
        request_fingerprint=sha("a"),
        model_digest=sha("b"),
        serving_revision_digest=sha("c"),
        candidates=candidates,
        selected_candidate_id=ranked.selected.candidate_id,
        execution_mode="eager",
        sampler_digest=sha("d"),
    )
    assert result.sample_seeds == ((4, 4), (9, 8))


def test_postprocessing_calibration_ranking_and_diagnostics_are_coherent() -> None:
    output = SimpleNamespace(
        coordinates=torch.tensor(
            [
                [
                    [[1.0, 0, 0], [-1.0, 0, 0], [0, 0, 0]],
                    [[2.0, 0, 0], [-2.0, 0, 0], [0, 0, 0]],
                ]
            ]
        ),
        confidence=torch.tensor([[[0.6, 0.6, 0.0], [0.8, 0.8, 0.0]]]),
    )
    mask = torch.tensor([[True, True, False]])
    calibrator = ConfidenceCalibrator(CalibrationParameters(temperature=2.0, bias=0.1))
    candidates = build_candidates(
        output,
        atom_mask=mask,
        seeds=(10, 11),
        steps=8,
        calibrator=calibrator,
    )
    ranked = CandidateRanker().rank(candidates, request_fingerprint=sha("a"))
    assert ranked.selected.candidate_id == "candidate-0001"
    assert candidates[0].batch_seeds == (10,)
    assert candidates[0].seed == 10
    assert ranked.evidence.ordered_candidate_ids == ("candidate-0001", "candidate-0000")

    summary = summarize_tensor("output", ranked.selected.coordinates)
    assert summary.valid
    trace = ExecutionTrace(clock_ns=lambda: 1)
    event = trace.record(
        "completed",
        tenant_id="tenant-secret",
        coordinates=ranked.selected.coordinates,
        execution_mode="eager",
        subject="customer-subject",
        id="raw-identifier",
        error="artifact for customer-subject failed",
    )
    assert event.attributes["tenant_id"] == "<redacted>"
    assert event.attributes["coordinates"] == "<redacted>"
    assert event.attributes["execution_mode"] == "eager"
    assert event.attributes["subject"] == trace.pseudonym("customer-subject")
    assert event.attributes["id"] == trace.pseudonym("raw-identifier")
    assert event.attributes["error"] == trace.pseudonym("artifact for customer-subject failed")


def test_model_sample_confidence_controls_candidate_ranking() -> None:
    output = SimpleNamespace(
        coordinates=torch.tensor(
            [
                [
                    [[1.0, 0, 0], [-1.0, 0, 0], [0, 0, 0]],
                    [[2.0, 0, 0], [-2.0, 0, 0], [0, 0, 0]],
                ]
            ]
        ),
        atom_confidence=torch.tensor([[[0.9, 0.9, 0.0], [0.1, 0.1, 0.0]]]),
        sample_confidence=torch.tensor([[0.2, 0.8]]),
    )
    candidates = build_candidates(
        output,
        atom_mask=torch.tensor([[True, True, False]]),
        seeds=(10, 11),
        steps=8,
    )

    assert candidates[0].confidence > candidates[1].confidence
    assert candidates[0].calibrated_confidence == pytest.approx(0.2)
    assert candidates[1].calibrated_confidence == pytest.approx(0.8)
    ranked = CandidateRanker().rank(candidates, request_fingerprint=sha("a"))
    assert ranked.selected.candidate_id == "candidate-0001"


@pytest.mark.parametrize(
    ("sample_confidence", "error"),
    [
        (torch.tensor([[float("nan"), 0.5]]), FloatingPointError),
        (torch.tensor([[1.1, 0.5]]), ValueError),
        (torch.tensor([0.5, 0.5]), ValueError),
    ],
)
def test_postprocessing_rejects_invalid_sample_confidence(
    sample_confidence: torch.Tensor,
    error: type[Exception],
) -> None:
    output = SimpleNamespace(
        coordinates=torch.zeros((1, 2, 2, 3)),
        atom_confidence=torch.full((1, 2, 2), 0.5),
        sample_confidence=sample_confidence,
    )
    with pytest.raises(error):
        build_candidates(
            output,
            atom_mask=torch.tensor([[True, True]]),
            seeds=(10, 11),
            steps=8,
        )


def test_confidence_representation_is_explicit() -> None:
    values = torch.tensor([0.2, 0.8])
    mask = torch.tensor([True, True])

    probabilities = estimate_confidence(
        values,
        mask,
        representation=ConfidenceRepresentation.PROBABILITIES,
    )
    logits = estimate_confidence(
        values,
        mask,
        representation=ConfidenceRepresentation.LOGITS,
    )

    assert probabilities == pytest.approx(0.5)
    assert logits == pytest.approx(float(torch.sigmoid(values).mean()))
    assert logits != pytest.approx(probabilities)
    with pytest.raises(ValueError, match="within"):
        estimate_confidence(
            torch.tensor([1.1]),
            torch.tensor([True]),
            representation=ConfidenceRepresentation.PROBABILITIES,
        )


def test_default_structure_validation_skips_pairwise_scan(monkeypatch) -> None:
    def unexpected_cdist(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("pairwise collision scan was not requested")

    monkeypatch.setattr(torch, "cdist", unexpected_cdist)
    report = validate_structure(
        torch.zeros((1, 4096, 3)),
        atom_mask=torch.ones((1, 4096), dtype=torch.bool),
    )

    assert report.minimum_nonbonded_distance is None


def test_collision_rejection_forces_pairwise_scan() -> None:
    with pytest.raises(ValueError, match="collision"):
        validate_structure(
            torch.zeros((1, 2, 3)),
            atom_mask=torch.ones((1, 2), dtype=torch.bool),
            reject_collisions_below=0.1,
        )
