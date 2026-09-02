"""Convert model outputs into validated inference candidates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from ..confidence.calibration import ConfidenceCalibrator
from ..confidence.confidence_estimation import ConfidenceRepresentation, estimate_confidence
from ..contracts.result_contract import InferenceCandidate
from ..postprocessing.coordinate_projection import center_coordinates
from ..postprocessing.structure_validation import validate_structure

type SampleSeedInput = torch.Tensor | Sequence[int] | Sequence[Sequence[int]]


def _normalize_sample_seeds(
    seeds: SampleSeedInput, *, batch_size: int, sample_count: int
) -> tuple[tuple[int, ...], ...]:
    raw_rows: list[object]
    if isinstance(seeds, torch.Tensor):
        if seeds.dtype is not torch.int64:
            raise TypeError("sample seed tensors must use torch.int64")
        if seeds.ndim != 2 or tuple(seeds.shape) != (batch_size, sample_count):
            raise ValueError("sample seed tensor must have shape [B, S]")
        raw_rows = list(seeds.detach().cpu().tolist())
    else:
        if isinstance(seeds, (str, bytes, bytearray)):
            raise TypeError("sample seeds must be an integer sequence or [B, S] matrix")
        raw_values = list(seeds)
        if raw_values and all(type(value) is int for value in raw_values):
            if batch_size != 1:
                raise ValueError("flat sample seeds are only valid for batch size 1")
            raw_rows = [raw_values]
        else:
            raw_rows = raw_values
    if len(raw_rows) != batch_size:
        raise ValueError("sample seed matrix batch dimension does not match coordinates")

    normalized: list[tuple[int, ...]] = []
    for row in raw_rows:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise TypeError("sample seeds must be an integer sequence or [B, S] matrix")
        values = tuple(row)
        if len(values) != sample_count:
            raise ValueError("sample seed matrix sample dimension does not match coordinates")
        if any(type(value) is not int or not 0 <= value < 2**63 for value in values):
            raise ValueError("sample seeds must be integers within [0, 2**63)")
        normalized.append(values)
    return tuple(normalized)


def _field(output: Any, *names: str) -> Any:
    for name in names:
        if isinstance(output, dict) and name in output:
            return output[name]
        if hasattr(output, name):
            return getattr(output, name)
    raise ValueError(f"model output is missing one of: {', '.join(names)}")


def _optional_field(output: Any, name: str) -> Any | None:
    if isinstance(output, dict):
        return output.get(name)
    return getattr(output, name, None)


def build_candidates(
    output: Any,
    *,
    atom_mask: torch.Tensor,
    seeds: SampleSeedInput,
    steps: int,
    calibrator: ConfidenceCalibrator | None = None,
) -> tuple[InferenceCandidate, ...]:
    """Build one sample candidate with one replay seed for every batch row.

    The canonical seed layout is ``[B, S]``. A flat ``[S]`` sequence remains
    accepted for batch size one. Any future execution-time request co-batching
    must preserve each request's row-to-seed mapping before calling this
    boundary; the current dynamic batcher only groups request envelopes.
    Model-provided ``sample_confidence`` is already calibrated and is the
    authoritative ranking score; ``calibrator`` applies only when it is absent.
    """

    coordinates = _field(output, "coordinates", "atom_coordinates")
    confidence = _field(output, "confidence", "atom_confidence")
    sample_confidence = _optional_field(output, "sample_confidence")
    if not isinstance(coordinates, torch.Tensor) or not isinstance(confidence, torch.Tensor):
        raise TypeError("coordinates and atom confidence must be tensors")
    if coordinates.ndim == 3:
        coordinates = coordinates.unsqueeze(1)
    if coordinates.ndim != 4 or coordinates.shape[-1] != 3:
        raise ValueError("fold coordinates must have shape [B, S, A, 3]")
    seed_matrix = _normalize_sample_seeds(
        seeds,
        batch_size=int(coordinates.shape[0]),
        sample_count=int(coordinates.shape[1]),
    )
    if confidence.ndim == 2:
        confidence = confidence.unsqueeze(1).expand(-1, coordinates.shape[1], -1)
    if confidence.ndim != 3:
        raise ValueError("confidence must have shape [B, S, A] or [B, A]")
    if confidence.shape != coordinates.shape[:3]:
        raise ValueError("confidence dimensions must match coordinates")
    if sample_confidence is not None:
        if not isinstance(sample_confidence, torch.Tensor):
            raise TypeError("sample_confidence must be a tensor")
        if sample_confidence.ndim != 2 or sample_confidence.shape != coordinates.shape[:2]:
            raise ValueError("sample_confidence must have shape [B, S]")

    active_calibrator = calibrator or ConfidenceCalibrator.identity()
    candidates: list[InferenceCandidate] = []
    for index in range(coordinates.shape[1]):
        candidate_coordinates = center_coordinates(coordinates[:, index], atom_mask)
        validate_structure(candidate_coordinates, atom_mask=atom_mask)
        raw = estimate_confidence(
            confidence[:, index],
            atom_mask,
            representation=ConfidenceRepresentation.PROBABILITIES,
        )
        if sample_confidence is None:
            calibrated = active_calibrator.calibrate_scalar(raw)
        else:
            calibrated = estimate_confidence(
                sample_confidence[:, index],
                torch.ones_like(sample_confidence[:, index], dtype=torch.bool),
                representation=ConfidenceRepresentation.PROBABILITIES,
            )
        candidates.append(
            InferenceCandidate(
                candidate_id=f"candidate-{index:04d}",
                coordinates=candidate_coordinates,
                confidence=raw,
                calibrated_confidence=calibrated,
                batch_seeds=tuple(row[index] for row in seed_matrix),
                steps=steps,
            )
        )
    return tuple(candidates)
