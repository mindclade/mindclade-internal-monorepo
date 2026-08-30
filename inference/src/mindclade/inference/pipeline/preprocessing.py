"""Tensor-manifest validation and model-batch construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeVar

import torch

from .._identity import tensor_digest
from ..batching.batch_limits import BatchLimits
from ..contracts.request_contract import InferenceRequest

TBatch = TypeVar("TBatch")

_MASK_SUFFIXES = ("_mask", "mask")
_COORDINATE_NAMES = {"noisy_coordinates", "target_coordinates", "coordinates"}


@dataclass(frozen=True, slots=True)
class PreparedInference:
    request: InferenceRequest
    inputs: Mapping[str, torch.Tensor]
    batch_size: int
    tokens: int
    atoms: int
    bonds: int
    tensor_bytes: int
    device: torch.device

    def verify_integrity(self) -> None:
        """Reject mutation between preprocessing and model execution."""

        self.request.verify_integrity()
        current = {name: tensor_digest(value) for name, value in self.inputs.items()}
        if current != dict(self.request.input_digests):
            raise ValueError("prepared input tensor changed after preprocessing")

    def as_model_batch(
        self, factory: Callable[..., TBatch], *, validation_mode: str | None = None
    ) -> TBatch:
        batch = factory(**dict(self.inputs))
        validator = getattr(batch, "validate", None)
        if callable(validator):
            if validation_mode is None:
                validator()
            else:
                validator(mode=validation_mode)
        return batch


def _dimension(inputs: Mapping[str, torch.Tensor], names: tuple[str, ...], axis: int) -> int:
    for name in names:
        tensor = inputs.get(name)
        if tensor is not None and tensor.ndim > axis:
            return int(tensor.shape[axis])
    return 0


def preprocess_request(
    request: InferenceRequest,
    *,
    limits: BatchLimits | None = None,
) -> PreparedInference:
    """Validate generic tensor invariants without parsing biological file formats."""

    request.verify_integrity()
    # Keep an execution-owned snapshot separate from the admitted request.
    tensors = {name: tensor.detach().clone() for name, tensor in request.inputs.items()}
    devices = {tensor.device for tensor in tensors.values()}
    if len(devices) != 1:
        raise ValueError("all request tensors must be on the same device")
    device = next(iter(devices))
    batch_sizes = {int(tensor.shape[0]) for tensor in tensors.values() if tensor.ndim > 0}
    if len(batch_sizes) != 1:
        raise ValueError("all non-scalar tensors must share the leading batch dimension")
    if not batch_sizes:
        raise ValueError("at least one input tensor must have a batch dimension")
    batch_size = next(iter(batch_sizes))
    if batch_size < 1:
        raise ValueError("empty batches are not supported")

    for name, tensor in tensors.items():
        if name.endswith(_MASK_SUFFIXES) and tensor.dtype is not torch.bool:
            raise TypeError(f"{name} must use torch.bool")
        if name in _COORDINATE_NAMES:
            if tensor.ndim != 3 or tensor.shape[-1] != 3:
                raise ValueError(f"{name} must have shape [B, A, 3]")
            if not tensor.is_floating_point():
                raise TypeError(f"{name} must be floating point")
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values")

    tokens = _dimension(tensors, ("token_type", "token_type_ids", "token_mask"), 1)
    atoms = _dimension(tensors, ("atomic_number", "atom_atomic_number", "atom_mask"), 1)
    bonds = _dimension(tensors, ("bond_indices", "bond_mask"), 1)
    selected_limits = limits or BatchLimits.async_()
    tensor_bytes = selected_limits.validate(
        batch_size=batch_size,
        tokens=tokens,
        atoms=atoms,
        bonds=bonds,
        samples=request.num_samples,
        steps=request.num_steps,
        tensors=tensors,
    )
    return PreparedInference(
        request=request,
        inputs=MappingProxyType(tensors),
        batch_size=batch_size,
        tokens=tokens,
        atoms=atoms,
        bonds=bonds,
        tensor_bytes=tensor_bytes,
        device=device,
    )
