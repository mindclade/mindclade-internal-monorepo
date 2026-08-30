"""Tensor contracts shared by training and inference."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import Any, Literal, Protocol, runtime_checkable

import torch
from torch import Tensor


class BatchValidationError(ValueError):
    """Raised at the model boundary for an invalid tensor batch."""


@runtime_checkable
class ModelBatch(Protocol):
    def validate(self, mode: str = "forward") -> None: ...

    def to(self, device: torch.device | str) -> ModelBatch: ...


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


@dataclasses.dataclass(frozen=True)
class CladeFoldBatch:
    """Padded CladeFold tensor batch.

    Symbolic dimensions are ``B`` batches, ``T`` tokens, ``A`` atoms, and ``E``
    padded undirected bonds. Masks are ``True`` for valid entries. Coordinates
    are measured in Angstrom and remain on the caller-selected device.
    """

    token_type: Tensor  # integer [B, T]
    molecule_type: Tensor  # integer [B, T]
    chain_id: Tensor  # integer [B, T]
    position_id: Tensor  # integer [B, T]
    token_mask: Tensor  # bool [B, T]
    anchor_atom_index: Tensor  # integer [B, T], -1 for padding

    atomic_number: Tensor  # integer [B, A]
    formal_charge: Tensor  # integer [B, A]
    chirality: Tensor  # integer [B, A]
    aromatic_mask: Tensor  # bool [B, A]
    atom_to_token: Tensor  # integer [B, A], -1 for padding
    atom_mask: Tensor  # bool [B, A]

    bond_indices: Tensor  # integer [B, E, 2], -1 for padding
    bond_type: Tensor  # integer [B, E]
    bond_stereo: Tensor  # integer [B, E]
    bond_mask: Tensor  # bool [B, E]

    noisy_coordinates: Tensor | None = None  # floating [B, A, 3]
    diffusion_time: Tensor | None = None  # floating [B]
    target_coordinates: Tensor | None = None  # floating [B, A, 3]
    target_mask: Tensor | None = None  # bool [B, A]

    @property
    def batch_size(self) -> int:
        return int(self.token_type.shape[0])

    @property
    def token_count(self) -> int:
        return int(self.token_type.shape[1])

    @property
    def atom_count(self) -> int:
        return int(self.atomic_number.shape[1])

    @property
    def bond_count(self) -> int:
        return int(self.bond_indices.shape[1])

    @property
    def device(self) -> torch.device:
        return self.token_type.device

    def tensor_items(self) -> Iterator[tuple[str, Tensor]]:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Tensor):
                yield field.name, value

    def replace(self, **changes: Any) -> CladeFoldBatch:
        return dataclasses.replace(self, **changes)

    def static(self) -> CladeFoldBatch:
        return self.replace(
            noisy_coordinates=None,
            diffusion_time=None,
            target_coordinates=None,
            target_mask=None,
        )

    def to(self, device: torch.device | str) -> CladeFoldBatch:
        values: dict[str, Any] = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.to(device) if isinstance(value, Tensor) else value
        return type(self)(**values)

    def validate(self, mode: Literal["forward", "static", "either"] = "forward") -> None:
        if mode not in {"forward", "static", "either"}:
            raise BatchValidationError(f"unknown validation mode {mode!r}")
        self._validate_shapes_and_dtypes()
        self._validate_devices_and_finiteness()
        self._validate_semantics()
        if mode == "forward" and (self.noisy_coordinates is None or self.diffusion_time is None):
            raise BatchValidationError("forward requires noisy_coordinates and diffusion_time")
        if mode == "static" and (
            self.noisy_coordinates is not None
            or self.diffusion_time is not None
            or self.target_coordinates is not None
            or self.target_mask is not None
        ):
            raise BatchValidationError("static/fold batches cannot contain noise, time, or labels")

    def _validate_shapes_and_dtypes(self) -> None:
        if self.token_type.ndim != 2:
            raise BatchValidationError("token_type must have shape [B, T]")
        b, t = self.token_type.shape
        if b < 1 or t < 1:
            raise BatchValidationError("B and T must be nonzero")
        if self.atomic_number.ndim != 2 or self.atomic_number.shape[0] != b:
            raise BatchValidationError("atomic_number must have shape [B, A]")
        a = self.atomic_number.shape[1]
        if a < 1:
            raise BatchValidationError("A must be nonzero")
        token_fields = {
            "token_type": self.token_type,
            "molecule_type": self.molecule_type,
            "chain_id": self.chain_id,
            "position_id": self.position_id,
            "anchor_atom_index": self.anchor_atom_index,
        }
        atom_fields = {
            "atomic_number": self.atomic_number,
            "formal_charge": self.formal_charge,
            "chirality": self.chirality,
            "atom_to_token": self.atom_to_token,
        }
        for name, tensor in token_fields.items():
            self._expect_shape(name, tensor, (b, t))
            self._expect_integer(name, tensor)
        for name, tensor in atom_fields.items():
            self._expect_shape(name, tensor, (b, a))
            self._expect_integer(name, tensor)
        for name, tensor, shape in (
            ("token_mask", self.token_mask, (b, t)),
            ("atom_mask", self.atom_mask, (b, a)),
            ("aromatic_mask", self.aromatic_mask, (b, a)),
        ):
            self._expect_shape(name, tensor, shape)
            self._expect_bool(name, tensor)
        if (
            self.bond_indices.ndim != 3
            or self.bond_indices.shape[0] != b
            or self.bond_indices.shape[2] != 2
        ):
            raise BatchValidationError("bond_indices must have shape [B, E, 2]")
        e = self.bond_indices.shape[1]
        self._expect_integer("bond_indices", self.bond_indices)
        for name, tensor in (("bond_type", self.bond_type), ("bond_stereo", self.bond_stereo)):
            self._expect_shape(name, tensor, (b, e))
            self._expect_integer(name, tensor)
        self._expect_shape("bond_mask", self.bond_mask, (b, e))
        self._expect_bool("bond_mask", self.bond_mask)
        if self.noisy_coordinates is not None:
            self._expect_shape("noisy_coordinates", self.noisy_coordinates, (b, a, 3))
            self._expect_floating("noisy_coordinates", self.noisy_coordinates)
        if self.diffusion_time is not None:
            self._expect_shape("diffusion_time", self.diffusion_time, (b,))
            self._expect_floating("diffusion_time", self.diffusion_time)
        if (self.target_coordinates is None) != (self.target_mask is None):
            raise BatchValidationError(
                "target_coordinates and target_mask must be supplied together"
            )
        if self.target_coordinates is not None:
            self._expect_shape("target_coordinates", self.target_coordinates, (b, a, 3))
            self._expect_floating("target_coordinates", self.target_coordinates)
            assert self.target_mask is not None
            self._expect_shape("target_mask", self.target_mask, (b, a))
            self._expect_bool("target_mask", self.target_mask)

    def _validate_devices_and_finiteness(self) -> None:
        devices = {tensor.device for _, tensor in self.tensor_items()}
        if len(devices) != 1:
            detail = ", ".join(f"{name}={tensor.device}" for name, tensor in self.tensor_items())
            raise BatchValidationError(f"all tensors must share one device: {detail}")
        for name in ("noisy_coordinates", "diffusion_time", "target_coordinates"):
            tensor = getattr(self, name)
            if tensor is not None and not torch.isfinite(tensor).all():
                raise BatchValidationError(f"{name} contains non-finite values")

    def _validate_semantics(self) -> None:
        if not self.token_mask.any(dim=1).all():
            raise BatchValidationError("every batch item must contain a valid token")
        if not self.atom_mask.any(dim=1).all():
            raise BatchValidationError("every batch item must contain a valid atom")
        valid_tokens = self.token_type[self.token_mask]
        if ((valid_tokens < 1) | (valid_tokens > 33)).any():
            raise BatchValidationError(
                "valid token_type IDs must be in [1, 33]; 34-63 are reserved"
            )
        invalid_tokens = ~self.token_mask
        if (self.token_type[invalid_tokens] != 0).any():
            raise BatchValidationError("padded token_type entries must use sentinel 0")
        if (self.anchor_atom_index[invalid_tokens] != -1).any():
            raise BatchValidationError("padded anchor_atom_index entries must use sentinel -1")
        for name, values in (
            ("molecule_type", self.molecule_type),
            ("chain_id", self.chain_id),
            ("position_id", self.position_id),
        ):
            if (values[invalid_tokens] != 0).any():
                raise BatchValidationError(f"padded {name} entries must use sentinel 0")
        if (self.chain_id[self.token_mask] < 0).any() or (
            self.position_id[self.token_mask] < 0
        ).any():
            raise BatchValidationError("valid chain_id and position_id values cannot be negative")
        valid_atoms = self.atomic_number[self.atom_mask]
        if ((valid_atoms < 1) | (valid_atoms > 118)).any():
            raise BatchValidationError("valid atomic_number values must be in [1, 118]")
        invalid_atoms = ~self.atom_mask
        if (self.atomic_number[invalid_atoms] != 0).any():
            raise BatchValidationError("padded atomic_number entries must use sentinel 0")
        if (self.atom_to_token[invalid_atoms] != -1).any():
            raise BatchValidationError("padded atom_to_token entries must use sentinel -1")
        for name, values in (("formal_charge", self.formal_charge), ("chirality", self.chirality)):
            if (values[invalid_atoms] != 0).any():
                raise BatchValidationError(f"padded {name} entries must use sentinel 0")
        if (self.aromatic_mask & ~self.atom_mask).any():
            raise BatchValidationError("aromatic_mask must be a subset of atom_mask")
        if self.target_mask is not None and (self.target_mask & ~self.atom_mask).any():
            raise BatchValidationError("target_mask must be a subset of atom_mask")
        if self.diffusion_time is not None and (
            (self.diffusion_time < 0.0).any() or (self.diffusion_time > 1.0).any()
        ):
            raise BatchValidationError("diffusion_time must be normalized to [0, 1]")

        b, a = self.atomic_number.shape
        token_count = self.token_type.shape[1]
        atom_to_token = self.atom_to_token[self.atom_mask]
        if ((atom_to_token < 0) | (atom_to_token >= token_count)).any():
            raise BatchValidationError("valid atom_to_token index is out of range")
        for batch_index in range(b):
            atom_indices = self.atom_to_token[batch_index, self.atom_mask[batch_index]]
            if not self.token_mask[batch_index, atom_indices].all():
                raise BatchValidationError("valid atoms must map to valid tokens")
            anchors = self.anchor_atom_index[batch_index, self.token_mask[batch_index]]
            if ((anchors < 0) | (anchors >= a)).any():
                raise BatchValidationError("valid anchor_atom_index is out of range")
            if not self.atom_mask[batch_index, anchors].all():
                raise BatchValidationError("anchor atoms must be valid")
            expected_tokens = torch.arange(token_count, device=self.device)[
                self.token_mask[batch_index]
            ]
            if not torch.equal(self.atom_to_token[batch_index, anchors], expected_tokens):
                raise BatchValidationError("each anchor atom must belong to its token")

            seen: set[tuple[int, int]] = set()
            for edge_index in range(self.bond_count):
                if not bool(self.bond_mask[batch_index, edge_index]):
                    if not torch.equal(
                        self.bond_indices[batch_index, edge_index],
                        self.bond_indices.new_full((2,), -1),
                    ):
                        raise BatchValidationError("padded bond indices must use sentinel [-1, -1]")
                    if int(self.bond_type[batch_index, edge_index]) != 0:
                        raise BatchValidationError("padded bond_type entries must use sentinel 0")
                    if int(self.bond_stereo[batch_index, edge_index]) != 0:
                        raise BatchValidationError("padded bond_stereo entries must use sentinel 0")
                    continue
                left, right = (int(v) for v in self.bond_indices[batch_index, edge_index].tolist())
                if left == right:
                    raise BatchValidationError("self bonds are not allowed")
                if left < 0 or right < 0 or left >= a or right >= a:
                    raise BatchValidationError("bond index is out of range")
                if not bool(self.atom_mask[batch_index, left]) or not bool(
                    self.atom_mask[batch_index, right]
                ):
                    raise BatchValidationError("bonds must connect valid atoms")
                key = (min(left, right), max(left, right))
                if key in seen:
                    raise BatchValidationError("valid bonds must be unique and undirected")
                seen.add(key)

    @staticmethod
    def _expect_shape(name: str, tensor: Tensor, shape: tuple[int, ...]) -> None:
        if tuple(tensor.shape) != tuple(shape):
            raise BatchValidationError(
                f"{name} must have shape {list(shape)}, got {list(tensor.shape)}"
            )

    @staticmethod
    def _expect_integer(name: str, tensor: Tensor) -> None:
        if tensor.dtype not in _INTEGER_DTYPES:
            raise BatchValidationError(f"{name} must use an integer dtype, got {tensor.dtype}")

    @staticmethod
    def _expect_bool(name: str, tensor: Tensor) -> None:
        if tensor.dtype is not torch.bool:
            raise BatchValidationError(f"{name} must use torch.bool, got {tensor.dtype}")

    @staticmethod
    def _expect_floating(name: str, tensor: Tensor) -> None:
        if not tensor.is_floating_point():
            raise BatchValidationError(f"{name} must use a floating dtype, got {tensor.dtype}")


__all__ = ["BatchValidationError", "CladeFoldBatch", "ModelBatch"]
