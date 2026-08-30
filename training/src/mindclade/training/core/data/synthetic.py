"""Stateless, deterministic CladeFold engineering fixtures."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
from mindclade.training.api.task import BatchEnvelope
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

DATASET_VERSION = "synthetic-cladefold-v1"


class SyntheticSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


_SPLIT_RANGES = {
    SyntheticSplit.TRAIN: (0, 128),
    SyntheticSplit.VALIDATION: (128, 144),
    SyntheticSplit.TEST: (144, 160),
}


@dataclass(frozen=True)
class SyntheticExample:
    sample_id: str
    fields: dict[str, Tensor]


def _sample_seed(base_seed: int, global_index: int) -> int:
    material = f"{DATASET_VERSION}:{base_seed}:{global_index}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


class DeterministicSyntheticDataset(Dataset[SyntheticExample]):
    """Fixed 128/16/16 train/validation/test examples with stable identities."""

    def __init__(
        self,
        split: SyntheticSplit,
        *,
        seed: int = 17,
        token_count: int = 8,
        atom_count: int = 16,
    ) -> None:
        if token_count <= 0 or atom_count < token_count:
            raise ValueError("synthetic data requires atom_count >= token_count > 0")
        if seed < 0:
            raise ValueError("synthetic seed must be non-negative")
        self.split = SyntheticSplit(split)
        self.seed = seed
        self.token_count = token_count
        self.atom_count = atom_count
        self._start, self._stop = _SPLIT_RANGES[self.split]

    def __len__(self) -> int:
        return self._stop - self._start

    def sample_id(self, index: int) -> str:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return f"{DATASET_VERSION}:{self.split.value}:{self._start + index:06d}"

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(self.sample_id(index) for index in range(len(self)))

    def __getitem__(self, index: int) -> SyntheticExample:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        global_index = self._start + index
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_sample_seed(self.seed, global_index))
        token_count = self.token_count
        atom_count = self.atom_count

        token_type = torch.randint(2, 22, (token_count,), generator=generator, dtype=torch.int64)
        molecule_type = torch.ones(token_count, dtype=torch.int64)
        chain_id = torch.zeros(token_count, dtype=torch.int64)
        position_id = torch.arange(token_count, dtype=torch.int64)
        token_mask = torch.ones(token_count, dtype=torch.bool)

        atom_to_token = torch.div(
            torch.arange(atom_count, dtype=torch.int64) * token_count,
            atom_count,
            rounding_mode="floor",
        ).clamp(max=token_count - 1)
        anchor_atom_index = torch.empty(token_count, dtype=torch.int64)
        for token_index in range(token_count):
            anchor_atom_index[token_index] = int(
                torch.nonzero(atom_to_token == token_index, as_tuple=False)[0, 0]
            )

        atomic_numbers = torch.tensor((6, 7, 8, 16), dtype=torch.int64)
        atomic_number = atomic_numbers[
            torch.randint(0, len(atomic_numbers), (atom_count,), generator=generator)
        ]
        formal_charge = torch.zeros(atom_count, dtype=torch.int64)
        chirality = torch.zeros(atom_count, dtype=torch.int64)
        aromatic_mask = torch.zeros(atom_count, dtype=torch.bool)
        atom_mask = torch.ones(atom_count, dtype=torch.bool)

        atom_positions = torch.arange(atom_count, dtype=torch.float32)
        phase = torch.rand((), generator=generator, dtype=torch.float32) * (2.0 * math.pi)
        angle = atom_positions * 0.55 + phase
        target_coordinates = torch.stack(
            (torch.cos(angle), torch.sin(angle), atom_positions * 0.12), dim=-1
        )
        target_coordinates += 0.01 * torch.randn(
            atom_count, 3, generator=generator, dtype=torch.float32
        )
        target_coordinates -= target_coordinates.mean(dim=0, keepdim=True)
        target_mask = atom_mask.clone()
        diffusion_time = 0.05 + 0.90 * torch.rand((), generator=generator, dtype=torch.float32)
        noisy_coordinates = target_coordinates + diffusion_time * torch.randn(
            atom_count, 3, generator=generator, dtype=torch.float32
        )

        bond_indices = torch.stack(
            (
                torch.arange(atom_count - 1, dtype=torch.int64),
                torch.arange(1, atom_count, dtype=torch.int64),
            ),
            dim=-1,
        )
        bond_count = atom_count - 1
        fields = {
            "token_type": token_type,
            "molecule_type": molecule_type,
            "chain_id": chain_id,
            "position_id": position_id,
            "token_mask": token_mask,
            "anchor_atom_index": anchor_atom_index,
            "atomic_number": atomic_number,
            "formal_charge": formal_charge,
            "chirality": chirality,
            "aromatic_mask": aromatic_mask,
            "atom_to_token": atom_to_token,
            "atom_mask": atom_mask,
            "bond_indices": bond_indices,
            "bond_type": torch.ones(bond_count, dtype=torch.int64),
            "bond_stereo": torch.zeros(bond_count, dtype=torch.int64),
            "bond_mask": torch.ones(bond_count, dtype=torch.bool),
            "noisy_coordinates": noisy_coordinates,
            "diffusion_time": diffusion_time,
            "target_coordinates": target_coordinates,
            "target_mask": target_mask,
        }
        return SyntheticExample(sample_id=self.sample_id(index), fields=fields)


def collate_cladefold_examples(examples: Sequence[SyntheticExample]) -> BatchEnvelope[Any]:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    expected = set(examples[0].fields)
    if any(set(example.fields) != expected for example in examples):
        raise ValueError("synthetic examples have inconsistent field sets")
    try:
        from mindclade.models import CladeFoldBatch
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "collating CladeFold fixtures requires the mindclade-models distribution"
        ) from exc
    fields = {
        name: torch.stack([example.fields[name] for example in examples], dim=0)
        for name in sorted(expected)
    }
    batch = CladeFoldBatch(**fields)
    return BatchEnvelope(payload=batch, sample_ids=tuple(item.sample_id for item in examples))


class DeterministicEpochSampler(Sampler[int]):
    """Epoch-addressable sampler whose permutation does not depend on prior iteration."""

    def __init__(self, size: int, *, seed: int, shuffle: bool) -> None:
        self.size = size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("sampler epoch must be non-negative")
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        if not self.shuffle:
            return iter(range(self.size))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_sample_seed(self.seed, self.epoch))
        return iter(torch.randperm(self.size, generator=generator).tolist())

    def __len__(self) -> int:
        return self.size


class DeterministicSyntheticLoader:
    """Reiterable loader with an explicit epoch cursor for exact resume."""

    def __init__(
        self,
        dataset: DeterministicSyntheticDataset,
        *,
        batch_size: int,
        seed: int,
        shuffle: bool,
    ) -> None:
        self.sampler = DeterministicEpochSampler(len(dataset), seed=seed, shuffle=shuffle)
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            num_workers=0,
            collate_fn=collate_cladefold_examples,
            drop_last=False,
        )

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)

    def __iter__(self) -> Iterator[BatchEnvelope[Any]]:
        return iter(self.loader)

    def __len__(self) -> int:
        return len(self.loader)


def build_synthetic_loader(
    split: SyntheticSplit,
    *,
    batch_size: int,
    seed: int = 17,
    token_count: int = 8,
    atom_count: int = 16,
    shuffle: bool = False,
) -> DeterministicSyntheticLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    dataset = DeterministicSyntheticDataset(
        split,
        seed=seed,
        token_count=token_count,
        atom_count=atom_count,
    )
    return DeterministicSyntheticLoader(
        dataset,
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle,
    )
