"""Stable seed derivation for model sampling contracts."""

from __future__ import annotations

import hashlib

_MAX_SEED = (1 << 63) - 1


def derive_sample_seed(base_seed: int, sample_index: int) -> int:
    """Derive a deterministic, signed-int64-compatible candidate seed.

    Candidate zero retains the request seed. Later candidates use a
    domain-separated hash so derivation cannot overflow at the maximum admitted
    request seed. ``sample_index`` is the flattened batch/candidate ordinal.
    """

    if type(base_seed) is not int or not 0 <= base_seed <= _MAX_SEED:
        raise ValueError(f"base_seed must be within [0, {_MAX_SEED}]")
    if type(sample_index) is not int or sample_index < 0:
        raise ValueError("sample_index must be a non-negative integer")
    if sample_index == 0:
        return base_seed
    payload = f"mindclade-sample-v1\0{base_seed}\0{sample_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & _MAX_SEED


__all__ = ["derive_sample_seed"]
