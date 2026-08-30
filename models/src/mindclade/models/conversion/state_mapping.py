"""Auditable state-key renaming."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

from torch import Tensor


def map_state_keys(
    state: Mapping[str, Tensor], mapping: Mapping[str, str]
) -> OrderedDict[str, Tensor]:
    unknown = set(mapping) - set(state)
    if unknown:
        raise KeyError(f"mapping references unknown source keys: {sorted(unknown)}")
    destination = [mapping.get(key, key) for key in state]
    if len(destination) != len(set(destination)):
        raise ValueError("state mapping produces duplicate destination keys")
    return OrderedDict((mapping.get(key, key), value) for key, value in state.items())


__all__ = ["map_state_keys"]
