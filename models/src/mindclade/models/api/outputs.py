"""Immutable structured model outputs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any


class ModelOutput(Mapping[str, Any]):
    """Small Hugging Face-style output with stable mapping and tuple access.

    ``None`` values remain addressable as attributes but are omitted from the
    mapping/tuple view, matching the useful part of the conventional model
    output behavior without importing a remote-model framework.
    """

    __slots__ = ("_all_fields", "_data")

    def __init__(self, **fields: Any) -> None:
        object.__setattr__(self, "_all_fields", dict(fields))
        object.__setattr__(
            self, "_data", {key: value for key, value in fields.items() if value is not None}
        )

    def __getitem__(self, key: str | int | slice) -> Any:
        if isinstance(key, (int, slice)):
            return self.to_tuple()[key]
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._all_fields[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError(f"{type(self).__name__} is immutable")

    def to_tuple(self) -> tuple[Any, ...]:
        return tuple(self._data.values())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)

    def __repr__(self) -> str:
        values = ", ".join(f"{key}={value!r}" for key, value in self._data.items())
        return f"{type(self).__name__}({values})"


__all__ = ["ModelOutput"]
