"""Immutable, canonical model configuration."""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Self

from .config_validation import ConfigurationError, reject_unknown_fields


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Base for local-only, schema-versioned model configurations."""

    model_type: ClassVar[str] = "model"
    supported_schema_versions: ClassVar[tuple[int, ...]] = (1,)

    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version not in self.supported_schema_versions
        ):
            raise ConfigurationError(
                f"unsupported schema_version={self.schema_version}; "
                f"supported={self.supported_schema_versions}"
            )

    def validate(self) -> None:
        """Validate subclass invariants; subclasses should call ``super``."""

        if (
            type(self.schema_version) is not int
            or self.schema_version not in self.supported_schema_versions
        ):
            raise ConfigurationError(f"unsupported schema version {self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["model_type"] = self.model_type
        return value

    def to_json_string(self) -> str:
        self.validate()
        return (
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        data = dict(value)
        model_type = data.pop("model_type", cls.model_type)
        if model_type != cls.model_type:
            raise ConfigurationError(
                f"configuration model_type {model_type!r} does not match {cls.model_type!r}"
            )
        reject_unknown_fields(data, {field.name for field in dataclasses.fields(cls)})
        result = cls(**data)
        result.validate()
        return result

    @classmethod
    def from_json_file(cls, path: os.PathLike[str] | str) -> Self:
        source = Path(path)
        if source.is_symlink() or not source.is_file():
            raise ConfigurationError(f"configuration path is not a regular local file: {source}")

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise ConfigurationError(f"duplicate configuration field {key!r}")
                result[key] = item
            return result

        try:
            value = json.loads(
                source.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
            )
        except ConfigurationError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"invalid configuration file {source}: {exc}") from exc
        if not isinstance(value, dict):
            raise ConfigurationError("configuration JSON must be an object")
        return cls.from_dict(value)

    @classmethod
    def from_pretrained(cls, directory: os.PathLike[str] | str) -> Self:
        """Load ``config.json`` from a local pretrained-style directory."""

        raw = os.fspath(directory)
        if "://" in raw:
            raise ConfigurationError("from_pretrained accepts local directories only")
        source = Path(raw)
        if source.is_symlink() or not source.is_dir():
            raise ConfigurationError(f"pretrained path is not a regular local directory: {source}")
        return cls.from_json_file(source / "config.json")

    def save_pretrained(self, directory: os.PathLike[str] | str) -> Path:
        target = Path(directory)
        if target.exists():
            if target.is_symlink() or not target.is_dir():
                raise ConfigurationError(
                    f"pretrained path is not a regular local directory: {target}"
                )
        else:
            target.mkdir(parents=True)
        path = target / "config.json"
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=target)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(self.to_json_string())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory_descriptor = os.open(target, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            temporary.unlink(missing_ok=True)
        return path


__all__ = ["ModelConfig"]
