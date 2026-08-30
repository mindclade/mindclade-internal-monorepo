"""Base class for integrity-checked local pretrained-style models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, Generic, Literal, Self, TypeVar, overload

from torch import nn

from mindclade.models.common.configuration.config_validation import ConfigurationError
from mindclade.models.common.configuration.model_config import ModelConfig

from .serialization import (
    SerializationError,
    load_state_directory,
    read_json_object,
    save_state_directory,
    verify_integrity,
)

ConfigT = TypeVar("ConfigT", bound=ModelConfig)


class PretrainedModel(nn.Module, Generic[ConfigT]):  # noqa: UP046
    """Safe local equivalent of the common pretrained-model lifecycle."""

    config_class: ClassVar[type[ModelConfig]] = ModelConfig

    def __init__(self, config: ConfigT) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self._loading_info: dict[str, list[str]] = {"missing_keys": [], "unexpected_keys": []}

    def save_pretrained(
        self,
        directory: os.PathLike[str] | str,
        *,
        max_shard_size: int | str = "4GiB",
    ) -> dict[str, Any]:
        return save_state_directory(
            directory,
            self.state_dict(),
            config_payload=self.config.to_json_string().encode("utf-8"),
            max_shard_size=max_shard_size,
        )

    @classmethod
    @overload
    def from_pretrained(
        cls,
        directory: os.PathLike[str] | str,
        *,
        strict: bool = True,
        output_loading_info: Literal[False] = False,
    ) -> Self: ...

    @classmethod
    @overload
    def from_pretrained(
        cls,
        directory: os.PathLike[str] | str,
        *,
        strict: bool = True,
        output_loading_info: Literal[True],
    ) -> tuple[Self, dict[str, list[str]]]: ...

    @classmethod
    def from_pretrained(
        cls,
        directory: os.PathLike[str] | str,
        *,
        strict: bool = True,
        output_loading_info: bool = False,
    ) -> Self | tuple[Self, dict[str, list[str]]]:
        raw = os.fspath(directory)
        if "://" in raw:
            raise SerializationError("from_pretrained accepts local directories only")
        source = Path(raw)
        if not source.is_dir():
            raise SerializationError(f"pretrained path is not a local directory: {source}")
        forbidden = sorted(
            path.name
            for path in source.iterdir()
            if path.suffix.lower() in {".bin", ".pkl", ".pickle", ".pt", ".pth"}
        )
        if forbidden:
            raise SerializationError(f"pickle-capable model files are forbidden: {forbidden}")
        # Verify all protected bytes, then validate configuration schema before
        # allocating state tensors or model parameters.
        verify_integrity(source)
        try:
            config_value = read_json_object(
                source / "config.json", description="verified model configuration"
            )
            config = cls.config_class.from_dict(config_value)
        except ConfigurationError as exc:
            raise SerializationError("invalid verified model configuration") from exc
        state = load_state_directory(source)
        model = cls(config)  # type: ignore[arg-type]
        try:
            incompatible = model.load_state_dict(state, strict=strict)
        except RuntimeError as exc:
            raise SerializationError(f"state dictionary is incompatible: {exc}") from exc
        info = {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
        model._loading_info = info
        model.eval()
        if output_loading_info:
            return model, info
        return model


__all__ = ["PretrainedModel"]
