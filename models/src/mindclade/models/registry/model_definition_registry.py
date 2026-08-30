"""Collision-safe model-type registry."""

from __future__ import annotations

import dataclasses
from typing import Any

from mindclade.models.api.model import PretrainedModel
from mindclade.models.common.configuration.model_config import ModelConfig


@dataclasses.dataclass(frozen=True)
class ModelDefinition:
    model_type: str
    config_class: type[ModelConfig]
    model_class: type[PretrainedModel[Any]]


class ModelDefinitionRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, ModelDefinition] = {}

    def register(
        self,
        model_type: str,
        config_class: type[ModelConfig],
        model_class: type[PretrainedModel[Any]],
    ) -> None:
        if model_type in self._definitions:
            raise ValueError(f"model type already registered: {model_type}")
        if config_class.model_type != model_type or model_class.config_class is not config_class:
            raise ValueError("registry model/config types do not agree")
        self._definitions[model_type] = ModelDefinition(model_type, config_class, model_class)

    def resolve(self, model_type: str) -> ModelDefinition:
        try:
            return self._definitions[model_type]
        except KeyError as exc:
            raise KeyError(f"unknown model type {model_type!r}") from exc

    def model_types(self) -> tuple[str, ...]:
        return tuple(sorted(self._definitions))


def _build_default_registry() -> ModelDefinitionRegistry:
    from mindclade.models.families.clade.cladefold.architecture.cladefold import CladeFoldModel
    from mindclade.models.families.clade.cladefold.configuration.cladefold_q0 import CladeFoldConfig

    registry = ModelDefinitionRegistry()
    registry.register(CladeFoldConfig.model_type, CladeFoldConfig, CladeFoldModel)
    return registry


default_registry = _build_default_registry()


__all__ = ["ModelDefinition", "ModelDefinitionRegistry", "default_registry"]
