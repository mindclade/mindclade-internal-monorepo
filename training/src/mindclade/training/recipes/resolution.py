"""Safe YAML recipe resolution with a canonical receipt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .schema import ResolvedRecipe, recipe_from_mapping


@dataclass(frozen=True)
class RecipeReceipt:
    source: Path
    resolved: ResolvedRecipe

    @property
    def sha256(self) -> str:
        return self.resolved.sha256


def resolve_recipe(path: Path) -> RecipeReceipt:
    source = path.resolve()
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load recipe {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("recipe root must be a YAML mapping")
    return RecipeReceipt(source=source, resolved=recipe_from_mapping(payload))
