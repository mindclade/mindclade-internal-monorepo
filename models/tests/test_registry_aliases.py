from __future__ import annotations

import pytest

from mindclade.models import CladeFoldModel
from mindclade.models.registry.alias_policy import AliasPolicyError, validate_alias
from mindclade.models.registry.model_definition_registry import default_registry


def test_registry_resolves_q0_without_constructing_weights() -> None:
    definition = default_registry.resolve("cladefold-q0")
    assert definition.model_class is CladeFoldModel


@pytest.mark.parametrize("alias", ["latest", "production", "scientific", "other-random"])
def test_unsafe_aliases_fail_closed(alias: str) -> None:
    with pytest.raises(AliasPolicyError):
        validate_alias(alias)


def test_explicit_random_initialization_alias_is_legal() -> None:
    assert validate_alias("cladefold-q0-random-init") == "cladefold-q0-random-init"
