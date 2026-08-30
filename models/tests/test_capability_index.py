from __future__ import annotations

import pytest

from mindclade.models.api.capabilities import ModelCapabilities
from mindclade.models.registry.capability_index import CapabilityIndex


@pytest.fixture
def capabilities() -> ModelCapabilities:
    return ModelCapabilities(
        schema_version="v1alpha1",
        model_type="cladefold-q0",
        inputs=("token_type",),
        outputs=("coordinates",),
        supports_training=True,
        supports_sampling=True,
    )


def test_capability_index_accepts_exact_lowercase_sha256_digest(
    capabilities: ModelCapabilities,
) -> None:
    digest = "sha256:" + "a" * 64
    index = CapabilityIndex()
    index.add(digest, capabilities)
    assert index.snapshot() == {digest: capabilities}


@pytest.mark.parametrize(
    "digest",
    [
        "sha256:",
        "sha256:" + "a" * 63,
        "sha256:" + "a" * 65,
        "sha256:" + "A" * 64,
        "sha512:" + "a" * 64,
        "prefix-sha256:" + "a" * 64,
        "sha256:" + "a" * 64 + "\n",
        7,
        None,
    ],
)
def test_capability_index_rejects_noncanonical_digest(
    digest: object, capabilities: ModelCapabilities
) -> None:
    with pytest.raises(ValueError, match="immutable sha256 model digest"):
        CapabilityIndex().add(digest, capabilities)  # type: ignore[arg-type]
