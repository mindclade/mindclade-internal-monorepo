"""Bounded systems-reference capability declaration."""

from __future__ import annotations

from mindclade.models.api.capabilities import ModelCapabilities


def cladefold_q0_capabilities() -> ModelCapabilities:
    return ModelCapabilities(
        schema_version="v1alpha1",
        model_type="cladefold-q0",
        inputs=("tokens", "atoms", "bonds", "noisy_coordinates", "diffusion_time"),
        outputs=("predicted_noise", "coordinates", "distogram", "confidence"),
        supports_training=True,
        supports_sampling=True,
        claim_level="systems-reference-only-random-initialization",
    )


__all__ = ["cladefold_q0_capabilities"]
