"""Export a verified CladeFold local bundle."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mindclade.models.families.clade.cladefold.architecture.cladefold import CladeFoldModel
from mindclade.models.packaging.bundle_manifest import BundleManifest
from mindclade.models.packaging.bundle_signing import BundleSigner
from mindclade.models.packaging.model_bundle import ModelBundle


def export_bundle(
    model: CladeFoldModel,
    directory: str | Path,
    *,
    signer: BundleSigner,
    qualification: Mapping[str, Any],
    conversion_receipt: Mapping[str, Any],
    sbom: Mapping[str, Any],
    provenance: Mapping[str, Any],
    source_revision: str,
) -> BundleManifest:
    from mindclade.models.families.clade.cladefold.capabilities.capability_manifest import (
        cladefold_q0_capabilities,
    )

    return ModelBundle.create(
        directory,
        model,
        signer=signer,
        capabilities=cladefold_q0_capabilities().to_dict(),
        qualification=qualification,
        conversion_receipt=conversion_receipt,
        sbom=sbom,
        provenance=provenance,
        architecture_version="q0",
        source_revision=source_revision,
    )


__all__ = ["export_bundle"]
