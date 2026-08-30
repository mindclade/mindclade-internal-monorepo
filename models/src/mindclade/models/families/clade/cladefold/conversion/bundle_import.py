"""Import a signed CladeFold local bundle."""

from __future__ import annotations

from pathlib import Path

from mindclade.models.families.clade.cladefold.architecture.cladefold import CladeFoldModel
from mindclade.models.packaging.bundle_signing import BundleVerifier
from mindclade.models.packaging.model_bundle import ModelBundle


def import_bundle(directory: str | Path, *, verifier: BundleVerifier) -> CladeFoldModel:
    return ModelBundle.load_model(directory, CladeFoldModel, verifier=verifier)


__all__ = ["import_bundle"]
