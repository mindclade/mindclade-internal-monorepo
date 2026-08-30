from __future__ import annotations

import json

import pytest
import torch

from mindclade.models import CladeFoldModel
from mindclade.models.api.serialization import SerializationError
from mindclade.models.packaging.bundle_manifest import BundleManifest
from mindclade.models.packaging.bundle_signing import (
    Ed25519PrivateKeySigner,
    Ed25519PublicKeyVerifier,
)
from mindclade.models.packaging.model_bundle import ModelBundle


def test_pretrained_shards_round_trip_and_corruption_is_rejected(tmp_path, cladefold_model) -> None:
    cladefold_model.eval()
    cladefold_model.save_pretrained(tmp_path, max_shard_size="64KiB")
    restored, info = CladeFoldModel.from_pretrained(tmp_path, output_loading_info=True)
    assert not restored.training
    assert info == {"missing_keys": [], "unexpected_keys": []}
    for name, value in cladefold_model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)

    integrity = json.loads((tmp_path / "model.integrity.json").read_text("utf-8"))
    shard = next(name for name in integrity["files"] if name.endswith(".safetensors"))
    payload = bytearray((tmp_path / shard).read_bytes())
    payload[-1] ^= 1
    (tmp_path / shard).write_bytes(payload)
    with pytest.raises(SerializationError, match="integrity verification failed"):
        CladeFoldModel.from_pretrained(tmp_path)


def test_signed_bundle_round_trip(tmp_path, cladefold_model) -> None:
    cryptography = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    private = cryptography.Ed25519PrivateKey.generate()
    signer = Ed25519PrivateKeySigner(private, "test-ed25519")
    verifier = Ed25519PublicKeyVerifier(private.public_key(), "test-ed25519")
    ModelBundle.create(
        tmp_path,
        cladefold_model,
        signer=signer,
        capabilities={"claim": "systems-reference-only"},
        qualification={"status": "REFERENCE_ONLY"},
        conversion_receipt={"converter": "identity"},
        sbom={"spdxVersion": "SPDX-2.3"},
        provenance={"builder": "test"},
        architecture_version="q0",
        source_revision="0123456789abcdef",
    )
    manifest = ModelBundle.verify(tmp_path, verifier=verifier)
    assert manifest.model_type == "cladefold-q0"
    assert manifest.architecture_version == "q0"
    assert manifest.source_revision == "0123456789abcdef"
    assert all(layer["digest"].startswith("sha256:") for layer in ModelBundle.oci_layers(tmp_path))
    restored = ModelBundle.load_model(tmp_path, CladeFoldModel, verifier=verifier)
    assert not restored.training


def test_bundle_contract_rejects_invalid_claim_refs_unknown_signature_fields_and_extras(
    tmp_path, cladefold_model
) -> None:
    cryptography = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    private = cryptography.Ed25519PrivateKey.generate()
    signer = Ed25519PrivateKeySigner(private, "test-ed25519")
    verifier = Ed25519PublicKeyVerifier(private.public_key(), "test-ed25519")
    manifest = ModelBundle.create(
        tmp_path,
        cladefold_model,
        signer=signer,
        capabilities={},
        qualification={},
        conversion_receipt={},
        sbom={"spdxVersion": "SPDX-2.3"},
        provenance={},
        architecture_version="q0",
        source_revision="0123456789abcdef",
    )
    invalid = dict(manifest.to_dict())
    invalid["capability_claim_refs"] = ["not-a-valid-claim"]
    with pytest.raises(ValueError, match="capability claim reference"):
        BundleManifest.from_dict(invalid)

    signature_path = tmp_path / ModelBundle.SIGNATURE
    signature = json.loads(signature_path.read_text("utf-8"))
    signature["untrusted"] = True
    signature_path.write_text(json.dumps(signature), encoding="utf-8")
    with pytest.raises(ValueError, match="signature envelope fields"):
        ModelBundle.verify(tmp_path, verifier=verifier)

    signature.pop("untrusted")
    signature_path.write_text(json.dumps(signature), encoding="utf-8")
    (tmp_path / "unmanifested.txt").write_text("not signed", encoding="utf-8")
    with pytest.raises(SerializationError, match="unmanifested"):
        ModelBundle.verify(tmp_path, verifier=verifier)
