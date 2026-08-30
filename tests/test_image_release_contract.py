from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import tarfile
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
IMAGE_COMPONENTS = {
    "artifact-proxy": ROOT / "services" / "artifact_proxy",
    "control-plane": ROOT / "services" / "control_plane",
    "inference-worker": ROOT / "workers" / "inference_worker",
    "runtime-gateway": ROOT / "services" / "runtime_gateway",
}
PINNED_FROM = re.compile(r"^FROM\s+\S+@sha256:[0-9a-f]{64}(?:\s+AS\s+\S+)?$", re.MULTILINE)


def _builder_module():
    path = ROOT / "tools" / "release" / "build_images.py"
    spec = importlib.util.spec_from_file_location("build_images", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_image_component_has_a_hardened_pinned_definition() -> None:
    for name, component_root in IMAGE_COMPONENTS.items():
        manifest = yaml.safe_load((component_root / "component.yaml").read_text(encoding="utf-8"))
        assert manifest["metadata"]["name"] == name
        assert manifest["spec"]["release"]["mode"] == "oci-image"
        containerfile = (component_root / "Containerfile").read_text(encoding="utf-8")
        assert re.match(r"^# syntax=\S+@sha256:[0-9a-f]{64}$", containerfile.splitlines()[0])
        from_lines = [line for line in containerfile.splitlines() if line.startswith("FROM ")]
        assert from_lines and all(PINNED_FROM.fullmatch(line) for line in from_lines)
        assert re.search(r"^USER\s+(?!0(?:[:\s]|$))", containerfile, re.MULTILINE)
        assert "LICENSE NOTICE /licenses" in containerfile
        assert "ENTRYPOINT" in containerfile


def test_oci_builder_covers_exact_image_release_set() -> None:
    module = _builder_module()
    assert set(module._IMAGES) == set(IMAGE_COMPONENTS)
    assert all((ROOT / path).is_file() for path in module._IMAGES.values())


def test_artifact_proxy_builder_copies_every_rust_workspace_member() -> None:
    workspace = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    members = workspace["workspace"]["members"]
    containerfile = (IMAGE_COMPONENTS["artifact-proxy"] / "Containerfile").read_text(
        encoding="utf-8"
    )
    for member in members:
        assert re.search(rf"^COPY\s+{re.escape(member)}(?:\s|/)", containerfile, re.MULTILINE), (
            f"artifact-proxy build context omits Cargo workspace member {member}"
        )
    assert "COPY protocols/generated/rust ./protocols/generated/rust" in containerfile


def _json_blob(value: object) -> tuple[str, bytes]:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest(), payload


def _oci_archive(
    path: Path, *, statement_subject_mode: str = "bound"
) -> tuple[dict[str, object], str, str]:
    image_manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {},
        "layers": [],
    }
    image_digest, image_payload = _json_blob(image_manifest)
    if statement_subject_mode == "bound":
        subject = [{"name": "_", "digest": {"sha256": image_digest.removeprefix("sha256:")}}]
    elif statement_subject_mode == "empty":
        subject = []
    elif statement_subject_mode == "wrong":
        subject = [{"name": "_", "digest": {"sha256": "f" * 64}}]
    else:
        raise ValueError(f"unsupported statement subject mode: {statement_subject_mode}")
    spdx_statement = {
        "_type": "https://in-toto.io/Statement/v1",
        "predicateType": "https://spdx.dev/Document",
        "subject": subject,
        "predicate": {
            "SPDXID": "SPDXRef-DOCUMENT",
            "creationInfo": {"created": "2026-08-29T00:00:00Z", "creators": ["Tool: test"]},
            "name": "test-image",
            "spdxVersion": "SPDX-2.3",
        },
    }
    provenance_statement = {
        "_type": "https://in-toto.io/Statement/v1",
        "predicateType": "https://slsa.dev/provenance/v0.2",
        "subject": subject,
        "predicate": {
            "builder": {"id": "test-builder"},
            "buildType": "test-build",
            "invocation": {},
            "materials": [],
            "metadata": {},
        },
    }
    spdx_digest, spdx_payload = _json_blob(spdx_statement)
    provenance_digest, provenance_payload = _json_blob(provenance_statement)
    attestation_manifest = {
        "artifactType": "application/vnd.docker.attestation.manifest.v1+json",
        "config": {},
        "layers": [
            {
                "annotations": {"in-toto.io/predicate-type": "https://spdx.dev/Document"},
                "digest": spdx_digest,
                "mediaType": "application/vnd.in-toto+json",
                "size": len(spdx_payload),
            },
            {
                "annotations": {"in-toto.io/predicate-type": "https://slsa.dev/provenance/v0.2"},
                "digest": provenance_digest,
                "mediaType": "application/vnd.in-toto+json",
                "size": len(provenance_payload),
            },
        ],
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "schemaVersion": 2,
        "subject": {
            "digest": image_digest,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "size": len(image_payload),
        },
    }
    attestation_digest, attestation_payload = _json_blob(attestation_manifest)
    image_descriptor = {
        "digest": image_digest,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "size": len(image_payload),
    }
    attestation_descriptor = {
        "annotations": {
            "vnd.docker.reference.digest": image_digest,
            "vnd.docker.reference.type": "attestation-manifest",
        },
        "digest": attestation_digest,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "size": len(attestation_payload),
    }
    output_index = {
        "manifests": [image_descriptor, attestation_descriptor],
        "mediaType": "application/vnd.oci.image.index.v1+json",
        "schemaVersion": 2,
    }
    output_digest, output_payload = _json_blob(output_index)
    output_descriptor = {
        "digest": output_digest,
        "mediaType": "application/vnd.oci.image.index.v1+json",
        "size": len(output_payload),
    }
    root_index = {
        "manifests": [output_descriptor],
        "mediaType": "application/vnd.oci.image.index.v1+json",
        "schemaVersion": 2,
    }
    blobs = {
        output_digest: output_payload,
        image_digest: image_payload,
        attestation_digest: attestation_payload,
        spdx_digest: spdx_payload,
        provenance_digest: provenance_payload,
    }
    with tarfile.open(path, "w") as archive:
        for name, payload in {
            "index.json": json.dumps(root_index).encode(),
            **{
                f"blobs/sha256/{digest.removeprefix('sha256:')}": payload
                for digest, payload in blobs.items()
            },
        }.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return output_descriptor, provenance_digest, image_digest


def test_oci_evidence_parser_binds_metadata_sbom_and_provenance(tmp_path: Path) -> None:
    module = _builder_module()
    archive = tmp_path / "image.oci.tar"
    output_descriptor, provenance_digest, image_digest = _oci_archive(archive)
    sbom = tmp_path / "image.spdx.json"

    parsed_provenance, parsed_subject = module._extract_attestations(archive, sbom)
    assert parsed_provenance == provenance_digest
    assert parsed_subject == image_digest
    assert json.loads(sbom.read_text(encoding="utf-8"))["SPDXID"] == "SPDXRef-DOCUMENT"
    metadata = {
        "containerimage.descriptor": output_descriptor,
        "containerimage.digest": output_descriptor["digest"],
    }
    assert (
        module._validate_output_descriptor(archive, metadata, parsed_subject)
        == output_descriptor["digest"]
    )


def test_oci_evidence_parser_accepts_buildkit_empty_statement_subjects(tmp_path: Path) -> None:
    module = _builder_module()
    archive = tmp_path / "image.oci.tar"
    _, provenance_digest, image_digest = _oci_archive(archive, statement_subject_mode="empty")

    parsed_provenance, parsed_subject = module._extract_attestations(
        archive, tmp_path / "image.spdx.json"
    )

    assert parsed_provenance == provenance_digest
    assert parsed_subject == image_digest


def test_oci_evidence_parser_rejects_mismatched_statement_subjects(tmp_path: Path) -> None:
    module = _builder_module()
    archive = tmp_path / "image.oci.tar"
    _oci_archive(archive, statement_subject_mode="wrong")

    with pytest.raises(RuntimeError, match="not bound"):
        module._extract_attestations(archive, tmp_path / "image.spdx.json")


def test_oci_evidence_parser_rejects_unbound_metadata(tmp_path: Path) -> None:
    module = _builder_module()
    archive = tmp_path / "image.oci.tar"
    output_descriptor, _, image_digest = _oci_archive(archive)
    metadata = {
        "containerimage.descriptor": {**output_descriptor, "size": 1},
        "containerimage.digest": output_descriptor["digest"],
    }
    with pytest.raises(RuntimeError, match="descriptor mismatch"):
        module._validate_output_descriptor(archive, metadata, image_digest)
