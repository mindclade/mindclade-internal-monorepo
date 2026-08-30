"""Build local OCI archives with BuildKit SBOM and provenance attestations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

_IMAGES = {
    "artifact-proxy": Path("services/artifact_proxy/Containerfile"),
    "control-plane": Path("services/control_plane/Containerfile"),
    "inference-worker": Path("workers/inference_worker/Containerfile"),
    "runtime-gateway": Path("services/runtime_gateway/Containerfile"),
}
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_IN_TOTO_STATEMENT_TYPES = {
    "https://in-toto.io/Statement/v0.1",
    "https://in-toto.io/Statement/v1",
}
_SPDX_PREDICATE_TYPE = "https://spdx.dev/Document"
_SLSA_PREDICATE_TYPES = {
    "https://slsa.dev/provenance/v0.2",
    "https://slsa.dev/provenance/v1",
}
_OCI_IMAGE_INDEX = "application/vnd.oci.image.index.v1+json"
_OCI_IMAGE_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
_ATTESTATION_ARTIFACT = "application/vnd.docker.attestation.manifest.v1+json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _run(command: list[str], *, root: Path) -> None:
    subprocess.run(command, cwd=root, check=True)


def _oci_json(archive: Path, member_name: str) -> dict[str, object]:
    with tarfile.open(archive, "r") as source:
        member = source.getmember(member_name)
        stream = source.extractfile(member)
        if stream is None or member.size > 32 * 1024 * 1024:
            raise RuntimeError(f"invalid OCI archive member: {member_name}")
        value = json.loads(stream.read())
    if not isinstance(value, dict):
        raise RuntimeError(f"OCI archive member is not an object: {member_name}")
    return value


def _oci_blob_json(archive: Path, digest: str) -> dict[str, object]:
    if not _DIGEST.fullmatch(digest):
        raise RuntimeError("OCI descriptor contains an invalid digest")
    algorithm, hexadecimal = digest.split(":", 1)
    member_name = f"blobs/{algorithm}/{hexadecimal}"
    with tarfile.open(archive, "r") as source:
        member = source.getmember(member_name)
        stream = source.extractfile(member)
        if stream is None or member.size > 32 * 1024 * 1024:
            raise RuntimeError(f"invalid OCI blob: {digest}")
        payload = stream.read()
    if "sha256:" + hashlib.sha256(payload).hexdigest() != digest:
        raise RuntimeError(f"OCI blob digest mismatch: {digest}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise RuntimeError(f"OCI blob is not an object: {digest}")
    return value


def _validate_attestation_statement(
    statement: dict[str, object], predicate_type: str
) -> tuple[str, dict[str, object], frozenset[str]]:
    if statement.get("_type") not in _IN_TOTO_STATEMENT_TYPES:
        raise RuntimeError("attestation is not a supported in-toto statement")
    if statement.get("predicateType") != predicate_type:
        raise RuntimeError("attestation predicate type does not match its OCI annotation")
    raw_subjects = statement.get("subject")
    if not isinstance(raw_subjects, list):
        raise RuntimeError("attestation omitted its subject")
    subjects: set[str] = set()
    for subject in raw_subjects:
        if not isinstance(subject, dict) or not isinstance(subject.get("name"), str):
            raise RuntimeError("attestation contains an invalid subject")
        digests = subject.get("digest")
        hexadecimal = digests.get("sha256") if isinstance(digests, dict) else None
        digest = f"sha256:{hexadecimal}"
        if not isinstance(hexadecimal, str) or not _DIGEST.fullmatch(digest):
            raise RuntimeError("attestation subject omitted its SHA-256 digest")
        subjects.add(digest)
    predicate = statement.get("predicate")
    if not isinstance(predicate, dict):
        raise RuntimeError("attestation omitted its predicate")
    if predicate_type == _SPDX_PREDICATE_TYPE:
        spdx_version = predicate.get("spdxVersion")
        creation = predicate.get("creationInfo")
        if (
            not isinstance(spdx_version, str)
            or not spdx_version.startswith("SPDX-")
            or predicate.get("SPDXID") != "SPDXRef-DOCUMENT"
            or not isinstance(predicate.get("name"), str)
            or not isinstance(creation, dict)
            or not isinstance(creation.get("created"), str)
            or not isinstance(creation.get("creators"), list)
        ):
            raise RuntimeError("SPDX attestation predicate is incomplete")
        return "spdx", predicate, frozenset(subjects)
    if predicate_type == "https://slsa.dev/provenance/v0.2":
        builder = predicate.get("builder")
        if (
            not isinstance(builder, dict)
            or not isinstance(builder.get("id"), str)
            or not isinstance(predicate.get("buildType"), str)
            or not isinstance(predicate.get("invocation"), dict)
            or not isinstance(predicate.get("metadata"), dict)
            or not isinstance(predicate.get("materials"), list)
        ):
            raise RuntimeError("SLSA v0.2 provenance predicate is incomplete")
        return "provenance", predicate, frozenset(subjects)
    if predicate_type == "https://slsa.dev/provenance/v1":
        if not isinstance(predicate.get("buildDefinition"), dict) or not isinstance(
            predicate.get("runDetails"), dict
        ):
            raise RuntimeError("SLSA v1 provenance predicate is incomplete")
        return "provenance", predicate, frozenset(subjects)
    raise RuntimeError(f"unsupported attestation predicate type: {predicate_type}")


def _find_oci_descriptor(
    archive: Path,
    index: dict[str, object],
    target_digest: str,
    visited_indexes: set[str] | None = None,
) -> dict[str, object] | None:
    visited = visited_indexes if visited_indexes is not None else set()
    descriptors = index.get("manifests")
    if not isinstance(descriptors, list):
        raise RuntimeError("OCI index omitted manifests")
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            continue
        digest = descriptor.get("digest")
        if digest == target_digest:
            return descriptor
        if descriptor.get("mediaType") != _OCI_IMAGE_INDEX or not isinstance(digest, str):
            continue
        if digest in visited:
            continue
        visited.add(digest)
        found = _find_oci_descriptor(
            archive, _oci_blob_json(archive, digest), target_digest, visited
        )
        if found is not None:
            return found
    return None


def _runnable_manifest_digests(
    archive: Path, descriptor: dict[str, object], visited_indexes: set[str] | None = None
) -> frozenset[str]:
    annotations = descriptor.get("annotations")
    if (
        isinstance(annotations, dict)
        and annotations.get("vnd.docker.reference.type") == "attestation-manifest"
    ):
        return frozenset()
    digest = descriptor.get("digest")
    media_type = descriptor.get("mediaType")
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise RuntimeError("OCI image descriptor contains an invalid digest")
    if media_type == _OCI_IMAGE_MANIFEST:
        manifest = _oci_blob_json(archive, digest)
        if manifest.get("schemaVersion") != 2 or not isinstance(manifest.get("layers"), list):
            raise RuntimeError("OCI image manifest is incomplete")
        return frozenset({digest})
    if media_type != _OCI_IMAGE_INDEX:
        raise RuntimeError(f"unsupported OCI image descriptor media type: {media_type}")
    visited = visited_indexes if visited_indexes is not None else set()
    if digest in visited:
        raise RuntimeError("OCI image index graph contains a cycle")
    visited.add(digest)
    nested = _oci_blob_json(archive, digest)
    descriptors = nested.get("manifests")
    if not isinstance(descriptors, list):
        raise RuntimeError("OCI image index omitted manifests")
    subjects: set[str] = set()
    for child in descriptors:
        if isinstance(child, dict):
            subjects.update(_runnable_manifest_digests(archive, child, visited))
    return frozenset(subjects)


def _validate_output_descriptor(
    archive: Path, build_metadata: dict[str, object], attestation_subject_digest: str
) -> str:
    image_digest = build_metadata.get("containerimage.digest")
    if not isinstance(image_digest, str) or not _DIGEST.fullmatch(image_digest):
        raise RuntimeError("BuildKit did not report an immutable image digest")
    metadata_descriptor = build_metadata.get("containerimage.descriptor")
    if not isinstance(metadata_descriptor, dict):
        raise RuntimeError("BuildKit did not report its OCI image descriptor")
    archive_descriptor = _find_oci_descriptor(
        archive, _oci_json(archive, "index.json"), image_digest
    )
    if archive_descriptor is None:
        raise RuntimeError("BuildKit image digest is absent from the OCI archive")
    for field in ("digest", "mediaType", "size"):
        if archive_descriptor.get(field) != metadata_descriptor.get(field):
            raise RuntimeError(f"BuildKit OCI descriptor mismatch: {field}")
    if attestation_subject_digest not in _runnable_manifest_digests(archive, archive_descriptor):
        raise RuntimeError("BuildKit attestations are not bound to the output image")
    return image_digest


def _extract_attestations(archive: Path, sbom: Path) -> tuple[str, str]:
    indexes = [_oci_json(archive, "index.json")]
    visited_indexes: set[str] = set()
    attestation_descriptors: list[tuple[dict[str, object], frozenset[str]]] = []
    while indexes:
        index = indexes.pop()
        descriptors = index.get("manifests")
        if not isinstance(descriptors, list):
            raise RuntimeError("OCI index omitted manifests")
        sibling_digests = frozenset(
            descriptor["digest"]
            for descriptor in descriptors
            if isinstance(descriptor, dict)
            and isinstance(descriptor.get("digest"), str)
            and not (
                isinstance(descriptor.get("annotations"), dict)
                and descriptor["annotations"].get("vnd.docker.reference.type")
                == "attestation-manifest"
            )
        )
        for descriptor in descriptors:
            if not isinstance(descriptor, dict):
                continue
            digest = descriptor.get("digest")
            media_type = descriptor.get("mediaType")
            if not isinstance(digest, str):
                continue
            if media_type == _OCI_IMAGE_INDEX:
                if digest in visited_indexes:
                    continue
                visited_indexes.add(digest)
                indexes.append(_oci_blob_json(archive, digest))
                continue
            annotations = descriptor.get("annotations")
            if (
                isinstance(annotations, dict)
                and annotations.get("vnd.docker.reference.type") == "attestation-manifest"
            ):
                attestation_descriptors.append((descriptor, sibling_digests))
    attestation_digest = ""
    attestation_subject_digest = ""
    spdx_predicate: dict[str, object] | None = None
    for descriptor, sibling_digests in attestation_descriptors:
        digest = descriptor.get("digest")
        if not isinstance(digest, str):
            continue
        descriptor_annotations = descriptor.get("annotations")
        reference_digest = (
            descriptor_annotations.get("vnd.docker.reference.digest")
            if isinstance(descriptor_annotations, dict)
            else None
        )
        if not isinstance(reference_digest, str) or not _DIGEST.fullmatch(reference_digest):
            raise RuntimeError("attestation manifest omitted its image subject digest")
        if reference_digest not in sibling_digests:
            raise RuntimeError("attestation references no runnable sibling image manifest")
        manifest = _oci_blob_json(archive, digest)
        manifest_subject = manifest.get("subject")
        if (
            manifest.get("artifactType") != _ATTESTATION_ARTIFACT
            or not isinstance(manifest_subject, dict)
            or manifest_subject.get("digest") != reference_digest
            or manifest_subject.get("mediaType") != _OCI_IMAGE_MANIFEST
        ):
            raise RuntimeError("attestation manifest is not bound to its OCI image subject")
        layers = manifest.get("layers")
        if not isinstance(layers, list):
            raise RuntimeError("attestation manifest omitted its layers")
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            layer_annotations = layer.get("annotations")
            predicate_type = (
                layer_annotations.get("in-toto.io/predicate-type", "")
                if isinstance(layer_annotations, dict)
                else ""
            )
            layer_digest = layer.get("digest")
            if not isinstance(predicate_type, str) or not isinstance(layer_digest, str):
                continue
            if layer.get("mediaType") != "application/vnd.in-toto+json":
                continue
            if predicate_type not in {_SPDX_PREDICATE_TYPE, *_SLSA_PREDICATE_TYPES}:
                continue
            statement = _oci_blob_json(archive, layer_digest)
            kind, predicate, subjects = _validate_attestation_statement(statement, predicate_type)
            # BuildKit may leave an in-toto statement's subject array empty when the
            # enclosing OCI attestation manifest carries the subject descriptor. A
            # non-empty statement remains an additional binding and must agree.
            if subjects and reference_digest not in subjects:
                raise RuntimeError("attestation subject is not bound to its image manifest")
            if attestation_subject_digest and attestation_subject_digest != reference_digest:
                raise RuntimeError("SBOM and provenance attest different image manifests")
            attestation_subject_digest = reference_digest
            if kind == "spdx":
                if spdx_predicate is not None:
                    raise RuntimeError("BuildKit emitted more than one SPDX attestation")
                spdx_predicate = predicate
            else:
                if attestation_digest:
                    raise RuntimeError("BuildKit emitted more than one provenance attestation")
                attestation_digest = layer_digest
    if spdx_predicate is None or not attestation_digest or not attestation_subject_digest:
        raise RuntimeError("BuildKit did not emit both SPDX and provenance attestations")
    sbom.write_text(json.dumps(spdx_predicate, sort_keys=True) + "\n", encoding="utf-8")
    return attestation_digest, attestation_subject_digest


def _build(name: str, containerfile: Path, *, platform: str, root: Path) -> dict[str, str]:
    output_root = root / "dist" / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=output_root) as temporary:
        temporary_root = Path(temporary)
        archive = temporary_root / f"{name}.oci.tar"
        metadata = temporary_root / f"{name}.buildkit.json"
        sbom = temporary_root / f"{name}.spdx.json"
        _run(
            [
                "docker",
                "buildx",
                "build",
                "--file",
                str(containerfile),
                "--platform",
                platform,
                "--provenance=mode=max",
                "--sbom=true",
                "--output",
                f"type=oci,dest={archive},oci-artifact=true",
                "--metadata-file",
                str(metadata),
                ".",
            ],
            root=root,
        )
        provenance_attestation_digest, attestation_subject_digest = _extract_attestations(
            archive, sbom
        )
        build_metadata = json.loads(metadata.read_text(encoding="utf-8"))
        if not isinstance(build_metadata, dict):
            raise RuntimeError(f"BuildKit metadata is not an object for {name}")
        image_digest = _validate_output_descriptor(
            archive, build_metadata, attestation_subject_digest
        )
        evidence = {
            "archive_digest": _sha256(archive),
            "attestation_subject_digest": attestation_subject_digest,
            "image_digest": image_digest,
            "platform": platform,
            "provenance_attestation_digest": provenance_attestation_digest,
            "sbom_digest": _sha256(sbom),
            "source_revision": os.environ.get("BUILDKITE_COMMIT")
            or os.environ.get("GITHUB_SHA", ""),
        }
        evidence_path = temporary_root / f"{name}.evidence.json"
        evidence_path.write_text(json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8")
        for source in (archive, metadata, sbom, evidence_path):
            os.replace(source, output_root / source.name)
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", action="append", choices=sorted(_IMAGES))
    parser.add_argument("--platform", default="linux/amd64")
    args = parser.parse_args(argv)
    root = Path(os.environ.get("BUILD_WORKSPACE_DIRECTORY", Path.cwd())).resolve()
    selected = sorted(set(args.image or _IMAGES))
    evidence = {
        name: _build(name, _IMAGES[name], platform=args.platform, root=root) for name in selected
    }
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
