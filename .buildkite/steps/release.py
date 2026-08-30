"""Build and validate local, unsigned release evidence without publishing."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_IMAGE_NAMES = {
    "artifact-proxy": "artifactProxy",
    "control-plane": "controlPlane",
    "inference-worker": "inferenceWorker",
    "runtime-gateway": "runtimeGateway",
}


def _trusted() -> bool:
    module_path = Path(__file__).parents[1] / "lib" / "trusted_context.py"
    spec = spec_from_file_location("trusted_context", module_path)
    if spec is None or spec.loader is None:
        return False
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return bool(module.buildkite_context_is_trusted(dict(os.environ)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid JSON evidence file {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"JSON evidence file is not an object: {path}")
    return value


def _image_builder() -> Any:
    module_path = Path(__file__).parents[2] / "tools" / "release" / "build_images.py"
    spec = spec_from_file_location("build_images", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit("image evidence validator is unavailable")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_gpu_evidence(path: Path, source_revision: str) -> None:
    evidence = _load_object(path)
    accelerators = evidence.get("accelerators")
    device_count = evidence.get("device_count")
    if (
        evidence.get("source_revision") != source_revision
        or evidence.get("cuda_available") is not True
        or evidence.get("bf16_supported") is not True
        or not isinstance(evidence.get("torch_version"), str)
        or not str(evidence.get("compiled_cuda", "")).startswith("13.0")
        or evidence.get("qualification_dtype") != "bfloat16"
        or evidence.get("world_size") != 2
        or not isinstance(device_count, int)
        or isinstance(device_count, bool)
        or device_count < 2
        or not isinstance(accelerators, list)
        or len(accelerators) < 2
        or not all(isinstance(accelerator, str) and accelerator for accelerator in accelerators)
    ):
        raise SystemExit("GPU evidence is incomplete or is not bound to this release profile")


def _validate_image_evidence(
    evidence_path: Path, source_revision: str, builder: Any
) -> dict[str, str]:
    image = _load_object(evidence_path)
    name = evidence_path.name.removesuffix(".evidence.json")
    if name not in _IMAGE_NAMES:
        raise SystemExit(f"unexpected image evidence: {evidence_path.name}")
    required = {
        "archive_digest",
        "attestation_subject_digest",
        "image_digest",
        "platform",
        "provenance_attestation_digest",
        "sbom_digest",
        "source_revision",
    }
    if not required.issubset(image) or not all(isinstance(image[key], str) for key in required):
        raise SystemExit(f"image evidence schema is incomplete: {evidence_path.name}")
    normalized = {key: str(image[key]) for key in required}
    for field in (
        "archive_digest",
        "attestation_subject_digest",
        "image_digest",
        "provenance_attestation_digest",
        "sbom_digest",
    ):
        if not _DIGEST.fullmatch(normalized[field]):
            raise SystemExit(f"invalid {field} in {evidence_path.name}")
    if normalized["source_revision"] != source_revision:
        raise SystemExit(f"image evidence is not revision-bound: {evidence_path.name}")
    if normalized["platform"] != "linux/amd64":
        raise SystemExit(f"unexpected image platform in {evidence_path.name}")

    archive = evidence_path.with_name(f"{name}.oci.tar")
    metadata_path = evidence_path.with_name(f"{name}.buildkit.json")
    sbom = evidence_path.with_name(f"{name}.spdx.json")
    if not all(path.is_file() for path in (archive, metadata_path, sbom)):
        raise SystemExit(f"retained image artifacts are incomplete for {name}")
    if _sha256(archive) != normalized["archive_digest"]:
        raise SystemExit(f"OCI archive digest mismatch for {name}")
    if _sha256(sbom) != normalized["sbom_digest"]:
        raise SystemExit(f"SPDX document digest mismatch for {name}")
    metadata = _load_object(metadata_path)
    with tempfile.TemporaryDirectory(prefix=f"mindclade-{name}-evidence-") as directory:
        parsed_sbom = Path(directory) / "parsed.spdx.json"
        try:
            provenance_digest, subject_digest = builder._extract_attestations(archive, parsed_sbom)
            image_digest = builder._validate_output_descriptor(archive, metadata, subject_digest)
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            raise SystemExit(f"invalid OCI evidence for {name}: {error}") from error
        if _sha256(parsed_sbom) != normalized["sbom_digest"]:
            raise SystemExit(f"retained SPDX document is not the attested document for {name}")
    if provenance_digest != normalized["provenance_attestation_digest"]:
        raise SystemExit(f"provenance attestation digest mismatch for {name}")
    if subject_digest != normalized["attestation_subject_digest"]:
        raise SystemExit(f"attestation subject digest mismatch for {name}")
    if image_digest != normalized["image_digest"]:
        raise SystemExit(f"image digest mismatch for {name}")
    return normalized


def _deployment_binding(
    images: dict[str, dict[str, str]], raw_values: object
) -> tuple[dict[str, str], str]:
    if not isinstance(raw_values, dict) or set(raw_values) != set(_IMAGE_NAMES.values()):
        raise SystemExit("development deployment image set is incomplete")
    values: dict[str, str] = {}
    bound = True
    for image_name, deployment_key in _IMAGE_NAMES.items():
        reference = raw_values.get(deployment_key)
        if not isinstance(reference, str) or not re.fullmatch(
            r"[^\s@]+@sha256:[0-9a-f]{64}", reference
        ):
            raise SystemExit(f"deployment image is not digest-pinned: {deployment_key}")
        values[deployment_key] = reference
        bound &= reference.rsplit("@", 1)[1] == images[image_name]["image_digest"]
    status = "matches-local-image-digests" if bound else "not-bound"
    return values, status


def main() -> None:
    if not _trusted():
        raise SystemExit("release evidence requires a trusted GitHub dispatcher context")
    subprocess.run(
        [
            "buildkite-agent",
            "artifact",
            "download",
            "gpu-evidence.json",
            ".",
            "--step",
            "gpu",
        ],
        check=True,
    )
    gpu_evidence_path = Path("gpu-evidence.json")
    source_revision = os.environ.get("BUILDKITE_COMMIT", "")
    if not source_revision:
        raise SystemExit("release evidence requires a source revision")
    _validate_gpu_evidence(gpu_evidence_path, source_revision)
    subprocess.run(["just", "build-wheels"], check=True)
    wheel_paths = sorted(Path("dist").glob("*.whl"))
    if len(wheel_paths) != 4:
        raise SystemExit(f"expected four wheels, found {len(wheel_paths)}")
    subprocess.run(
        ["uv", "run", "check-wheel-contents", *(str(path) for path in wheel_paths)],
        check=True,
    )
    subprocess.run(
        ["uv", "run", "python", "protocols/python/wheel_smoke.py", "--wheel-dir", "dist"],
        check=True,
    )
    subprocess.run(["uv", "run", "python", "tools/release/build_images.py"], check=True)
    image_evidence_paths = sorted(Path("dist/images").glob("*.evidence.json"))
    expected_image_evidence = {f"{name}.evidence.json" for name in _IMAGE_NAMES}
    if {path.name for path in image_evidence_paths} != expected_image_evidence:
        raise SystemExit(
            f"expected image evidence {sorted(expected_image_evidence)}, "
            f"found {sorted(path.name for path in image_evidence_paths)}"
        )
    builder = _image_builder()
    images: dict[str, dict[str, str]] = {}
    for evidence_path in image_evidence_paths:
        images[evidence_path.name.removesuffix(".evidence.json")] = _validate_image_evidence(
            evidence_path, source_revision, builder
        )
    deployment_values = _load_object(Path("deploy/values.development.json"))
    deployment_images, deployment_binding_status = _deployment_binding(
        images, deployment_values.get("images")
    )
    wheels = {path.name: _sha256(path) for path in wheel_paths}
    evidence = {
        "gpu_evidence_digest": _sha256(gpu_evidence_path),
        "images": images,
        "deployment_binding_status": deployment_binding_status,
        "deployment_images": deployment_images,
        "publication_status": "local-unsigned",
        "source_revision": source_revision,
        "wheels": wheels,
    }
    Path("release-evidence.json").write_text(
        json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
