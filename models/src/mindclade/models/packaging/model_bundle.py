"""Integrity-checked local model release bundle."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

from mindclade.models.api.model import PretrainedModel
from mindclade.models.api.serialization import SerializationError, sha256_file

from .bundle_manifest import BundleFile, BundleManifest, sorted_bundle_files
from .bundle_signing import BundleSigner, BundleVerifier, SignatureEnvelope
from .capability_claim_refs import validate_capability_claim_ref

ModelT = TypeVar("ModelT", bound=PretrainedModel[Any])


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _media_type(path: str) -> str:
    if path.endswith(".safetensors"):
        return "application/vnd.safetensors"
    if path.endswith(".json"):
        return "application/json"
    return "application/octet-stream"


class ModelBundle:
    MANIFEST = "bundle.manifest.json"
    SIGNATURE = "bundle.manifest.sig.json"

    @classmethod
    def create(
        cls,
        directory: os.PathLike[str] | str,
        model: PretrainedModel[Any],
        *,
        signer: BundleSigner,
        capabilities: Mapping[str, Any],
        qualification: Mapping[str, Any],
        conversion_receipt: Mapping[str, Any],
        sbom: Mapping[str, Any],
        provenance: Mapping[str, Any],
        architecture_version: str,
        source_revision: str,
        capability_claim_refs: tuple[str, ...] = (),
        max_shard_size: int | str = "4GiB",
    ) -> BundleManifest:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        if any(target.iterdir()):
            raise FileExistsError("bundle output directory must be empty")
        model.save_pretrained(target, max_shard_size=max_shard_size)
        documents = {
            "capabilities.json": capabilities,
            "qualification.json": qualification,
            "conversion-receipt.json": conversion_receipt,
            "sbom.spdx.json": sbom,
            "provenance.json": provenance,
        }
        for filename, value in documents.items():
            _atomic_json(target / filename, value)
        refs = tuple(
            validate_capability_claim_ref(reference) for reference in capability_claim_refs
        )
        entries = []
        for path in target.iterdir():
            if not path.is_file() or path.name in {cls.MANIFEST, cls.SIGNATURE}:
                continue
            entries.append(
                BundleFile(
                    path=path.name,
                    digest=sha256_file(path),
                    size_bytes=path.stat().st_size,
                    media_type=_media_type(path.name),
                )
            )
        config_entry = next(entry for entry in entries if entry.path == "config.json")
        manifest = BundleManifest(
            model_type=model.config.model_type,
            architecture_version=architecture_version,
            source_revision=source_revision,
            config_digest=config_entry.digest,
            files=sorted_bundle_files(entries),
            capability_claim_refs=refs,
        )
        payload = manifest.canonical_bytes()
        signature = SignatureEnvelope(key_id=signer.key_id, signature=signer.sign(payload))
        temporary = target / f".{cls.MANIFEST}.{os.getpid()}.tmp"
        temporary.write_bytes(payload)
        os.replace(temporary, target / cls.MANIFEST)
        _atomic_json(target / cls.SIGNATURE, signature.to_dict())
        return manifest

    @classmethod
    def verify(
        cls, directory: os.PathLike[str] | str, *, verifier: BundleVerifier
    ) -> BundleManifest:
        source = Path(directory)
        try:
            payload = (source / cls.MANIFEST).read_bytes()
            manifest_value = json.loads(payload)
            signature_value = json.loads((source / cls.SIGNATURE).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SerializationError(f"invalid model bundle metadata: {exc}") from exc
        manifest = BundleManifest.from_dict(manifest_value)
        if payload != manifest.canonical_bytes():
            raise SerializationError("bundle manifest is not canonical")
        signature = SignatureEnvelope.from_dict(signature_value)
        if signature.key_id != verifier.key_id:
            raise SerializationError("bundle signature key identity mismatch")
        try:
            verifier.verify(payload, signature.signature)
        except Exception as exc:
            raise SerializationError("bundle signature verification failed") from exc
        expected_files = {entry.path for entry in manifest.files} | {cls.MANIFEST, cls.SIGNATURE}
        actual_files: set[str] = set()
        for path in source.rglob("*"):
            if path.is_symlink():
                raise SerializationError("model bundle cannot contain symbolic links")
            if path.is_file():
                actual_files.add(path.relative_to(source).as_posix())
        if actual_files != expected_files:
            raise SerializationError("model bundle contains unmanifested or missing files")
        for entry in manifest.files:
            path = source / entry.path
            if (
                not path.is_file()
                or path.stat().st_size != entry.size_bytes
                or sha256_file(path) != entry.digest
            ):
                raise SerializationError(f"bundle file verification failed: {entry.path}")
        return manifest

    @classmethod
    def load_model(
        cls,
        directory: os.PathLike[str] | str,
        model_class: type[ModelT],
        *,
        verifier: BundleVerifier,
    ) -> ModelT:
        manifest = cls.verify(directory, verifier=verifier)
        if manifest.model_type != model_class.config_class.model_type:
            raise SerializationError("bundle model type does not match requested model class")
        return model_class.from_pretrained(directory)

    @classmethod
    def oci_layers(cls, directory: os.PathLike[str] | str) -> tuple[Mapping[str, Any], ...]:
        """Return one OCI descriptor per file; no tar wrapper is introduced."""

        source = Path(directory)
        manifest = BundleManifest.from_dict(json.loads((source / cls.MANIFEST).read_text("utf-8")))
        entries = list(manifest.files)
        for name in (cls.MANIFEST, cls.SIGNATURE):
            path = source / name
            entries.append(
                BundleFile(name, sha256_file(path), path.stat().st_size, _media_type(name))
            )
        return tuple(
            {
                "mediaType": entry.media_type,
                "digest": entry.digest,
                "size": entry.size_bytes,
                "annotations": {"org.opencontainers.image.title": entry.path},
            }
            for entry in sorted(entries, key=lambda item: item.path)
        )


__all__ = ["ModelBundle"]
