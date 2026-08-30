"""Canonical inner model-bundle manifest."""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from .capability_claim_refs import validate_capability_claim_ref


@dataclasses.dataclass(frozen=True)
class BundleFile:
    path: str
    digest: str
    size_bytes: int
    media_type: str

    def __post_init__(self) -> None:
        if (
            type(self.path) is not str
            or not self.path
            or self.path.startswith("/")
            or "\\" in self.path
            or any(part in {"", ".", ".."} for part in self.path.split("/"))
        ):
            raise ValueError("bundle file path must be relative and cannot traverse")
        if (
            type(self.digest) is not str
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", self.digest)
            or type(self.size_bytes) is not int
            or self.size_bytes < 1
        ):
            raise ValueError("bundle file must have a sha256 digest and positive size")
        if type(self.media_type) is not str or not self.media_type:
            raise ValueError("bundle file media_type cannot be empty")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BundleFile:
        required = {field.name for field in dataclasses.fields(cls)}
        if set(value) != required:
            raise ValueError("bundle file fields are invalid")
        return cls(
            path=value["path"],
            digest=value["digest"],
            size_bytes=value["size_bytes"],
            media_type=value["media_type"],
        )


@dataclasses.dataclass(frozen=True)
class BundleManifest:
    model_type: str
    architecture_version: str
    source_revision: str
    config_digest: str
    files: tuple[BundleFile, ...]
    capability_claim_refs: tuple[str, ...]
    claims: tuple[str, ...] = ()
    schema_version: str = "v1alpha1"
    artifact_type: str = "application/vnd.mindclade.model.bundle.v1"

    def __post_init__(self) -> None:
        for name in (
            "model_type",
            "architecture_version",
            "source_revision",
            "config_digest",
            "schema_version",
            "artifact_type",
        ):
            if type(getattr(self, name)) is not str:
                raise ValueError(f"bundle manifest {name} must be a string")
        if self.schema_version != "v1alpha1":
            raise ValueError("unsupported bundle manifest schema")
        if self.artifact_type != "application/vnd.mindclade.model.bundle.v1":
            raise ValueError("unsupported bundle artifact type")
        if not self.model_type or not self.architecture_version:
            raise ValueError("model_type and architecture_version cannot be empty")
        if not re.fullmatch(r"[0-9a-f]{7,64}", self.source_revision):
            raise ValueError("source_revision must be an immutable Git revision")
        if type(self.claims) is not tuple or any(type(claim) is not str for claim in self.claims):
            raise ValueError("bundle claims must be a tuple of strings")
        if self.claims:
            raise ValueError("Q0 bundles cannot carry scientific performance claims")
        if type(self.files) is not tuple or any(
            not isinstance(entry, BundleFile) for entry in self.files
        ):
            raise ValueError("bundle files must be BundleFile entries")
        paths = [entry.path for entry in self.files]
        if not paths or paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("bundle files must be unique and sorted")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.config_digest):
            raise ValueError("config_digest must be sha256-addressed")
        config = next((entry for entry in self.files if entry.path == "config.json"), None)
        if config is None or config.digest != self.config_digest:
            raise ValueError("config_digest must identify files/config.json")
        if type(self.capability_claim_refs) is not tuple or any(
            type(reference) is not str for reference in self.capability_claim_refs
        ):
            raise ValueError("capability claim references must be a tuple of strings")
        if len(set(self.capability_claim_refs)) != len(self.capability_claim_refs):
            raise ValueError("capability claim references must be unique")
        for reference in self.capability_claim_refs:
            validate_capability_claim_ref(reference)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "architecture_version": self.architecture_version,
            "capability_claim_refs": list(self.capability_claim_refs),
            "claims": list(self.claims),
            "config_digest": self.config_digest,
            "files": [dataclasses.asdict(entry) for entry in self.files],
            "model_type": self.model_type,
            "schema_version": self.schema_version,
            "source_revision": self.source_revision,
        }

    def canonical_bytes(self) -> bytes:
        # Values in this schema are strings/integers/lists only; canonical key
        # ordering and compact UTF-8 therefore coincide with RFC 8785 output.
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BundleManifest:
        required = {field.name for field in dataclasses.fields(cls)}
        if set(value) != required:
            raise ValueError("bundle manifest fields are invalid")
        raw_files = value.get("files")
        if not isinstance(raw_files, list) or not all(
            isinstance(entry, dict) for entry in raw_files
        ):
            raise ValueError("bundle manifest files must be an array of objects")
        raw_refs = value["capability_claim_refs"]
        raw_claims = value["claims"]
        if not isinstance(raw_refs, list) or not isinstance(raw_claims, list):
            raise TypeError("bundle claims and claim references must be arrays")
        return cls(
            model_type=value["model_type"],
            architecture_version=value["architecture_version"],
            source_revision=value["source_revision"],
            config_digest=value["config_digest"],
            files=tuple(BundleFile.from_dict(entry) for entry in raw_files),
            capability_claim_refs=tuple(raw_refs),
            claims=tuple(raw_claims),
            schema_version=value["schema_version"],
            artifact_type=value["artifact_type"],
        )


def sorted_bundle_files(entries: Iterable[BundleFile]) -> tuple[BundleFile, ...]:
    return tuple(sorted(entries, key=lambda entry: entry.path))


__all__ = ["BundleFile", "BundleManifest", "sorted_bundle_files"]
