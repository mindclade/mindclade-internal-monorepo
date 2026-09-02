"""Canonical inference result artifact manifest."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .._identity import canonical_json_bytes, content_digest, require_sha256_digest


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class ArtifactFile:
    path: str
    digest: str
    size_bytes: int
    media_type: str

    def __post_init__(self) -> None:
        if type(self.path) is not str or "\\" in self.path or "\0" in self.path:
            raise ValueError("artifact file path must be a normalized POSIX string")
        normalized = PurePosixPath(self.path)
        if normalized.is_absolute() or ".." in normalized.parts or str(normalized) != self.path:
            raise ValueError("artifact file path must be normalized and relative")
        if not self.path or self.path == ".":
            raise ValueError("artifact file path cannot be empty")
        if type(self.digest) is not str:
            raise ValueError("file digest must be a string")
        object.__setattr__(self, "digest", require_sha256_digest(self.digest, field="file digest"))
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("file size cannot be negative")
        if type(self.media_type) is not str or "/" not in self.media_type:
            raise ValueError("media_type must be a MIME media type")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactFile:
        if set(value) != {"path", "digest", "size_bytes", "media_type"}:
            raise ValueError("artifact file fields are invalid")
        return cls(
            path=value["path"],
            digest=value["digest"],
            size_bytes=value["size_bytes"],
            media_type=value["media_type"],
        )

    @classmethod
    def from_path(cls, root: Path, path: Path, *, media_type: str) -> ArtifactFile:
        if path.is_symlink() or not path.is_file():
            raise ValueError("artifact entries must be regular, non-symlink files")
        resolved_root = root.resolve()
        resolved_path = path.resolve()
        try:
            relative = resolved_path.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError("artifact file is outside the staging root") from exc
        return cls(
            path=relative.as_posix(),
            digest=file_sha256(resolved_path),
            size_bytes=resolved_path.stat().st_size,
            media_type=media_type,
        )


@dataclass(frozen=True, slots=True)
class ResultManifest:
    request_fingerprint: str
    model_digest: str
    serving_revision_digest: str
    sampler_digest: str
    execution_mode: str
    files: tuple[ArtifactFile, ...]
    ranking_evidence_digest: str | None = None
    schema_version: str = "inference-result-manifest.v1alpha1"

    def __post_init__(self) -> None:
        if self.schema_version != "inference-result-manifest.v1alpha1":
            raise ValueError("unsupported result manifest schema")
        for name in (
            "request_fingerprint",
            "model_digest",
            "serving_revision_digest",
            "sampler_digest",
        ):
            value = getattr(self, name)
            if type(value) is not str:
                raise ValueError(f"{name} must be a string")
            object.__setattr__(self, name, require_sha256_digest(value, field=name))
        if self.ranking_evidence_digest is not None:
            if type(self.ranking_evidence_digest) is not str:
                raise ValueError("ranking_evidence_digest must be a string")
            object.__setattr__(
                self,
                "ranking_evidence_digest",
                require_sha256_digest(
                    self.ranking_evidence_digest, field="ranking_evidence_digest"
                ),
            )
        if (
            type(self.files) is not tuple
            or not self.files
            or any(not isinstance(entry, ArtifactFile) for entry in self.files)
        ):
            raise ValueError("artifact manifest must contain ArtifactFile payload entries")
        ordered = tuple(sorted(self.files, key=lambda item: item.path))
        if len({entry.path for entry in ordered}) != len(ordered):
            raise ValueError("artifact manifest paths must be unique")
        if any(entry.path == "manifest.json" for entry in ordered):
            raise ValueError("manifest.json cannot list itself")
        if type(self.execution_mode) is not str or not self.execution_mode:
            raise ValueError("execution_mode must be a non-empty string")
        object.__setattr__(self, "files", ordered)

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_fingerprint": self.request_fingerprint,
            "model_digest": self.model_digest,
            "serving_revision_digest": self.serving_revision_digest,
            "sampler_digest": self.sampler_digest,
            "execution_mode": self.execution_mode,
            "ranking_evidence_digest": self.ranking_evidence_digest,
            "files": [asdict(entry) for entry in self.files],
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResultManifest:
        expected = {
            "schema_version",
            "request_fingerprint",
            "model_digest",
            "serving_revision_digest",
            "sampler_digest",
            "execution_mode",
            "ranking_evidence_digest",
            "files",
        }
        if set(value) != expected:
            raise ValueError("result manifest fields are invalid")
        files = value.get("files")
        if not isinstance(files, list) or not all(isinstance(entry, dict) for entry in files):
            raise ValueError("manifest files must be a list of objects")
        return cls(
            request_fingerprint=value["request_fingerprint"],
            model_digest=value["model_digest"],
            serving_revision_digest=value["serving_revision_digest"],
            sampler_digest=value["sampler_digest"],
            execution_mode=value["execution_mode"],
            ranking_evidence_digest=(
                None
                if value.get("ranking_evidence_digest") is None
                else value["ranking_evidence_digest"]
            ),
            files=tuple(ArtifactFile.from_dict(entry) for entry in files),
            schema_version=value["schema_version"],
        )

    def verify(self, root: Path) -> None:
        if root.is_symlink():
            raise ValueError("artifact root cannot be a symlink")
        resolved = root.resolve()
        manifest_path = resolved / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("artifact manifest is missing or unsafe")
        if manifest_path.read_bytes() != self.to_bytes():
            raise ValueError("artifact manifest bytes are not canonical or do not match")
        expected_paths = {entry.path for entry in self.files} | {"manifest.json"}
        actual_paths: set[str] = set()
        for path in resolved.rglob("*"):
            if path.is_symlink():
                raise ValueError("artifact cannot contain symlinks")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError("artifact contains an unsupported file type")
            actual_paths.add(path.relative_to(resolved).as_posix())
        if actual_paths != expected_paths:
            raise ValueError("artifact file set differs from its manifest")
        for entry in self.files:
            path = (resolved / entry.path).resolve()
            try:
                path.relative_to(resolved)
            except ValueError as exc:
                raise ValueError("artifact manifest path escapes root") from exc
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"artifact file missing or unsafe: {entry.path}")
            if path.stat().st_size != entry.size_bytes:
                raise ValueError(f"artifact file size mismatch: {entry.path}")
            if file_sha256(path) != entry.digest:
                raise ValueError(f"artifact file digest mismatch: {entry.path}")
