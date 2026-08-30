"""Canonical checkpoint commit manifests and integrity verification."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from mindclade.training.api.checkpoint import (
    CheckpointIntegrityError,
    IncompleteCheckpointError,
)

MANIFEST_FILENAME = "manifest.json"
SCHEMA_VERSION = "v1alpha1"
FORMAT_NAME = "torch.distributed.checkpoint"


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckpointFile:
    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        parsed = PurePosixPath(self.path)
        if parsed.is_absolute() or ".." in parsed.parts or self.path in {"", "."}:
            raise ValueError(f"checkpoint file path is unsafe: {self.path!r}")
        if self.path == MANIFEST_FILENAME:
            raise ValueError("manifest must not include itself")
        if self.size < 0:
            raise ValueError("checkpoint file size cannot be negative")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("checkpoint file sha256 must be 64 lowercase hexadecimal characters")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


@dataclass(frozen=True)
class CheckpointManifest:
    checkpoint_id: str
    created_at: str
    global_step: int
    world_size: int
    torch_version: str
    model_schema_sha256: str
    program_sha256: str
    files: tuple[CheckpointFile, ...]
    schema_version: str = SCHEMA_VERSION
    format: str = FORMAT_NAME

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported checkpoint schema {self.schema_version!r}")
        if self.format != FORMAT_NAME:
            raise ValueError(f"unsupported checkpoint format {self.format!r}")
        if not self.checkpoint_id or "/" in self.checkpoint_id or "\\" in self.checkpoint_id:
            raise ValueError("checkpoint_id must be one safe path component")
        if self.global_step < 0 or self.world_size <= 0:
            raise ValueError("global_step must be non-negative and world_size positive")
        if len(self.files) == 0:
            raise ValueError("checkpoint manifest must include payload files")
        paths = [entry.path for entry in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("checkpoint files must be unique and sorted by path")
        for digest in (self.model_schema_sha256, self.program_sha256):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("schema and program digests must be lowercase sha256 values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "created_at": self.created_at,
            "files": [entry.to_dict() for entry in self.files],
            "format": self.format,
            "global_step": self.global_step,
            "model_schema_sha256": self.model_schema_sha256,
            "program_sha256": self.program_sha256,
            "schema_version": self.schema_version,
            "torch_version": self.torch_version,
            "world_size": self.world_size,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointManifest:
        expected = {
            "checkpoint_id",
            "created_at",
            "files",
            "format",
            "global_step",
            "model_schema_sha256",
            "program_sha256",
            "schema_version",
            "torch_version",
            "world_size",
        }
        unknown = set(value) - expected
        missing = expected - set(value)
        if unknown or missing:
            raise ValueError(
                f"invalid checkpoint manifest fields; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        files_value = value["files"]
        if not isinstance(files_value, list):
            raise ValueError("checkpoint files must be a list")
        files = tuple(
            CheckpointFile(
                path=str(item["path"]),
                size=int(item["size"]),
                sha256=str(item["sha256"]),
            )
            for item in files_value
        )
        return cls(
            checkpoint_id=str(value["checkpoint_id"]),
            created_at=str(value["created_at"]),
            global_step=int(value["global_step"]),
            world_size=int(value["world_size"]),
            torch_version=str(value["torch_version"]),
            model_schema_sha256=str(value["model_schema_sha256"]),
            program_sha256=str(value["program_sha256"]),
            files=files,
            schema_version=str(value["schema_version"]),
            format=str(value["format"]),
        )


def inventory_files(root: Path) -> tuple[CheckpointFile, ...]:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CheckpointIntegrityError(f"checkpoint payload cannot contain symlink: {path}")
        if not path.is_file() or path.name == MANIFEST_FILENAME:
            continue
        relative = path.relative_to(root).as_posix()
        entries.append(
            CheckpointFile(path=relative, size=path.stat().st_size, sha256=sha256_file(path))
        )
    return tuple(entries)


def load_manifest(checkpoint: Path) -> CheckpointManifest:
    path = checkpoint / MANIFEST_FILENAME
    if not path.is_file():
        raise IncompleteCheckpointError(f"checkpoint has no commit manifest: {checkpoint}")
    if path.is_symlink():
        raise CheckpointIntegrityError("checkpoint manifest cannot be a symlink")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("manifest root must be an object")
        manifest = CheckpointManifest.from_dict(raw)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CheckpointIntegrityError(f"invalid checkpoint manifest: {exc}") from exc
    if manifest.checkpoint_id != checkpoint.name:
        raise CheckpointIntegrityError(
            f"checkpoint directory {checkpoint.name!r} does not match manifest id "
            f"{manifest.checkpoint_id!r}"
        )
    return manifest


def verify_checkpoint(checkpoint: Path) -> CheckpointManifest:
    manifest = load_manifest(checkpoint)
    actual = inventory_files(checkpoint)
    expected = manifest.files
    actual_paths = {entry.path for entry in actual}
    expected_paths = {entry.path for entry in expected}
    if actual_paths != expected_paths:
        raise CheckpointIntegrityError(
            "checkpoint file set differs from manifest; "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"unexpected={sorted(actual_paths - expected_paths)}"
        )
    actual_by_path = {entry.path: entry for entry in actual}
    for expected_entry in expected:
        actual_entry = actual_by_path[expected_entry.path]
        if actual_entry.size != expected_entry.size or actual_entry.sha256 != expected_entry.sha256:
            raise CheckpointIntegrityError(
                f"checkpoint file integrity mismatch: {expected_entry.path}"
            )
    return manifest


def digest_mapping(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def digest_model_schema(items: Iterable[tuple[str, tuple[int, ...], str]]) -> str:
    rows = [
        {"dtype": dtype, "name": name, "shape": list(shape)} for name, shape, dtype in sorted(items)
    ]
    return digest_mapping({"parameters_and_buffers": rows})
