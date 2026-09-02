"""Atomic local artifact commit; remote stores consume the same manifest contract."""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .result_manifest import ArtifactFile, ResultManifest

_ATTEMPT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class CommittedArtifact:
    path: Path
    manifest: ResultManifest
    created: bool

    @property
    def digest(self) -> str:
        return self.manifest.digest


class ArtifactCommitter:
    """Publish a complete directory by same-filesystem rename, manifest last."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.staging_root = self.root / ".staging"
        self.objects_root = self.root / "sha256"
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.objects_root.mkdir(parents=True, exist_ok=True)

    def begin(self, attempt_id: str) -> Path:
        if not _ATTEMPT_ID.fullmatch(attempt_id):
            raise ValueError("attempt_id contains unsafe characters")
        return Path(tempfile.mkdtemp(prefix=f"{attempt_id}-", dir=self.staging_root))

    def commit(
        self,
        staging: Path,
        *,
        request_fingerprint: str,
        model_digest: str,
        serving_revision_digest: str,
        sampler_digest: str,
        execution_mode: str,
        media_types: Mapping[str, str],
        ranking_evidence_digest: str | None = None,
    ) -> CommittedArtifact:
        if staging.is_symlink():
            raise ValueError("staging path must be a non-symlink directory")
        staging = staging.resolve()
        try:
            staging.relative_to(self.staging_root)
        except ValueError as exc:
            raise ValueError("staging directory is outside the artifact staging root") from exc
        if not staging.is_dir():
            raise ValueError("staging path must be a non-symlink directory")
        if (staging / "manifest.json").exists():
            raise ValueError("staging directory already contains a manifest")

        files: list[ArtifactFile] = []
        discovered = []
        for path in staging.rglob("*"):
            if path.is_symlink():
                raise ValueError("artifact staging cannot contain symlinks")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError("artifact staging contains an unsupported file type")
            discovered.append(path)
        for path in discovered:
            relative = path.relative_to(staging).as_posix()
            try:
                media_type = media_types[relative]
            except KeyError as exc:
                raise ValueError(f"media type missing for artifact file: {relative}") from exc
            files.append(ArtifactFile.from_path(staging, path, media_type=media_type))
        if set(media_types) != {entry.path for entry in files}:
            raise ValueError("media types include a file absent from staging")
        manifest = ResultManifest(
            request_fingerprint=request_fingerprint,
            model_digest=model_digest,
            serving_revision_digest=serving_revision_digest,
            sampler_digest=sampler_digest,
            execution_mode=execution_mode,
            files=tuple(files),
            ranking_evidence_digest=ranking_evidence_digest,
        )
        manifest_path = staging / "manifest.json"
        temporary_manifest = staging / ".manifest.json.tmp"
        with temporary_manifest.open("xb") as handle:
            handle.write(manifest.to_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_manifest, manifest_path)
        self._fsync_directory(staging)

        destination = self.objects_root / manifest.digest[7:]
        try:
            os.rename(staging, destination)
            created = True
            self._fsync_directory(self.objects_root)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            existing = self._load_manifest(destination / "manifest.json")
            if existing.digest != manifest.digest:
                raise RuntimeError("artifact digest collision or corrupt existing object") from exc
            existing.verify(destination)
            shutil.rmtree(staging)
            created = False
        manifest.verify(destination)
        return CommittedArtifact(destination, manifest, created)

    @staticmethod
    def _load_manifest(path: Path) -> ResultManifest:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return ResultManifest.from_dict(value)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
