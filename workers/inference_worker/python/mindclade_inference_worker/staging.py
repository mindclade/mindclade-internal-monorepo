"""Capability-scoped artifact staging for the worker init container."""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Protocol
from urllib.parse import quote
from urllib.request import Request, urlopen

from .contracts import JobManifest
from .io import new_path_under, prepare_directory, require_clean_tree, sha256_file
from .trust import TrustedKeyring

_CHUNK = 1024 * 1024


class HttpResponse(Protocol):
    status: int

    def read(self, size: int = -1) -> bytes: ...

    def __enter__(self) -> HttpResponse: ...

    def __exit__(self, *args: object) -> None: ...


OpenUrl = Callable[[Request, float], HttpResponse]


def stage_job(
    manifest: JobManifest,
    *,
    trust: TrustedKeyring,
    artifact_root: Path,
    artifact_proxy_url: str,
    maximum_bundle_bytes: int = 20 * 1024**3,
    maximum_input_bytes: int = 4 * 1024**3,
    open_url: OpenUrl = urlopen,  # type: ignore[assignment]
) -> None:
    """Authenticate a job, download exact artifacts, and safely unpack its bundle."""

    trust.verify_job(manifest)
    root = prepare_directory(artifact_root, artifact_root)
    bundle_destination = new_path_under(root, manifest.bundle_path)
    input_destination = new_path_under(root, manifest.input_path)
    archive_path = new_path_under(
        root, root / f".bundle-archive.fence-{manifest.fencing_token}.tar"
    )
    staging_directory = Path(tempfile.mkdtemp(prefix=".bundle-stage-", dir=root))
    try:
        _download(
            _artifact_url(artifact_proxy_url, manifest.bundle_archive_digest),
            manifest.bundle_download_capability,
            archive_path,
            expected_digest=manifest.bundle_archive_digest,
            maximum_bytes=maximum_bundle_bytes,
            open_url=open_url,
        )
        _extract_bundle(archive_path, staging_directory, maximum_bytes=maximum_bundle_bytes)
        require_clean_tree(staging_directory)
        _verify_bundle_identity(staging_directory, manifest, trust)
        _download(
            _artifact_url(artifact_proxy_url, manifest.input_digest),
            manifest.input_download_capability,
            input_destination,
            expected_digest=manifest.input_digest,
            maximum_bytes=maximum_input_bytes,
            open_url=open_url,
        )
        os.rename(staging_directory, bundle_destination)
    finally:
        archive_path.unlink(missing_ok=True)
        if staging_directory.exists():
            shutil.rmtree(staging_directory)


def _artifact_url(base: str, digest: str) -> str:
    normalized = base.rstrip("/")
    if not normalized.startswith(("http://", "https://")):
        raise ValueError("artifact proxy URL must use HTTP or HTTPS")
    return f"{normalized}/v1alpha1/artifacts/{quote(digest, safe=':')}"


def _download(
    url: str,
    capability: str,
    destination: Path,
    *,
    expected_digest: str,
    maximum_bytes: int,
    open_url: OpenUrl,
) -> None:
    request = Request(url, headers={"X-Mindclade-Capability": capability})
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    total = 0
    try:
        with open_url(request, 30.0) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError("artifact proxy rejected the staged download")
            with temporary.open("wb") as stream:
                while chunk := response.read(_CHUNK):
                    total += len(chunk)
                    if total > maximum_bytes:
                        raise ValueError("staged artifact exceeds its configured size limit")
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        if sha256_file(temporary) != expected_digest:
            raise ValueError("staged artifact digest verification failed")
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError("refusing to replace a staged artifact") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _extract_bundle(archive: Path, destination: Path, *, maximum_bytes: int) -> None:
    total = 0
    seen: set[PurePosixPath] = set()
    with tarfile.open(archive, mode="r:*") as stream:
        members = stream.getmembers()
        if not members or len(members) > 100_000:
            raise ValueError("bundle archive member count is invalid")
        for member in members:
            relative = PurePosixPath(member.name)
            if (
                relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
                or relative in seen
            ):
                raise ValueError("bundle archive contains an unsafe path")
            seen.add(relative)
            target = destination.joinpath(*relative.parts)
            target.relative_to(destination)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError("bundle archive may contain only files and directories")
            total += member.size
            if total > maximum_bytes:
                raise ValueError("expanded model bundle exceeds its configured size limit")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = stream.extractfile(member)
            if source is None:
                raise ValueError("bundle archive file could not be read")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=_CHUNK)


def _verify_bundle_identity(
    bundle_path: Path, manifest: JobManifest, trust: TrustedKeyring
) -> None:
    from mindclade.models.packaging.bundle_signing import Ed25519PublicKeyVerifier
    from mindclade.models.packaging.model_bundle import ModelBundle

    inner_manifest = bundle_path / ModelBundle.MANIFEST
    identity = sha256_file(inner_manifest)
    if identity != manifest.bundle_manifest_digest or identity != manifest.model_digest:
        raise ValueError("staged model identity does not match the signed job")
    key = trust.bundle_key(manifest.bundle_signing_key_id)
    verifier = Ed25519PublicKeyVerifier(key, manifest.bundle_signing_key_id)
    ModelBundle.verify(bundle_path, verifier=verifier)


__all__ = ["stage_job"]
