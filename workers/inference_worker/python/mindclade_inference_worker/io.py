"""Integrity checking and crash-safe local writes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_bytes_exclusive(path: Path, value: bytes) -> None:
    """Atomically publish bytes without ever replacing an existing result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite published result: {path.name}") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    atomic_bytes_exclusive(path, payload)


def resolve_existing_path(root: Path, candidate: Path, *, directory: bool) -> Path:
    """Resolve an existing path beneath a root and reject every symlink hop."""

    root_absolute, candidate_absolute = _lexical_paths(root, candidate)
    _reject_symlink_components(root_absolute, candidate_absolute)
    try:
        resolved_root = root_absolute.resolve(strict=True)
        resolved = candidate_absolute.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError("worker path must exist beneath its configured root") from exc
    if directory and not resolved.is_dir():
        raise ValueError("worker path must be a directory")
    if not directory and not resolved.is_file():
        raise ValueError("worker path must be a regular file")
    return resolved


def prepare_directory(root: Path, candidate: Path) -> Path:
    """Create a result/staging directory without accepting symlink traversal."""

    root_absolute, candidate_absolute = _lexical_paths(root, candidate)
    root_absolute.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(root_absolute, candidate_absolute, allow_missing=True)
    candidate_absolute.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(root_absolute, candidate_absolute)
    if not candidate_absolute.is_dir():
        raise ValueError("worker output path must be a directory")
    return candidate_absolute.resolve(strict=True)


def new_path_under(root: Path, candidate: Path) -> Path:
    """Validate an unpublished destination beneath a configured root."""

    root_absolute, candidate_absolute = _lexical_paths(root, candidate)
    root_absolute.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(root_absolute, candidate_absolute, allow_missing=True)
    if candidate_absolute.exists() or candidate_absolute.is_symlink():
        raise FileExistsError(f"staging destination already exists: {candidate.name}")
    parent = candidate_absolute.parent
    parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(root_absolute, parent)
    return candidate_absolute


def require_clean_tree(root: Path) -> None:
    """Reject symlinked bundle members before a model loader can follow them."""

    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("staged model bundle cannot contain symlinks")


def _lexical_paths(root: Path, candidate: Path) -> tuple[Path, Path]:
    if not root.is_absolute() or not candidate.is_absolute():
        raise ValueError("worker paths and configured roots must be absolute")
    if ".." in root.parts or ".." in candidate.parts:
        raise ValueError("worker paths cannot contain traversal components")
    root_absolute = Path(os.path.abspath(root))
    candidate_absolute = Path(os.path.abspath(candidate))
    try:
        candidate_absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise ValueError("worker path escapes its configured root") from exc
    return root_absolute, candidate_absolute


def _reject_symlink_components(root: Path, candidate: Path, *, allow_missing: bool = False) -> None:
    # Build cumulatively: root / a / b, rather than root / b.
    paths = [root]
    accumulated = root
    for part in candidate.relative_to(root).parts:
        accumulated /= part
        paths.append(accumulated)
    for current in paths:
        if current.is_symlink():
            raise ValueError("worker path cannot traverse a symlink")
        if not current.exists():
            if allow_missing:
                break
            raise ValueError("worker path component does not exist")
