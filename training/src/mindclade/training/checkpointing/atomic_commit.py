"""Same-filesystem, manifest-last checkpoint publication."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path

from mindclade.training.api.checkpoint import CheckpointError

from .manifest import MANIFEST_FILENAME, CheckpointManifest, canonical_json_bytes

_CHECKPOINT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class AtomicCheckpointWriter:
    """Owns a private staging directory until an immutable commit succeeds."""

    def __init__(self, root: Path, checkpoint_id: str) -> None:
        if _CHECKPOINT_ID.fullmatch(checkpoint_id) is None:
            raise ValueError("checkpoint_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}")
        self.root = root.resolve()
        self.checkpoint_id = checkpoint_id
        self.target = self.root / checkpoint_id
        self.staging: Path | None = None
        self.committed = False

    def __enter__(self) -> AtomicCheckpointWriter:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.target.exists():
            raise CheckpointError(
                f"checkpoint already exists and will not be overwritten: {self.target}"
            )
        self.staging = Path(
            tempfile.mkdtemp(prefix=f".{self.checkpoint_id}.staging-", dir=str(self.root))
        )
        return self

    def commit(self, manifest: CheckpointManifest) -> Path:
        if self.staging is None:
            raise RuntimeError("atomic checkpoint writer is not open")
        if self.committed:
            raise RuntimeError("atomic checkpoint writer has already committed")
        if manifest.checkpoint_id != self.checkpoint_id:
            raise ValueError("manifest checkpoint id does not match writer target")
        manifest_path = self.staging / MANIFEST_FILENAME
        with manifest_path.open("xb") as handle:
            handle.write(canonical_json_bytes(manifest.to_dict()))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(self.staging)
        if self.target.exists():
            raise CheckpointError(f"checkpoint target appeared during save: {self.target}")
        os.rename(self.staging, self.target)
        _fsync_directory(self.root)
        self.committed = True
        self.staging = None
        return self.target

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        if self.staging is not None and self.staging.exists():
            shutil.rmtree(self.staging)
            self.staging = None
