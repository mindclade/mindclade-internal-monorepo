"""Deployment-owned Ed25519 trust roots for jobs and model bundles."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key

from .contracts import JobManifest


class TrustedKeyring:
    """Immutable public keys supplied by the deployment, never by a job."""

    def __init__(
        self,
        *,
        scheduler_keys: Mapping[str, Ed25519PublicKey],
        bundle_keys: Mapping[str, Ed25519PublicKey],
    ) -> None:
        if not scheduler_keys or not bundle_keys:
            raise ValueError("trusted keyring requires scheduler and bundle keys")
        self._scheduler_keys = dict(scheduler_keys)
        self._bundle_keys = dict(bundle_keys)

    @classmethod
    def from_file(cls, path: Path) -> TrustedKeyring:
        """Load a strict key index whose PEM paths remain beneath its directory."""

        try:
            raw: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("trusted keyring could not be loaded") from exc
        if not isinstance(raw, dict):
            raise ValueError("trusted keyring must be a JSON object")
        if set(raw) != {"schema_version", "scheduler_keys", "bundle_keys"}:
            raise ValueError("trusted keyring fields are invalid")
        if raw["schema_version"] != "v1alpha1":
            raise ValueError("trusted keyring schema is unsupported")
        root = path.parent.resolve(strict=True)
        return cls(
            scheduler_keys=_load_key_map(root, raw["scheduler_keys"]),
            bundle_keys=_load_key_map(root, raw["bundle_keys"]),
        )

    def verify_job(self, manifest: JobManifest) -> None:
        """Authenticate every job field before paths or capabilities are used."""

        key = self._scheduler_keys.get(manifest.scheduler_signing_key_id)
        if key is None:
            raise ValueError("job manifest signing key is not trusted")
        try:
            signature = base64.b64decode(manifest.manifest_signature, validate=True)
            key.verify(signature, manifest.canonical_unsigned_bytes())
        except (InvalidSignature, ValueError) as exc:
            raise ValueError("job manifest signature verification failed") from exc

    def bundle_key(self, key_id: str) -> Ed25519PublicKey:
        """Resolve a bundle key ID exclusively through deployment policy."""

        key = self._bundle_keys.get(key_id)
        if key is None:
            raise ValueError("model bundle signing key is not trusted")
        return key


def _load_key_map(root: Path, raw: Any) -> dict[str, Ed25519PublicKey]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("trusted key group must be a non-empty object")
    result: dict[str, Ed25519PublicKey] = {}
    for key_id, relative_value in raw.items():
        if not isinstance(key_id, str) or not key_id or len(key_id) > 128:
            raise ValueError("trusted key ID is invalid")
        if not isinstance(relative_value, str):
            raise ValueError("trusted key path must be a string")
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("trusted key path must stay beneath the keyring")
        key_path = (root / relative).resolve(strict=True)
        try:
            key_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("trusted key path escapes the keyring") from exc
        try:
            key = load_pem_public_key(key_path.read_bytes())
        except (OSError, ValueError) as exc:
            raise ValueError("trusted Ed25519 public key could not be loaded") from exc
        if not isinstance(key, Ed25519PublicKey):
            raise ValueError("trusted key must be an Ed25519 public key")
        result[key_id] = key
    return result


__all__ = ["TrustedKeyring"]
