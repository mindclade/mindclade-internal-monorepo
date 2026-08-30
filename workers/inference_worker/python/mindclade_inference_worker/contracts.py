"""Strict manifests at the scheduler-to-worker boundary."""

from __future__ import annotations

import base64
import dataclasses
import json
import re
from pathlib import Path
from typing import Any

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_JOB_ID = re.compile(r"^job-[0-9a-f]{32}-[0-9]{8}$")


def _require_digest(name: str, value: str) -> str:
    if not _DIGEST.fullmatch(value):
        raise ValueError(f"{name} must be an immutable sha256 digest")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class JobManifest:
    """All immutable inputs needed to execute one fenced inference attempt."""

    job_id: str
    tenant_id: str
    project_id: str
    model_digest: str
    bundle_manifest_digest: str
    bundle_archive_digest: str
    input_digest: str
    serving_revision_digest: str
    bundle_path: Path
    input_path: Path
    output_directory: Path
    bundle_signing_key_id: str
    scheduler_signing_key_id: str
    manifest_signature: str
    bundle_download_capability: str = dataclasses.field(repr=False)
    input_download_capability: str = dataclasses.field(repr=False)
    completion_capability: str = dataclasses.field(repr=False)
    completion_signing_private_key: str = dataclasses.field(repr=False)
    fencing_token: int
    seed: int
    num_samples: int = 1
    num_steps: int = 32
    device: str = "auto"
    schema_version: str = "v1alpha1"

    def __post_init__(self) -> None:
        if self.schema_version != "v1alpha1":
            raise ValueError("unsupported worker manifest schema")
        if not _JOB_ID.fullmatch(self.job_id):
            raise ValueError("job_id must contain a restart-unique immutable identity")
        for name in (
            "tenant_id",
            "project_id",
            "bundle_signing_key_id",
            "scheduler_signing_key_id",
        ):
            value = getattr(self, name)
            if not value or len(value) > 128:
                raise ValueError(f"{name} must contain 1..128 characters")
        for name in (
            "model_digest",
            "bundle_manifest_digest",
            "bundle_archive_digest",
            "input_digest",
            "serving_revision_digest",
        ):
            _require_digest(name, getattr(self, name))
        for name in (
            "bundle_download_capability",
            "input_download_capability",
            "completion_capability",
        ):
            value = getattr(self, name)
            if not value or len(value) > 16_384 or "\n" in value or "\r" in value:
                raise ValueError(f"{name} must be a bounded single-line capability")
        try:
            completion_key = base64.b64decode(self.completion_signing_private_key, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("completion_signing_private_key must be canonical base64") from exc
        if len(completion_key) != 32:
            raise ValueError("completion_signing_private_key must contain an Ed25519 seed")
        try:
            signature = base64.b64decode(self.manifest_signature, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("manifest_signature must be canonical base64") from exc
        if len(signature) != 64:
            raise ValueError("manifest_signature must contain an Ed25519 signature")
        if self.fencing_token < 1:
            raise ValueError("fencing_token must be positive")
        if not 0 <= self.seed < 2**63:
            raise ValueError("seed must be in [0, 2**63)")
        if not 1 <= self.num_samples <= 16:
            raise ValueError("num_samples must be within [1, 16]")
        if not 2 <= self.num_steps <= 128:
            raise ValueError("num_steps must be within [2, 128]")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> JobManifest:
        allowed = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown job manifest fields: {', '.join(unknown)}")
        converted = dict(value)
        for name in (
            "bundle_path",
            "input_path",
            "output_directory",
        ):
            if name in converted:
                converted[name] = Path(converted[name])
        return cls(**converted)

    def canonical_unsigned_bytes(self) -> bytes:
        """Return deterministic bytes authenticated by the scheduler."""

        value: dict[str, Any] = {}
        for field in dataclasses.fields(self):
            if field.name == "manifest_signature":
                continue
            item = getattr(self, field.name)
            value[field.name] = str(item) if isinstance(item, Path) else item
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")


@dataclasses.dataclass(frozen=True, slots=True)
class ResultReceipt:
    """Control-plane completion payload without raw tensors."""

    job_id: str
    tenant_id: str
    project_id: str
    model_digest: str
    input_digest: str
    serving_revision_digest: str
    result_digest: str
    result_size_bytes: int
    result_manifest_path: str
    fencing_token: int
    request_fingerprint: str
    selected_candidate_id: str
    execution_mode: str
    sampler_digest: str
    schema_version: str = "v1alpha1"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
