"""Authenticated, digest-exact publication of fenced result artifacts."""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote
from urllib.request import Request, urlopen

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .contracts import JobManifest, ResultReceipt
from .io import resolve_existing_path, sha256_file

_CHUNK = 4 * 1024 * 1024
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{15,127}$")


class HttpResponse(Protocol):
    status: int

    def read(self, size: int = -1) -> bytes: ...

    def __enter__(self) -> HttpResponse: ...

    def __exit__(self, *args: object) -> None: ...


OpenUrl = Callable[[Request, float], HttpResponse]


def publish_result_artifact(
    control_plane_url: str,
    artifact_proxy_url: str,
    manifest: JobManifest,
    receipt: ResultReceipt,
    *,
    result_root: Path,
    open_url: OpenUrl = urlopen,  # type: ignore[assignment]
) -> None:
    """Authorize, upload, and commit the exact tensor file in the receipt."""

    result_path = resolve_existing_path(
        result_root,
        Path(receipt.result_manifest_path).with_name(
            f"result.fence-{receipt.fencing_token}.safetensors"
        ),
        directory=False,
    )
    size_bytes = result_path.stat().st_size
    if size_bytes != receipt.result_size_bytes or sha256_file(result_path) != receipt.result_digest:
        raise ValueError("result tensor no longer matches its fenced receipt")

    authorization_payload = _canonical_json(
        {
            "fencing_token": receipt.fencing_token,
            "job_id": receipt.job_id,
            "project_id": receipt.project_id,
            "result_digest": receipt.result_digest,
            "result_size_bytes": receipt.result_size_bytes,
            "schema_version": "v1alpha1",
            "tenant_id": receipt.tenant_id,
        }
    )
    signature = _completion_signature(manifest, authorization_payload)
    authorization_request = Request(
        _endpoint(
            control_plane_url,
            f"/internal/v1alpha1/jobs/{quote(manifest.job_id, safe='')}/result-upload-capability",
        ),
        data=authorization_payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Mindclade-Completion-Capability": manifest.completion_capability,
            "X-Mindclade-Completion-Signature": signature,
        },
    )
    with open_url(authorization_request, 10.0) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError("control plane rejected result upload authorization")
        authorization = _strict_json(response.read(64 * 1024 + 1), 64 * 1024)
    if set(authorization) != {"upload_capability", "session_id"}:
        raise ValueError("result upload authorization fields are invalid")
    capability = authorization["upload_capability"]
    session_id = authorization["session_id"]
    if (
        not isinstance(capability, str)
        or not capability
        or len(capability) > 16_384
        or "\n" in capability
        or "\r" in capability
        or not isinstance(session_id, str)
        or not _SAFE_ID.fullmatch(session_id)
    ):
        raise ValueError("result upload authorization values are invalid")

    begin_payload = _canonical_json(
        {
            "digest": receipt.result_digest,
            "project_id": receipt.project_id,
            "session_id": session_id,
            "size_bytes": receipt.result_size_bytes,
            "tenant_id": receipt.tenant_id,
        }
    )
    begin_request = Request(
        _endpoint(artifact_proxy_url, "/v1alpha1/uploads"),
        data=begin_payload,
        method="POST",
        headers={"Content-Type": "application/json", "X-Mindclade-Capability": capability},
    )
    with open_url(begin_request, 30.0) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError("artifact proxy rejected result upload creation")
        begin = _session_response(response.read(64 * 1024 + 1), session_id)
    if begin["committed_bytes"] != 0:
        raise ValueError("new result upload did not begin at offset zero")

    offset = 0
    with result_path.open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            append_request = Request(
                _endpoint(
                    artifact_proxy_url,
                    f"/v1alpha1/uploads/{quote(session_id, safe='')}?offset={offset}",
                ),
                data=chunk,
                method="PUT",
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Mindclade-Capability": capability,
                },
            )
            with open_url(append_request, 30.0) as response:
                if not 200 <= response.status < 300:
                    raise RuntimeError("artifact proxy rejected a result upload chunk")
                progress = _session_response(response.read(64 * 1024 + 1), session_id)
            offset += len(chunk)
            if progress["committed_bytes"] != offset:
                raise ValueError("artifact proxy returned inconsistent upload progress")

    commit_request = Request(
        _endpoint(
            artifact_proxy_url,
            f"/v1alpha1/uploads/{quote(session_id, safe='')}/commit",
        ),
        data=b"",
        method="POST",
        headers={"X-Mindclade-Capability": capability},
    )
    with open_url(commit_request, 30.0) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError("artifact proxy rejected result upload commit")
        committed = _strict_json(response.read(64 * 1024 + 1), 64 * 1024)
    if set(committed) != {"digest"} or committed["digest"] != receipt.result_digest:
        raise ValueError("artifact proxy committed an unexpected result identity")


def _completion_signature(manifest: JobManifest, payload: bytes) -> str:
    private_key = Ed25519PrivateKey.from_private_bytes(
        base64.b64decode(manifest.completion_signing_private_key, validate=True)
    )
    return base64.b64encode(private_key.sign(payload)).decode("ascii")


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _strict_json(payload: bytes, maximum: int) -> dict[str, Any]:
    if len(payload) > maximum:
        raise ValueError("artifact control response exceeds its size limit")
    try:
        value: Any = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("artifact control response is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("artifact control response must be an object")
    return value


def _session_response(payload: bytes, session_id: str) -> dict[str, Any]:
    value = _strict_json(payload, 64 * 1024)
    if set(value) != {"upload_id", "committed_bytes"}:
        raise ValueError("artifact upload response fields are invalid")
    if value["upload_id"] != session_id or not isinstance(value["committed_bytes"], int):
        raise ValueError("artifact upload response values are invalid")
    return value


def _endpoint(base_url: str, path: str) -> str:
    normalized = base_url.rstrip("/")
    if not normalized.startswith(("http://", "https://")):
        raise ValueError("service URL must use HTTP or HTTPS")
    return normalized + path


__all__ = ["publish_result_artifact"]
