"""Fenced completion handoff to the internal control-plane endpoint."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from typing import Protocol
from urllib.parse import quote
from urllib.request import Request, urlopen

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .contracts import JobManifest, ResultReceipt


class HttpResponse(Protocol):
    status: int

    def __enter__(self) -> HttpResponse: ...

    def __exit__(self, *args: object) -> None: ...


OpenUrl = Callable[[Request, float], HttpResponse]


def publish_completion(
    base_url: str,
    manifest: JobManifest,
    receipt: ResultReceipt,
    *,
    open_url: OpenUrl = urlopen,  # type: ignore[assignment]
) -> None:
    """Report a result only through the control plane's fenced transition."""

    normalized = base_url.rstrip("/")
    if not normalized.startswith(("http://", "https://")):
        raise ValueError("control-plane URL must use HTTP or HTTPS")
    url = f"{normalized}/internal/v1alpha1/jobs/{quote(manifest.job_id, safe='')}/complete"
    payload = json.dumps(
        receipt.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    private_key = Ed25519PrivateKey.from_private_bytes(
        base64.b64decode(manifest.completion_signing_private_key, validate=True)
    )
    signature = base64.b64encode(private_key.sign(payload)).decode("ascii")
    request = Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Mindclade-Completion-Capability": manifest.completion_capability,
            "X-Mindclade-Completion-Signature": signature,
        },
    )
    with open_url(request, 10.0) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError("control plane rejected the fenced completion")


__all__ = ["publish_completion"]
