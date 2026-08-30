"""Trust-gated GitHub-to-Buildkite dispatch with a non-secret receipt."""

from __future__ import annotations

import json
import os
import urllib.request
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any


def _trust_module() -> Any:
    path = Path(__file__).parents[2] / ".buildkite/lib/trusted_context.py"
    spec = spec_from_file_location("trusted_context", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("trusted-context policy is unavailable")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"{name} is required in a trusted dispatch context")
    return value


def main() -> int:
    event_path = Path(_required("GITHUB_EVENT_PATH"))
    event_name = _required("GITHUB_EVENT_NAME")
    trusted = bool(_trust_module().github_context_is_trusted(event_name, event_path))
    receipt_path = Path("buildkite-dispatch-receipt.json")
    if not trusted:
        receipt_path.write_text(
            json.dumps({"dispatched": False, "reason": "untrusted event context"}, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return 0

    commit = _required("GITHUB_SHA")
    branch = _required("GITHUB_REF_NAME")
    github_ref = _required("GITHUB_REF")
    organization = _required("BUILDKITE_ORGANIZATION_SLUG")
    pipeline = _required("BUILDKITE_PIPELINE_SLUG")
    payload = {
        "commit": commit,
        "branch": branch,
        "message": f"GitHub {event_name}: {os.environ.get('GITHUB_REPOSITORY', '')}@{commit[:12]}",
        "env": {
            "GITHUB_RUN_ID": _required("GITHUB_RUN_ID"),
            "GITHUB_REPOSITORY": _required("GITHUB_REPOSITORY"),
            "MINDCLADE_GITHUB_EVENT_NAME": event_name,
            "MINDCLADE_GITHUB_REF": github_ref,
            "MINDCLADE_GITHUB_SHA": commit,
            "MINDCLADE_GITHUB_TRUSTED": "true",
        },
    }
    request = urllib.request.Request(
        f"https://api.buildkite.com/v2/organizations/{organization}/pipelines/{pipeline}/builds",
        data=json.dumps(payload, sort_keys=True).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {_required('BUILDKITE_API_TOKEN')}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    receipt = {
        "build_number": result.get("number"),
        "commit": commit,
        "dispatched": True,
        "state": result.get("state"),
        "web_url": result.get("web_url"),
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
