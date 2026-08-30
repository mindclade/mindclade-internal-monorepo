"""Fail closed when a release-capable CI context is not trusted."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SAFE_GITHUB_EVENTS = frozenset({"push", "workflow_dispatch"})


def github_context_is_trusted(event_name: str, event_path: Path | None) -> bool:
    """Return whether a GitHub event may access release-capable credentials."""
    if event_name not in _SAFE_GITHUB_EVENTS or event_path is None:
        return False
    try:
        event = json.loads(event_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    repository = event.get("repository")
    ref = event.get("ref")
    return bool(
        isinstance(repository, dict)
        and repository.get("full_name")
        and ref in {"main", "refs/heads/main"}
        and event.get("deleted") is not True
    )


def buildkite_context_is_trusted(environment: dict[str, str]) -> bool:
    """Trust only main builds carrying the GitHub dispatcher's immutable evidence."""
    build_commit = environment.get("BUILDKITE_COMMIT", "")
    github_commit = environment.get("MINDCLADE_GITHUB_SHA", "")
    return (
        environment.get("MINDCLADE_GITHUB_TRUSTED") == "true"
        and environment.get("BUILDKITE_SOURCE") == "api"
        and environment.get("BUILDKITE_BRANCH") == "main"
        and environment.get("BUILDKITE_PULL_REQUEST") == "false"
        and environment.get("MINDCLADE_GITHUB_EVENT_NAME") in _SAFE_GITHUB_EVENTS
        and environment.get("MINDCLADE_GITHUB_REF") == "refs/heads/main"
        and bool(_GIT_SHA.fullmatch(build_commit))
        and github_commit == build_commit
    )


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    provider = args[0] if args else "buildkite"
    if provider == "github":
        event_path = os.environ.get("GITHUB_EVENT_PATH")
        trusted = github_context_is_trusted(
            os.environ.get("GITHUB_EVENT_NAME", ""),
            Path(event_path) if event_path else None,
        )
    elif provider == "buildkite":
        trusted = buildkite_context_is_trusted(dict(os.environ))
    else:
        raise ValueError(f"unsupported CI provider: {provider}")
    print(json.dumps({"provider": provider, "trusted": trusted}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
