from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / ".buildkite" / "lib" / "trusted_context.py"
    spec = importlib.util.spec_from_file_location("trusted_context", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dispatch_module():
    path = Path(__file__).parents[1] / "tools" / "ci" / "dispatch_buildkite.py"
    spec = importlib.util.spec_from_file_location("dispatch_buildkite", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trusted_buildkite_environment() -> dict[str, str]:
    revision = "a" * 40
    return {
        "BUILDKITE_BRANCH": "main",
        "BUILDKITE_COMMIT": revision,
        "BUILDKITE_PULL_REQUEST": "false",
        "BUILDKITE_SOURCE": "api",
        "MINDCLADE_GITHUB_EVENT_NAME": "push",
        "MINDCLADE_GITHUB_REF": "refs/heads/main",
        "MINDCLADE_GITHUB_SHA": revision,
        "MINDCLADE_GITHUB_TRUSTED": "true",
    }


def test_buildkite_release_context_is_fail_closed() -> None:
    trusted_context = _module()
    assert trusted_context.buildkite_context_is_trusted({}) is False
    assert trusted_context.buildkite_context_is_trusted(_trusted_buildkite_environment()) is True

    branch_only = {"BUILDKITE_BRANCH": "main", "BUILDKITE_PULL_REQUEST": "false"}
    assert trusted_context.buildkite_context_is_trusted(branch_only) is False

    for field in _trusted_buildkite_environment():
        environment = _trusted_buildkite_environment()
        del environment[field]
        assert trusted_context.buildkite_context_is_trusted(environment) is False

    for field, unsafe_value in (
        ("MINDCLADE_GITHUB_TRUSTED", "false"),
        ("BUILDKITE_SOURCE", "webhook"),
        ("BUILDKITE_BRANCH", "feature"),
        ("BUILDKITE_PULL_REQUEST", "123"),
        ("MINDCLADE_GITHUB_EVENT_NAME", "pull_request"),
        ("MINDCLADE_GITHUB_REF", "refs/heads/feature"),
        ("MINDCLADE_GITHUB_SHA", "b" * 40),
        ("BUILDKITE_COMMIT", "not-a-commit"),
    ):
        environment = {**_trusted_buildkite_environment(), field: unsafe_value}
        assert trusted_context.buildkite_context_is_trusted(environment) is False


def test_only_main_github_dispatch_events_are_trusted(tmp_path: Path) -> None:
    trusted_context = _module()
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps(
            {
                "ref": "refs/heads/main",
                "repository": {"full_name": "mindclade/internal"},
            }
        ),
        encoding="utf-8",
    )
    assert trusted_context.github_context_is_trusted("push", event) is True
    event.write_text(
        json.dumps(
            {
                "ref": "refs/heads/feature",
                "repository": {"full_name": "mindclade/internal"},
            }
        ),
        encoding="utf-8",
    )
    assert trusted_context.github_context_is_trusted("push", event) is False
    assert trusted_context.github_context_is_trusted("push", None) is False
    assert trusted_context.github_context_is_trusted("pull_request", None) is False


def test_dispatcher_propagates_revision_bound_trust_evidence(tmp_path: Path, monkeypatch) -> None:
    dispatcher = _dispatch_module()
    revision = "a" * 40
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps(
            {
                "after": revision,
                "ref": "refs/heads/main",
                "repository": {"full_name": "mindclade/internal"},
            }
        ),
        encoding="utf-8",
    )
    environment = {
        "BUILDKITE_API_TOKEN": "test-token",
        "BUILDKITE_ORGANIZATION_SLUG": "mindclade",
        "BUILDKITE_PIPELINE_SLUG": "internal",
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_REF_NAME": "main",
        "GITHUB_REPOSITORY": "mindclade/internal",
        "GITHUB_RUN_ID": "12345",
        "GITHUB_SHA": revision,
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.chdir(tmp_path)
    requests = []

    def _urlopen(request, *, timeout: int):
        requests.append((request, timeout))
        return io.BytesIO(b'{"number": 42, "state": "scheduled", "web_url": "https://ci"}')

    monkeypatch.setattr(dispatcher.urllib.request, "urlopen", _urlopen)

    assert dispatcher.main() == 0
    assert len(requests) == 1
    request, timeout = requests[0]
    assert timeout == 30
    assert request.data is not None
    payload = json.loads(request.data)
    assert payload["commit"] == revision
    assert payload["branch"] == "main"
    assert payload["env"] == {
        "GITHUB_REPOSITORY": "mindclade/internal",
        "GITHUB_RUN_ID": "12345",
        "MINDCLADE_GITHUB_EVENT_NAME": "push",
        "MINDCLADE_GITHUB_REF": "refs/heads/main",
        "MINDCLADE_GITHUB_SHA": revision,
        "MINDCLADE_GITHUB_TRUSTED": "true",
    }
