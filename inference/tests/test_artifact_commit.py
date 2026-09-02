from __future__ import annotations

import json

import pytest
from mindclade.inference.artifacts.artifact_commit import ArtifactCommitter
from mindclade.inference.artifacts.result_manifest import ResultManifest
from mindclade.inference.artifacts.stream_writer import StreamWriter
from mindclade.inference.contracts.stream_contract import (
    InferenceStreamEvent,
    StreamEventKind,
)

from .conftest import sha


def test_stream_and_artifact_commit_are_atomic_and_idempotent(tmp_path) -> None:
    committer = ArtifactCommitter(tmp_path / "objects")
    staging = committer.begin("attempt-1")
    with StreamWriter(staging / "events.jsonl", request_id="request-1") as writer:
        writer.write(InferenceStreamEvent("request-1", 0, StreamEventKind.ACCEPTED))
        writer.write(
            InferenceStreamEvent("request-1", 1, StreamEventKind.COMPLETED, {"candidate": "0"})
        )
    (staging / "result.json").write_text('{"selected":"candidate-0000"}\n', encoding="utf-8")
    kwargs = {
        "request_fingerprint": sha("a"),
        "model_digest": sha("b"),
        "serving_revision_digest": sha("c"),
        "sampler_digest": sha("d"),
        "execution_mode": "eager",
        "media_types": {
            "events.jsonl": "application/x-ndjson",
            "result.json": "application/json",
        },
    }
    committed = committer.commit(staging, **kwargs)
    assert committed.created
    assert (committed.path / "manifest.json").is_file()
    committed.manifest.verify(committed.path)

    duplicate = committer.begin("attempt-2")
    (duplicate / "events.jsonl").write_bytes((committed.path / "events.jsonl").read_bytes())
    (duplicate / "result.json").write_bytes((committed.path / "result.json").read_bytes())
    second = committer.commit(duplicate, **kwargs)
    assert not second.created
    assert second.path == committed.path
    assert second.digest == committed.digest


def test_commit_rejects_undeclared_files_and_detects_corruption(tmp_path) -> None:
    committer = ArtifactCommitter(tmp_path / "objects")
    staging = committer.begin("attempt-1")
    (staging / "result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="media type missing"):
        committer.commit(
            staging,
            request_fingerprint=sha("a"),
            model_digest=sha("b"),
            serving_revision_digest=sha("c"),
            sampler_digest=sha("d"),
            execution_mode="eager",
            media_types={},
        )

    committed = committer.commit(
        staging,
        request_fingerprint=sha("a"),
        model_digest=sha("b"),
        serving_revision_digest=sha("c"),
        sampler_digest=sha("d"),
        execution_mode="eager",
        media_types={"result.json": "application/json"},
    )
    (committed.path / "result.json").write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        committed.manifest.verify(committed.path)


def test_committed_artifact_verification_requires_exact_canonical_closure(tmp_path) -> None:
    committer = ArtifactCommitter(tmp_path / "objects")
    staging = committer.begin("attempt-closure")
    (staging / "result.json").write_text("{}", encoding="utf-8")
    committed = committer.commit(
        staging,
        request_fingerprint=sha("a"),
        model_digest=sha("b"),
        serving_revision_digest=sha("c"),
        sampler_digest=sha("d"),
        execution_mode="eager",
        media_types={"result.json": "application/json"},
    )

    (committed.path / "unmanifested.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="file set"):
        committed.manifest.verify(committed.path)

    (committed.path / "unmanifested.json").unlink()
    value = committed.manifest.to_dict()
    value["unknown"] = True
    with pytest.raises(ValueError, match="fields"):
        ResultManifest.from_dict(value)


def test_stream_rejects_events_after_terminal(tmp_path) -> None:
    writer = StreamWriter(tmp_path / "events.jsonl", request_id="request-1")
    writer.write(InferenceStreamEvent("request-1", 0, StreamEventKind.COMPLETED))
    with pytest.raises(ValueError, match="terminal"):
        writer.write(InferenceStreamEvent("request-1", 1, StreamEventKind.PROGRESS))
    receipt = writer.close()
    assert receipt.event_count == 1
    payload = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8"))
    assert payload["kind"] == "completed"


def test_stream_size_rejection_does_not_advance_sequence(tmp_path) -> None:
    writer = StreamWriter(tmp_path / "events.jsonl", request_id="request-1", max_bytes=256)
    writer.write(InferenceStreamEvent("request-1", 0, StreamEventKind.ACCEPTED))

    with pytest.raises(ValueError, match="max_bytes"):
        writer.write(
            InferenceStreamEvent(
                "request-1",
                1,
                StreamEventKind.PROGRESS,
                {"detail": "x" * 512},
            )
        )

    writer.write(InferenceStreamEvent("request-1", 1, StreamEventKind.COMPLETED))
    receipt = writer.close()
    assert receipt.event_count == 2
