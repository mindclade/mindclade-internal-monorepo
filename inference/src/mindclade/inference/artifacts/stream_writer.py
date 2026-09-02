"""Bounded JSON-lines writer for progressive inference results."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from .._identity import canonical_json_bytes
from ..contracts.stream_contract import InferenceStreamEvent, StreamSequence


@dataclass(frozen=True, slots=True)
class StreamWriteReceipt:
    path: str
    event_count: int
    size_bytes: int
    digest: str


class StreamWriter:
    def __init__(self, path: Path, *, request_id: str, max_bytes: int = 128 * 1024 * 1024) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.path = path
        self.max_bytes = max_bytes
        self._sequence = StreamSequence(request_id)
        self._handle = path.open("xb")
        self._hasher = hashlib.sha256()
        self._size = 0
        self._events = 0
        self._closed = False

    def write(self, event: InferenceStreamEvent) -> None:
        if self._closed:
            raise ValueError("stream writer is closed")
        line = (
            canonical_json_bytes(
                {
                    "request_id": event.request_id,
                    "sequence": event.sequence,
                    "kind": event.kind.value,
                    "payload": event.payload,
                }
            )
            + b"\n"
        )
        if self._size + len(line) > self.max_bytes:
            raise ValueError("stream artifact exceeds max_bytes")
        self._sequence.accept(event)
        self._handle.write(line)
        self._hasher.update(line)
        self._size += len(line)
        self._events += 1

    def close(self) -> StreamWriteReceipt:
        if not self._closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._closed = True
        return StreamWriteReceipt(
            path=self.path.name,
            event_count=self._events,
            size_bytes=self._size,
            digest="sha256:" + self._hasher.hexdigest(),
        )

    def __enter__(self) -> StreamWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
