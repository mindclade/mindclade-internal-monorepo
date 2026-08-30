"""Thread-safe, non-blocking dynamic batching with bounded queue age."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass

from ..contracts.request_contract import InferenceRequest
from .batch_key import BatchKey
from .batch_limits import BatchLimits


class QueueFullError(RuntimeError):
    """Raised before enqueue when the bounded request queue has no capacity."""


@dataclass(frozen=True, slots=True)
class _QueuedRequest:
    request: InferenceRequest
    enqueued_ns: int


@dataclass(frozen=True, slots=True)
class BatchEnvelope:
    key: BatchKey
    requests: tuple[InferenceRequest, ...]
    oldest_queue_age_ms: float


class DynamicBatcher:
    """Group only requests with identical semantic execution keys.

    Callers own scheduling: ``enqueue`` is non-blocking and ``pop_ready`` returns
    batches that reached size or age limits. ``flush`` is intended for shutdown or
    low-traffic drains.
    """

    def __init__(
        self,
        limits: BatchLimits,
        *,
        max_batch_size: int = 4,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not 1 <= max_batch_size <= 64:
            raise ValueError("max_batch_size must be within [1, 64]")
        self.limits = limits
        self.max_batch_size = max_batch_size
        self._clock_ns = clock_ns
        self._queues: OrderedDict[BatchKey, deque[_QueuedRequest]] = OrderedDict()
        self._depth = 0
        self._lock = threading.Lock()

    @property
    def depth(self) -> int:
        with self._lock:
            return self._depth

    def enqueue(self, request: InferenceRequest) -> bool:
        key = BatchKey.from_request(request)
        with self._lock:
            if self._depth >= self.limits.max_queue_depth:
                raise QueueFullError("dynamic batch queue is full")
            queue = self._queues.setdefault(key, deque())
            queue.append(_QueuedRequest(request, self._clock_ns()))
            self._depth += 1
            return len(queue) >= self.max_batch_size

    def pop_ready(self) -> tuple[BatchEnvelope, ...]:
        now = self._clock_ns()
        cutoff_ns = self.limits.max_queue_delay_ms * 1_000_000
        batches: list[BatchEnvelope] = []
        with self._lock:
            for key in tuple(self._queues):
                queue = self._queues[key]
                oldest_age = now - queue[0].enqueued_ns
                if len(queue) < self.max_batch_size and oldest_age < cutoff_ns:
                    continue
                batches.append(self._pop_one(key, now))
        return tuple(batches)

    def flush(self) -> tuple[BatchEnvelope, ...]:
        now = self._clock_ns()
        batches: list[BatchEnvelope] = []
        with self._lock:
            while self._queues:
                key = next(iter(self._queues))
                batches.append(self._pop_one(key, now))
        return tuple(batches)

    def _pop_one(self, key: BatchKey, now_ns: int) -> BatchEnvelope:
        queue = self._queues[key]
        take = min(self.max_batch_size, len(queue))
        items = tuple(queue.popleft() for _ in range(take))
        self._depth -= take
        if not queue:
            del self._queues[key]
        return BatchEnvelope(
            key=key,
            requests=tuple(item.request for item in items),
            oldest_queue_age_ms=(now_ns - items[0].enqueued_ns) / 1_000_000.0,
        )
