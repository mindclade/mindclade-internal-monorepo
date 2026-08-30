"""Atomic accounting for bounded adaptive inference work."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .compute_policy import ComputePolicy


class BudgetExceeded(RuntimeError):
    """Raised when a reservation would exceed a resolved compute policy."""


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    max_steps: int
    consumed_steps: int
    max_candidates: int
    consumed_candidates: int

    @property
    def remaining_steps(self) -> int:
        return self.max_steps - self.consumed_steps

    @property
    def remaining_candidates(self) -> int:
        return self.max_candidates - self.consumed_candidates


class BudgetLedger:
    """Thread-safe ledger; reservations either commit completely or not at all."""

    def __init__(self, policy: ComputePolicy, *, steps: int = 0, candidates: int = 0) -> None:
        self.policy = policy
        self._steps = steps
        self._candidates = candidates
        self._lock = threading.Lock()
        self._validate_existing()

    def _validate_existing(self) -> None:
        if not 0 <= self._steps <= self.policy.max_steps:
            raise ValueError("existing step consumption is outside the policy budget")
        if not 0 <= self._candidates <= self.policy.max_candidates:
            raise ValueError("existing candidate consumption is outside the policy budget")

    def consume(self, *, steps: int = 0, candidates: int = 0) -> BudgetSnapshot:
        if steps < 0 or candidates < 0:
            raise ValueError("budget consumption cannot be negative")
        with self._lock:
            next_steps = self._steps + steps
            next_candidates = self._candidates + candidates
            if next_steps > self.policy.max_steps:
                raise BudgetExceeded("adaptive step budget exhausted")
            if next_candidates > self.policy.max_candidates:
                raise BudgetExceeded("adaptive candidate budget exhausted")
            self._steps = next_steps
            self._candidates = next_candidates
            return self._snapshot_unlocked()

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            max_steps=self.policy.max_steps,
            consumed_steps=self._steps,
            max_candidates=self.policy.max_candidates,
            consumed_candidates=self._candidates,
        )
