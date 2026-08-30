from __future__ import annotations

import pytest
import torch
from mindclade.inference.adaptive_compute.budget_accounting import BudgetExceeded, BudgetLedger
from mindclade.inference.adaptive_compute.candidate_receipt import CandidateReceipt
from mindclade.inference.adaptive_compute.compute_policy import ComputePolicy
from mindclade.inference.adaptive_compute.stopping_rule import Observation, StoppingRule

from .conftest import sha


def test_budget_consumption_is_bounded_and_atomic() -> None:
    ledger = BudgetLedger(ComputePolicy(min_steps=4, max_steps=8, max_candidates=2))
    assert ledger.consume(steps=4, candidates=1).remaining_steps == 4
    before = ledger.snapshot()
    with pytest.raises(BudgetExceeded, match="step"):
        ledger.consume(steps=5, candidates=1)
    assert ledger.snapshot() == before
    assert ledger.consume(steps=4, candidates=1).remaining_candidates == 0


def test_stopping_requires_both_signals_for_consecutive_observations() -> None:
    policy = ComputePolicy(
        min_steps=4,
        max_steps=16,
        evaluation_interval=4,
        patience=2,
        confidence_gain_threshold=0.002,
        displacement_threshold_angstrom=0.05,
    )
    rule = StoppingRule(policy)
    assert not rule.observe(Observation(4, 0.7, 0.5)).stop
    assert not rule.observe(Observation(8, 0.701, 0.04)).stop
    decision = rule.observe(Observation(12, 0.7015, 0.03))
    assert decision.stop
    assert decision.reason == "converged"


def test_candidate_receipt_binds_coordinates_and_execution_identity() -> None:
    coordinates = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)
    receipt = CandidateReceipt.create(
        candidate_id="candidate-0000",
        request_fingerprint=sha("a"),
        model_digest=sha("b"),
        sampler_digest=sha("c"),
        coordinates=coordinates,
        seed=1,
        completed_steps=8,
        raw_confidence=0.5,
        calibrated_confidence=0.6,
        stop_reason="budget-exhausted",
    )
    changed = CandidateReceipt.create(
        candidate_id="candidate-0000",
        request_fingerprint=sha("a"),
        model_digest=sha("b"),
        sampler_digest=sha("c"),
        coordinates=coordinates + 1,
        seed=1,
        completed_steps=8,
        raw_confidence=0.5,
        calibrated_confidence=0.6,
        stop_reason="budget-exhausted",
    )
    assert receipt.digest != changed.digest
