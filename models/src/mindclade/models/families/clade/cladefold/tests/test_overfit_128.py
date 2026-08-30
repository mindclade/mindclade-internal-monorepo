from __future__ import annotations

import torch


def test_objective_learns_on_repeated_128_example_fixture(cladefold_model, cladefold_batch) -> None:
    values = {}
    for name, tensor in cladefold_batch.tensor_items():
        values[name] = tensor.repeat_interleave(64, dim=0)
    batch = type(cladefold_batch)(**values)
    optimizer = torch.optim.AdamW(cladefold_model.parameters(), lr=3e-3, weight_decay=0.0)
    losses = []
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        loss = cladefold_model(batch).loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert sum(losses[-3:]) / 3 < sum(losses[:3]) / 3
