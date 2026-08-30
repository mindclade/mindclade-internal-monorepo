from __future__ import annotations

import torch

from mindclade.models import CladeFoldModel


def test_integrity_checkpoint_round_trip_preserves_outputs(
    tmp_path, cladefold_model, cladefold_batch
) -> None:
    cladefold_model.eval()
    reference = cladefold_model(cladefold_batch)
    cladefold_model.save_pretrained(tmp_path)
    restored = CladeFoldModel.from_pretrained(tmp_path)
    candidate = restored(cladefold_batch)
    torch.testing.assert_close(reference.predicted_noise, candidate.predicted_noise, atol=0, rtol=0)
    torch.testing.assert_close(
        reference.distogram_logits, candidate.distogram_logits, atol=0, rtol=0
    )
