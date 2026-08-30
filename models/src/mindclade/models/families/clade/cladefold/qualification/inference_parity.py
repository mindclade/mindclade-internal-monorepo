"""Eager inference parity assertion."""

from __future__ import annotations

import torch

from mindclade.models.api.outputs import ModelOutput


def assert_inference_parity(
    reference: ModelOutput, candidate: ModelOutput, *, atol: float = 1e-5, rtol: float = 1e-5
) -> None:
    if tuple(reference.keys()) != tuple(candidate.keys()):
        raise AssertionError("model output fields differ")
    for name in reference:
        left, right = reference[name], candidate[name]
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, atol=atol, rtol=rtol, msg=f"{name} differs")
        elif left != right:
            raise AssertionError(f"{name} differs")


__all__ = ["assert_inference_parity"]
