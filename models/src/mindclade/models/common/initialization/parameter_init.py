"""Deterministic-under-caller-RNG parameter initialization."""

from __future__ import annotations

import torch
from torch import nn

from .init_policy import InitializationPolicy


def initialize_module(module: nn.Module, policy: InitializationPolicy | None = None) -> None:
    selected = policy or InitializationPolicy()
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.normal_(child.weight, mean=0.0, std=selected.standard_deviation)
            if child.bias is not None and selected.zero_bias:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.Embedding):
            nn.init.normal_(child.weight, mean=0.0, std=selected.standard_deviation)
            if child.padding_idx is not None:
                with torch.no_grad():
                    child.weight[child.padding_idx].zero_()
        elif isinstance(child, nn.LayerNorm):
            if selected.unit_norm_scale and child.elementwise_affine:
                nn.init.ones_(child.weight)
            if child.elementwise_affine:
                nn.init.zeros_(child.bias)


__all__ = ["initialize_module"]
