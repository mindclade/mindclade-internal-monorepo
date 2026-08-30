"""Stable configuration identities."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from .model_config import ModelConfig


def config_digest(config: ModelConfig | Mapping[str, Any]) -> str:
    if isinstance(config, ModelConfig):
        payload = config.to_json_string().encode("utf-8")
    else:
        import json

        payload = (
            json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


__all__ = ["config_digest"]
