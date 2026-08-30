"""Content-addressed compiled-region cache key."""

from __future__ import annotations

from dataclasses import dataclass

from .._identity import content_digest, require_sha256_digest


@dataclass(frozen=True, slots=True)
class CompileKey:
    model_digest: str
    serving_revision_digest: str
    runtime_config_digest: str
    sampler_digest: str
    output_contract_digest: str
    device_type: str
    device_capability: str
    dtype: str
    shape_signature: tuple[int, ...]
    torch_version: str
    compiler_version: str

    def __post_init__(self) -> None:
        for name in (
            "model_digest",
            "serving_revision_digest",
            "runtime_config_digest",
            "sampler_digest",
            "output_contract_digest",
        ):
            object.__setattr__(self, name, require_sha256_digest(getattr(self, name), field=name))
        if any(value < 0 for value in self.shape_signature):
            raise ValueError("shape signature cannot contain negative dimensions")
        for name in (
            "device_type",
            "device_capability",
            "dtype",
            "torch_version",
            "compiler_version",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} cannot be empty")

    @property
    def digest(self) -> str:
        return content_digest(self)
