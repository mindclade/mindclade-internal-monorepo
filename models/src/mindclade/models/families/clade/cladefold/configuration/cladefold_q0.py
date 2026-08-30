"""CladeFold Q0 architecture and diffusion configuration."""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

from mindclade.models.common.configuration.config_validation import (
    ConfigurationError,
    require_nonnegative,
    require_positive,
    require_positive_integer,
    require_probability,
)
from mindclade.models.common.configuration.model_config import ModelConfig


@dataclasses.dataclass(frozen=True)
class CladeFoldConfig(ModelConfig):
    """Frozen Q0 configuration.

    The default shape is production-sized and random initialized. ``tiny`` is
    the CPU contract fixture and intentionally uses the same code path.
    """

    model_type: ClassVar[str] = "cladefold-q0"

    token_vocab_size: int = 64
    molecule_vocab_size: int = 8
    max_chain_id: int = 1024
    max_position_id: int = 8192
    max_atomic_number: int = 118
    charge_vocab_size: int = 17
    chirality_vocab_size: int = 8
    bond_type_vocab_size: int = 8
    bond_stereo_vocab_size: int = 8

    token_dim: int = 384
    pair_dim: int = 192
    atom_dim: int = 256
    time_dim: int = 128
    pairformer_blocks: int = 24
    denoiser_blocks: int = 12
    token_heads: int = 12
    triangle_heads: int = 6
    atom_heads: int = 8
    transition_multiplier: int = 4
    outer_product_dim: int = 32
    radial_basis_functions: int = 32
    atom_knn: int = 32

    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initialization_std: float = 0.02
    distogram_bins: int = 64
    distogram_min_angstrom: float = 2.0
    distogram_max_angstrom: float = 22.0
    confidence_bins: int = 50
    sigma_min: float = 0.01
    sigma_max: float = 20.0
    default_sampling_steps: int = 50
    output_hidden_states: bool = False

    noise_loss_weight: float = 1.0
    distogram_loss_weight: float = 0.30
    confidence_loss_weight: float = 0.10
    calibration_loss_weight: float = 0.05
    geometry_loss_weight: float = 0.10

    def __post_init__(self) -> None:
        super().__post_init__()
        self.validate()

    def validate(self) -> None:
        super().validate()
        positive_ints = (
            "token_vocab_size",
            "molecule_vocab_size",
            "max_chain_id",
            "max_position_id",
            "max_atomic_number",
            "charge_vocab_size",
            "chirality_vocab_size",
            "bond_type_vocab_size",
            "bond_stereo_vocab_size",
            "token_dim",
            "pair_dim",
            "atom_dim",
            "time_dim",
            "pairformer_blocks",
            "denoiser_blocks",
            "token_heads",
            "triangle_heads",
            "atom_heads",
            "transition_multiplier",
            "outer_product_dim",
            "radial_basis_functions",
            "atom_knn",
            "distogram_bins",
            "confidence_bins",
            "default_sampling_steps",
        )
        for name in positive_ints:
            require_positive_integer(name, getattr(self, name))
        if self.token_vocab_size < 34:
            raise ConfigurationError("token_vocab_size must include Q0 IDs 0 through 33")
        if self.max_atomic_number < 118:
            raise ConfigurationError("max_atomic_number must include elements 1 through 118")
        if self.charge_vocab_size < 17:
            raise ConfigurationError(
                "charge_vocab_size must represent formal charges -8 through +8"
            )
        for dimension, heads in (
            (self.token_dim, self.token_heads),
            (self.pair_dim, self.triangle_heads),
            (self.atom_dim, self.atom_heads),
        ):
            if dimension % heads:
                raise ConfigurationError(
                    f"dimension {dimension} must be divisible by heads {heads}"
                )
        require_probability("dropout", self.dropout)
        require_positive("layer_norm_epsilon", self.layer_norm_epsilon)
        require_positive("initialization_std", self.initialization_std)
        require_positive("distogram_min_angstrom", self.distogram_min_angstrom)
        require_positive("distogram_max_angstrom", self.distogram_max_angstrom)
        require_positive("sigma_min", self.sigma_min)
        require_positive("sigma_max", self.sigma_max)
        if self.sigma_max <= self.sigma_min:
            raise ConfigurationError("sigma_max must be greater than sigma_min")
        if self.distogram_max_angstrom <= self.distogram_min_angstrom:
            raise ConfigurationError("distogram maximum must exceed minimum")
        if self.default_sampling_steps < 2:
            raise ConfigurationError("default_sampling_steps must be at least two")
        for name in (
            "noise_loss_weight",
            "distogram_loss_weight",
            "confidence_loss_weight",
            "calibration_loss_weight",
            "geometry_loss_weight",
        ):
            require_nonnegative(name, getattr(self, name))
        if type(self.output_hidden_states) is not bool:
            raise ConfigurationError("output_hidden_states must be a boolean")

    @classmethod
    def tiny(cls, **overrides: Any) -> CladeFoldConfig:
        values: dict[str, Any] = {
            "token_dim": 64,
            "pair_dim": 32,
            "atom_dim": 64,
            "time_dim": 32,
            "pairformer_blocks": 2,
            "denoiser_blocks": 2,
            "token_heads": 4,
            "triangle_heads": 4,
            "atom_heads": 4,
            "transition_multiplier": 2,
            "outer_product_dim": 8,
            "radial_basis_functions": 8,
            "atom_knn": 8,
            "default_sampling_steps": 4,
            "dropout": 0.0,
        }
        values.update(overrides)
        return cls(**values)


__all__ = ["CladeFoldConfig"]
