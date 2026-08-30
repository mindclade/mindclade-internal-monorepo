"""CladeFold denoiser composition."""

from __future__ import annotations

from mindclade.models.components.diffusion.coordinate_denoiser import CoordinateDenoiser
from mindclade.models.families.clade.cladefold.configuration.cladefold_q0 import CladeFoldConfig


def build_diffusion_head(config: CladeFoldConfig) -> CoordinateDenoiser:
    return CoordinateDenoiser(
        atom_dim=config.atom_dim,
        pair_dim=config.pair_dim,
        time_dim=config.time_dim,
        blocks=config.denoiser_blocks,
        heads=config.atom_heads,
        radial_basis_functions=config.radial_basis_functions,
        atom_knn=config.atom_knn,
        bond_type_vocab_size=config.bond_type_vocab_size,
        bond_stereo_vocab_size=config.bond_stereo_vocab_size,
        dropout=config.dropout,
        epsilon=config.layer_norm_epsilon,
        sigma_max=config.sigma_max,
    )


__all__ = ["build_diffusion_head"]
