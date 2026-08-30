"""CladeFold Q0 reference architecture.

Q0 is a systems-reference model with random initialization. Its tensor and
artifact contracts are production-shaped; it is not a scientifically qualified
predictor and ships no weights.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from mindclade.models.api.batch import BatchValidationError, CladeFoldBatch
from mindclade.models.api.model import PretrainedModel
from mindclade.models.api.outputs import ModelOutput
from mindclade.models.api.sampling import derive_sample_seed
from mindclade.models.common.embeddings.pair_embedding import PairEmbedding
from mindclade.models.common.embeddings.sequence_embedding import SequenceEmbedding
from mindclade.models.common.embeddings.time_embedding import TimeEmbedding
from mindclade.models.common.initialization.init_policy import InitializationPolicy
from mindclade.models.common.initialization.parameter_init import initialize_module
from mindclade.models.common.losses.loss_reduction import differentiable_zero
from mindclade.models.common.losses.masked_losses import (
    masked_cross_entropy,
    masked_huber,
    masked_mse,
)
from mindclade.models.common.masking.coordinate_mask import (
    apply_coordinate_mask,
    center_coordinates,
)
from mindclade.models.common.masking.pair_mask import make_pair_mask
from mindclade.models.components.confidence.calibration_head import CalibrationHead
from mindclade.models.components.confidence.confidence_head import ConfidenceHead
from mindclade.models.components.diffusion.diffusion_objective import noise_prediction_loss
from mindclade.models.components.diffusion.noise_schedule import VESchedule
from mindclade.models.components.geometry.distogram import DistogramHead
from mindclade.models.components.heads.coordinate_diffusion_head import CoordinateDiffusionHead
from mindclade.models.families.clade.cladefold.configuration.cladefold_q0 import CladeFoldConfig

from .diffusion_head import build_diffusion_head
from .pairformer_stack import PairformerStack


class CladeFoldModelOutput(ModelOutput):
    def __init__(
        self,
        *,
        loss: Tensor,
        noise_loss: Tensor,
        distogram_loss: Tensor,
        confidence_loss: Tensor,
        calibration_loss: Tensor,
        geometry_loss: Tensor,
        predicted_noise: Tensor,
        denoised_coordinates: Tensor,
        distogram_logits: Tensor,
        atom_confidence: Tensor,
        token_confidence: Tensor,
        calibration_temperature: Tensor,
        calibrated_confidence: Tensor,
        atom_confidence_logits: Tensor,
        token_confidence_logits: Tensor,
        token_hidden_states: Tensor | None = None,
        pair_hidden_states: Tensor | None = None,
        atom_hidden_states: Tensor | None = None,
    ) -> None:
        super().__init__(
            loss=loss,
            noise_loss=noise_loss,
            distogram_loss=distogram_loss,
            confidence_loss=confidence_loss,
            calibration_loss=calibration_loss,
            geometry_loss=geometry_loss,
            predicted_noise=predicted_noise,
            denoised_coordinates=denoised_coordinates,
            distogram_logits=distogram_logits,
            atom_confidence=atom_confidence,
            token_confidence=token_confidence,
            calibration_temperature=calibration_temperature,
            calibrated_confidence=calibrated_confidence,
            atom_confidence_logits=atom_confidence_logits,
            token_confidence_logits=token_confidence_logits,
            token_hidden_states=token_hidden_states,
            pair_hidden_states=pair_hidden_states,
            atom_hidden_states=atom_hidden_states,
        )


class CladeFoldFoldOutput(ModelOutput):
    def __init__(
        self,
        *,
        atom_coordinates: Tensor,
        atom_confidence: Tensor,
        token_confidence: Tensor,
        sample_confidence: Tensor,
        distogram_logits: Tensor,
        sample_seeds: Tensor,
        trajectories: Tensor | None = None,
    ) -> None:
        super().__init__(
            atom_coordinates=atom_coordinates,
            atom_confidence=atom_confidence,
            token_confidence=token_confidence,
            sample_confidence=sample_confidence,
            distogram_logits=distogram_logits,
            sample_seeds=sample_seeds,
            trajectories=trajectories,
        )


class _AtomEmbedding(nn.Module):
    def __init__(self, config: CladeFoldConfig) -> None:
        super().__init__()
        self.atomic_number = nn.Embedding(
            config.max_atomic_number + 1, config.atom_dim, padding_idx=0
        )
        self.formal_charge = nn.Embedding(config.charge_vocab_size, config.atom_dim)
        self.chirality = nn.Embedding(config.chirality_vocab_size, config.atom_dim)
        self.aromatic = nn.Embedding(2, config.atom_dim)
        self.token_projection = nn.Linear(config.token_dim, config.atom_dim, bias=False)
        self.norm = nn.LayerNorm(config.atom_dim, eps=config.layer_norm_epsilon)

    def forward(self, batch: CladeFoldBatch, tokens: Tensor) -> Tensor:
        batch_index = torch.arange(batch.batch_size, device=batch.device).unsqueeze(1)
        token_context = tokens[batch_index, batch.atom_to_token.clamp_min(0)]
        values = (
            self.atomic_number(batch.atomic_number)
            + self.formal_charge(batch.formal_charge + 8)
            + self.chirality(batch.chirality)
            + self.aromatic(batch.aromatic_mask.long())
            + self.token_projection(token_context)
        )
        return cast(Tensor, self.norm(values)) * batch.atom_mask.to(dtype=values.dtype).unsqueeze(
            -1
        )


class CladeFoldModel(PretrainedModel[CladeFoldConfig]):
    """Pairformer plus proper-rotation-equivariant VE denoiser."""

    config_class = CladeFoldConfig

    def __init__(self, config: CladeFoldConfig) -> None:
        super().__init__(config)
        self.sequence_embedding = SequenceEmbedding(
            token_vocab_size=config.token_vocab_size,
            molecule_vocab_size=config.molecule_vocab_size,
            max_chain_id=config.max_chain_id,
            max_position_id=config.max_position_id,
            dimension=config.token_dim,
        )
        self.pair_embedding = PairEmbedding(config.token_dim, config.pair_dim)
        self.pairformer = PairformerStack(config)
        self.atom_embedding = _AtomEmbedding(config)
        self.time_embedding = TimeEmbedding(config.time_dim)
        self.denoiser = build_diffusion_head(config)
        self.coordinate_head = CoordinateDiffusionHead()
        self.distogram_head = DistogramHead(config.pair_dim, config.distogram_bins)
        self.confidence_head = ConfidenceHead(
            config.atom_dim, config.token_dim, config.confidence_bins
        )
        self.calibration_head = CalibrationHead(config.token_dim)
        self.schedule = VESchedule(config.sigma_min, config.sigma_max)
        initialize_module(self, InitializationPolicy(config.initialization_std))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate_config_ranges(self, batch: CladeFoldBatch) -> None:
        model_device = next(self.parameters()).device
        if batch.device != model_device:
            raise BatchValidationError(
                f"batch is on {batch.device}, but model parameters are on {model_device}"
            )
        checks = (
            ("token_type", batch.token_type, 0, self.config.token_vocab_size - 1, batch.token_mask),
            (
                "molecule_type",
                batch.molecule_type,
                0,
                self.config.molecule_vocab_size - 1,
                batch.token_mask,
            ),
            ("chain_id", batch.chain_id, 0, self.config.max_chain_id - 1, batch.token_mask),
            (
                "position_id",
                batch.position_id,
                0,
                self.config.max_position_id - 1,
                batch.token_mask,
            ),
            ("formal_charge", batch.formal_charge, -8, 8, batch.atom_mask),
            (
                "chirality",
                batch.chirality,
                0,
                self.config.chirality_vocab_size - 1,
                batch.atom_mask,
            ),
            (
                "bond_type",
                batch.bond_type,
                0,
                self.config.bond_type_vocab_size - 1,
                batch.bond_mask,
            ),
            (
                "bond_stereo",
                batch.bond_stereo,
                0,
                self.config.bond_stereo_vocab_size - 1,
                batch.bond_mask,
            ),
            (
                "atomic_number",
                batch.atomic_number,
                1,
                self.config.max_atomic_number,
                batch.atom_mask,
            ),
        )
        for name, values, lower, upper, mask in checks:
            selected = values[mask]
            if selected.numel() and ((selected < lower) | (selected > upper)).any():
                raise BatchValidationError(f"{name} values must be in [{lower}, {upper}]")

    def _encode_trunk(self, batch: CladeFoldBatch) -> tuple[Tensor, Tensor, Tensor]:
        pair_mask = make_pair_mask(batch.token_mask)
        tokens = self.sequence_embedding(
            batch.token_type,
            batch.molecule_type,
            batch.chain_id,
            batch.position_id,
            batch.token_mask,
        )
        pair = self.pair_embedding(tokens, batch.position_id, batch.chain_id, pair_mask)
        tokens, pair = self.pairformer(tokens, pair, batch.token_mask, pair_mask)
        atoms = self.atom_embedding(batch, tokens)
        return tokens, pair, atoms

    def _predict(
        self,
        batch: CladeFoldBatch,
        tokens: Tensor,
        pair: Tensor,
        initial_atoms: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        assert batch.noisy_coordinates is not None and batch.diffusion_time is not None
        time = self.time_embedding(batch.diffusion_time)
        predicted_noise, atoms = self.denoiser(
            initial_atoms,
            batch.noisy_coordinates,
            pair,
            batch.atom_to_token,
            batch.atom_mask,
            batch.bond_indices,
            batch.bond_type,
            batch.bond_stereo,
            batch.bond_mask,
            time,
        )
        sigma = self.schedule.sigma(batch.diffusion_time).to(dtype=batch.noisy_coordinates.dtype)
        denoised = self.coordinate_head(
            batch.noisy_coordinates, predicted_noise, sigma, batch.atom_mask
        )
        return predicted_noise, denoised, atoms

    def forward(
        self, batch: CladeFoldBatch, *, return_hidden_states: bool | None = None
    ) -> CladeFoldModelOutput:
        batch.validate("forward")
        self._validate_config_ranges(batch)
        tokens, pair, initial_atoms = self._encode_trunk(batch)
        predicted_noise, denoised, atoms = self._predict(batch, tokens, pair, initial_atoms)
        pair_mask = make_pair_mask(batch.token_mask)
        distogram_logits = self.distogram_head(pair, pair_mask)
        atom_logits, token_logits, atom_confidence, token_confidence = self.confidence_head(
            atoms, tokens, batch.atom_mask, batch.token_mask
        )
        calibration_temperature, calibrated_tokens = self.calibration_head(
            tokens, batch.token_mask, token_confidence
        )
        calibrated_confidence = (
            calibrated_tokens * batch.token_mask.to(calibrated_tokens.dtype)
        ).sum(dim=1) / batch.token_mask.sum(dim=1).clamp_min(1).to(calibrated_tokens.dtype)

        zero = differentiable_zero(predicted_noise)
        noise_loss = distogram_loss = confidence_loss = calibration_loss = geometry_loss = zero
        if batch.target_coordinates is not None:
            assert batch.target_mask is not None
            assert batch.noisy_coordinates is not None and batch.diffusion_time is not None
            sigma = self.schedule.sigma(batch.diffusion_time)
            noise_loss = noise_prediction_loss(
                predicted_noise,
                batch.noisy_coordinates,
                batch.target_coordinates,
                sigma,
                batch.target_mask,
            )
            distogram_loss = self._distogram_loss(batch, distogram_logits)
            confidence_loss, accuracy = self._confidence_loss(
                batch, denoised, atom_logits, token_logits
            )
            calibration_loss = masked_mse(
                calibrated_confidence, accuracy, torch.ones_like(accuracy, dtype=torch.bool)
            )
            geometry_loss = self._geometry_loss(batch, denoised)
        loss = (
            self.config.noise_loss_weight * noise_loss
            + self.config.distogram_loss_weight * distogram_loss
            + self.config.confidence_loss_weight * confidence_loss
            + self.config.calibration_loss_weight * calibration_loss
            + self.config.geometry_loss_weight * geometry_loss
        ).float()
        include_hidden = (
            self.config.output_hidden_states
            if return_hidden_states is None
            else return_hidden_states
        )
        return CladeFoldModelOutput(
            loss=loss,
            noise_loss=noise_loss.float(),
            distogram_loss=distogram_loss.float(),
            confidence_loss=confidence_loss.float(),
            calibration_loss=calibration_loss.float(),
            geometry_loss=geometry_loss.float(),
            predicted_noise=predicted_noise,
            denoised_coordinates=denoised,
            distogram_logits=distogram_logits,
            atom_confidence=atom_confidence,
            token_confidence=token_confidence,
            calibration_temperature=calibration_temperature,
            calibrated_confidence=calibrated_confidence,
            atom_confidence_logits=atom_logits,
            token_confidence_logits=token_logits,
            token_hidden_states=tokens if include_hidden else None,
            pair_hidden_states=pair if include_hidden else None,
            atom_hidden_states=atoms if include_hidden else None,
        )

    def _anchor_coordinates(self, batch: CladeFoldBatch, coordinates: Tensor) -> Tensor:
        batch_index = torch.arange(batch.batch_size, device=batch.device).unsqueeze(1)
        return coordinates[batch_index, batch.anchor_atom_index.clamp_min(0)]

    def _distogram_loss(self, batch: CladeFoldBatch, logits: Tensor) -> Tensor:
        assert batch.target_coordinates is not None and batch.target_mask is not None
        anchors = self._anchor_coordinates(batch, batch.target_coordinates)
        distance = torch.cdist(anchors.float(), anchors.float())
        boundaries = torch.linspace(
            self.config.distogram_min_angstrom,
            self.config.distogram_max_angstrom,
            self.config.distogram_bins - 1,
            device=distance.device,
        )
        target = torch.bucketize(distance, boundaries)
        anchor_target = self._anchor_coordinates(
            batch, batch.target_mask.to(dtype=anchors.dtype).unsqueeze(-1).expand(-1, -1, 3)
        )[..., 0].bool()
        mask = (
            make_pair_mask(batch.token_mask)
            & anchor_target.unsqueeze(2)
            & anchor_target.unsqueeze(1)
        )
        return masked_cross_entropy(logits, target, mask)

    def _confidence_loss(
        self,
        batch: CladeFoldBatch,
        denoised: Tensor,
        atom_logits: Tensor,
        token_logits: Tensor,
    ) -> tuple[Tensor, Tensor]:
        assert batch.target_coordinates is not None and batch.target_mask is not None
        error = torch.linalg.vector_norm(
            denoised.float() - batch.target_coordinates.float(), dim=-1
        )
        accuracy_per_atom = torch.exp(-error / 4.0)
        target_bin = (
            torch.floor(accuracy_per_atom * (self.config.confidence_bins - 1))
            .long()
            .clamp(0, self.config.confidence_bins - 1)
        )
        atom_loss = masked_cross_entropy(atom_logits, target_bin, batch.target_mask)
        anchor_error = self._anchor_coordinates(batch, error.unsqueeze(-1))[..., 0]
        anchor_valid = self._anchor_coordinates(
            batch, batch.target_mask.to(error.dtype).unsqueeze(-1).expand(-1, -1, 3)
        )[..., 0].bool()
        token_target = (
            torch.floor(torch.exp(-anchor_error / 4.0) * (self.config.confidence_bins - 1))
            .long()
            .clamp(0, self.config.confidence_bins - 1)
        )
        token_loss = masked_cross_entropy(
            token_logits, token_target, batch.token_mask & anchor_valid
        )
        weights = batch.target_mask.to(dtype=accuracy_per_atom.dtype)
        aggregate = (accuracy_per_atom * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return 0.5 * (atom_loss + token_loss), aggregate

    def _geometry_loss(self, batch: CladeFoldBatch, denoised: Tensor) -> Tensor:
        assert batch.target_coordinates is not None and batch.target_mask is not None
        predicted_distance = torch.cdist(denoised.float(), denoised.float())
        target_distance = torch.cdist(
            batch.target_coordinates.float(), batch.target_coordinates.float()
        )
        valid = batch.target_mask.unsqueeze(2) & batch.target_mask.unsqueeze(1)
        atoms = batch.atom_count
        diagonal = torch.eye(atoms, device=batch.device, dtype=torch.bool).unsqueeze(0)
        candidates = valid & ~diagonal
        k = min(self.config.atom_knn, max(atoms - 1, 1))
        nearest = torch.zeros_like(candidates)
        nearest_index = (
            target_distance.masked_fill(~candidates, float("inf"))
            .topk(k, dim=-1, largest=False, sorted=True)
            .indices
        )
        nearest.scatter_(2, nearest_index, True)
        nearest &= candidates
        bonds = torch.zeros_like(candidates)
        for batch_index in range(batch.batch_size):
            active = batch.bond_mask[batch_index]
            if bool(active.any()):
                edge = batch.bond_indices[batch_index, active]
                bonds[batch_index, edge[:, 0], edge[:, 1]] = True
                bonds[batch_index, edge[:, 1], edge[:, 0]] = True
        return masked_huber(predicted_distance, target_distance, nearest | bonds, delta=1.0)

    @torch.no_grad()
    def fold(
        self,
        batch: CladeFoldBatch,
        *,
        seed: int,
        num_samples: int = 1,
        num_steps: int | None = None,
        return_trajectory: bool = False,
    ) -> CladeFoldFoldOutput:
        if self.training:
            raise RuntimeError("fold is eval-only; call model.eval() first")
        batch.validate("static")
        self._validate_config_ranges(batch)
        if type(seed) is not int or seed < 0 or seed >= 2**63:
            raise ValueError("seed must be in [0, 2**63)")
        if type(num_samples) is not int or num_samples < 1:
            raise ValueError("num_samples must be positive")
        steps = self.config.default_sampling_steps if num_steps is None else num_steps
        if type(steps) is not int or steps < 2:
            raise ValueError("fold requires at least two steps")

        tokens, pair, atoms = self._encode_trunk(batch)
        repeated = self._repeat_static_batch(batch, num_samples)
        repeated_tokens = tokens.repeat_interleave(num_samples, dim=0)
        repeated_pair = pair.repeat_interleave(num_samples, dim=0)
        repeated_atoms = atoms.repeat_interleave(num_samples, dim=0)
        sample_seeds = torch.empty(
            (batch.batch_size, num_samples), device=batch.device, dtype=torch.int64
        )
        coordinate_samples: list[Tensor] = []
        for batch_index in range(batch.batch_size):
            for sample_index in range(num_samples):
                sample_seed = derive_sample_seed(seed, batch_index * num_samples + sample_index)
                sample_seeds[batch_index, sample_index] = sample_seed
                generator = torch.Generator(device=batch.device)
                generator.manual_seed(sample_seed)
                coordinate_samples.append(
                    torch.randn(
                        (batch.atom_count, 3),
                        generator=generator,
                        device=batch.device,
                        dtype=next(self.parameters()).dtype,
                    )
                )
        coordinates = torch.stack(coordinate_samples, dim=0) * self.config.sigma_max
        coordinates = center_coordinates(coordinates, repeated.atom_mask)
        levels = self.schedule.levels(steps, device=batch.device, dtype=coordinates.dtype)
        trajectory = [coordinates] if return_trajectory else None
        final_atom_hidden = repeated_atoms
        for index in range(steps):
            sigma = levels[index]
            next_sigma = levels[index + 1] if index + 1 < steps else levels.new_zeros(())
            time = self.schedule.normalized_time(sigma.expand(repeated.batch_size)).clamp(0.0, 1.0)
            step_batch = repeated.replace(
                noisy_coordinates=coordinates,
                diffusion_time=time,
            )
            noise, _denoised, atom_hidden = self._predict(
                step_batch, repeated_tokens, repeated_pair, repeated_atoms
            )
            delta = next_sigma - sigma
            proposal = apply_coordinate_mask(coordinates + delta * noise, repeated.atom_mask)
            if float(next_sigma) > 0.0:
                next_time = self.schedule.normalized_time(
                    next_sigma.expand(repeated.batch_size)
                ).clamp(0.0, 1.0)
                proposal_batch = repeated.replace(
                    noisy_coordinates=proposal,
                    diffusion_time=next_time,
                )
                next_noise, _next_denoised, next_atoms = self._predict(
                    proposal_batch, repeated_tokens, repeated_pair, repeated_atoms
                )
                coordinates = apply_coordinate_mask(
                    coordinates + delta * 0.5 * (noise + next_noise), repeated.atom_mask
                )
                final_atom_hidden = next_atoms
            else:
                coordinates = proposal
                final_atom_hidden = atom_hidden
            if trajectory is not None:
                trajectory.append(coordinates)

        pair_mask = make_pair_mask(batch.token_mask)
        distogram_logits = self.distogram_head(pair, pair_mask)
        repeated_token_mask = repeated.token_mask
        _, _, atom_confidence, token_confidence = self.confidence_head(
            final_atom_hidden,
            repeated_tokens,
            repeated.atom_mask,
            repeated_token_mask,
        )
        _, calibrated_tokens = self.calibration_head(
            repeated_tokens, repeated_token_mask, token_confidence
        )
        sample_confidence = (
            calibrated_tokens * repeated_token_mask.to(calibrated_tokens.dtype)
        ).sum(dim=1) / repeated_token_mask.sum(dim=1).clamp_min(1).to(calibrated_tokens.dtype)
        b, s, a = batch.batch_size, num_samples, batch.atom_count
        final_coordinates = coordinates.reshape(b, s, a, 3)
        atom_confidence = atom_confidence.reshape(b, s, a)
        token_confidence = token_confidence.reshape(b, s, batch.token_count)
        sample_confidence = sample_confidence.reshape(b, s)
        trajectories = None
        if trajectory is not None:
            trajectories = torch.stack(trajectory, dim=1).reshape(b, s, steps + 1, a, 3)
        return CladeFoldFoldOutput(
            atom_coordinates=final_coordinates,
            atom_confidence=atom_confidence,
            token_confidence=token_confidence,
            sample_confidence=sample_confidence,
            distogram_logits=distogram_logits,
            sample_seeds=sample_seeds,
            trajectories=trajectories,
        )

    @staticmethod
    def _repeat_static_batch(batch: CladeFoldBatch, samples: int) -> CladeFoldBatch:
        values = {}
        for name, tensor in batch.tensor_items():
            values[name] = tensor.repeat_interleave(samples, dim=0)
        return CladeFoldBatch(**values)


__all__ = ["CladeFoldFoldOutput", "CladeFoldModel", "CladeFoldModelOutput"]
