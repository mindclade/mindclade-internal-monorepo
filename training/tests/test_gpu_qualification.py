from __future__ import annotations

import os

import pytest
import torch
from mindclade.models import CladeFoldConfig, CladeFoldModel
from mindclade.models.components.pairformer.pairformer_block import PairformerBlock
from mindclade.training import PrecisionConfig, PrecisionMode
from mindclade.training.core.data import (
    DeterministicSyntheticDataset,
    SyntheticSplit,
    collate_cladefold_examples,
)
from mindclade.training.execution.single_process.engine import SingleProcessEngine
from mindclade.training.providers.pytorch.fsdp2_adapter import apply_fsdp2, fsdp2_capability


def qualification_config() -> CladeFoldConfig:
    return CladeFoldConfig.tiny(
        token_dim=32,
        pair_dim=16,
        atom_dim=32,
        time_dim=16,
        pairformer_blocks=1,
        denoiser_blocks=1,
        atom_knn=4,
        default_sampling_steps=2,
    )


def qualification_batch():
    example = DeterministicSyntheticDataset(
        SyntheticSplit.TEST,
        seed=17,
        token_count=4,
        atom_count=8,
    )[0]
    return collate_cladefold_examples([example]).payload


@pytest.mark.gpu
def test_cladefold_cuda_bfloat16_forward_is_finite() -> None:
    assert torch.cuda.is_available()
    assert torch.cuda.is_bf16_supported()
    device = torch.device("cuda", 0)
    torch.manual_seed(17)
    model = CladeFoldModel(qualification_config()).to(device).eval()
    batch = qualification_batch().to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(batch)
    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.denoised_coordinates).all()


@pytest.mark.gpu
def test_cuda_fp16_amp_update_keeps_fp32_parameters_and_gradients() -> None:
    assert torch.cuda.is_available()
    device = torch.device("cuda", 0)
    torch.manual_seed(23)
    engine = SingleProcessEngine.create(
        PrecisionConfig(mode=PrecisionMode.FP16),
        device=device,
    )
    model = engine.prepare_model(torch.nn.Linear(4, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    inputs = torch.randn((8, 4), device=device)
    targets = torch.randn((8, 2), device=device)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    optimizer.zero_grad(set_to_none=True)
    with engine.precision.autocast():
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
    engine.backward(loss)
    engine.unscale_(optimizer)

    assert all(parameter.dtype is torch.float32 for parameter in model.parameters())
    assert all(
        parameter.grad is not None
        and parameter.grad.dtype is torch.float32
        and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    engine.optimizer_step(optimizer)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert any(
        not torch.equal(previous, parameter)
        for previous, parameter in zip(before, model.parameters(), strict=True)
    )


@pytest.mark.gpu
@pytest.mark.distributed
def test_two_gpu_fsdp2_bottom_up_sharding() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group("nccl")
    try:
        assert torch.distributed.get_world_size() == 2
        model = CladeFoldModel(qualification_config()).to(torch.device("cuda", local_rank))
        sharded = apply_fsdp2(model, block_types=(PairformerBlock,))
        assert sharded is model
        assert fsdp2_capability().ready
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()
