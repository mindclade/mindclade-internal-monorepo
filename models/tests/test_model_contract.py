from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest
import torch

from mindclade.models import CladeFoldConfig, CladeFoldModel
from mindclade.models.api.batch import BatchValidationError
from mindclade.models.common.configuration.config_validation import ConfigurationError
from models.conftest import make_batch


def test_forward_backward_and_padding_contract(
    cladefold_model: CladeFoldModel, cladefold_batch
) -> None:
    output = cladefold_model(cladefold_batch, return_hidden_states=True)
    assert output.predicted_noise.shape == (2, 6, 3)
    assert output.distogram_logits.shape == (2, 4, 4, 16)
    assert output.loss.dtype == torch.float32
    assert all(torch.isfinite(value).all() for value in output.to_tuple())
    assert torch.count_nonzero(output.predicted_noise[~cladefold_batch.atom_mask]) == 0
    assert torch.count_nonzero(output.atom_confidence[~cladefold_batch.atom_mask]) == 0
    assert torch.count_nonzero(output.token_confidence[~cladefold_batch.token_mask]) == 0
    pair_mask = cladefold_batch.token_mask.unsqueeze(2) & cladefold_batch.token_mask.unsqueeze(1)
    assert torch.count_nonzero(output.distogram_logits[~pair_mask]) == 0

    output.loss.backward()
    gradients = [parameter.grad for parameter in cladefold_model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_batch_size_one_and_actionable_reserved_token_error(
    cladefold_model: CladeFoldModel,
) -> None:
    batch = make_batch(batch_size=1)
    assert cladefold_model(batch).predicted_noise.shape[0] == 1
    invalid = dataclasses.replace(batch, token_type=batch.token_type.clone())
    invalid.token_type[0, 0] = 34
    with pytest.raises(BatchValidationError, match="reserved"):
        cladefold_model(invalid)


def test_model_output_is_immutable_and_mapping_compatible(cladefold_model, cladefold_batch) -> None:
    output = cladefold_model(cladefold_batch)
    assert output["loss"] is output.loss
    assert output[0] is output.loss
    assert output[:2] == output.to_tuple()[:2]
    with pytest.raises(TypeError, match="immutable"):
        output.loss = torch.tensor(0.0)


def test_config_pretrained_directory_round_trip(tmp_path) -> None:
    config = CladeFoldConfig.tiny()
    assert config.save_pretrained(tmp_path).name == "config.json"
    assert CladeFoldConfig.from_pretrained(tmp_path) == config


def test_configuration_schema_covers_the_serialized_config_exactly() -> None:
    schema_path = Path(
        "models/src/mindclade/models/families/clade/cladefold/"
        "configuration/configuration.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert set(schema["properties"]) == set(CladeFoldConfig.tiny().to_dict())
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"schema_version": True}, "schema_version"),
        ({"token_dim": 32.0}, "positive integer"),
        ({"sigma_max": math.nan}, "finite real"),
        ({"noise_loss_weight": math.inf}, "finite real"),
        ({"output_hidden_states": 1}, "boolean"),
    ],
)
def test_config_rejects_ambiguous_types_and_nonfinite_values(overrides, match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        CladeFoldConfig.tiny(**overrides)


@pytest.mark.parametrize(
    ("option", "value"),
    [("seed", False), ("num_samples", 1.5), ("num_steps", 2.0)],
)
def test_fold_rejects_non_integer_sampling_controls(
    cladefold_model, cladefold_batch, option: str, value: object
) -> None:
    cladefold_model.eval()
    arguments = {"seed": 1, "num_samples": 1, "num_steps": 2}
    arguments[option] = value
    with pytest.raises(ValueError):
        cladefold_model.fold(cladefold_batch.static(), **arguments)
