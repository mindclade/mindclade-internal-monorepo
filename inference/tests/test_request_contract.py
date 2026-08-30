from __future__ import annotations

import pytest
import torch
from mindclade.inference.batching.batch_key import BatchKey
from mindclade.inference.batching.batch_limits import BatchLimits
from mindclade.inference.batching.dynamic_batcher import DynamicBatcher
from mindclade.inference.pipeline.feature_resolution import resolve_features
from mindclade.inference.pipeline.preprocessing import preprocess_request


def test_request_fingerprint_is_content_stable_and_excludes_routing_ids(request_factory) -> None:
    first = request_factory()
    second = request_factory(
        request_id="request-2",
        tenant_id="tenant-b",
        project_id="project-b",
        inputs={name: tensor.clone() for name, tensor in first.inputs.items()},
    )
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != request_factory(seed=8).fingerprint

    first.inputs["token_type"][0, 0] = 4
    with pytest.raises(ValueError, match="changed after admission"):
        preprocess_request(first)


def test_request_requires_digest_seed_and_tensor_only_inputs(request_factory) -> None:
    with pytest.raises(ValueError, match="sha256"):
        request_factory(model_digest="cladefold-q0-random-init")
    with pytest.raises(ValueError, match="seed"):
        request_factory(seed=-1)
    with pytest.raises(TypeError, match=r"torch\.Tensor"):
        request_factory(inputs={"token_type": [1, 2]})


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"seed": True}, "seed"),
        ({"num_samples": 1.5}, "num_samples"),
        ({"num_steps": 32.0}, "num_steps"),
        ({"output_fields": "coordinates"}, "tuple of strings"),
    ],
)
def test_request_rejects_ambiguous_scalar_and_collection_types(
    request_factory, overrides, match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        request_factory(**overrides)


def test_request_rejects_non_strided_tensor_layout(request_factory) -> None:
    sparse = torch.sparse_coo_tensor(
        indices=[[0], [0]], values=[1], size=(1, 1), check_invariants=True
    )
    with pytest.raises(TypeError, match="strided"):
        request_factory(inputs={"token_type": sparse})


def test_request_owns_admitted_tensor_bytes(request_factory) -> None:
    source = torch.tensor([[1, 2]], dtype=torch.int64)
    request = request_factory(inputs={"token_type": source})
    fingerprint = request.fingerprint

    source.fill_(7)

    assert request.fingerprint == fingerprint
    request.verify_integrity()
    assert not torch.equal(request.inputs["token_type"], source)


def test_preprocessing_checks_device_dtype_finiteness_and_limits(request_factory) -> None:
    request = request_factory()
    prepared = preprocess_request(request, limits=BatchLimits.sync())
    assert (prepared.batch_size, prepared.tokens, prepared.atoms, prepared.bonds) == (1, 2, 3, 2)
    assert prepared.tensor_bytes > 0

    invalid = dict(request.inputs)
    invalid["atom_mask"] = torch.ones((1, 3), dtype=torch.int64)
    with pytest.raises(TypeError, match="atom_mask"):
        preprocess_request(request_factory(inputs=invalid))

    invalid = dict(request.inputs)
    invalid["coordinates"] = torch.full((1, 3, 3), float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        preprocess_request(request_factory(inputs=invalid))


def test_feature_receipt_and_batching_are_deterministic_and_tenant_isolated(
    request_factory,
) -> None:
    clock = [0]
    batcher = DynamicBatcher(BatchLimits.sync(), max_batch_size=2, clock_ns=lambda: clock[0])
    first = request_factory(request_id="one")
    second = request_factory(request_id="two")
    other_tenant = request_factory(request_id="three", tenant_id="tenant-b")
    assert BatchKey.from_request(first) == BatchKey.from_request(second)
    assert BatchKey.from_request(first) != BatchKey.from_request(other_tenant)

    assert not batcher.enqueue(first)
    assert batcher.enqueue(second)
    assert not batcher.enqueue(other_tenant)
    ready = batcher.pop_ready()
    assert len(ready) == 1
    assert [request.request_id for request in ready[0].requests] == ["one", "two"]
    assert batcher.depth == 1
    clock[0] = 11_000_000
    assert batcher.pop_ready()[0].requests == (other_tenant,)

    _, first_receipt = resolve_features(preprocess_request(first))
    _, second_receipt = resolve_features(preprocess_request(second))
    assert first_receipt.derivation_digest == second_receipt.derivation_digest
