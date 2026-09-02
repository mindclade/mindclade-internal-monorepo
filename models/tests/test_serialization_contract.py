from __future__ import annotations

import json
import struct

import pytest
import torch

from mindclade.models import CladeFoldModel
from mindclade.models.api.serialization import (
    SerializationError,
    decode_safetensors,
    encode_safetensors,
    sha256_file,
)


def _rewrite_integrity_entry(directory, filename: str) -> None:
    manifest_path = directory / "model.integrity.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = directory / filename
    manifest["files"][filename] = {
        "digest": sha256_file(path),
        "size": path.stat().st_size,
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_repeated_save_pretrained_removes_only_stale_model_shards(
    tmp_path, cladefold_model
) -> None:
    documentation = tmp_path / "README.md"
    documentation.write_text("local model notes\n", encoding="utf-8")

    cladefold_model.save_pretrained(tmp_path, max_shard_size="64KiB")
    assert (tmp_path / "model.safetensors.index.json").is_file()
    assert list(tmp_path.glob("model-*-of-*.safetensors"))

    cladefold_model.save_pretrained(tmp_path, max_shard_size="4GiB")
    assert (tmp_path / "model.safetensors").is_file()
    assert not (tmp_path / "model.safetensors.index.json").exists()
    assert not list(tmp_path.glob("model-*-of-*.safetensors"))
    assert documentation.read_text(encoding="utf-8") == "local model notes\n"
    CladeFoldModel.from_pretrained(tmp_path)

    cladefold_model.save_pretrained(tmp_path, max_shard_size="64KiB")
    assert not (tmp_path / "model.safetensors").exists()
    CladeFoldModel.from_pretrained(tmp_path)


@pytest.mark.parametrize("filename", ["untracked.safetensors", "legacy-checkpoint.pt"])
def test_pretrained_loading_rejects_unmanifested_or_pickle_capable_weights(
    tmp_path, cladefold_model, filename: str
) -> None:
    cladefold_model.save_pretrained(tmp_path)
    (tmp_path / filename).write_bytes(b"not model data")
    with pytest.raises(SerializationError, match="unmanifested|pickle-capable"):
        CladeFoldModel.from_pretrained(tmp_path)


def test_pretrained_loading_rejects_protected_symlinks(tmp_path, cladefold_model) -> None:
    cladefold_model.save_pretrained(tmp_path)
    weights = tmp_path / "model.safetensors"
    backing = tmp_path / "weights.data"
    backing.write_bytes(weights.read_bytes())
    weights.unlink()
    weights.symlink_to(backing.name)

    with pytest.raises(SerializationError, match="unsafe|symbolic"):
        CladeFoldModel.from_pretrained(tmp_path)


def test_sharded_weight_map_must_name_the_shard_containing_each_tensor(
    tmp_path, cladefold_model
) -> None:
    cladefold_model.save_pretrained(tmp_path, max_shard_size="64KiB")
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    first_name = next(iter(weight_map))
    first_shard = weight_map[first_name]
    second_name = next(name for name, shard in weight_map.items() if shard != first_shard)
    weight_map[first_name], weight_map[second_name] = (
        weight_map[second_name],
        weight_map[first_name],
    )
    index_path.write_text(
        json.dumps(index, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rewrite_integrity_entry(tmp_path, index_path.name)

    with pytest.raises(SerializationError, match="weight map does not match shard contents"):
        CladeFoldModel.from_pretrained(tmp_path)


def test_malformed_safetensors_header_types_raise_serialization_error() -> None:
    header = {
        "value": {
            "dtype": "F32",
            "shape": [True],
            "data_offsets": [0, 4],
        }
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    payload = struct.pack("<Q", len(encoded)) + encoded + b"\0" * 4

    with pytest.raises(SerializationError, match="invalid tensor shape"):
        decode_safetensors(payload)


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((0,), torch.float32),
        ((2, 0, 3), torch.bfloat16),
        ((1, 0), torch.bool),
    ],
)
def test_zero_element_safetensors_round_trip(shape, dtype: torch.dtype) -> None:
    tensors = {
        "empty": torch.empty(shape, dtype=dtype),
        "value": torch.tensor([1, 2, 3], dtype=torch.int64),
    }

    decoded = decode_safetensors(encode_safetensors(tensors))

    assert decoded["empty"].shape == shape
    assert decoded["empty"].dtype is dtype
    assert decoded["empty"].numel() == 0
    torch.testing.assert_close(decoded["value"], tensors["value"])
