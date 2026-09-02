"""Local-only safetensors serialization with digest verification.

The codec implements the public safetensors container layout directly so the
core package has no optional import at runtime. It deliberately supports only
the dtypes used in PyTorch state dictionaries and never evaluates code or
unpickles data.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import tempfile
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


class SerializationError(RuntimeError):
    """Raised when a local model artifact is malformed or fails integrity."""


_DTYPE_TO_CODE = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}
_CODE_TO_DTYPE = {value: key for key, value in _DTYPE_TO_CODE.items()}
_TENSOR_FILE = re.compile(r"^model(?:-\d{5}-of-\d{5})?\.safetensors$")
_SHARDED_TENSOR_FILE = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")
_INDEX_FILE = "model.safetensors.index.json"
_INTEGRITY_FILE = "model.integrity.json"
_FORBIDDEN_MODEL_SUFFIXES = {".bin", ".pkl", ".pickle", ".pt", ".pth"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _decode_json_object(payload: bytes | str, *, description: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SerializationError(f"{description} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        value = json.loads(text, object_pairs_hook=reject_duplicates)
    except SerializationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SerializationError(f"invalid {description}") from exc
    if not isinstance(value, dict):
        raise SerializationError(f"{description} must be an object")
    return value


def read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    """Read a regular local JSON object with duplicate-key rejection."""

    if path.is_symlink() or not path.is_file():
        raise SerializationError(f"{description} is missing or is not a regular file")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise SerializationError(f"cannot read {description}") from exc
    return _decode_json_object(payload, description=description)


def _tensor_bytes(tensor: Tensor) -> bytes:
    if tensor.layout is not torch.strided:
        raise SerializationError("only strided tensors can be serialized")
    # Clone ensures storage_offset=0 and that the untyped storage contains only
    # this tensor, including for contiguous views into a larger allocation.
    contiguous = tensor.detach().cpu().contiguous().clone()
    # Reading untyped storage preserves bfloat16/integer bit patterns and keeps
    # NumPy out of the core serialization dependency surface.
    return bytes(contiguous.untyped_storage())


def encode_safetensors(tensors: Mapping[str, Tensor]) -> bytes:
    header: dict[str, Any] = {"__metadata__": {"format": "pt", "schema_version": "1"}}
    chunks: list[bytes] = []
    offset = 0
    seen_storage: dict[tuple[int, int], str] = {}
    names = list(tensors)
    if any(type(name) is not str or not name or name == "__metadata__" for name in names):
        raise SerializationError("state tensor names must be non-empty strings")
    for name in sorted(names):
        tensor = tensors[name]
        if not isinstance(tensor, Tensor):
            raise SerializationError(f"state value {name!r} is not a tensor")
        if tensor.dtype not in _DTYPE_TO_CODE:
            raise SerializationError(f"unsupported dtype {tensor.dtype} for {name!r}")
        if tensor.numel() > 0:
            key = (tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())
            previous = seen_storage.get(key)
            if previous is not None:
                raise SerializationError(
                    f"undeclared tensor aliasing between {previous!r} and {name!r} is not supported"
                )
            seen_storage[key] = name
        data = _tensor_bytes(tensor)
        header[name] = {
            "dtype": _DTYPE_TO_CODE[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        chunks.append(data)
        offset += len(data)
    raw_header = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    padding = (-len(raw_header)) % 8
    raw_header += b" " * padding
    return struct.pack("<Q", len(raw_header)) + raw_header + b"".join(chunks)


def decode_safetensors(payload: bytes) -> OrderedDict[str, Tensor]:
    if len(payload) < 10:
        raise SerializationError("safetensors payload is truncated")
    (header_size,) = struct.unpack("<Q", payload[:8])
    if header_size < 2 or header_size > 100 * 1024 * 1024 or 8 + header_size > len(payload):
        raise SerializationError("invalid safetensors header length")
    header = _decode_json_object(
        payload[8 : 8 + header_size], description="safetensors JSON header"
    )
    metadata = header.get("__metadata__")
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(type(key) is not str or type(value) is not str for key, value in metadata.items())
    ):
        raise SerializationError("safetensors metadata must contain string keys and values")
    data = payload[8 + header_size :]
    result: OrderedDict[str, Tensor] = OrderedDict()
    occupied: list[tuple[int, int]] = []
    names = [key for key in header if key != "__metadata__"]
    if any(type(name) is not str or not name for name in names):
        raise SerializationError("safetensors tensor names must be non-empty strings")
    for name in sorted(names):
        entry = header[name]
        if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
            raise SerializationError(f"invalid tensor header for {name!r}")
        dtype_code = entry["dtype"]
        raw_shape = entry["shape"]
        raw_offsets = entry["data_offsets"]
        if type(dtype_code) is not str or dtype_code not in _CODE_TO_DTYPE:
            raise SerializationError(f"invalid tensor dtype for {name!r}")
        if not isinstance(raw_shape, list) or any(type(value) is not int for value in raw_shape):
            raise SerializationError(f"invalid tensor shape for {name!r}")
        if (
            not isinstance(raw_offsets, list)
            or len(raw_offsets) != 2
            or any(type(value) is not int for value in raw_offsets)
        ):
            raise SerializationError(f"invalid tensor offsets for {name!r}")
        dtype = _CODE_TO_DTYPE[dtype_code]
        shape = tuple(raw_shape)
        start, end = raw_offsets
        if any(value < 0 for value in shape) or start < 0 or end < start or end > len(data):
            raise SerializationError(f"invalid tensor bounds for {name!r}")
        if any(
            not (end <= prior_start or start >= prior_end) for prior_start, prior_end in occupied
        ):
            raise SerializationError(f"overlapping tensor data for {name!r}")
        element_size = torch.empty((), dtype=dtype).element_size()
        expected = element_size
        for dimension in shape:
            expected *= dimension
        if end - start != expected:
            raise SerializationError(f"tensor byte size does not match shape for {name!r}")
        occupied.append((start, end))
        if expected == 0:
            tensor = torch.empty(shape, dtype=dtype)
        else:
            raw = bytearray(data[start:end])
            try:
                tensor = torch.frombuffer(raw, dtype=dtype).clone().reshape(shape)
            except (RuntimeError, TypeError, ValueError) as exc:
                raise SerializationError(f"cannot materialize tensor {name!r}") from exc
        result[name] = tensor
    cursor = 0
    for start, end in sorted(occupied):
        if start != cursor:
            raise SerializationError("safetensors data offsets must cover a contiguous buffer")
        cursor = end
    if cursor != len(data):
        raise SerializationError("safetensors payload contains unreferenced data")
    return result


def _parse_size(value: int | str) -> int:
    if type(value) is int:
        if value <= 0:
            raise ValueError("max_shard_size must be positive")
        return value
    if not isinstance(value, str):
        raise TypeError(f"invalid size type {type(value).__name__}")
    match = re.fullmatch(r"\s*(\d+)\s*(B|KiB|MiB|GiB)\s*", value)
    if not match:
        raise ValueError(f"invalid size {value!r}")
    factors = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}
    return int(match.group(1)) * factors[match.group(2)]


def _partition_state(state: Mapping[str, Tensor], max_bytes: int) -> list[OrderedDict[str, Tensor]]:
    shards: list[OrderedDict[str, Tensor]] = []
    current: OrderedDict[str, Tensor] = OrderedDict()
    current_bytes = 0
    for name in sorted(state):
        tensor = state[name]
        size = tensor.numel() * tensor.element_size()
        if current and current_bytes + size > max_bytes:
            shards.append(current)
            current = OrderedDict()
            current_bytes = 0
        current[name] = tensor
        current_bytes += size
    if current or not shards:
        shards.append(current)
    return shards


def save_state_directory(
    directory: os.PathLike[str] | str,
    state: Mapping[str, Tensor],
    *,
    config_payload: bytes,
    max_shard_size: int | str = "4GiB",
) -> dict[str, Any]:
    target = Path(directory)
    if target.exists():
        if target.is_symlink() or not target.is_dir():
            raise SerializationError(f"model path is not a regular local directory: {target}")
    else:
        target.mkdir(parents=True)
    _atomic_write(target / "config.json", config_payload)
    shards = _partition_state(state, _parse_size(max_shard_size))
    filenames: list[str] = []
    weight_map: dict[str, str] = {}
    if len(shards) == 1:
        filenames = ["model.safetensors"]
    else:
        filenames = [
            f"model-{index:05d}-of-{len(shards):05d}.safetensors"
            for index in range(1, len(shards) + 1)
        ]
    for index, shard in enumerate(shards):
        filename = filenames[index]
        _atomic_write(target / filename, encode_safetensors(shard))
        for name in shard:
            weight_map[name] = filename
    generated = ["config.json", *filenames]
    if len(shards) > 1:
        index_document = {
            "metadata": {
                "total_size": sum(value.numel() * value.element_size() for value in state.values()),
                "schema_version": 1,
            },
            "weight_map": weight_map,
        }
        _atomic_write(target / _INDEX_FILE, _canonical_json(index_document))
        generated.append(_INDEX_FILE)
    generated_set = set(generated)
    for path in target.iterdir():
        if (path.name == _INDEX_FILE or _TENSOR_FILE.fullmatch(path.name)) and (
            path.name not in generated_set
        ):
            if path.is_symlink() or not path.is_file():
                raise SerializationError(f"stale model artifact is not a regular file: {path.name}")
            path.unlink()
    _fsync_directory(target)
    integrity = {
        "schema_version": 1,
        "algorithm": "sha256",
        "files": {
            name: {"digest": sha256_file(target / name), "size": (target / name).stat().st_size}
            for name in sorted(generated)
        },
    }
    _atomic_write(target / _INTEGRITY_FILE, _canonical_json(integrity))
    return integrity


def verify_integrity(directory: os.PathLike[str] | str) -> dict[str, Any]:
    source = Path(directory)
    if source.is_symlink() or not source.is_dir():
        raise SerializationError(f"model path is not a regular local directory: {source}")
    manifest = read_json_object(source / _INTEGRITY_FILE, description="model integrity manifest")
    if set(manifest) != {"schema_version", "algorithm", "files"}:
        raise SerializationError("integrity manifest fields are invalid")
    if manifest.get("schema_version") != 1 or manifest.get("algorithm") != "sha256":
        raise SerializationError("unsupported integrity manifest")
    files = manifest.get("files")
    if not isinstance(files, dict) or "config.json" not in files:
        raise SerializationError("integrity manifest has no configuration")
    for name, expected in files.items():
        if type(name) is not str or not isinstance(expected, dict):
            raise SerializationError("integrity manifest file entries are invalid")
        if name != "config.json" and name != _INDEX_FILE and not _TENSOR_FILE.fullmatch(name):
            raise SerializationError(f"integrity manifest contains unsupported file {name!r}")
        if set(expected) != {"digest", "size"}:
            raise SerializationError(f"integrity metadata fields are invalid for {name!r}")
        size = expected["size"]
        digest = expected["digest"]
        if (
            type(size) is not int
            or size < 1
            or type(digest) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
        ):
            raise SerializationError(f"integrity metadata is invalid for {name!r}")
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise SerializationError(f"integrity-protected file is missing or unsafe: {name}")
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise SerializationError(f"integrity verification failed for {name}")

    protected = set(files)
    for path in source.rglob("*"):
        if path.is_symlink():
            raise SerializationError("model directory cannot contain symbolic links")
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        suffix = path.suffix.lower()
        if suffix in _FORBIDDEN_MODEL_SUFFIXES:
            raise SerializationError(f"pickle-capable model file is forbidden: {relative}")
        artifact_like = (
            path.name == "config.json"
            or path.name == _INTEGRITY_FILE
            or path.name.endswith(".safetensors")
            or path.name.endswith(".safetensors.index.json")
        )
        if artifact_like and (
            len(relative.parts) != 1 or path.name not in protected | {_INTEGRITY_FILE}
        ):
            raise SerializationError(f"unmanifested model artifact is forbidden: {relative}")
    return manifest


def _validate_sharded_filenames(filenames: list[str]) -> None:
    parsed = []
    for filename in filenames:
        match = _SHARDED_TENSOR_FILE.fullmatch(filename)
        if match is None:
            raise SerializationError("safetensors index must reference numbered shards")
        parsed.append((int(match.group(1)), int(match.group(2))))
    shard_count = len(filenames)
    if sorted(index for index, _ in parsed) != list(range(1, shard_count + 1)) or any(
        total != shard_count for _, total in parsed
    ):
        raise SerializationError("safetensors shard filenames are not a complete sequence")


def load_state_directory(directory: os.PathLike[str] | str) -> OrderedDict[str, Tensor]:
    source = Path(directory)
    manifest = verify_integrity(source)
    protected = set(manifest["files"])
    index_path = source / _INDEX_FILE
    expected_total_size: int | None = None
    if index_path.exists():
        if index_path.name not in protected:
            raise SerializationError("unprotected safetensors index")
        index = read_json_object(index_path, description="safetensors index")
        if set(index) != {"metadata", "weight_map"}:
            raise SerializationError("safetensors index fields are invalid")
        metadata = index["metadata"]
        weight_map = index["weight_map"]
        if (
            not isinstance(metadata, dict)
            or set(metadata) != {"schema_version", "total_size"}
            or metadata.get("schema_version") != 1
            or type(metadata.get("total_size")) is not int
            or metadata["total_size"] < 0
        ):
            raise SerializationError("safetensors index metadata is invalid")
        if not isinstance(weight_map, dict) or not weight_map:
            raise SerializationError("safetensors weight map must be a non-empty object")
        if any(
            type(name) is not str or not name or type(filename) is not str
            for name, filename in weight_map.items()
        ):
            raise SerializationError("safetensors weight map entries are invalid")
        filenames = sorted(set(weight_map.values()))
        if not filenames or any(
            name not in protected or not _TENSOR_FILE.fullmatch(name) for name in filenames
        ):
            raise SerializationError("index references an unprotected or invalid shard")
        _validate_sharded_filenames(filenames)
        if protected != {"config.json", _INDEX_FILE, *filenames}:
            raise SerializationError("integrity manifest and safetensors index file sets differ")
        expected_total_size = metadata["total_size"]
    else:
        filenames = ["model.safetensors"]
        if protected != {"config.json", filenames[0]}:
            raise SerializationError("integrity manifest does not describe one model.safetensors")
        weight_map = None
    result: OrderedDict[str, Tensor] = OrderedDict()
    for filename in filenames:
        try:
            payload = (source / filename).read_bytes()
        except OSError as exc:
            raise SerializationError(f"cannot read safetensors shard {filename!r}") from exc
        shard = decode_safetensors(payload)
        overlap = set(result).intersection(shard)
        if overlap:
            raise SerializationError(f"duplicate tensor keys across shards: {sorted(overlap)}")
        if weight_map is not None and any(weight_map.get(name) != filename for name in shard):
            raise SerializationError("safetensors weight map does not match shard contents")
        result.update(shard)
    if weight_map is not None:
        if set(result) != set(weight_map):
            raise SerializationError("safetensors index does not match shard contents")
        actual_total_size = sum(value.numel() * value.element_size() for value in result.values())
        if actual_total_size != expected_total_size:
            raise SerializationError("safetensors index total_size does not match tensor data")
    return result


__all__ = [
    "SerializationError",
    "decode_safetensors",
    "encode_safetensors",
    "load_state_directory",
    "read_json_object",
    "save_state_directory",
    "sha256_file",
    "verify_integrity",
]
