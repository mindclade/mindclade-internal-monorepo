"""Ed25519 inner-manifest signing contract."""

from __future__ import annotations

import base64
import dataclasses
from typing import Protocol, runtime_checkable


@runtime_checkable
class BundleSigner(Protocol):
    @property
    def key_id(self) -> str: ...

    def sign(self, payload: bytes) -> bytes: ...


@runtime_checkable
class BundleVerifier(Protocol):
    @property
    def key_id(self) -> str: ...

    def verify(self, payload: bytes, signature: bytes) -> None: ...


@dataclasses.dataclass(frozen=True)
class SignatureEnvelope:
    key_id: str
    signature: bytes
    algorithm: str = "Ed25519"
    schema_version: str = "v1alpha1"

    def __post_init__(self) -> None:
        if (
            type(self.key_id) is not str
            or not self.key_id
            or len(self.key_id) > 128
            or any(not (character.isalnum() or character in "._:-") for character in self.key_id)
        ):
            raise ValueError("signature key_id is invalid")
        if type(self.signature) is not bytes or len(self.signature) != 64:
            raise ValueError("Ed25519 signature must contain 64 bytes")
        if self.algorithm != "Ed25519" or self.schema_version != "v1alpha1":
            raise ValueError("unsupported signature envelope")

    def to_dict(self) -> dict[str, str]:
        return {
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "schema_version": self.schema_version,
            "signature": base64.b64encode(self.signature).decode("ascii"),
        }

    @classmethod
    def from_dict(cls, value: dict[str, str]) -> SignatureEnvelope:
        if set(value) != {"algorithm", "key_id", "schema_version", "signature"}:
            raise ValueError("signature envelope fields are invalid")
        if any(type(item) is not str for item in value.values()):
            raise ValueError("signature envelope values must be strings")
        if value.get("algorithm") != "Ed25519" or value.get("schema_version") != "v1alpha1":
            raise ValueError("unsupported signature envelope")
        try:
            signature = base64.b64decode(value["signature"], validate=True)
        except Exception as exc:
            raise ValueError("signature is not valid base64") from exc
        return cls(key_id=value["key_id"], signature=signature)


class Ed25519PrivateKeySigner:
    """Optional in-memory signer for tests; production supplies a KMS signer."""

    def __init__(self, private_key: object, key_id: str) -> None:
        if (
            not key_id
            or len(key_id) > 128
            or any(not (character.isalnum() or character in "._:-") for character in key_id)
        ):
            raise ValueError("key_id is required")
        self._private_key = private_key
        self._key_id = key_id

    @property
    def key_id(self) -> str:
        return self._key_id

    def sign(self, payload: bytes) -> bytes:
        return self._private_key.sign(payload)  # type: ignore[attr-defined,no-any-return]


class Ed25519PublicKeyVerifier:
    def __init__(self, public_key: object, key_id: str) -> None:
        if (
            not key_id
            or len(key_id) > 128
            or any(not (character.isalnum() or character in "._:-") for character in key_id)
        ):
            raise ValueError("key_id is required")
        self._public_key = public_key
        self._key_id = key_id

    @property
    def key_id(self) -> str:
        return self._key_id

    def verify(self, payload: bytes, signature: bytes) -> None:
        self._public_key.verify(signature, payload)  # type: ignore[attr-defined]


__all__ = [
    "BundleSigner",
    "BundleVerifier",
    "Ed25519PrivateKeySigner",
    "Ed25519PublicKeyVerifier",
    "SignatureEnvelope",
]
