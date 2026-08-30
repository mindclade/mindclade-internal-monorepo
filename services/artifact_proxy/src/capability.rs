//! Narrow, signed capabilities used instead of ambient artifact permissions.

use crate::digest::Digest;
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use ed25519_dalek::{Signature, Verifier as _, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    time::{SystemTime, UNIX_EPOCH},
};
use thiserror::Error;

/// Tenant and project isolation boundary.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Scope {
    /// Stable tenant identifier.
    pub tenant_id: String,
    /// Stable project identifier within the tenant.
    pub project_id: String,
}

/// Single permitted operation encoded in a capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CapabilityOperation {
    /// Retrieve verified bytes.
    Download,
    /// Create, append, or commit an upload.
    Upload,
}

/// Ed25519-signed authority for one scope, digest, operation, and deadline.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Capability {
    /// Authorized tenant/project.
    pub scope: Scope,
    /// Exact immutable artifact digest.
    pub digest: Digest,
    /// Authorized operation.
    pub operation: CapabilityOperation,
    /// Unix epoch seconds after which the capability is invalid.
    pub expires_unix: u64,
    /// Maximum object size authorized by this capability.
    pub max_size_bytes: u64,
    /// Client-generated transfer identifier bound into upload sessions.
    pub session_id: String,
    /// Single-use random value preventing upload capability replay.
    pub nonce: String,
    /// Rotation-friendly public key identifier.
    pub key_id: String,
    /// URL-safe base64 Ed25519 signature.
    pub signature: String,
}

impl Capability {
    /// Canonical, unambiguous bytes signed by the issuer.
    #[must_use]
    pub fn signing_bytes(&self) -> Vec<u8> {
        fn push_field(target: &mut Vec<u8>, value: &[u8]) {
            target.extend_from_slice(&(value.len() as u64).to_be_bytes());
            target.extend_from_slice(value);
        }
        let mut bytes = b"mindclade-artifact-capability-v2".to_vec();
        bytes.push(match self.operation {
            CapabilityOperation::Download => 0,
            CapabilityOperation::Upload => 1,
        });
        for value in [
            self.scope.tenant_id.as_bytes(),
            self.scope.project_id.as_bytes(),
            self.digest.to_string().as_bytes(),
            self.session_id.as_bytes(),
            self.nonce.as_bytes(),
            self.key_id.as_bytes(),
        ] {
            push_field(&mut bytes, value);
        }
        bytes.extend_from_slice(&self.expires_unix.to_be_bytes());
        bytes.extend_from_slice(&self.max_size_bytes.to_be_bytes());
        bytes
    }
}

/// Capability validation failure.
#[derive(Debug, Error)]
pub enum CapabilityError {
    /// The key identifier is not active.
    #[error("capability signing key is unknown")]
    UnknownKey,
    /// The capability has expired.
    #[error("capability has expired")]
    Expired,
    /// Scope, digest, or operation does not match the request.
    #[error("capability does not authorize this request")]
    ScopeMismatch,
    /// Signature text or cryptographic verification failed.
    #[error("capability signature is invalid")]
    InvalidSignature,
    /// Capability fields do not use the shared safe identifier grammar.
    #[error("capability identifiers or size are invalid")]
    InvalidFields,
}

/// Verifies capabilities against a rotation set of public keys.
pub struct CapabilityVerifier {
    keys: HashMap<String, VerifyingKey>,
    now: Box<dyn Fn() -> u64 + Send + Sync>,
}

impl CapabilityVerifier {
    /// Build a verifier using the system clock.
    #[must_use]
    pub fn new(keys: HashMap<String, VerifyingKey>) -> Self {
        Self {
            keys,
            now: Box::new(|| {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            }),
        }
    }

    /// Build a verifier with a deterministic clock for contract tests.
    #[must_use]
    pub fn with_clock(
        keys: HashMap<String, VerifyingKey>,
        now: impl Fn() -> u64 + Send + Sync + 'static,
    ) -> Self {
        Self {
            keys,
            now: Box::new(now),
        }
    }

    /// Verify cryptographic integrity and exact request authority.
    ///
    /// # Errors
    ///
    /// Returns [`CapabilityError`] when fields, scope, expiry, key, or signature are invalid.
    pub fn verify(
        &self,
        capability: &Capability,
        scope: &Scope,
        digest: &Digest,
        operation: CapabilityOperation,
    ) -> Result<(), CapabilityError> {
        if capability.expires_unix <= (self.now)() {
            return Err(CapabilityError::Expired);
        }
        if capability.scope != *scope
            || capability.digest != *digest
            || capability.operation != operation
        {
            return Err(CapabilityError::ScopeMismatch);
        }
        if !safe_scope_identifier(&capability.scope.tenant_id)
            || !safe_scope_identifier(&capability.scope.project_id)
            || !safe_capability_identifier(&capability.key_id, 1, 128)
            || !safe_capability_identifier(&capability.session_id, 16, 128)
            || !safe_capability_identifier(&capability.nonce, 16, 128)
            || capability.max_size_bytes == 0
        {
            return Err(CapabilityError::InvalidFields);
        }
        let key = self
            .keys
            .get(&capability.key_id)
            .ok_or(CapabilityError::UnknownKey)?;
        let signature_bytes = URL_SAFE_NO_PAD
            .decode(&capability.signature)
            .map_err(|_| CapabilityError::InvalidSignature)?;
        let signature = Signature::from_slice(&signature_bytes)
            .map_err(|_| CapabilityError::InvalidSignature)?;
        key.verify(&capability.signing_bytes(), &signature)
            .map_err(|_| CapabilityError::InvalidSignature)
    }

    /// Verify an upload and bind its declared size and session identifier.
    ///
    /// # Errors
    ///
    /// Returns [`CapabilityError`] when verification fails or transfer fields differ.
    pub fn verify_upload(
        &self,
        capability: &Capability,
        scope: &Scope,
        digest: &Digest,
        size_bytes: u64,
        session_id: &str,
    ) -> Result<(), CapabilityError> {
        self.verify(capability, scope, digest, CapabilityOperation::Upload)?;
        if capability.max_size_bytes != size_bytes || capability.session_id != session_id {
            return Err(CapabilityError::ScopeMismatch);
        }
        Ok(())
    }
}

fn safe_scope_identifier(value: &str) -> bool {
    let length = value.len();
    (1..=64).contains(&length)
        && value
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
        && value
            .bytes()
            .all(|character| character.is_ascii_alphanumeric() || b"._-".contains(&character))
}

fn safe_capability_identifier(value: &str, minimum: usize, maximum: usize) -> bool {
    let length = value.len();
    (minimum..=maximum).contains(&length)
        && value
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
        && value
            .bytes()
            .all(|character| character.is_ascii_alphanumeric() || b"._:-".contains(&character))
}
