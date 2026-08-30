//! Cross-module artifact integrity and capability contracts.

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use ed25519_dalek::{Signer as _, SigningKey};
use mindclade_artifact_proxy::{
    Capability, CapabilityOperation, CapabilityVerifier, Digest, FilesystemStore, Scope,
};
use std::collections::HashMap;
use std::fs;

fn signed_capability(
    scope: Scope,
    digest: Digest,
    operation: CapabilityOperation,
) -> (CapabilityVerifier, Capability) {
    let signing_key = SigningKey::from_bytes(&[7_u8; 32]);
    let mut capability = Capability {
        scope,
        digest,
        operation,
        expires_unix: 2_000,
        max_size_bytes: 1_024,
        session_id: "session-test-0001".to_owned(),
        nonce: "nonce-test-0000001".to_owned(),
        key_id: "test-key".to_owned(),
        signature: String::new(),
    };
    capability.signature =
        URL_SAFE_NO_PAD.encode(signing_key.sign(&capability.signing_bytes()).to_bytes());
    let verifier = CapabilityVerifier::with_clock(
        HashMap::from([("test-key".to_owned(), signing_key.verifying_key())]),
        || 1_000,
    );
    (verifier, capability)
}

#[test]
fn upload_capability_binds_size_session_and_single_use_nonce() {
    let temporary = tempfile::tempdir().unwrap();
    let store = FilesystemStore::with_limits(temporary.path(), 16, 1).unwrap();
    let scope = Scope {
        tenant_id: "tenant-a".to_owned(),
        project_id: "project-a".to_owned(),
    };
    let bytes = b"artifact";
    let digest = Digest::of(bytes);
    let (verifier, mut capability) =
        signed_capability(scope.clone(), digest.clone(), CapabilityOperation::Upload);
    capability.max_size_bytes = bytes.len() as u64;
    capability.signature = URL_SAFE_NO_PAD.encode(
        SigningKey::from_bytes(&[7_u8; 32])
            .sign(&capability.signing_bytes())
            .to_bytes(),
    );
    verifier
        .verify_upload(
            &capability,
            &scope,
            &digest,
            bytes.len() as u64,
            &capability.session_id,
        )
        .unwrap();
    let session = store
        .begin_authorized(
            scope.clone(),
            digest.clone(),
            bytes.len() as u64,
            &capability.session_id,
            &capability.nonce,
            u64::MAX,
        )
        .unwrap();
    assert!(
        store
            .begin_authorized(
                scope,
                digest,
                bytes.len() as u64,
                "session-test-0002",
                &capability.nonce,
                u64::MAX,
            )
            .is_err()
    );
    store.append(&session.upload_id, 0, bytes).unwrap();
    store.commit(&session.upload_id).unwrap();
}

#[test]
fn store_rejects_zero_oversized_and_excess_concurrent_uploads() {
    let temporary = tempfile::tempdir().unwrap();
    let store = FilesystemStore::with_limits(temporary.path(), 8, 1).unwrap();
    let scope = Scope {
        tenant_id: "tenant-a".to_owned(),
        project_id: "project-a".to_owned(),
    };
    assert!(store.begin(scope.clone(), Digest::of(b""), 0).is_err());
    assert!(
        store
            .begin(scope.clone(), Digest::of(b"too-large"), 9)
            .is_err()
    );
    store.begin(scope.clone(), Digest::of(b"one"), 3).unwrap();
    assert!(store.begin(scope, Digest::of(b"two"), 3).is_err());
}

#[test]
fn resumable_upload_is_digest_verified() {
    let temporary = tempfile::tempdir().unwrap();
    let store = FilesystemStore::new(temporary.path()).unwrap();
    let bytes = b"immutable model bytes";
    let digest = Digest::of(bytes);
    let scope = Scope {
        tenant_id: "tenant-a".to_owned(),
        project_id: "project-a".to_owned(),
    };
    let session = store
        .begin(scope, digest.clone(), bytes.len() as u64)
        .unwrap();
    let first = &bytes[..8];
    assert_eq!(
        store
            .append(&session.upload_id, 0, first)
            .unwrap()
            .committed_bytes,
        8
    );
    assert!(store.append(&session.upload_id, 0, b"replay").is_err());
    store.append(&session.upload_id, 8, &bytes[8..]).unwrap();
    assert_eq!(store.commit(&session.upload_id).unwrap(), digest);
    assert_eq!(store.get(&digest).unwrap(), bytes);
    assert_eq!(store.verified_size(&digest).unwrap(), bytes.len() as u64);

    let hex = digest.hex();
    fs::write(
        temporary.path().join("objects").join(&hex[..2]).join(hex),
        b"same-length-corruption!",
    )
    .unwrap();
    assert!(store.verified_size(&digest).is_err());
}

#[test]
fn corrupt_upload_never_becomes_visible() {
    let temporary = tempfile::tempdir().unwrap();
    let store = FilesystemStore::new(temporary.path()).unwrap();
    let expected = Digest::of(b"expected");
    let scope = Scope {
        tenant_id: "tenant-a".to_owned(),
        project_id: "project-a".to_owned(),
    };
    let session = store.begin(scope, expected.clone(), 8).unwrap();
    store.append(&session.upload_id, 0, b"tampered").unwrap();
    assert!(store.commit(&session.upload_id).is_err());
    assert!(store.get(&expected).is_err());
}

#[test]
fn capability_is_exactly_tenant_project_digest_and_operation_scoped() {
    let scope = Scope {
        tenant_id: "tenant-a".to_owned(),
        project_id: "project-a".to_owned(),
    };
    let digest = Digest::of(b"artifact");
    let (verifier, capability) =
        signed_capability(scope.clone(), digest.clone(), CapabilityOperation::Download);
    verifier
        .verify(&capability, &scope, &digest, CapabilityOperation::Download)
        .unwrap();
    let other = Scope {
        tenant_id: "tenant-b".to_owned(),
        project_id: "project-a".to_owned(),
    };
    assert!(
        verifier
            .verify(&capability, &other, &digest, CapabilityOperation::Download)
            .is_err()
    );
    assert!(
        verifier
            .verify(&capability, &scope, &digest, CapabilityOperation::Upload)
            .is_err()
    );
    let mut invalid_scope_capability = capability.clone();
    invalid_scope_capability.scope.tenant_id = "tenant:escape".to_owned();
    assert!(
        verifier
            .verify(
                &invalid_scope_capability,
                &invalid_scope_capability.scope,
                &digest,
                CapabilityOperation::Download,
            )
            .is_err()
    );
}
