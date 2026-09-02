//! Artifact proxy process composition root.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use ed25519_dalek::VerifyingKey;
use mindclade_artifact_proxy::{CapabilityVerifier, FilesystemStore, http};
use std::{collections::HashMap, env, fs, sync::Arc, time::Duration};

fn required(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| panic!("{name} is required"))
}

fn required_seconds(name: &str, minimum: u64, maximum: u64) -> Duration {
    let value = required(name);
    let seconds = value
        .parse::<u64>()
        .unwrap_or_else(|_| panic!("{name} must be an integer number of seconds"));
    assert!(
        (minimum..=maximum).contains(&seconds),
        "{name} must be within {minimum}..{maximum} seconds"
    );
    Duration::from_secs(seconds)
}

#[tokio::main]
async fn main() {
    let root = required("MINDCLADE_ARTIFACT_ROOT");
    let key_file = required("MINDCLADE_ARTIFACT_PUBLIC_KEYS_FILE");
    let key_document: HashMap<String, String> =
        serde_json::from_slice(&fs::read(key_file).expect("public key file must be readable"))
            .expect("public key file must be a key-id to base64 map");
    let keys = key_document
        .into_iter()
        .map(|(id, encoded)| {
            let bytes = STANDARD.decode(encoded).expect("public key must be base64");
            let bytes: [u8; 32] = bytes
                .try_into()
                .expect("Ed25519 public key must contain 32 bytes");
            (
                id,
                VerifyingKey::from_bytes(&bytes).expect("Ed25519 public key must be valid"),
            )
        })
        .collect();
    let store = Arc::new(FilesystemStore::new(root).expect("artifact root must be writable"));
    let verifier = Arc::new(CapabilityVerifier::new(keys));
    let address = env::var("MINDCLADE_ARTIFACT_PROXY_ADDRESS")
        .unwrap_or_else(|_| "127.0.0.1:8082".to_owned());
    let body_read_timeout =
        required_seconds("MINDCLADE_ARTIFACT_BODY_READ_TIMEOUT_SECONDS", 1, 300);
    http::serve(&address, &store, &verifier, body_read_timeout)
        .await
        .expect("artifact proxy failed");
}
