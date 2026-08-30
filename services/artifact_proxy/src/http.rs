//! Minimal synchronous HTTP adapter for the artifact capability and store contracts.

use crate::{Capability, CapabilityOperation, CapabilityVerifier, Digest, FilesystemStore, Scope};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use serde::{Deserialize, Serialize};
use std::{io::Read, str::FromStr, sync::Arc, thread};
use tiny_http::{Header, Method, Request, Response, ResponseBox, Server, StatusCode};

const HTTP_WORKERS: usize = 16;

/// Serve the artifact API until the process is terminated.
///
/// # Errors
///
/// Returns an error if the listener cannot start or an HTTP worker panics.
pub fn serve(
    address: &str,
    store: &Arc<FilesystemStore>,
    verifier: &Arc<CapabilityVerifier>,
) -> Result<(), String> {
    let server = Arc::new(Server::http(address).map_err(|error| error.to_string())?);
    let mut workers = Vec::with_capacity(HTTP_WORKERS);
    for _ in 0..HTTP_WORKERS {
        let server = Arc::clone(&server);
        let store = Arc::clone(store);
        let verifier = Arc::clone(verifier);
        workers.push(thread::spawn(move || {
            while let Ok(request) = server.recv() {
                handle(request, &store, &verifier);
            }
        }));
    }
    for worker in workers {
        worker
            .join()
            .map_err(|_| "artifact HTTP worker panicked".to_owned())?;
    }
    Ok(())
}

#[derive(Deserialize)]
struct BeginRequest {
    tenant_id: String,
    project_id: String,
    digest: Digest,
    size_bytes: u64,
    session_id: String,
}

#[derive(Serialize)]
struct SessionResponse<'a> {
    upload_id: &'a str,
    committed_bytes: u64,
}

fn handle(mut request: Request, store: &FilesystemStore, verifier: &CapabilityVerifier) {
    if request.method() == &Method::Get && request.url() == "/healthz" {
        let _ = request.respond(Response::empty(StatusCode(204)));
        return;
    }
    let path = request
        .url()
        .split('?')
        .next()
        .unwrap_or_default()
        .to_owned();
    let parts: Vec<&str> = path.trim_matches('/').split('/').collect();
    let result = match (request.method(), parts.as_slice()) {
        (&Method::Post, ["v1alpha1", "uploads"]) => begin(&mut request, store, verifier),
        (&Method::Put, ["v1alpha1", "uploads", upload_id]) => {
            append(&mut request, store, verifier, upload_id)
        }
        (&Method::Post, ["v1alpha1", "uploads", upload_id, "commit"]) => {
            commit(&request, store, verifier, upload_id)
        }
        (&Method::Get, ["v1alpha1", "artifacts", digest]) => {
            download(&request, store, verifier, digest)
        }
        (&Method::Head, ["v1alpha1", "artifacts", digest]) => {
            metadata(&request, store, verifier, digest)
        }
        _ => Err((404, "route not found".to_owned())),
    };
    match result {
        Ok(response) => {
            let _ = request.respond(response);
        }
        Err((status, message)) => {
            let body = serde_json::to_string(&serde_json::json!({"error": message}))
                .unwrap_or_else(|_| "{}".to_owned());
            let _ = request.respond(json_response(body, status));
        }
    }
}

fn begin(
    request: &mut Request,
    store: &FilesystemStore,
    verifier: &CapabilityVerifier,
) -> Result<ResponseBox, (u16, String)> {
    let body = read_limited(request, 64 * 1024)?;
    let body: BeginRequest =
        serde_json::from_slice(&body).map_err(|_| (400, "invalid begin request".to_owned()))?;
    let scope = Scope {
        tenant_id: body.tenant_id,
        project_id: body.project_id,
    };
    let capability = capability_header(request)?;
    verifier
        .verify_upload(
            &capability,
            &scope,
            &body.digest,
            body.size_bytes,
            &body.session_id,
        )
        .map_err(|error| (403, error.to_string()))?;
    let session = store
        .begin_authorized(
            scope,
            body.digest,
            body.size_bytes,
            &body.session_id,
            &capability.nonce,
            capability.expires_unix,
        )
        .map_err(|error| store_error(&error))?;
    let json = serde_json::to_string(&SessionResponse {
        upload_id: &session.upload_id,
        committed_bytes: 0,
    })
    .map_err(|_| (500, "serialization failed".to_owned()))?;
    Ok(json_response(json, 201))
}

fn append(
    request: &mut Request,
    store: &FilesystemStore,
    verifier: &CapabilityVerifier,
    upload_id: &str,
) -> Result<ResponseBox, (u16, String)> {
    let session = store
        .session(upload_id)
        .map_err(|error| store_error(&error))?;
    let capability = capability_header(request)?;
    verifier
        .verify_upload(
            &capability,
            &session.scope,
            &session.digest,
            session.size_bytes,
            &session.upload_id,
        )
        .map_err(|error| (403, error.to_string()))?;
    if capability.nonce != session.authorization_nonce {
        return Err((
            403,
            "capability does not authorize this upload session".to_owned(),
        ));
    }
    let offset = request
        .url()
        .split_once("offset=")
        .and_then(|(_, value)| value.parse::<u64>().ok())
        .ok_or((400, "offset is required".to_owned()))?;
    let bytes = read_limited(request, 8 * 1024 * 1024)?;
    let updated = store
        .append(upload_id, offset, &bytes)
        .map_err(|error| store_error(&error))?;
    let json = serde_json::to_string(&SessionResponse {
        upload_id: &updated.upload_id,
        committed_bytes: updated.committed_bytes,
    })
    .map_err(|_| (500, "serialization failed".to_owned()))?;
    Ok(json_response(json, 200))
}

fn commit(
    request: &Request,
    store: &FilesystemStore,
    verifier: &CapabilityVerifier,
    upload_id: &str,
) -> Result<ResponseBox, (u16, String)> {
    let session = store
        .session(upload_id)
        .map_err(|error| store_error(&error))?;
    let capability = capability_header(request)?;
    verifier
        .verify_upload(
            &capability,
            &session.scope,
            &session.digest,
            session.size_bytes,
            &session.upload_id,
        )
        .map_err(|error| (403, error.to_string()))?;
    if capability.nonce != session.authorization_nonce {
        return Err((
            403,
            "capability does not authorize this upload session".to_owned(),
        ));
    }
    let digest = store
        .commit(upload_id)
        .map_err(|error| store_error(&error))?;
    Ok(json_response(
        serde_json::json!({"digest": digest.to_string()}).to_string(),
        200,
    ))
}

fn download(
    request: &Request,
    store: &FilesystemStore,
    verifier: &CapabilityVerifier,
    digest: &str,
) -> Result<ResponseBox, (u16, String)> {
    let decoded = decode_path_segment(digest)?;
    let digest = Digest::from_str(&decoded).map_err(|error| (400, error))?;
    let capability = capability_header(request)?;
    verifier
        .verify(
            &capability,
            &capability.scope,
            &digest,
            CapabilityOperation::Download,
        )
        .map_err(|error| (403, error.to_string()))?;
    let (file, size) = store
        .open_verified(&digest)
        .map_err(|error| store_error(&error))?;
    if size > capability.max_size_bytes {
        return Err((403, "artifact exceeds capability size bound".to_owned()));
    }
    let length = usize::try_from(size).map_err(|_| (413, "artifact is too large".to_owned()))?;
    let mut response = Response::new(StatusCode(200), Vec::new(), file, Some(length), None).boxed();
    response.add_header(
        Header::from_bytes("Content-Type", "application/octet-stream")
            .map_err(|()| (500, "header failed".to_owned()))?,
    );
    response.add_header(
        Header::from_bytes("ETag", digest.to_string())
            .map_err(|()| (500, "header failed".to_owned()))?,
    );
    Ok(response)
}

fn metadata(
    request: &Request,
    store: &FilesystemStore,
    verifier: &CapabilityVerifier,
    digest: &str,
) -> Result<ResponseBox, (u16, String)> {
    let decoded = decode_path_segment(digest)?;
    let digest = Digest::from_str(&decoded).map_err(|error| (400, error))?;
    let capability = capability_header(request)?;
    verifier
        .verify(
            &capability,
            &capability.scope,
            &digest,
            CapabilityOperation::Download,
        )
        .map_err(|error| (403, error.to_string()))?;
    let size = store
        .verified_size(&digest)
        .map_err(|error| store_error(&error))?;
    if size != capability.max_size_bytes {
        return Err((
            403,
            "artifact size differs from capability bound".to_owned(),
        ));
    }
    let mut response = Response::empty(StatusCode(200)).boxed();
    response.add_header(
        Header::from_bytes("X-Mindclade-Artifact-Digest", digest.to_string())
            .map_err(|()| (500, "header failed".to_owned()))?,
    );
    response.add_header(
        Header::from_bytes("X-Mindclade-Artifact-Size", size.to_string())
            .map_err(|()| (500, "header failed".to_owned()))?,
    );
    Ok(response)
}

fn decode_path_segment(value: &str) -> Result<String, (u16, String)> {
    fn hexadecimal(value: u8) -> Option<u8> {
        match value {
            b'0'..=b'9' => Some(value - b'0'),
            b'a'..=b'f' => Some(value - b'a' + 10),
            b'A'..=b'F' => Some(value - b'A' + 10),
            _ => None,
        }
    }

    let source = value.as_bytes();
    let mut decoded = Vec::with_capacity(source.len());
    let mut index = 0;
    while index < source.len() {
        if source[index] != b'%' {
            decoded.push(source[index]);
            index += 1;
            continue;
        }
        if index + 2 >= source.len() {
            return Err((400, "artifact path encoding is invalid".to_owned()));
        }
        let high = hexadecimal(source[index + 1])
            .ok_or((400, "artifact path encoding is invalid".to_owned()))?;
        let low = hexadecimal(source[index + 2])
            .ok_or((400, "artifact path encoding is invalid".to_owned()))?;
        let byte = (high << 4) | low;
        if matches!(byte, b'/' | b'\\' | b'%') {
            return Err((400, "encoded path separators are forbidden".to_owned()));
        }
        decoded.push(byte);
        index += 3;
    }
    String::from_utf8(decoded).map_err(|_| (400, "artifact path is not UTF-8".to_owned()))
}

fn capability_header(request: &Request) -> Result<Capability, (u16, String)> {
    let encoded = request
        .headers()
        .iter()
        .find(|header| header.field.equiv("X-Mindclade-Capability"))
        .map(|header| header.value.as_str())
        .ok_or((401, "capability is required".to_owned()))?;
    let bytes = URL_SAFE_NO_PAD
        .decode(encoded)
        .map_err(|_| (401, "capability encoding is invalid".to_owned()))?;
    serde_json::from_slice(&bytes).map_err(|_| (401, "capability payload is invalid".to_owned()))
}

fn read_limited(request: &mut Request, maximum: u64) -> Result<Vec<u8>, (u16, String)> {
    let mut bytes = Vec::new();
    request
        .as_reader()
        .take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| (400, "request body could not be read".to_owned()))?;
    if bytes.len() as u64 > maximum {
        return Err((413, "request body is too large".to_owned()));
    }
    Ok(bytes)
}

fn json_response(body: String, status: u16) -> ResponseBox {
    Response::from_string(body)
        .with_status_code(status)
        .with_header(
            Header::from_bytes("Content-Type", "application/json").expect("static header is valid"),
        )
        .boxed()
}

fn store_error(error: &crate::ArtifactError) -> (u16, String) {
    let status = match error {
        crate::ArtifactError::UploadNotFound => 404,
        crate::ArtifactError::Io(value) if value.kind() == std::io::ErrorKind::NotFound => 404,
        crate::ArtifactError::OffsetMismatch { .. }
        | crate::ArtifactError::SizeExceeded
        | crate::ArtifactError::Integrity
        | crate::ArtifactError::Replay => 409,
        crate::ArtifactError::Capacity => 429,
        _ => 500,
    };
    (status, error.to_string())
}

#[cfg(test)]
mod tests {
    use super::{decode_path_segment, serve};
    use crate::{
        Capability, CapabilityOperation, CapabilityVerifier, Digest, FilesystemStore, Scope,
    };
    use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
    use ed25519_dalek::{Signer as _, SigningKey};
    use std::{
        collections::HashMap,
        io::{Read as _, Write as _},
        net::{TcpListener, TcpStream},
        sync::Arc,
        thread,
        time::Duration,
    };

    #[test]
    fn artifact_digest_segment_accepts_literal_or_encoded_colon() {
        assert_eq!(decode_path_segment("sha256:abcd").unwrap(), "sha256:abcd");
        assert_eq!(decode_path_segment("sha256%3Aabcd").unwrap(), "sha256:abcd");
    }

    #[test]
    fn artifact_digest_segment_rejects_ambiguous_or_malformed_encoding() {
        for value in [
            "sha256%",
            "sha256%2",
            "sha256%GG",
            "sha256%25253A",
            "sha256%2Fabcd",
        ] {
            assert!(decode_path_segment(value).is_err(), "accepted {value}");
        }
    }

    #[test]
    fn authenticated_head_attests_exact_digest_and_size() {
        let temporary = tempfile::tempdir().unwrap();
        let store = Arc::new(FilesystemStore::new(temporary.path()).unwrap());
        let bytes = b"durably committed result";
        let digest = Digest::of(bytes);
        let scope = Scope {
            tenant_id: "tenant-a".to_owned(),
            project_id: "project-a".to_owned(),
        };
        let upload = store
            .begin(scope.clone(), digest.clone(), bytes.len() as u64)
            .unwrap();
        store.append(&upload.upload_id, 0, bytes).unwrap();
        store.commit(&upload.upload_id).unwrap();

        let signing_key = SigningKey::from_bytes(&[9_u8; 32]);
        let verifier = Arc::new(CapabilityVerifier::with_clock(
            HashMap::from([("head-test-key".to_owned(), signing_key.verifying_key())]),
            || 1_000,
        ));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let server_store = Arc::clone(&store);
        thread::spawn(move || {
            serve(&address.to_string(), &server_store, &verifier).unwrap();
        });

        let capability =
            signed_download_capability(&signing_key, scope, digest.clone(), bytes.len() as u64);
        let response = head(address, &digest, Some(&capability));
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        assert!(
            response.contains(&format!("X-Mindclade-Artifact-Digest: {digest}\r\n")),
            "{response}"
        );
        assert!(
            response.contains(&format!("X-Mindclade-Artifact-Size: {}\r\n", bytes.len())),
            "{response}"
        );

        let mut wrong_size: Capability =
            serde_json::from_slice(&URL_SAFE_NO_PAD.decode(&capability).unwrap()).unwrap();
        wrong_size.max_size_bytes += 1;
        wrong_size.signature =
            URL_SAFE_NO_PAD.encode(signing_key.sign(&wrong_size.signing_bytes()).to_bytes());
        let wrong_size = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&wrong_size).unwrap());
        assert!(head(address, &digest, Some(&wrong_size)).starts_with("HTTP/1.1 403"));
        assert!(head(address, &digest, None).starts_with("HTTP/1.1 401"));
    }

    fn signed_download_capability(
        signing_key: &SigningKey,
        scope: Scope,
        digest: Digest,
        size_bytes: u64,
    ) -> String {
        let mut capability = Capability {
            scope,
            digest,
            operation: CapabilityOperation::Download,
            expires_unix: 2_000,
            max_size_bytes: size_bytes,
            session_id: "download-head-test-session".to_owned(),
            nonce: "download-head-test-nonce".to_owned(),
            key_id: "head-test-key".to_owned(),
            signature: String::new(),
        };
        capability.signature =
            URL_SAFE_NO_PAD.encode(signing_key.sign(&capability.signing_bytes()).to_bytes());
        URL_SAFE_NO_PAD.encode(serde_json::to_vec(&capability).unwrap())
    }

    fn head(address: std::net::SocketAddr, digest: &Digest, capability: Option<&str>) -> String {
        let mut stream = (0..100)
            .find_map(|_| {
                if let Ok(stream) = TcpStream::connect(address) {
                    Some(stream)
                } else {
                    thread::sleep(Duration::from_millis(1));
                    None
                }
            })
            .expect("artifact HTTP server did not start");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let capability_header = capability
            .map(|value| format!("X-Mindclade-Capability: {value}\r\n"))
            .unwrap_or_default();
        write!(
            stream,
            "HEAD /v1alpha1/artifacts/{digest} HTTP/1.1\r\nHost: {address}\r\n{capability_header}Connection: close\r\n\r\n"
        )
        .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        response
    }
}
