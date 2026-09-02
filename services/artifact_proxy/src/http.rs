//! Bounded asynchronous HTTP adapter for artifact capabilities and storage.

use crate::{
    ArtifactError, Capability, CapabilityOperation, CapabilityVerifier, Digest, FilesystemStore,
    Scope,
};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use bytes::Bytes;
use futures_util::TryStreamExt as _;
use http_body_util::{BodyExt as _, Full, StreamBody, combinators::BoxBody};
use hyper::{
    HeaderMap, Method, Request, Response, StatusCode,
    body::{Body, Frame, Incoming},
    header::{CONTENT_LENGTH, CONTENT_TYPE, ETAG, HeaderValue},
    server::conn::http1,
    service::service_fn,
};
use hyper_util::rt::{TokioIo, TokioTimer};
use serde::{Deserialize, Serialize};
use std::{
    convert::Infallible,
    error::Error,
    io,
    pin::Pin,
    str::FromStr,
    sync::Arc,
    task::{Context as TaskContext, Poll},
    time::Duration,
};
use tokio::{
    net::TcpListener,
    sync::{OwnedSemaphorePermit, Semaphore},
    task, time,
};
use tokio_util::io::ReaderStream;

const HEADER_READ_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_CONCURRENT_CONNECTIONS: usize = 32;
const MAX_CONCURRENT_REQUESTS: usize = 16;
const MAX_CONCURRENT_STORAGE_OPERATIONS: usize = 8;

type BoxError = Box<dyn Error + Send + Sync>;
type HttpBody = BoxBody<Bytes, BoxError>;
type HttpError = (StatusCode, String);

#[derive(Clone)]
struct WorkLimits {
    connections: Arc<Semaphore>,
    requests: Arc<Semaphore>,
    storage: Arc<Semaphore>,
}

struct PermitBody<B> {
    inner: B,
    _permit: OwnedSemaphorePermit,
}

impl<B> Body for PermitBody<B>
where
    B: Body + Unpin,
{
    type Data = B::Data;
    type Error = B::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        context: &mut TaskContext<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        Pin::new(&mut self.inner).poll_frame(context)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> hyper::body::SizeHint {
        self.inner.size_hint()
    }
}

impl WorkLimits {
    fn new() -> Self {
        Self {
            connections: Arc::new(Semaphore::new(MAX_CONCURRENT_CONNECTIONS)),
            requests: Arc::new(Semaphore::new(MAX_CONCURRENT_REQUESTS)),
            storage: Arc::new(Semaphore::new(MAX_CONCURRENT_STORAGE_OPERATIONS)),
        }
    }
}

/// Serve the artifact API until the listener fails or the process is terminated.
///
/// Header parsing and every upload-body read are time bounded. Filesystem hashing
/// and synchronization run on Tokio's blocking pool rather than occupying HTTP
/// executor threads.
///
/// # Errors
///
/// Returns an error when the body timeout is invalid, the listener cannot start,
/// or accepting a connection fails.
pub async fn serve(
    address: &str,
    store: &Arc<FilesystemStore>,
    verifier: &Arc<CapabilityVerifier>,
    body_read_timeout: Duration,
) -> Result<(), String> {
    serve_with_limits(
        address,
        store,
        verifier,
        body_read_timeout,
        WorkLimits::new(),
    )
    .await
}

async fn serve_with_limits(
    address: &str,
    store: &Arc<FilesystemStore>,
    verifier: &Arc<CapabilityVerifier>,
    body_read_timeout: Duration,
    limits: WorkLimits,
) -> Result<(), String> {
    if body_read_timeout.is_zero() {
        return Err("artifact body-read timeout must be positive".to_owned());
    }
    let listener = TcpListener::bind(address)
        .await
        .map_err(|error| error.to_string())?;
    loop {
        let (stream, _) = listener.accept().await.map_err(|error| error.to_string())?;
        let Ok(connection_permit) = Arc::clone(&limits.connections).try_acquire_owned() else {
            continue;
        };
        let store = Arc::clone(store);
        let verifier = Arc::clone(verifier);
        let limits = limits.clone();
        task::spawn(async move {
            let _connection_permit = connection_permit;
            let service = service_fn(move |request| {
                let store = Arc::clone(&store);
                let verifier = Arc::clone(&verifier);
                let limits = limits.clone();
                async move {
                    Ok::<_, Infallible>(
                        handle(request, &store, &verifier, &limits, body_read_timeout).await,
                    )
                }
            });
            let _ = http1::Builder::new()
                .timer(TokioTimer::new())
                .header_read_timeout(HEADER_READ_TIMEOUT)
                .serve_connection(TokioIo::new(stream), service)
                .await;
        });
    }
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

async fn handle(
    request: Request<Incoming>,
    store: &Arc<FilesystemStore>,
    verifier: &Arc<CapabilityVerifier>,
    limits: &WorkLimits,
    body_read_timeout: Duration,
) -> Response<HttpBody> {
    if request.method() == Method::GET && request.uri().path() == "/healthz" {
        return empty_response(StatusCode::NO_CONTENT);
    }
    let Ok(request_permit) = Arc::clone(&limits.requests).try_acquire_owned() else {
        return json_response(
            serde_json::json!({"error": "artifact service is at request capacity"}).to_string(),
            StatusCode::SERVICE_UNAVAILABLE,
        );
    };
    let path = request.uri().path().to_owned();
    let parts: Vec<&str> = path.trim_matches('/').split('/').collect();
    let result = match (request.method().clone(), parts.as_slice()) {
        (Method::POST, ["v1alpha1", "uploads"]) => {
            begin(request, store, verifier, limits, body_read_timeout).await
        }
        (Method::PUT, ["v1alpha1", "uploads", upload_id]) => {
            append(
                request,
                store,
                verifier,
                limits,
                upload_id,
                body_read_timeout,
            )
            .await
        }
        (Method::POST, ["v1alpha1", "uploads", upload_id, "commit"]) => {
            commit(&request, store, verifier, limits, upload_id).await
        }
        (Method::GET, ["v1alpha1", "artifacts", digest]) => {
            download(&request, store, verifier, limits, digest).await
        }
        (Method::HEAD, ["v1alpha1", "artifacts", digest]) => {
            metadata(&request, store, verifier, limits, digest).await
        }
        _ => Err((StatusCode::NOT_FOUND, "route not found".to_owned())),
    };
    let response = match result {
        Ok(response) => response,
        Err((status, message)) => {
            let body = serde_json::to_string(&serde_json::json!({"error": message}))
                .unwrap_or_else(|_| "{}".to_owned());
            json_response(body, status)
        }
    };
    retain_request_capacity(response, request_permit)
}

fn retain_request_capacity(
    response: Response<HttpBody>,
    request_permit: OwnedSemaphorePermit,
) -> Response<HttpBody> {
    response.map(|body| {
        PermitBody {
            inner: body,
            _permit: request_permit,
        }
        .boxed()
    })
}

async fn begin(
    request: Request<Incoming>,
    store: &Arc<FilesystemStore>,
    verifier: &CapabilityVerifier,
    limits: &WorkLimits,
    body_read_timeout: Duration,
) -> Result<Response<HttpBody>, HttpError> {
    let headers = request.headers().clone();
    let body = read_limited(request.into_body(), 64 * 1024, body_read_timeout).await?;
    let body: BeginRequest = serde_json::from_slice(&body)
        .map_err(|_| (StatusCode::BAD_REQUEST, "invalid begin request".to_owned()))?;
    let scope = Scope {
        tenant_id: body.tenant_id,
        project_id: body.project_id,
    };
    let capability = capability_header(&headers)?;
    verifier
        .verify_upload(
            &capability,
            &scope,
            &body.digest,
            body.size_bytes,
            &body.session_id,
        )
        .map_err(|error| (StatusCode::FORBIDDEN, error.to_string()))?;
    let store = Arc::clone(store);
    let nonce = capability.nonce;
    let session = store_call(limits, move || {
        store.begin_authorized(
            scope,
            body.digest,
            body.size_bytes,
            &body.session_id,
            &nonce,
            capability.expires_unix,
        )
    })
    .await?;
    let json = serde_json::to_string(&SessionResponse {
        upload_id: &session.upload_id,
        committed_bytes: 0,
    })
    .map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "serialization failed".to_owned(),
        )
    })?;
    Ok(json_response(json, StatusCode::CREATED))
}

async fn append(
    request: Request<Incoming>,
    store: &Arc<FilesystemStore>,
    verifier: &CapabilityVerifier,
    limits: &WorkLimits,
    upload_id: &str,
    body_read_timeout: Duration,
) -> Result<Response<HttpBody>, HttpError> {
    let upload_id = upload_id.to_owned();
    let session_store = Arc::clone(store);
    let session_id = upload_id.clone();
    let session = store_call(limits, move || session_store.session(&session_id)).await?;
    let capability = capability_header(request.headers())?;
    verifier
        .verify_upload(
            &capability,
            &session.scope,
            &session.digest,
            session.size_bytes,
            &session.upload_id,
        )
        .map_err(|error| (StatusCode::FORBIDDEN, error.to_string()))?;
    if capability.nonce != session.authorization_nonce {
        return Err((
            StatusCode::FORBIDDEN,
            "capability does not authorize this upload session".to_owned(),
        ));
    }
    let offset = request
        .uri()
        .query()
        .and_then(|query| {
            query
                .split('&')
                .find_map(|field| field.strip_prefix("offset="))
        })
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or((StatusCode::BAD_REQUEST, "offset is required".to_owned()))?;
    let bytes = read_limited(request.into_body(), 8 * 1024 * 1024, body_read_timeout).await?;
    let append_store = Arc::clone(store);
    let updated = store_call(limits, move || {
        append_store.append(&upload_id, offset, &bytes)
    })
    .await?;
    let json = serde_json::to_string(&SessionResponse {
        upload_id: &updated.upload_id,
        committed_bytes: updated.committed_bytes,
    })
    .map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "serialization failed".to_owned(),
        )
    })?;
    Ok(json_response(json, StatusCode::OK))
}

async fn commit(
    request: &Request<Incoming>,
    store: &Arc<FilesystemStore>,
    verifier: &CapabilityVerifier,
    limits: &WorkLimits,
    upload_id: &str,
) -> Result<Response<HttpBody>, HttpError> {
    let upload_id = upload_id.to_owned();
    let session_store = Arc::clone(store);
    let session_id = upload_id.clone();
    let session = store_call(limits, move || session_store.session(&session_id)).await?;
    let capability = capability_header(request.headers())?;
    verifier
        .verify_upload(
            &capability,
            &session.scope,
            &session.digest,
            session.size_bytes,
            &session.upload_id,
        )
        .map_err(|error| (StatusCode::FORBIDDEN, error.to_string()))?;
    if capability.nonce != session.authorization_nonce {
        return Err((
            StatusCode::FORBIDDEN,
            "capability does not authorize this upload session".to_owned(),
        ));
    }
    let commit_store = Arc::clone(store);
    let digest = store_call(limits, move || commit_store.commit(&upload_id)).await?;
    Ok(json_response(
        serde_json::json!({"digest": digest.to_string()}).to_string(),
        StatusCode::OK,
    ))
}

async fn download(
    request: &Request<Incoming>,
    store: &Arc<FilesystemStore>,
    verifier: &CapabilityVerifier,
    limits: &WorkLimits,
    digest: &str,
) -> Result<Response<HttpBody>, HttpError> {
    let decoded = decode_path_segment(digest)?;
    let digest = Digest::from_str(&decoded).map_err(|error| (StatusCode::BAD_REQUEST, error))?;
    let capability = capability_header(request.headers())?;
    verifier
        .verify(
            &capability,
            &capability.scope,
            &digest,
            CapabilityOperation::Download,
        )
        .map_err(|error| (StatusCode::FORBIDDEN, error.to_string()))?;
    let open_store = Arc::clone(store);
    let requested_digest = digest.clone();
    let (file, size) =
        store_call(limits, move || open_store.open_verified(&requested_digest)).await?;
    if size > capability.max_size_bytes {
        return Err((
            StatusCode::FORBIDDEN,
            "artifact exceeds capability size bound".to_owned(),
        ));
    }
    let stream = ReaderStream::new(tokio::fs::File::from_std(file))
        .map_ok(Frame::data)
        .map_err(|error| -> BoxError { Box::new(error) });
    let body = StreamBody::new(stream).boxed();
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_static("application/octet-stream"),
    );
    response.headers_mut().insert(
        CONTENT_LENGTH,
        HeaderValue::from_str(&size.to_string()).map_err(internal_header_error)?,
    );
    response.headers_mut().insert(
        ETAG,
        HeaderValue::from_str(&digest.to_string()).map_err(internal_header_error)?,
    );
    Ok(response)
}

async fn metadata(
    request: &Request<Incoming>,
    store: &Arc<FilesystemStore>,
    verifier: &CapabilityVerifier,
    limits: &WorkLimits,
    digest: &str,
) -> Result<Response<HttpBody>, HttpError> {
    let decoded = decode_path_segment(digest)?;
    let digest = Digest::from_str(&decoded).map_err(|error| (StatusCode::BAD_REQUEST, error))?;
    let capability = capability_header(request.headers())?;
    verifier
        .verify(
            &capability,
            &capability.scope,
            &digest,
            CapabilityOperation::Download,
        )
        .map_err(|error| (StatusCode::FORBIDDEN, error.to_string()))?;
    let size_store = Arc::clone(store);
    let requested_digest = digest.clone();
    let size = store_call(limits, move || size_store.verified_size(&requested_digest)).await?;
    if size != capability.max_size_bytes {
        return Err((
            StatusCode::FORBIDDEN,
            "artifact size differs from capability bound".to_owned(),
        ));
    }
    let mut response = empty_response(StatusCode::OK);
    response.headers_mut().insert(
        "x-mindclade-artifact-digest",
        HeaderValue::from_str(&digest.to_string()).map_err(internal_header_error)?,
    );
    response.headers_mut().insert(
        "x-mindclade-artifact-size",
        HeaderValue::from_str(&size.to_string()).map_err(internal_header_error)?,
    );
    Ok(response)
}

async fn store_call<T, F>(limits: &WorkLimits, operation: F) -> Result<T, HttpError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, ArtifactError> + Send + 'static,
{
    let storage_permit = Arc::clone(&limits.storage)
        .acquire_owned()
        .await
        .map_err(|_| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "artifact storage capacity is unavailable".to_owned(),
            )
        })?;
    task::spawn_blocking(move || {
        let _storage_permit = storage_permit;
        operation()
    })
    .await
    .map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "artifact storage operation failed".to_owned(),
        )
    })?
    .map_err(|error| store_error(&error))
}

async fn read_limited(
    mut body: Incoming,
    maximum: u64,
    body_read_timeout: Duration,
) -> Result<Vec<u8>, HttpError> {
    if body.size_hint().lower() > maximum
        || body
            .size_hint()
            .upper()
            .is_some_and(|upper| upper > maximum)
    {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            "request body is too large".to_owned(),
        ));
    }
    let read = async {
        let mut bytes = Vec::new();
        while let Some(frame) = body.frame().await {
            let frame = frame.map_err(|_| {
                (
                    StatusCode::BAD_REQUEST,
                    "request body could not be read".to_owned(),
                )
            })?;
            if let Ok(data) = frame.into_data() {
                let new_length = (bytes.len() as u64).checked_add(data.len() as u64).ok_or((
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "request body is too large".to_owned(),
                ))?;
                if new_length > maximum {
                    return Err((
                        StatusCode::PAYLOAD_TOO_LARGE,
                        "request body is too large".to_owned(),
                    ));
                }
                bytes.extend_from_slice(&data);
            }
        }
        Ok(bytes)
    };
    time::timeout(body_read_timeout, read).await.map_err(|_| {
        (
            StatusCode::REQUEST_TIMEOUT,
            "request body read timed out".to_owned(),
        )
    })?
}

fn decode_path_segment(value: &str) -> Result<String, HttpError> {
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
            return Err((
                StatusCode::BAD_REQUEST,
                "artifact path encoding is invalid".to_owned(),
            ));
        }
        let high = hexadecimal(source[index + 1]).ok_or((
            StatusCode::BAD_REQUEST,
            "artifact path encoding is invalid".to_owned(),
        ))?;
        let low = hexadecimal(source[index + 2]).ok_or((
            StatusCode::BAD_REQUEST,
            "artifact path encoding is invalid".to_owned(),
        ))?;
        let byte = (high << 4) | low;
        if matches!(byte, b'/' | b'\\' | b'%') {
            return Err((
                StatusCode::BAD_REQUEST,
                "encoded path separators are forbidden".to_owned(),
            ));
        }
        decoded.push(byte);
        index += 3;
    }
    String::from_utf8(decoded).map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            "artifact path is not UTF-8".to_owned(),
        )
    })
}

fn capability_header(headers: &HeaderMap) -> Result<Capability, HttpError> {
    let encoded = headers
        .get("x-mindclade-capability")
        .and_then(|value| value.to_str().ok())
        .ok_or((
            StatusCode::UNAUTHORIZED,
            "capability is required".to_owned(),
        ))?;
    let bytes = URL_SAFE_NO_PAD.decode(encoded).map_err(|_| {
        (
            StatusCode::UNAUTHORIZED,
            "capability encoding is invalid".to_owned(),
        )
    })?;
    serde_json::from_slice(&bytes).map_err(|_| {
        (
            StatusCode::UNAUTHORIZED,
            "capability payload is invalid".to_owned(),
        )
    })
}

fn empty_response(status: StatusCode) -> Response<HttpBody> {
    let mut response = Response::new(full_body(Bytes::new()));
    *response.status_mut() = status;
    response
}

fn json_response(body: String, status: StatusCode) -> Response<HttpBody> {
    let mut response = Response::new(full_body(Bytes::from(body)));
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response
}

fn full_body(bytes: Bytes) -> HttpBody {
    Full::new(bytes)
        .map_err(|never| -> BoxError { match never {} })
        .boxed()
}

fn internal_header_error(_: hyper::header::InvalidHeaderValue) -> HttpError {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        "response header failed".to_owned(),
    )
}

fn store_error(error: &ArtifactError) -> HttpError {
    let status = match error {
        ArtifactError::UploadNotFound => StatusCode::NOT_FOUND,
        ArtifactError::Io(value) if value.kind() == io::ErrorKind::NotFound => {
            StatusCode::NOT_FOUND
        }
        ArtifactError::OffsetMismatch { .. }
        | ArtifactError::SizeExceeded
        | ArtifactError::Integrity
        | ArtifactError::Replay => StatusCode::CONFLICT,
        ArtifactError::Capacity => StatusCode::TOO_MANY_REQUESTS,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, error.to_string())
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_CONCURRENT_REQUESTS, WorkLimits, decode_path_segment, empty_response,
        retain_request_capacity, serve, serve_with_limits, store_call,
    };
    use crate::{
        Capability, CapabilityOperation, CapabilityVerifier, Digest, FilesystemStore, Scope,
    };
    use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
    use ed25519_dalek::{Signer as _, SigningKey};
    use hyper::StatusCode;
    use std::{
        collections::HashMap,
        io::{Read as _, Write as _},
        net::{TcpListener, TcpStream},
        sync::Arc,
        thread,
        time::Duration,
    };
    use tokio::{net::TcpStream as TokioTcpStream, sync::Semaphore};

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
    fn response_body_holds_request_capacity_until_dropped() {
        let limits = WorkLimits::new();
        let permit = Arc::clone(&limits.requests).try_acquire_owned().unwrap();
        let response = retain_request_capacity(empty_response(StatusCode::OK), permit);
        assert_eq!(
            limits.requests.available_permits(),
            MAX_CONCURRENT_REQUESTS - 1
        );
        drop(response);
        assert_eq!(limits.requests.available_permits(), MAX_CONCURRENT_REQUESTS);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_store_call_retains_capacity_until_blocking_work_finishes() {
        let limits = WorkLimits {
            connections: Arc::new(Semaphore::new(1)),
            requests: Arc::new(Semaphore::new(1)),
            storage: Arc::new(Semaphore::new(1)),
        };
        let worker_limits = limits.clone();
        let (started_send, started_receive) = tokio::sync::oneshot::channel();
        let (finish_send, finish_receive) = tokio::sync::oneshot::channel();
        let worker = tokio::spawn(async move {
            store_call(&worker_limits, move || {
                let _ = started_send.send(());
                let _ = finish_receive.blocking_recv();
                Ok(())
            })
            .await
        });

        started_receive.await.unwrap();
        assert_eq!(limits.storage.available_permits(), 0);
        worker.abort();
        assert!(worker.await.unwrap_err().is_cancelled());
        assert_eq!(limits.storage.available_permits(), 0);

        finish_send.send(()).unwrap();
        wait_for_permits(&limits.storage, 1).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn slow_headers_cannot_create_unbounded_connection_tasks() {
        let temporary = tempfile::tempdir().unwrap();
        let store = Arc::new(FilesystemStore::new(temporary.path()).unwrap());
        let verifier = Arc::new(CapabilityVerifier::with_clock(HashMap::new(), || 1_000));
        let limits = WorkLimits {
            connections: Arc::new(Semaphore::new(1)),
            requests: Arc::new(Semaphore::new(1)),
            storage: Arc::new(Semaphore::new(1)),
        };
        let observed_limits = limits.clone();
        let address = unused_address();
        let server_store = Arc::clone(&store);
        let server = tokio::spawn(async move {
            serve_with_limits(
                &address.to_string(),
                &server_store,
                &verifier,
                Duration::from_secs(1),
                limits,
            )
            .await
            .unwrap();
        });

        let stalled = connect(address).await;
        write_all(&stalled, b"GET /").await;
        wait_for_permits(&observed_limits.connections, 0).await;

        let rejected = connect(address).await;
        write_all(&rejected, b"GET /healthz HTTP/1.1\r\nHost: test\r\n\r\n").await;
        let mut byte = [0_u8; 1];
        let closed =
            tokio::time::timeout(Duration::from_millis(500), read_once(&rejected, &mut byte))
                .await
                .expect("over-capacity connection remained open");
        assert!(
            matches!(closed, Ok(0) | Err(_)),
            "unexpected read: {closed:?}"
        );

        drop(stalled);
        wait_for_permits(&observed_limits.connections, 1).await;
        let health = tokio::task::spawn_blocking(move || {
            raw_response(address, |stream| {
                write!(
                    stream,
                    "GET /healthz HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n"
                )
            })
        })
        .await
        .unwrap();
        assert!(health.starts_with("HTTP/1.1 204"), "{health}");
        server.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn authenticated_head_attests_exact_digest_and_size() {
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
        let address = unused_address();
        let server_store = Arc::clone(&store);
        let server = tokio::spawn(async move {
            serve(
                &address.to_string(),
                &server_store,
                &verifier,
                Duration::from_secs(1),
            )
            .await
            .unwrap();
        });

        let capability =
            signed_download_capability(&signing_key, scope, digest.clone(), bytes.len() as u64);
        let response = head(address, &digest, Some(&capability));
        let lower = response.to_ascii_lowercase();
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        assert!(
            lower.contains(&format!("x-mindclade-artifact-digest: {digest}\r\n")),
            "{response}"
        );
        assert!(
            lower.contains(&format!("x-mindclade-artifact-size: {}\r\n", bytes.len())),
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
        server.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stalled_upload_body_times_out_without_blocking_health() {
        let temporary = tempfile::tempdir().unwrap();
        let store = Arc::new(FilesystemStore::new(temporary.path()).unwrap());
        let verifier = Arc::new(CapabilityVerifier::with_clock(HashMap::new(), || 1_000));
        let address = unused_address();
        let server_store = Arc::clone(&store);
        let server = tokio::spawn(async move {
            serve(
                &address.to_string(),
                &server_store,
                &verifier,
                Duration::from_millis(500),
            )
            .await
            .unwrap();
        });

        let (ready_send, mut ready_receive) = tokio::sync::mpsc::channel(1);
        let stalled = tokio::task::spawn_blocking(move || {
            raw_response(address, |stream| {
                write!(
                    stream,
                    "POST /v1alpha1/uploads HTTP/1.1\r\nHost: {address}\r\nContent-Length: 100\r\nConnection: close\r\n\r\n{{"
                )?;
                ready_send.blocking_send(()).unwrap();
                Ok(())
            })
        });
        tokio::time::timeout(Duration::from_millis(250), ready_receive.recv())
            .await
            .expect("stalled client did not connect")
            .expect("stalled client readiness channel closed");
        assert!(
            !stalled.is_finished(),
            "stalled body finished before timeout"
        );

        let health = tokio::task::spawn_blocking(move || {
            raw_response(address, |stream| {
                write!(
                    stream,
                    "GET /healthz HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n"
                )
            })
        });
        let health = tokio::time::timeout(Duration::from_millis(250), health)
            .await
            .expect("health check waited for stalled upload bodies")
            .unwrap();
        assert!(health.starts_with("HTTP/1.1 204"), "{health}");
        assert!(
            !stalled.is_finished(),
            "health check outlasted body timeout"
        );
        let stalled = stalled.await.unwrap();
        assert!(stalled.starts_with("HTTP/1.1 408"), "{stalled}");
        server.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn request_capacity_is_bounded_and_health_bypasses_it() {
        let temporary = tempfile::tempdir().unwrap();
        let store = Arc::new(FilesystemStore::new(temporary.path()).unwrap());
        let verifier = Arc::new(CapabilityVerifier::with_clock(HashMap::new(), || 1_000));
        let limits = WorkLimits::new();
        let capacity_permit = Arc::clone(&limits.requests)
            .acquire_many_owned(u32::try_from(MAX_CONCURRENT_REQUESTS).unwrap())
            .await
            .unwrap();
        let address = unused_address();
        let server_store = Arc::clone(&store);
        let server = tokio::spawn(async move {
            serve_with_limits(
                &address.to_string(),
                &server_store,
                &verifier,
                Duration::from_secs(1),
                limits,
            )
            .await
            .unwrap();
        });

        let rejected = raw_response(address, |stream| {
            write!(
                stream,
                "POST /v1alpha1/uploads HTTP/1.1\r\nHost: {address}\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}"
            )
        });
        assert!(rejected.starts_with("HTTP/1.1 503"), "{rejected}");
        let health = raw_response(address, |stream| {
            write!(
                stream,
                "GET /healthz HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n"
            )
        });
        assert!(health.starts_with("HTTP/1.1 204"), "{health}");
        drop(capacity_permit);
        server.abort();
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
        raw_response(address, |stream| {
            let capability_header = capability
                .map(|value| format!("X-Mindclade-Capability: {value}\r\n"))
                .unwrap_or_default();
            write!(
                stream,
                "HEAD /v1alpha1/artifacts/{digest} HTTP/1.1\r\nHost: {address}\r\n{capability_header}Connection: close\r\n\r\n"
            )
        })
    }

    fn unused_address() -> std::net::SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        address
    }

    async fn connect(address: std::net::SocketAddr) -> TokioTcpStream {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                match TokioTcpStream::connect(address).await {
                    Ok(stream) => return stream,
                    Err(_) => tokio::time::sleep(Duration::from_millis(1)).await,
                }
            }
        })
        .await
        .expect("artifact HTTP server did not start")
    }

    async fn write_all(stream: &TokioTcpStream, mut bytes: &[u8]) {
        while !bytes.is_empty() {
            stream.writable().await.unwrap();
            match stream.try_write(bytes) {
                Ok(0) => panic!("artifact HTTP connection closed while writing"),
                Ok(written) => bytes = &bytes[written..],
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(error) => panic!("artifact HTTP write failed: {error}"),
            }
        }
    }

    async fn read_once(stream: &TokioTcpStream, buffer: &mut [u8]) -> std::io::Result<usize> {
        loop {
            stream.readable().await?;
            match stream.try_read(buffer) {
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
                result => return result,
            }
        }
    }

    async fn wait_for_permits(semaphore: &Semaphore, expected: usize) {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if semaphore.available_permits() == expected {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("semaphore did not reach the expected capacity");
    }

    fn raw_response(
        address: std::net::SocketAddr,
        write_request: impl FnOnce(&mut TcpStream) -> std::io::Result<()>,
    ) -> String {
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
        write_request(&mut stream).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        response
    }
}
