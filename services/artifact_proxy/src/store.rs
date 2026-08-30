//! Recoverable filesystem implementation of resumable content-addressed storage.

use crate::{Digest, Scope};
use sha2::{Digest as _, Sha256};
use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};
use thiserror::Error;

/// Artifact store failure with stable caller semantics.
#[derive(Debug, Error)]
pub enum ArtifactError {
    /// Underlying durable storage failed.
    #[error("artifact storage operation failed: {0}")]
    Io(#[from] std::io::Error),
    /// The upload identifier does not exist in this process.
    #[error("upload session not found")]
    UploadNotFound,
    /// A chunk does not begin at the currently committed offset.
    #[error("chunk offset mismatch: expected {expected}, received {received}")]
    OffsetMismatch {
        /// Next offset accepted by the session.
        expected: u64,
        /// Offset supplied by the caller.
        received: u64,
    },
    /// The data exceeded the declared object length.
    #[error("upload exceeds declared size")]
    SizeExceeded,
    /// Length or SHA-256 digest did not match at commit.
    #[error("artifact integrity check failed")]
    Integrity,
    /// A poisoned process-local lock prevents safe operation.
    #[error("artifact session state is unavailable")]
    Lock,
    /// Store or tenant safety limits reject additional work.
    #[error("artifact store capacity limit exceeded")]
    Capacity,
    /// A session identifier or nonce was previously consumed.
    #[error("upload capability nonce or session was already used")]
    Replay,
}

/// Public upload progress; no temporary filesystem path crosses the API boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UploadSession {
    /// Opaque process-local upload identifier.
    pub upload_id: String,
    /// Expected immutable digest.
    pub digest: Digest,
    /// Declared total object size.
    pub size_bytes: u64,
    /// Bytes durably appended so far.
    pub committed_bytes: u64,
    /// Tenant and project authorized for this upload.
    pub scope: Scope,
    /// Capability nonce bound to this session.
    pub authorization_nonce: String,
}

#[derive(Clone)]
struct SessionState {
    public: UploadSession,
    path: PathBuf,
    expires_unix: u64,
}

/// Local backend with the same offset and integrity contracts as a resumable GCS backend.
pub struct FilesystemStore {
    root: PathBuf,
    sessions: Mutex<HashMap<String, SessionState>>,
    used_nonces: Mutex<HashMap<String, u64>>,
    sequence: AtomicU64,
    max_object_bytes: u64,
    max_sessions: usize,
}

impl FilesystemStore {
    /// Create the store beneath an explicit, narrow root directory.
    ///
    /// # Errors
    ///
    /// Returns [`ArtifactError`] when directories cannot be created.
    pub fn new(root: impl AsRef<Path>) -> Result<Self, ArtifactError> {
        Self::with_limits(root, 8 * 1024 * 1024 * 1024, 1_024)
    }

    /// Create a store with explicit object and concurrent-session limits.
    ///
    /// # Errors
    ///
    /// Returns [`ArtifactError`] for invalid limits or filesystem failures.
    pub fn with_limits(
        root: impl AsRef<Path>,
        max_object_bytes: u64,
        max_sessions: usize,
    ) -> Result<Self, ArtifactError> {
        if max_object_bytes == 0 || max_sessions == 0 {
            return Err(ArtifactError::Capacity);
        }
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join("objects"))?;
        fs::create_dir_all(root.join("uploads"))?;
        Ok(Self {
            root,
            sessions: Mutex::new(HashMap::new()),
            used_nonces: Mutex::new(HashMap::new()),
            sequence: AtomicU64::new(0),
            max_object_bytes,
            max_sessions,
        })
    }

    /// Start or resume a process-local upload session.
    ///
    /// # Errors
    ///
    /// Returns [`ArtifactError`] when limits, locking, or filesystem operations fail.
    pub fn begin(
        &self,
        scope: Scope,
        digest: Digest,
        size_bytes: u64,
    ) -> Result<UploadSession, ArtifactError> {
        let sequence = self.sequence.fetch_add(1, Ordering::Relaxed) + 1;
        let upload_id = format!("upload-{sequence:016x}");
        self.begin_authorized(
            scope,
            digest,
            size_bytes,
            &upload_id,
            &format!("nonce-{sequence:016x}"),
            u64::MAX,
        )
    }

    /// Start a capability-bound upload, rejecting replay and unsafe sizes.
    ///
    /// # Errors
    ///
    /// Returns [`ArtifactError`] for replay, capacity, expiry cleanup, or I/O failures.
    pub fn begin_authorized(
        &self,
        scope: Scope,
        digest: Digest,
        size_bytes: u64,
        upload_id: &str,
        nonce: &str,
        expires_unix: u64,
    ) -> Result<UploadSession, ArtifactError> {
        if size_bytes == 0 || size_bytes > self.max_object_bytes {
            return Err(ArtifactError::SizeExceeded);
        }
        let now = unix_now();
        let mut sessions = self.sessions.lock().map_err(|_| ArtifactError::Lock)?;
        let expired: Vec<String> = sessions
            .iter()
            .filter(|(_, state)| state.expires_unix <= now)
            .map(|(id, _)| id.clone())
            .collect();
        for id in expired {
            if let Some(state) = sessions.remove(&id) {
                let _ = fs::remove_file(state.path);
            }
        }
        let mut used_nonces = self.used_nonces.lock().map_err(|_| ArtifactError::Lock)?;
        used_nonces.retain(|_, expiry| *expiry > now);
        if sessions.len() >= self.max_sessions
            || sessions.contains_key(upload_id)
            || used_nonces.contains_key(nonce)
        {
            return Err(if sessions.len() >= self.max_sessions {
                ArtifactError::Capacity
            } else {
                ArtifactError::Replay
            });
        }
        let path = self.root.join("uploads").join(format!("{upload_id}.part"));
        File::create(&path)?.sync_all()?;
        let public = UploadSession {
            upload_id: upload_id.to_owned(),
            digest,
            size_bytes,
            committed_bytes: 0,
            scope,
            authorization_nonce: nonce.to_owned(),
        };
        used_nonces.insert(nonce.to_owned(), expires_unix);
        sessions.insert(
            upload_id.to_owned(),
            SessionState {
                public: public.clone(),
                path,
                expires_unix,
            },
        );
        Ok(public)
    }

    /// Inspect upload authority and progress before capability validation.
    ///
    /// # Errors
    ///
    /// Returns [`ArtifactError::UploadNotFound`] or [`ArtifactError::Lock`].
    pub fn session(&self, upload_id: &str) -> Result<UploadSession, ArtifactError> {
        self.sessions
            .lock()
            .map_err(|_| ArtifactError::Lock)?
            .get(upload_id)
            .map(|state| state.public.clone())
            .ok_or(ArtifactError::UploadNotFound)
    }

    /// Append exactly one chunk at the expected byte offset.
    ///
    /// # Errors
    ///
    /// Returns [`ArtifactError`] for unknown sessions, offsets, limits, locking, or I/O.
    pub fn append(
        &self,
        upload_id: &str,
        offset: u64,
        bytes: &[u8],
    ) -> Result<UploadSession, ArtifactError> {
        let mut sessions = self.sessions.lock().map_err(|_| ArtifactError::Lock)?;
        let state = sessions
            .get_mut(upload_id)
            .ok_or(ArtifactError::UploadNotFound)?;
        if offset != state.public.committed_bytes {
            return Err(ArtifactError::OffsetMismatch {
                expected: state.public.committed_bytes,
                received: offset,
            });
        }
        let new_length = offset
            .checked_add(bytes.len() as u64)
            .ok_or(ArtifactError::SizeExceeded)?;
        if new_length > state.public.size_bytes {
            return Err(ArtifactError::SizeExceeded);
        }
        let mut file = OpenOptions::new().write(true).open(&state.path)?;
        file.seek(SeekFrom::Start(offset))?;
        file.write_all(bytes)?;
        file.sync_data()?;
        state.public.committed_bytes = new_length;
        Ok(state.public.clone())
    }

    /// Verify length and SHA-256 before atomically exposing an object.
    ///
    /// # Errors
    ///
    /// Returns [`ArtifactError`] when the session, bytes, digest, lock, or I/O is invalid.
    pub fn commit(&self, upload_id: &str) -> Result<Digest, ArtifactError> {
        let state = self
            .sessions
            .lock()
            .map_err(|_| ArtifactError::Lock)?
            .get(upload_id)
            .cloned()
            .ok_or(ArtifactError::UploadNotFound)?;
        if state.public.committed_bytes != state.public.size_bytes {
            return Err(ArtifactError::Integrity);
        }
        let mut file = File::open(&state.path)?;
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 8 * 1024];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
        }
        let actual: [u8; 32] = hasher.finalize().into();
        let actual: Digest = format!("sha256:{}", hex::encode(actual))
            .parse()
            .map_err(|_| ArtifactError::Integrity)?;
        if actual != state.public.digest {
            let _ = fs::remove_file(&state.path);
            self.sessions
                .lock()
                .map_err(|_| ArtifactError::Lock)?
                .remove(upload_id);
            return Err(ArtifactError::Integrity);
        }
        let destination = self.object_path(&actual);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        if destination.exists() {
            fs::remove_file(&state.path)?;
        } else {
            fs::rename(&state.path, &destination)?;
        }
        self.sessions
            .lock()
            .map_err(|_| ArtifactError::Lock)?
            .remove(upload_id);
        Ok(actual)
    }

    /// Read an entire verified object. HTTP range projection is deliberately separate.
    ///
    /// # Errors
    ///
    /// Returns [`ArtifactError`] when verification, sizing, or I/O fails.
    pub fn get(&self, digest: &Digest) -> Result<Vec<u8>, ArtifactError> {
        let (mut file, size) = self.open_verified(digest)?;
        let capacity = usize::try_from(size).map_err(|_| ArtifactError::SizeExceeded)?;
        let mut bytes = Vec::with_capacity(capacity);
        file.read_to_end(&mut bytes)?;
        Ok(bytes)
    }

    /// Open a verified object for bounded streaming without buffering it in memory.
    ///
    /// # Errors
    ///
    /// Returns [`ArtifactError`] when verification, sizing, or I/O fails.
    pub fn open_verified(&self, digest: &Digest) -> Result<(File, u64), ArtifactError> {
        let mut file = File::open(self.object_path(digest))?;
        let size = file.metadata()?.len();
        if size == 0 || size > self.max_object_bytes {
            return Err(ArtifactError::SizeExceeded);
        }
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 8 * 1024];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
        }
        let actual = format!(
            "sha256:{}",
            hex::encode(<[u8; 32]>::from(hasher.finalize()))
        );
        if actual != digest.to_string() {
            return Err(ArtifactError::Integrity);
        }
        file.seek(SeekFrom::Start(0))?;
        Ok((file, size))
    }

    /// Re-hash an immutable object and return its exact committed size.
    ///
    /// This is the metadata attestation boundary used by the control plane
    /// before it accepts a worker completion receipt.
    ///
    /// # Errors
    ///
    /// Returns [`ArtifactError`] when the object is absent, oversized, or its
    /// on-disk bytes no longer match the requested digest.
    pub fn verified_size(&self, digest: &Digest) -> Result<u64, ArtifactError> {
        let (_file, size) = self.open_verified(digest)?;
        Ok(size)
    }

    fn object_path(&self, digest: &Digest) -> PathBuf {
        let body = digest.hex();
        self.root.join("objects").join(&body[..2]).join(body)
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
