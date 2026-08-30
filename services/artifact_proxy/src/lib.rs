//! Tenant-scoped, digest-verifying artifact access for model workers.

pub mod capability;
pub mod digest;
pub mod http;
pub mod store;

pub use capability::{Capability, CapabilityOperation, CapabilityVerifier, Scope};
pub use digest::Digest;
pub use store::{ArtifactError, FilesystemStore, UploadSession};
