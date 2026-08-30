//! Checked-in Rust bindings for the Mindclade protobuf APIs.

/// Mindclade protocol packages.
pub mod mindclade {
    /// Administrative resource APIs.
    pub mod admin {
        /// Alpha-version administrative messages.
        #[allow(
            clippy::all,
            clippy::pedantic,
            missing_docs,
            reason = "prost-generated bindings"
        )]
        pub mod v1alpha1 {
            include!("../../generated/rust/mindclade/admin/v1alpha1/mindclade.admin.v1alpha1.rs");
        }
    }

    /// Artifact reference, bundle, and transfer APIs.
    pub mod artifact {
        /// Alpha-version artifact messages and gRPC service bindings.
        #[allow(
            clippy::all,
            clippy::pedantic,
            missing_docs,
            reason = "prost- and tonic-generated bindings"
        )]
        pub mod v1alpha1 {
            include!(
                "../../generated/rust/mindclade/artifact/v1alpha1/mindclade.artifact.v1alpha1.rs"
            );
        }
    }

    /// Shared protocol primitives.
    pub mod common {
        /// Alpha-version common messages.
        #[allow(
            clippy::all,
            clippy::pedantic,
            missing_docs,
            reason = "prost-generated bindings"
        )]
        pub mod v1alpha1 {
            include!("../../generated/rust/mindclade/common/v1alpha1/mindclade.common.v1alpha1.rs");
        }
    }

    /// Inference request and lifecycle APIs.
    pub mod inference {
        /// Alpha-version inference messages and gRPC service bindings.
        #[allow(
            clippy::all,
            clippy::pedantic,
            missing_docs,
            reason = "prost- and tonic-generated bindings"
        )]
        pub mod v1alpha1 {
            include!(
                "../../generated/rust/mindclade/inference/v1alpha1/mindclade.inference.v1alpha1.rs"
            );
        }
    }

    /// Asynchronous job lifecycle APIs.
    pub mod job {
        /// Alpha-version job messages.
        #[allow(
            clippy::all,
            clippy::pedantic,
            missing_docs,
            reason = "prost-generated bindings"
        )]
        pub mod v1alpha1 {
            include!("../../generated/rust/mindclade/job/v1alpha1/mindclade.job.v1alpha1.rs");
        }
    }

    /// Model release APIs.
    pub mod model {
        /// Alpha-version model release messages.
        #[allow(
            clippy::all,
            clippy::pedantic,
            missing_docs,
            reason = "prost-generated bindings"
        )]
        pub mod v1alpha1 {
            include!("../../generated/rust/mindclade/model/v1alpha1/mindclade.model.v1alpha1.rs");
        }
    }
}
