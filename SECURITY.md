# Security policy

Report suspected vulnerabilities through Mindclade's private security channel; do
not open a public issue. Never attach customer tensors, model artifacts, credentials,
tokens, or kubeconfigs to a report.

Model loading is local and safetensors-only. The development serving path downloads
catalog-bound archives through operation-scoped capabilities, verifies their exact
digest before extraction, and verifies the bundle's Ed25519 signature and every
manifested file digest before loading. Production promotion additionally requires an
immutable OCI subject plus signature and attestation verification at the release
boundary; that registry flow is not implemented or claimed by this repository.
