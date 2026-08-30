# Q0 model serving contract

An external inference request names an immutable model bundle digest and an immutable
input artifact digest. The request is scoped by OIDC tenant/project claims and an
idempotency key. Accepted work progresses through `queued`, `admitted`, `running`,
and one terminal state. A worker completion is accepted only when its fencing token
matches the active attempt.

The development model-bundle path has two integrity layers:

1. Its canonical inner manifest is Ed25519-signed and protects every config, weight,
   qualification, conversion, SBOM, and provenance file.
2. Its transport archive is catalogued by an exact `sha256:` digest and byte size,
   capability-scoped during download, and re-hashed before extraction. The library
   can emit OCI layer descriptors, but assembling and Cosign-verifying a model OCI
   manifest remains a promotion requirement rather than a capability claimed here.

Only `safetensors` are loaded. Pickle checkpoints and remote-code model loading are
not supported. Tensor values are excluded from logs and receipts. Retention is not
enforced by the development stores; proposed 30-day tensor and one-year manifest and
audit retention periods require a separately implemented durable lifecycle policy.

The checked-in deployment is development-only and render-only. There is no repository
command that provisions GCP, applies Kubernetes resources, publishes an image, or
promotes a model.
