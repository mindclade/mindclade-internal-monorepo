# Inference worker

The scheduler starts one worker per fenced attempt and Ed25519-signs every immutable
job field. The worker resolves scheduler and bundle key IDs only through a
deployment-owned `keyring.json`; jobs cannot supply public-key paths. A dedicated init
command downloads capability-scoped artifacts from the artifact proxy, verifies their
digests, safely expands the signed bundle, and rejects path or symlink traversal.

Results and receipts use fence-specific, non-overwriting filenames. After computing
the result digest and size, the worker signs an upload-authorization request with its
per-attempt completion key. It streams the exact result through the artifact proxy's
begin/append/commit protocol, checks the committed identity, and only then submits the
signed completion receipt. Receipts and logs never include raw input/output tensors or
bearer capabilities.

The current one-job worker continues to admit batch size one. Its safetensors result
stores each candidate's `batch_seeds` vector beside the matching `[B, A, 3]`
coordinates, plus the selected candidate's seed vector. Seed provenance therefore
remains digest-bound inside the result artifact without adding large integers to the
external JSON completion receipt.

The public-key volume contains a deployment-owned index and relative PEM files:

```json
{
  "schema_version": "v1alpha1",
  "scheduler_keys": {"scheduler-2026-08": "scheduler-2026-08.pem"},
  "bundle_keys": {"bundle-2026-08": "bundle-2026-08.pem"}
}
```

Key IDs and relative filenames are policy configuration. The signed job contains key
IDs only, and neither the worker nor the stager accepts a keyring path from that job.

The current Q0 worker is a development reference and makes no biological accuracy,
scientific, clinical, or production-fitness claim.
