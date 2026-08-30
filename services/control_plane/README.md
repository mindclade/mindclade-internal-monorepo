# Control plane

The control plane owns idempotency, cost admission, job transitions, attempt fencing,
and an outbox-style event stream. Every repository key includes tenant and project.
The in-memory store is the executable reference implementation; production assembly
must provide a transactional Postgres implementation before the service is promoted
beyond development.

Each development-store process generates a cryptographic 128-bit incarnation and
assigns public IDs as `job-<32 lowercase hex>-<8 decimal digits>`. Idempotent replays
within that process retain the same ID, while a restart enters a new ID namespace so
an old public ID cannot alias an unrelated job created after volatile state is lost.

The composition root binds loopback by default and trusts identity headers only from
the runtime gateway. It is not an internet-facing deployment.

`Dispatcher` is the executable scheduler seam. The deployed composition uses
`KubernetesAttemptLauncher`: a first non-replayed submission is admitted, leased,
and converted into a suspended JobSet. The launcher first creates the JobSet without
a Kueue label, then an owner-bound immutable manifest Secret and a retained
per-attempt result PVC, and only then patches the configured queue label. A partial
launch is rolled back. The Secret is mounted mode `0400`; completion and artifact
capabilities never appear in JobSet metadata, events, responses, or logs.
At process start the composition root generates a 128-bit resource incarnation.
Kubernetes names are deterministic over that incarnation, the public job ID, and
the fencing token. This keeps cancellation and reconciliation exact within one
process while giving each restart a collision-resistant namespace distinct from
retained PVCs left after the development store loses its counter. The incarnation
is deployment-generated, never request-controlled, and is validated again from
JobSet annotations.

The launcher issues byte-compatible Ed25519 artifact-proxy v2 download capabilities
from catalog-bound archive/input digests and sizes. It Ed25519-signs the worker
manifest with a rotation-addressed scheduler key. Each lease also generates a fresh
completion signing keypair: the store retains only the public key, while the private
seed exists only in the signed attempt Secret. Completion requires both the
scope/job/fence-bound one-time capability and an Ed25519 signature over the exact
JSON request bytes. Running-job cancellation deletes and observes absence of the
exact fence-derived JobSet before changing state or releasing outstanding budget.

Queue wait, pod startup, and active execution each have explicit upper bounds. The
launcher refuses configuration unless the staging capability lifetime covers their
sum plus launch, reconciliation, and expiry slack. A fail-closed background
reconciler reads only launcher-labelled JobSets, validates the exact job/fence and
lifecycle annotations, and treats queue/startup/active deadline expiry, failed
children, a terminal JobSet without a receipt, or a missing JobSet as failure. It
deletes and observes absence of the exact fence-derived object before changing job
state or releasing budget.

Workers complete through
`POST /internal/v1alpha1/jobs/{job_id}/complete`. This route authenticates the exact
worker receipt with `X-Mindclade-Completion-Capability`, not with gateway-derived
end-user identity. Receipt decoding rejects unknown fields and binds the admitted
model/input identities, selected serving revision, execution mode, sampler,
fencing token, result digest, size, and fence-specific receipt path before the
single atomic terminal transition. Capabilities are single-use because a completed,
failed, or cancelled attempt has its stored binding erased.

Before completion, the worker must call the separately signed
`result-upload-capability` route. The first call fixes the result digest and size for
the fence; exact retries receive fresh short-lived session authority so a lost
response or worker restart can resume safely, while a changed identity is rejected.
Each Rust-v2-compatible upload capability binds the scope, digest, size, session, and
nonce. Completion first revalidates that binding and then performs an authenticated
artifact-proxy HEAD request; the proxy re-hashes committed bytes and returns exact
metadata. Only a matching attestation permits the terminal transition. Jobs expose an
`artifact://sha256/<hex>` reference whose retrieval still requires an authorized
capability, never a cluster-internal PVC path.

The per-attempt PVC remains crash evidence/staging and is not the success authority.
This is a development reference: before serving, reconciliation terminates managed
JobSets that have no matching row, but the process-local store cannot reconstruct
lost jobs, events, completion keys, or API responses after restart. Old IDs are
therefore rejected or remain absent rather than resolving to newly created jobs.
Promotion still requires a durable transactional store and an informer/controller
implementation; the bounded development poller is not a durability claim.
