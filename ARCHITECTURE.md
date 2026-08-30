# Architecture

Mindclade's Q0 slice separates definitions, execution libraries, control-plane
state, and artifact bytes by contract:

```text
client -> runtime gateway -> control plane -> Kueue / JobSet -> inference worker
                                                |                    |
                                                +-- artifact proxy --+

training ------> mindclade.models <------ mindclade.inference
```

The gateway validates OIDC identity and path scope. The development control plane
owns idempotency, admission, job transitions, cancellation, and attempt fencing in a
process-local reference store. Workers receive immutable model and input digests,
verify a signed local bundle, and atomically produce a digest-addressed result. The
artifact proxy grants no ambient access: each operation requires a short-lived
Ed25519 capability for one tenant, project, digest, and operation.

The model, training, inference, and generated-protocol wheels coexist in the PEP 420
`mindclade` namespace. Models never import execution policy. Training and inference
may depend on models but never on one another or on generated clients. See
[dependency law](docs/architecture/dependency-law.md).

## Evidence boundary

Q0 exercises contracts and production-shaped failure semantics on synthetic data. It
does not contain weights, real biological datasets, benchmark claims, production
Postgres/GCS adapters, restart-durable control state, or production deployment
approval. GPU, multi-node, and live cloud qualification require separately captured
evidence.
