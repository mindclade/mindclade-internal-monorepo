# Artifact proxy

The proxy mediates worker access to immutable model and tensor artifacts. Every
operation requires a short-lived Ed25519 capability scoped to one tenant, project,
digest, and operation. Uploads are offset checked, length checked, SHA-256 verified,
and atomically exposed only after commit.

Authenticated `HEAD /v1alpha1/artifacts/{digest}` uses the same exact download
capability and re-hashes the committed bytes before returning digest and size headers.
The control plane uses that attestation before accepting a result completion.

The development filesystem backend is mounted on a persistent volume so committed
objects survive proxy pod replacement, but it is single-replica and not a production
durability claim. A production GCS backend must preserve these semantics and use
Workload Identity; the proxy never accepts static cloud credentials.
