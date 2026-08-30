# Model runtime deployment

This package renders the development environment only. All images are immutable
digests, pods use the restricted security profile, and artifact-capable workloads use
GKE Workload Identity. Kueue admits suspended JobSets onto the L4 resource flavor.

The control plane creates attempt resources through namespace-scoped RBAC. No static
queue-labelled JobSet or bearer manifest is rendered: each attempt uses an immutable
owner-bound Secret and an independently retained PVC before becoming visible to
Kueue. The only projected Kubernetes token is a one-hour token mounted into the
control plane; every other workload has ambient token mounting disabled.

Development queue wait is capped at 15 minutes, worker startup at 5 minutes, and
active execution at 60 minutes. The 90-minute staging capability lifetime exceeds
that combined window plus launch/reconciliation slack. The control plane lists only
its labelled JobSets and fails closed if its reconciliation loop cannot observe
status. At startup it deletes and confirms absence of orphan managed JobSets; because
the job store is in memory, this stops stale compute but cannot restore lost job API
state or completion authority.

The development artifact proxy stores committed objects on
`artifact-store-development`, a single-writer persistent volume. Workers publish
results there before completion, and the control plane has narrowly scoped network
access for authenticated metadata verification. This survives pod replacement but is
not a replicated production storage design.

The deployment operator must provide, out of band:

- `control-plane-signing-private-keys` with PKCS#8 PEM files named
  `scheduler-private.pem` and `artifact-capability-private.pem`;
- `worker-trusted-public-keys` with the matching scheduler public key and catalog
  bundle public keys plus `keyring.json`;
- `artifact-signing-public-keys` with `public-keys.json` containing the matching
  artifact capability public key; and
- the catalog's digest-exact model bundle archives and input artifacts preloaded in
  the development artifact store (the manifests do not ingest or synthesize bytes);
- the OIDC and internal-identity secrets already named by the base manifests.

Key IDs in those documents must match the deployment environment and development
catalog. There is intentionally no `apply`, production overlay, secret generation, or
cloud provisioning command. Promotion requires signed model and image digests, a
durable catalog/job store, a result-retention controller, and separate operational
approval.
