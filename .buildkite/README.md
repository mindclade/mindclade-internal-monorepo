# Buildkite pipeline

GitHub runs required repository checks. A separate main-branch push workflow records
a receipt when it dispatches the immutable revision to Buildkite; pull-request code
never receives that dispatch credential. Buildkite is the authoritative compute plane
for CPU, GPU, service, and release-evidence jobs. Release steps retain exact wheels and
local OCI archives with BuildKit provenance and SPDX SBOM evidence. Registry
publication and Cosign signing are deliberately outside this repository's authority
boundary.

Release evidence fails closed unless the build arrived through the API dispatcher,
contains its explicit trust signal, identifies a safe GitHub event on `refs/heads/main`,
is not a pull request, and binds the dispatched GitHub SHA to `BUILDKITE_COMMIT`.
`just service-check` runs all Go tests under the race detector; the release step depends
on that service gate.
