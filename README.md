# Mindclade internal monorepo

This repository implements the governed Mindclade model, training, inference, and
runtime foundations described by `BLUEPRINT.md`.

The initial model is **CladeFold Q0**, a random-initialized reference architecture.
It ships no pretrained weights and carries no biological accuracy, scientific,
clinical, or production fitness claim.

## Developer workflow

The authoritative Python resolver is [uv](https://docs.astral.sh/uv/). The default
developer flow is:

```sh
just sync
just check
just test-cpu
just build-wheels
```

`just build-images --image runtime-gateway` builds a selected service as a local
OCI archive; omit `--image` to build all four deployables. The image lane requires a
BuildKit-capable Docker daemon and emits SPDX, SLSA provenance, and digest evidence
under `dist/images/`. It never pushes or signs an external registry artifact.

GPU and distributed checks require the documented CUDA 13.0 execution profile and
are never run implicitly on a developer workstation.

No command in this repository provisions cloud infrastructure, deploys to a live
cluster, publishes packages, or loads credentials during import or tests.
