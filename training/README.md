# mindclade-training

`mindclade-training` is the internal, typed training layer for Mindclade model
packages. It provides serializable training contracts, deterministic synthetic
fixtures, a small single-process PyTorch reference engine, FSDP2 capability
adapters, and integrity-checked Distributed Checkpoint (DCP) commits.

The package is intentionally model-light: `mindclade.models` is its only model
dependency, and neither models nor inference imports this package.

## Local smoke test

```bash
uv run pytest -m "not gpu and not distributed" training/tests
```

The selected tests are CPU-only, require no network access or model artifacts, and
leave production/cloud state untouched. Separate `gpu` and `distributed` markers
are exercised only by the accelerator qualification lane. The supported runtime is
Python 3.12 with PyTorch 2.13. Accelerator-specific PyTorch wheels are selected by
the workspace lock profiles, not by this package.

## CLI

```bash
mindclade-training inspect training/src/mindclade/training/recipes/smoke/cpu_contract.yaml
mindclade-training run training/src/mindclade/training/recipes/smoke/cpu_contract.yaml --output ./runs/smoke
mindclade-training resume ./runs/smoke/checkpoints/step-00000004 --recipe training/src/mindclade/training/recipes/smoke/cpu_contract.yaml
```

The bundled recipes use a randomly initialized tiny CladeFold model and
synthetic data. They are engineering fixtures only and make no scientific or
biological capability claim.
