# Repository instructions

## Scope and safety

- Never deploy, publish, provision cloud resources, change IAM, or access live GCP or
  Kubernetes environments from repository commands.
- Never commit credentials, model weights, checkpoints, customer tensors, or generated
  datasets.
- Use `uv` as the only Python dependency resolver and `Bazel` as the canonical build
  graph. Do not add requirements files by hand.

## Package boundaries

- `mindclade.models` owns model configuration, tensor contracts, architecture,
  serialization, model bundles, and model-definition registration.
- `mindclade.training` may depend on models and owns optimization, distributed
  execution, checkpoint/resume, recipes, and trainability qualification.
- `mindclade.inference` may depend on models and owns inference orchestration,
  batching, adaptive compute, execution modes, and result artifacts.
- Models must not import training or inference. Libraries must not import services,
  workers, deployment, or generated clients.

## Verification

- Run the narrowest affected tests first, then `just check` and `just test-cpu`.
- Keep tests deterministic by controlling time, seeds, network, and temporary paths.
- GPU evidence must name the accelerator, CUDA, PyTorch, dtype, and world size.
