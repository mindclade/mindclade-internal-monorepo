# Dependency law

The three model-execution distributions and the generated-protocol distribution
share a PEP 420 `mindclade` namespace but have independent versions and release
lifecycles.

```text
mindclade.training  ──>  mindclade.models  <──  mindclade.inference
```

`models` owns tensor and serialization contracts. `training` and `inference` may
consume those contracts, but neither may import the other. All three execution
libraries are independent of services, workers, deployment manifests, and generated
RPC clients. The fourth wheel packages those generated clients without changing the
library dependency direction. The `tools/repo/dependency_policy.py` presubmit check
enforces this direction.

Runtime services communicate through versioned protocol messages and immutable
artifact references. They do not exchange Python model objects across process
boundaries.
