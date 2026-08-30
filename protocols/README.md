# Protocol contracts

The `v1alpha1` contracts define tenant-scoped asynchronous inference, immutable
digest-addressed artifacts, model releases, and resumable event delivery. The
checked semantic fixture keeps the protobuf and REST/SSE representations aligned
with the executable service bounds. Deprecated protobuf fields remain solely for
wire compatibility and are not promises of executable HTTP behavior.

The current `v1alpha1` REST job identity grammar is intentionally strict and
restart-safe: `job-<32 lowercase hex>-<8 decimal digits>`. Earlier sequential-only
examples are not accepted and cannot alias jobs created by a restarted volatile
development store. The semantic fixture binds both the path parameter and returned
`Job.id` schema to this canonical grammar.

Run `buf lint` before changing a protobuf contract and validate changes against the
checked-in compatibility baselines. Generated clients are build outputs and are not
hand-edited.

## Checked client deliverables

- Python is distributed as the private `mindclade-protocols` wheel. It declares
  the exact generated-code runtime floors (`protobuf>=7.36,<8` and
  `grpcio>=1.78,<2`) and contributes PEP 420 packages such as
  `mindclade.job.v1alpha1`. The runtime's regular `mindclade.inference` package
  extends its path so `mindclade.inference.v1alpha1` coexists in the same
  environment. Empty `_pb2_grpc.py` placeholders for schemas with no service
  definition are intentionally omitted from the wheel.
- Rust is exposed by the workspace crate `mindclade-protocols`, which wraps all
  checked prost/tonic modules under `mindclade::<domain>::v1alpha1`.
- Go clients remain packages in the repository's root Go module. `just
  service-check` compiles every generated Go package with `go test ./...` and
  follows with `go vet ./...`; no separate Go module is published.

`//protocols:python_client_test` and `//protocols:rust_client_test` compile and
exercise the packaged generated clients. `just build-wheels && just check-wheels`
also inspects all wheel contents and performs an import/serialization smoke test
from extracted wheels. Regenerate checked outputs only after a source contract or
generator pin changes.

The generation-drift assertion resolves the exact remote plugins pinned in
`buf.gen.yaml` and therefore requires access to the Buf Schema Registry. The
deterministic `just test-cpu` lane excludes only that `network`-marked assertion;
`just protocol-check` and its CI gate continue to require it.
