set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default:
    @just --list

sync:
    uv sync --all-packages --locked

format:
    uv run ruff format models training inference workers protocols deploy tests tools .buildkite
    uv run ruff check --fix models training inference workers protocols deploy tests tools .buildkite

check:
    uv run python tools/bazel/export_uv_lock.py --check
    uv run ruff format --check models training inference workers protocols deploy tests tools .buildkite
    uv run ruff check models training inference workers protocols deploy tests tools .buildkite
    uv run mypy -p mindclade.models -p mindclade.training -p mindclade.inference
    MYPYPATH=workers/inference_worker/python uv run mypy -p mindclade_inference_worker
    uv run python tools/repo/dependency_policy.py
    uv run python tools/repo/component_metadata.py

protocol-check:
    buf format --diff --exit-code
    buf lint
    buf breaking --against protocols/compatibility/baselines/protobuf.binpb
    uv run pytest protocols/compatibility/tests

service-check:
    go test -race ./...
    go vet ./...
    cargo test --workspace --all-targets --locked
    cargo clippy --workspace --all-targets --locked -- -D warnings
    uv run python deploy/policies/verify.py

test-cpu:
    uv run pytest -m "not gpu and not distributed and not nightly and not network"

test-models:
    uv run pytest models/tests models/src/mindclade/models/families/clade/cladefold/tests

test-training:
    uv run pytest training/tests -m "not gpu and not distributed"

test-inference:
    uv run pytest inference/tests -m "not gpu"

build-wheels:
    uv build --package mindclade-models --wheel --out-dir dist --no-sources
    uv build --package mindclade-training --wheel --out-dir dist --no-sources
    uv build --package mindclade-inference --wheel --out-dir dist --no-sources
    uv build --package mindclade-protocols --wheel --out-dir dist --no-sources

check-wheels:
    uv run check-wheel-contents dist/*.whl
    uv run python protocols/python/wheel_smoke.py --wheel-dir dist

build-images *args:
    uv run python tools/release/build_images.py {{args}}

bazel-test:
    if command -v bazelisk >/dev/null 2>&1; then bazelisk test //...; else bazel test //...; fi

render-deploy:
    uv run pytest deploy/tests/test_deterministic_render.py
