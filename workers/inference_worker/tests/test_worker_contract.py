from __future__ import annotations

import base64
import dataclasses
import io
import json
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request

import pytest
import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from mindclade.inference.contracts.result_contract import InferenceCandidate
from mindclade.models.api.serialization import decode_safetensors
from mindclade.models.packaging.bundle_manifest import BundleFile, BundleManifest
from mindclade.models.packaging.bundle_signing import SignatureEnvelope
from mindclade_inference_worker.completion import publish_completion
from mindclade_inference_worker.contracts import JobManifest, ResultReceipt
from mindclade_inference_worker.io import sha256_bytes, sha256_file
from mindclade_inference_worker.publication import publish_result_artifact
from mindclade_inference_worker.runner import (
    ExecutionProduct,
    WorkerRoots,
    _encode_ranked_candidates,
    execute_job,
)
from mindclade_inference_worker.staging import stage_job
from mindclade_inference_worker.trust import TrustedKeyring

TEST_JOB_ID = "job-11111111111111111111111111111111-00000001"


class FakeExecutor:
    def run(self, manifest: JobManifest) -> ExecutionProduct:
        return ExecutionProduct(
            tensor_bytes=b"safe-tensor-result",
            request_fingerprint="sha256:" + "e" * 64,
            selected_candidate_id="candidate-0000",
            execution_mode="eager",
            sampler_digest="sha256:" + "f" * 64,
            diagnostics={"steps": manifest.num_steps},
        )


class FakeResponse(io.BytesIO):
    def __init__(self, value: bytes = b"", *, status: int = 200) -> None:
        super().__init__(value)
        self.status = status


@dataclasses.dataclass(frozen=True)
class WorkerTestEnvironment:
    manifest: JobManifest
    trust: TrustedKeyring
    roots: WorkerRoots
    scheduler_signer: Ed25519PrivateKey


def environment(tmp_path: Path) -> WorkerTestEnvironment:
    scheduler_signer = Ed25519PrivateKey.generate()
    bundle_key = Ed25519PrivateKey.generate()
    artifact_root = tmp_path / "artifacts"
    bundle = artifact_root / "bundle"
    bundle.mkdir(parents=True)
    config = bundle / "config.json"
    config.write_text('{"model_type":"cladefold"}', encoding="utf-8")
    inner = BundleManifest(
        model_type="cladefold",
        architecture_version="q0",
        source_revision="a" * 40,
        config_digest=sha256_file(config),
        files=(
            BundleFile(
                path="config.json",
                digest=sha256_file(config),
                size_bytes=config.stat().st_size,
                media_type="application/json",
            ),
        ),
        capability_claim_refs=(),
    )
    manifest_path = bundle / "bundle.manifest.json"
    manifest_path.write_bytes(inner.canonical_bytes())
    envelope = SignatureEnvelope(
        key_id="bundle-key", signature=bundle_key.sign(inner.canonical_bytes())
    )
    (bundle / "bundle.manifest.sig.json").write_text(
        json.dumps(envelope.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    input_path = artifact_root / "input.safetensors"
    input_path.write_bytes(b"input")
    identity = sha256_file(manifest_path)
    unsigned = JobManifest(
        job_id=TEST_JOB_ID,
        tenant_id="tenant-a",
        project_id="project-a",
        model_digest=identity,
        bundle_manifest_digest=identity,
        bundle_archive_digest="sha256:" + "b" * 64,
        input_digest=sha256_file(input_path),
        serving_revision_digest="sha256:" + "d" * 64,
        bundle_path=bundle,
        input_path=input_path,
        output_directory=tmp_path / "results",
        bundle_signing_key_id="bundle-key",
        scheduler_signing_key_id="scheduler-key",
        manifest_signature=base64.b64encode(bytes(64)).decode("ascii"),
        bundle_download_capability="bundle-capability",
        input_download_capability="input-capability",
        completion_capability="completion-capability",
        completion_signing_private_key=base64.b64encode(bytes(range(32))).decode("ascii"),
        fencing_token=9,
        seed=7,
        num_steps=2,
    )
    manifest = sign(unsigned, scheduler_signer)
    trust = TrustedKeyring(
        scheduler_keys={"scheduler-key": scheduler_signer.public_key()},
        bundle_keys={"bundle-key": bundle_key.public_key()},
    )
    roots = WorkerRoots(
        artifact_root=artifact_root,
        result_root=tmp_path / "results",
    )
    return WorkerTestEnvironment(manifest, trust, roots, scheduler_signer)


def sign(manifest: JobManifest, key: Ed25519PrivateKey) -> JobManifest:
    signature = base64.b64encode(key.sign(manifest.canonical_unsigned_bytes())).decode("ascii")
    return dataclasses.replace(manifest, manifest_signature=signature)


def run(env: WorkerTestEnvironment, manifest: JobManifest | None = None) -> ResultReceipt:
    return execute_job(
        manifest or env.manifest,
        FakeExecutor(),
        trust=env.trust,
        roots=env.roots,
    )


def test_execute_job_commits_digest_and_fence_receipt(tmp_path: Path) -> None:
    env = environment(tmp_path)
    receipt = run(env)
    result = env.roots.result_root / "result.fence-9.safetensors"
    receipt_path = env.roots.result_root / "result.fence-9.receipt.json"
    assert receipt.fencing_token == 9
    assert receipt.result_digest == sha256_file(result)
    document = json.loads(receipt_path.read_text())
    assert document["scientific_claim"] is None
    assert document["execution_mode"] == "eager"
    assert "batch_seeds" not in receipt.to_dict()
    rendered = json.dumps(document).lower()
    assert "coordinates" not in rendered
    assert "safe-tensor-result" not in rendered


def test_worker_safetensors_preserve_single_batch_seed_ergonomics() -> None:
    maximum_seed = (1 << 63) - 1
    candidate = InferenceCandidate(
        candidate_id="candidate-0000",
        coordinates=torch.zeros((1, 3, 3)),
        confidence=0.5,
        calibrated_confidence=0.5,
        batch_seeds=(maximum_seed,),
        steps=2,
    )
    payload = _encode_ranked_candidates(
        SimpleNamespace(candidates=(candidate,), selected=candidate)
    )
    tensors = decode_safetensors(payload)

    assert candidate.seed == maximum_seed
    assert tensors["candidates.0000.batch_seeds"].tolist() == [maximum_seed]
    assert tensors["selected.batch_seeds"].tolist() == [maximum_seed]


def test_tampered_manifest_fails_authentication(tmp_path: Path) -> None:
    env = environment(tmp_path)
    tampered = dataclasses.replace(env.manifest, num_steps=3)
    with pytest.raises(ValueError, match="signature verification"):
        run(env, tampered)


def test_untrusted_scheduler_key_is_rejected(tmp_path: Path) -> None:
    env = environment(tmp_path)
    untrusted = dataclasses.replace(env.manifest, scheduler_signing_key_id="attacker")
    with pytest.raises(ValueError, match="not trusted"):
        run(env, untrusted)


def test_model_digest_is_bound_to_verified_inner_manifest(tmp_path: Path) -> None:
    env = environment(tmp_path)
    unbound = dataclasses.replace(env.manifest, model_digest="sha256:" + "1" * 64)
    unbound = sign(unbound, env.scheduler_signer)
    with pytest.raises(ValueError, match="verified bundle manifest"):
        run(env, unbound)


def test_path_traversal_is_rejected_after_authentication(tmp_path: Path) -> None:
    env = environment(tmp_path)
    traversing = dataclasses.replace(
        env.manifest, bundle_path=env.roots.artifact_root / ".." / "outside"
    )
    traversing = sign(traversing, env.scheduler_signer)
    with pytest.raises(ValueError, match="traversal"):
        run(env, traversing)


def test_symlinked_input_is_rejected(tmp_path: Path) -> None:
    env = environment(tmp_path)
    symlink = env.roots.artifact_root / "linked-input.safetensors"
    symlink.symlink_to(env.manifest.input_path)
    linked = sign(dataclasses.replace(env.manifest, input_path=symlink), env.scheduler_signer)
    with pytest.raises(ValueError, match="symlink"):
        run(env, linked)


def test_stale_fence_outputs_are_isolated_and_never_overwritten(tmp_path: Path) -> None:
    env = environment(tmp_path)
    first = run(env)
    stale = sign(dataclasses.replace(env.manifest, fencing_token=8), env.scheduler_signer)
    second = run(env, stale)
    assert first.result_manifest_path != second.result_manifest_path
    assert (env.roots.result_root / "result.fence-9.safetensors").read_bytes() == (
        b"safe-tensor-result"
    )
    assert (env.roots.result_root / "result.fence-8.safetensors").is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run(env)


def test_input_digest_mismatch_fails_before_execution(tmp_path: Path) -> None:
    env = environment(tmp_path)
    env.manifest.input_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="input tensor artifact digest"):
        run(env)


def test_manifest_rejects_mutable_and_unknown_fields(tmp_path: Path) -> None:
    env = environment(tmp_path)
    value = dataclass_dict(env.manifest)
    value["job_id"] = "job-00000001"
    with pytest.raises(ValueError, match="restart-unique"):
        JobManifest.from_dict(value)
    value = dataclass_dict(env.manifest)
    value["model_digest"] = "latest"
    with pytest.raises(ValueError, match="immutable sha256"):
        JobManifest.from_dict(value)
    value = dataclass_dict(env.manifest)
    value["credential"] = "must-not-be-accepted"
    with pytest.raises(ValueError, match="unknown job manifest"):
        JobManifest.from_dict(value)


def test_completion_uses_capability_and_fenced_receipt(tmp_path: Path) -> None:
    env = environment(tmp_path)
    receipt = run(env)
    requests: list[Request] = []

    def accept(request: Request, timeout: float) -> FakeResponse:
        requests.append(request)
        assert timeout == 10.0
        return FakeResponse(status=204)

    publish_completion("http://control-plane:8081", env.manifest, receipt, open_url=accept)
    assert requests[0].full_url.endswith(f"/internal/v1alpha1/jobs/{TEST_JOB_ID}/complete")
    assert requests[0].get_header("X-mindclade-completion-capability") == ("completion-capability")
    payload = requests[0].data or b"{}"
    assert json.loads(payload)["fencing_token"] == 9
    completion_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    completion_key.public_key().verify(
        base64.b64decode(requests[0].get_header("X-mindclade-completion-signature")),
        payload,
    )


def test_rejected_fenced_completion_is_not_success(tmp_path: Path) -> None:
    env = environment(tmp_path)
    receipt = run(env)

    def reject(_request: Request, _timeout: float) -> FakeResponse:
        return FakeResponse(status=409)

    with pytest.raises(RuntimeError, match="rejected the fenced completion"):
        publish_completion("http://control-plane:8081", env.manifest, receipt, open_url=reject)


def test_result_is_authorized_uploaded_and_committed_before_completion(tmp_path: Path) -> None:
    env = environment(tmp_path)
    receipt = run(env)
    requests: list[Request] = []
    session_id = "upload-session-0001"
    result_bytes = b"safe-tensor-result"

    def accept(request: Request, timeout: float) -> FakeResponse:
        requests.append(request)
        index = len(requests)
        if index == 1:
            assert timeout == 10.0
            assert request.full_url.endswith(
                f"/internal/v1alpha1/jobs/{TEST_JOB_ID}/result-upload-capability"
            )
            payload = request.data or b"{}"
            completion_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
            completion_key.public_key().verify(
                base64.b64decode(request.get_header("X-mindclade-completion-signature")),
                payload,
            )
            assert json.loads(payload) == {
                "fencing_token": 9,
                "job_id": TEST_JOB_ID,
                "project_id": "project-a",
                "result_digest": receipt.result_digest,
                "result_size_bytes": len(result_bytes),
                "schema_version": "v1alpha1",
                "tenant_id": "tenant-a",
            }
            return FakeResponse(
                json.dumps(
                    {
                        "upload_capability": "encoded-upload-capability",
                        "session_id": session_id,
                    }
                ).encode(),
                status=201,
            )
        assert timeout == 30.0
        assert request.get_header("X-mindclade-capability") == ("encoded-upload-capability")
        if index == 2:
            assert request.get_method() == "POST"
            assert request.full_url.endswith("/v1alpha1/uploads")
            assert json.loads(request.data or b"{}") == {
                "digest": receipt.result_digest,
                "project_id": "project-a",
                "session_id": session_id,
                "size_bytes": len(result_bytes),
                "tenant_id": "tenant-a",
            }
            return FakeResponse(
                json.dumps({"upload_id": session_id, "committed_bytes": 0}).encode(),
                status=201,
            )
        if index == 3:
            assert request.get_method() == "PUT"
            assert request.full_url.endswith(f"/v1alpha1/uploads/{session_id}?offset=0")
            assert request.data == result_bytes
            return FakeResponse(
                json.dumps({"upload_id": session_id, "committed_bytes": len(result_bytes)}).encode()
            )
        assert index == 4
        assert request.get_method() == "POST"
        assert request.full_url.endswith(f"/v1alpha1/uploads/{session_id}/commit")
        return FakeResponse(json.dumps({"digest": receipt.result_digest}).encode())

    publish_result_artifact(
        "http://control-plane:8081",
        "http://artifact-proxy:8082",
        env.manifest,
        receipt,
        result_root=env.roots.result_root,
        open_url=accept,
    )
    assert len(requests) == 4


def test_result_publication_rejects_unexpected_commit_identity(tmp_path: Path) -> None:
    env = environment(tmp_path)
    receipt = run(env)
    session_id = "upload-session-0001"
    count = 0

    def wrong_commit(_request: Request, _timeout: float) -> FakeResponse:
        nonlocal count
        count += 1
        if count == 1:
            return FakeResponse(
                json.dumps(
                    {
                        "upload_capability": "encoded-upload-capability",
                        "session_id": session_id,
                    }
                ).encode(),
                status=201,
            )
        if count == 2:
            return FakeResponse(
                json.dumps({"upload_id": session_id, "committed_bytes": 0}).encode(),
                status=201,
            )
        if count == 3:
            return FakeResponse(
                json.dumps(
                    {
                        "upload_id": session_id,
                        "committed_bytes": receipt.result_size_bytes,
                    }
                ).encode()
            )
        return FakeResponse(json.dumps({"digest": "sha256:" + "0" * 64}).encode())

    with pytest.raises(ValueError, match="unexpected result identity"):
        publish_result_artifact(
            "http://control-plane:8081",
            "http://artifact-proxy:8082",
            env.manifest,
            receipt,
            result_root=env.roots.result_root,
            open_url=wrong_commit,
        )


def test_artifact_stager_downloads_and_verifies_exact_inputs(tmp_path: Path) -> None:
    env = environment(tmp_path)
    archive = bundle_archive(env.manifest.bundle_path)
    input_bytes = env.manifest.input_path.read_bytes()
    shutil.rmtree(env.manifest.bundle_path)
    env.manifest.input_path.unlink()
    staged_manifest = dataclasses.replace(
        env.manifest,
        bundle_archive_digest=sha256_bytes(archive),
    )
    staged_manifest = sign(staged_manifest, env.scheduler_signer)

    def download(request: Request, timeout: float) -> FakeResponse:
        assert timeout == 30.0
        if staged_manifest.bundle_archive_digest in request.full_url:
            assert request.get_header("X-mindclade-capability") == "bundle-capability"
            return FakeResponse(archive)
        assert request.get_header("X-mindclade-capability") == "input-capability"
        return FakeResponse(input_bytes)

    stage_job(
        staged_manifest,
        trust=env.trust,
        artifact_root=env.roots.artifact_root,
        artifact_proxy_url="http://artifact-proxy:8082",
        open_url=download,
    )
    assert staged_manifest.bundle_path.is_dir()
    assert sha256_file(staged_manifest.input_path) == staged_manifest.input_digest


def bundle_archive(bundle: Path) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for path in sorted(bundle.iterdir()):
            value = path.read_bytes()
            member = tarfile.TarInfo(path.name)
            member.size = len(value)
            member.mtime = 0
            archive.addfile(member, io.BytesIO(value))
    return output.getvalue()


def dataclass_dict(manifest: JobManifest) -> dict[str, object]:
    value = {field: getattr(manifest, field) for field in manifest.__dataclass_fields__}
    for field in ("bundle_path", "input_path", "output_directory"):
        value[field] = str(value[field])
    return value
