"""Validated execution from a local, artifact-proxy-staged job manifest."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from .contracts import JobManifest, ResultReceipt
from .io import (
    atomic_bytes_exclusive,
    atomic_json_exclusive,
    prepare_directory,
    require_clean_tree,
    resolve_existing_path,
    sha256_bytes,
    sha256_file,
)
from .trust import TrustedKeyring


@dataclasses.dataclass(frozen=True, slots=True)
class ExecutionProduct:
    """Safe tensor payload plus its non-sensitive execution metadata."""

    tensor_bytes: bytes
    request_fingerprint: str
    selected_candidate_id: str
    execution_mode: str
    sampler_digest: str
    diagnostics: Mapping[str, Any] = dataclasses.field(default_factory=dict)


class Executor(Protocol):
    def run(self, manifest: JobManifest) -> ExecutionProduct: ...


@dataclasses.dataclass(frozen=True, slots=True)
class WorkerRoots:
    """Deployment-owned roots that a signed manifest is allowed to address."""

    artifact_root: Path
    result_root: Path


def execute_job(
    manifest: JobManifest,
    executor: Executor,
    *,
    trust: TrustedKeyring,
    roots: WorkerRoots,
) -> ResultReceipt:
    """Verify staged inputs, execute once, and atomically commit a result receipt."""

    trust.verify_job(manifest)
    bundle_path = resolve_existing_path(roots.artifact_root, manifest.bundle_path, directory=True)
    input_path = resolve_existing_path(roots.artifact_root, manifest.input_path, directory=False)
    require_clean_tree(bundle_path)
    inner_manifest = resolve_existing_path(
        bundle_path, bundle_path / "bundle.manifest.json", directory=False
    )
    bundle_identity = sha256_file(inner_manifest)
    if bundle_identity != manifest.bundle_manifest_digest:
        raise ValueError("model bundle manifest digest does not match the job")
    _verify_bundle(bundle_path, manifest, trust)
    if manifest.model_digest != bundle_identity:
        raise ValueError("model_digest must identify the verified bundle manifest")
    if sha256_file(input_path) != manifest.input_digest:
        raise ValueError("input tensor artifact digest does not match the job")
    verified_manifest = dataclasses.replace(
        manifest, bundle_path=bundle_path, input_path=input_path
    )
    product = executor.run(verified_manifest)
    if product.execution_mode != "eager":
        raise ValueError("worker execution provenance must report the eager runtime")
    result_digest = sha256_bytes(product.tensor_bytes)
    output_directory = prepare_directory(roots.result_root, manifest.output_directory)
    fence = manifest.fencing_token
    result_path = output_directory / f"result.fence-{fence}.safetensors"
    receipt_path = output_directory / f"result.fence-{fence}.receipt.json"
    atomic_bytes_exclusive(result_path, product.tensor_bytes)
    receipt = ResultReceipt(
        job_id=manifest.job_id,
        tenant_id=manifest.tenant_id,
        project_id=manifest.project_id,
        model_digest=manifest.model_digest,
        input_digest=manifest.input_digest,
        serving_revision_digest=manifest.serving_revision_digest,
        result_digest=result_digest,
        result_size_bytes=len(product.tensor_bytes),
        result_manifest_path=str(receipt_path),
        fencing_token=manifest.fencing_token,
        request_fingerprint=product.request_fingerprint,
        selected_candidate_id=product.selected_candidate_id,
        execution_mode=product.execution_mode,
        sampler_digest=product.sampler_digest,
    )
    atomic_json_exclusive(
        receipt_path,
        {
            **receipt.to_dict(),
            "tensor_file": result_path.name,
            "diagnostics": dict(product.diagnostics),
            "scientific_claim": None,
        },
    )
    return receipt


class MindcladeModelExecutor:
    """Production adapter over the models and inference wheel APIs."""

    def __init__(self, trust: TrustedKeyring) -> None:
        self._trust = trust

    def run(self, manifest: JobManifest) -> ExecutionProduct:
        import torch
        from mindclade.inference import InferenceRequest
        from mindclade.inference.pipeline import (
            ModelExecutor,
            ModelResolver,
            ResolvedModel,
            build_candidates,
            preprocess_request,
        )
        from mindclade.inference.ranking.candidate_ranker import CandidateRanker
        from mindclade.inference.sampling.deterministic_sampler import DeterministicModelSampler
        from mindclade.models import CladeFoldBatch, CladeFoldModel
        from mindclade.models.api.serialization import decode_safetensors
        from mindclade.models.packaging.bundle_signing import Ed25519PublicKeyVerifier
        from mindclade.models.packaging.model_bundle import ModelBundle

        public_key = self._trust.bundle_key(manifest.bundle_signing_key_id)
        verifier = Ed25519PublicKeyVerifier(public_key, manifest.bundle_signing_key_id)
        model = ModelBundle.load_model(
            manifest.bundle_path,
            CladeFoldModel,
            verifier=verifier,
        )
        selected_device = manifest.device
        if selected_device == "auto":
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        if selected_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("the admitted CUDA worker has no visible CUDA device")
        device = torch.device(selected_device)
        model = model.to(device).eval()
        tensors = {
            name: tensor.to(device)
            for name, tensor in decode_safetensors(manifest.input_path.read_bytes()).items()
        }
        request = InferenceRequest(
            request_id=manifest.job_id,
            tenant_id=manifest.tenant_id,
            project_id=manifest.project_id,
            model_digest=manifest.model_digest,
            inputs=tensors,
            seed=manifest.seed,
            num_samples=manifest.num_samples,
            num_steps=manifest.num_steps,
        )
        prepared = preprocess_request(request)
        if prepared.batch_size != 1:
            raise ValueError("one-job workers currently require batch size 1")
        resolver: ModelResolver[CladeFoldModel] = ModelResolver()
        resolver.register(
            ResolvedModel(
                model_digest=manifest.model_digest,
                model=model,
                batch_factory=CladeFoldBatch,
            )
        )
        sampler = DeterministicModelSampler(ModelExecutor(resolver))
        output = sampler.sample(prepared)
        candidates = build_candidates(
            output,
            atom_mask=tensors["atom_mask"],
            seeds=output.sample_seeds,
            steps=manifest.num_steps,
        )
        ranked = CandidateRanker().rank(candidates, request_fingerprint=request.fingerprint)
        return ExecutionProduct(
            tensor_bytes=_encode_ranked_candidates(ranked),
            request_fingerprint=request.fingerprint,
            selected_candidate_id=ranked.selected.candidate_id,
            execution_mode="eager",
            sampler_digest=sampler.digest,
            diagnostics={
                "device": device.type,
                "dtype": str(next(model.parameters()).dtype),
                "samples": manifest.num_samples,
                "steps": manifest.num_steps,
                "q0_reference_only": True,
            },
        )


def _encode_ranked_candidates(ranked: Any) -> bytes:
    """Serialize coordinates and their exact per-batch replay seeds together."""

    import torch
    from mindclade.models.api.serialization import encode_safetensors

    tensor_values: dict[str, torch.Tensor] = {}
    for index, candidate in enumerate(ranked.candidates):
        prefix = f"candidates.{index:04d}"
        tensor_values[f"{prefix}.coordinates"] = candidate.coordinates.detach().cpu()
        tensor_values[f"{prefix}.confidence"] = torch.tensor(
            [candidate.confidence, candidate.calibrated_confidence], dtype=torch.float32
        )
        tensor_values[f"{prefix}.batch_seeds"] = torch.tensor(
            candidate.batch_seeds, dtype=torch.int64
        )
    tensor_values["selected.coordinates"] = ranked.selected.coordinates.detach().cpu().clone()
    tensor_values["selected.batch_seeds"] = torch.tensor(
        ranked.selected.batch_seeds, dtype=torch.int64
    )
    return encode_safetensors(tensor_values)


def _verify_bundle(bundle_path: Path, manifest: JobManifest, trust: TrustedKeyring) -> None:
    from mindclade.models.packaging.bundle_signing import Ed25519PublicKeyVerifier
    from mindclade.models.packaging.model_bundle import ModelBundle

    key = trust.bundle_key(manifest.bundle_signing_key_id)
    verifier = Ed25519PublicKeyVerifier(key, manifest.bundle_signing_key_id)
    ModelBundle.verify(bundle_path, verifier=verifier)
