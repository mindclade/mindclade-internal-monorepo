from __future__ import annotations

from mindclade.artifact.v1alpha1 import artifact_service_pb2, artifact_service_pb2_grpc
from mindclade.common.v1alpha1 import identifiers_pb2
from mindclade.inference import InferenceRequest
from mindclade.inference.v1alpha1 import inference_pb2
from mindclade.job.v1alpha1 import job_pb2


class _RecordingChannel:
    def __init__(self) -> None:
        self.paths: list[str] = []

    def unary_unary(self, path: str, **_: object) -> object:
        self.paths.append(path)
        return lambda request: request

    def unary_stream(self, path: str, **_: object) -> object:
        self.paths.append(path)
        return lambda request: iter((request,))


def test_generated_messages_round_trip_beside_runtime_namespace() -> None:
    job = job_pb2.Job(
        name=identifiers_pb2.ResourceName(
            scope=identifiers_pb2.Scope(tenant_id="tenant-a", project_id="project-a"),
            kind="inference-job",
            resource_id="job-11111111111111111111111111111111-00000001",
        ),
        state=job_pb2.JOB_STATE_RUNNING,
        model_digest="sha256:" + "a" * 64,
        seed=7,
        diffusion_steps=16,
    )

    encoded = job.SerializeToString(deterministic=True)
    decoded = job_pb2.Job.FromString(encoded)

    assert decoded == job
    assert decoded.name.scope.tenant_id == "tenant-a"
    assert decoded.state == job_pb2.JOB_STATE_RUNNING
    assert InferenceRequest.__module__.startswith("mindclade.inference.contracts.")
    assert inference_pb2.InferenceOptions(seed=7, diffusion_steps=16).seed == 7


def test_generated_grpc_stub_exposes_streaming_download_without_network() -> None:
    channel = _RecordingChannel()
    stub = artifact_service_pb2_grpc.ArtifactServiceStub(channel)
    request = artifact_service_pb2.DownloadArtifactRequest(
        context={"scope": {"tenant_id": "tenant-a", "project_id": "project-a"}},
        digest="sha256:" + "b" * 64,
    )

    assert request.context.scope.project_id == "project-a"
    assert callable(stub.DownloadArtifact)
    assert channel.paths[-1] == ("/mindclade.artifact.v1alpha1.ArtifactService/DownloadArtifact")
