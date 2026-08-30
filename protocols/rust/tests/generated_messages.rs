//! Smoke tests for the checked-in prost and tonic bindings.

use mindclade_protocols::mindclade::{artifact, common, inference, job};
use prost::Message;

#[test]
fn inference_request_round_trips_across_generated_packages() {
    let request = inference::v1alpha1::SubmitInferenceRequest {
        context: Some(common::v1alpha1::CommandContext {
            scope: Some(common::v1alpha1::Scope {
                tenant_id: "tenant-a".to_owned(),
                project_id: "project-a".to_owned(),
            }),
            request_id: "request-0001".to_owned(),
            idempotency_key: "test-key".to_owned(),
            principal_subject: "service:runtime-gateway".to_owned(),
            deadline: Some(prost_types::Timestamp {
                seconds: 1_800_000_000,
                nanos: 0,
            }),
        }),
        model_digest: "sha256:model".to_owned(),
        options: Some(inference::v1alpha1::InferenceOptions {
            seed: i64::MAX as u64,
            diffusion_steps: 128,
            recycles: 3,
            precision: "bfloat16".to_owned(),
        }),
        input: Some(
            inference::v1alpha1::submit_inference_request::Input::InputArtifact(
                artifact::v1alpha1::ArtifactReference {
                    digest: "sha256:input".to_owned(),
                    size_bytes: 42,
                    media_type: "application/vnd.mindclade.tensor-set+protobuf".to_owned(),
                },
            ),
        ),
    };

    let encoded = request.encode_to_vec();
    let decoded = inference::v1alpha1::SubmitInferenceRequest::decode(encoded.as_slice())
        .expect("generated request should decode");

    assert_eq!(decoded, request);
}

#[test]
fn job_result_fields_round_trip() {
    let message = job::v1alpha1::Job {
        name: Some(common::v1alpha1::ResourceName {
            scope: Some(common::v1alpha1::Scope {
                tenant_id: "tenant-a".to_owned(),
                project_id: "project-a".to_owned(),
            }),
            kind: "jobs".to_owned(),
            resource_id: "job-11111111111111111111111111111111-00000001".to_owned(),
        }),
        state: job::v1alpha1::JobState::Succeeded as i32,
        model_digest: "sha256:model".to_owned(),
        input: None,
        result: None,
        estimated_gpu_milliseconds: 2_000,
        fencing_token: 7,
        created_at: None,
        updated_at: None,
        seed: 17,
        diffusion_steps: 64,
        failure_code: String::new(),
        result_artifact_uri:
            "artifact://sha256/cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
                .to_owned(),
    };

    let decoded = job::v1alpha1::Job::decode(message.encode_to_vec().as_slice())
        .expect("generated job should decode");

    assert_eq!(decoded, message);
    assert_eq!(decoded.state(), job::v1alpha1::JobState::Succeeded);
}

#[test]
fn generated_grpc_clients_are_exported() {
    use artifact::v1alpha1::artifact_service_client::ArtifactServiceClient;
    use inference::v1alpha1::inference_service_client::InferenceServiceClient;

    let artifact_client: Option<ArtifactServiceClient<tonic::transport::Channel>> = None;
    let inference_client: Option<InferenceServiceClient<tonic::transport::Channel>> = None;

    assert!(artifact_client.is_none());
    assert!(inference_client.is_none());
}
