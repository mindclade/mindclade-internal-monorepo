from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import jsonschema
import pytest
import yaml
from mindclade.models.packaging.bundle_manifest import BundleManifest

ROOT = Path(__file__).parents[2]
REPOSITORY = ROOT.parent


def _document(path: str) -> object:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _require_buf() -> str:
    executable = shutil.which("buf")
    assert executable is not None, "buf is required for protocol qualification"
    return executable


def _proto_image(tmp_path: Path) -> dict[str, object]:
    output = tmp_path / "protobuf-image.json"
    subprocess.run(
        [_require_buf(), "build", "--exclude-imports", "--output", str(output)],
        cwd=REPOSITORY,
        check=True,
    )
    image = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(image, dict)
    return image


def _proto_files(image: dict[str, object], package: str) -> list[dict[str, object]]:
    files = image.get("file")
    assert isinstance(files, list)
    matching = [
        candidate
        for candidate in files
        if isinstance(candidate, dict) and candidate.get("package") == package
    ]
    if not matching:
        raise AssertionError(f"protobuf package is absent: {package}")
    return matching


def _proto_file(image: dict[str, object], package: str) -> dict[str, object]:
    files = _proto_files(image, package)
    assert len(files) == 1, f"protobuf package spans multiple files: {package}"
    return files[0]


def _proto_message(image: dict[str, object], qualified_name: str) -> dict[str, object]:
    package, name = qualified_name.rsplit(".", 1)
    for protobuf_file in _proto_files(image, package):
        messages = protobuf_file.get("messageType", [])
        assert isinstance(messages, list)
        for candidate in messages:
            if isinstance(candidate, dict) and candidate.get("name") == name:
                return candidate
    raise AssertionError(f"protobuf message is absent: {qualified_name}")


def _proto_service(image: dict[str, object], qualified_name: str) -> dict[str, object]:
    package, name = qualified_name.rsplit(".", 1)
    for protobuf_file in _proto_files(image, package):
        services = protobuf_file.get("service", [])
        assert isinstance(services, list)
        for candidate in services:
            if isinstance(candidate, dict) and candidate.get("name") == name:
                return candidate
    raise AssertionError(f"protobuf service is absent: {qualified_name}")


def _named_entries(container: dict[str, object], key: str) -> dict[str, dict[str, object]]:
    entries = container.get(key)
    assert isinstance(entries, list)
    result: dict[str, dict[str, object]] = {}
    for entry in entries:
        assert isinstance(entry, dict)
        name = entry.get("name")
        assert isinstance(name, str)
        result[name] = entry
    return result


def _runtime_source_if_available(relative_path: object) -> str | None:
    assert isinstance(relative_path, str)
    path = REPOSITORY / relative_path
    # The direct protocol gate runs from the repository and binds these
    # assertions to executable sources. Bazel's protocol-only sandbox omits
    # other packages; their compiled unit tests remain separate graph nodes.
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _flat_struct_source(source: str, declaration: str) -> str:
    start = source.index(declaration)
    end = source.index("\n}", start)
    return source[start:end]


def test_model_manifest_fixtures_match_runtime_contract() -> None:
    schema = _document("schemas/model_manifest/model_manifest.schema.json")
    positive = _document("schemas/model_manifest/positive.json")
    jsonschema.validate(positive, schema)
    assert isinstance(positive, dict)
    assert BundleManifest.from_dict(positive).to_dict() == positive
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_document("schemas/model_manifest/negative_checkpoint.json"), schema)
    invalid_ref = _document("schemas/model_manifest/negative_claim_ref.json")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid_ref, schema)
    assert isinstance(invalid_ref, dict)
    with pytest.raises(ValueError, match="capability claim reference"):
        BundleManifest.from_dict(invalid_ref)
    for name in ("negative_backslash_path.json", "negative_empty_path_segment.json"):
        invalid_path = _document(f"schemas/model_manifest/{name}")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid_path, schema)
        assert isinstance(invalid_path, dict)
        with pytest.raises(ValueError, match="bundle file path"):
            BundleManifest.from_dict(invalid_path)


def test_executable_semantics_match_openapi_and_protobuf(tmp_path: Path) -> None:
    contract = _document("compatibility/semantic_contract.json")
    assert isinstance(contract, dict)
    openapi = yaml.safe_load((ROOT / "openapi/external-api.yaml").read_text(encoding="utf-8"))
    image = _proto_image(tmp_path)

    submit_contract = contract["submit_inference"]
    assert isinstance(submit_contract, dict)
    submit_schema = openapi["components"]["schemas"][submit_contract["openapi_schema"]]
    idempotency_parameters = [
        parameter
        for parameter in openapi["paths"][
            "/tenants/{tenantId}/projects/{projectId}/inference-jobs"
        ]["post"]["parameters"]
        if isinstance(parameter, dict) and parameter.get("name") == "Idempotency-Key"
    ]
    assert len(idempotency_parameters) == 1
    idempotency_pattern = submit_contract["idempotency_key_pattern"]
    assert isinstance(idempotency_pattern, str)
    assert idempotency_parameters[0]["schema"]["pattern"] == idempotency_pattern
    options = _proto_message(image, submit_contract["proto_message"])
    option_fields = _named_entries(options, "field")
    numeric_fields = submit_contract["numeric_fields"]
    assert isinstance(numeric_fields, dict)
    for json_name, raw_expectation in numeric_fields.items():
        assert isinstance(json_name, str) and isinstance(raw_expectation, dict)
        json_property = submit_schema["properties"][json_name]
        assert json_property["minimum"] == raw_expectation["minimum"]
        assert json_property["maximum"] == raw_expectation["maximum"]
        proto_field = option_fields[raw_expectation["proto_field"]]
        assert proto_field["type"] == raw_expectation["proto_type"]
    inference_proto = (ROOT / "proto/mindclade/inference/v1alpha1/inference.proto").read_text(
        encoding="utf-8"
    )
    for raw_expectation in numeric_fields.values():
        assert isinstance(raw_expectation, dict)
        documented_range = (
            f"Inclusive valid range: {raw_expectation['minimum']}..{raw_expectation['maximum']}."
        )
        assert documented_range in inference_proto
    control_source = _runtime_source_if_available(submit_contract["runtime_source"])
    if control_source is not None:
        assert f"regexp.MustCompile(`{idempotency_pattern}`)" in control_source
        seed_limits = numeric_fields["seed"]
        step_limits = numeric_fields["diffusion_steps"]
        assert isinstance(seed_limits, dict) and isinstance(step_limits, dict)
        assert "r.Seed >= 1<<63" in control_source
        assert (
            f"r.DiffusionSteps < {step_limits['minimum']} || "
            f"r.DiffusionSteps > {step_limits['maximum']}"
        ) in control_source

    job_contract = contract["job"]
    assert isinstance(job_contract, dict)
    job_schema = openapi["components"]["schemas"][job_contract["openapi_schema"]]
    job_id_pattern = job_contract["job_id_pattern"]
    assert isinstance(job_id_pattern, str)
    assert job_schema["properties"]["id"]["pattern"] == job_id_pattern
    assert openapi["components"]["parameters"]["JobId"]["schema"]["pattern"] == job_id_pattern
    assert sorted(job_schema["required"]) == job_contract["required_openapi_fields"]
    assert job_schema["properties"]["state"]["enum"] == job_contract["states"]
    openapi_job_schema = dict(openapi)
    openapi_job_schema["$ref"] = f"#/components/schemas/{job_contract['openapi_schema']}"
    jsonschema.validate(job_contract["example_terminal_projection"], openapi_job_schema)
    assert (
        job_schema["properties"]["result_artifact_uri"]["pattern"]
        == job_contract["result_artifact_uri_pattern"]
    )
    job = _proto_message(image, job_contract["proto_message"])
    job_fields = _named_entries(job, "field")
    field_mappings = job_contract["field_mappings"]
    assert isinstance(field_mappings, dict)
    for json_name, raw_mapping in field_mappings.items():
        assert json_name in job_schema["properties"]
        assert isinstance(raw_mapping, dict)
        proto_field = job_fields[raw_mapping["proto_field"]]
        assert proto_field["type"] == raw_mapping["proto_type"]
    job_file = _proto_file(image, "mindclade.job.v1alpha1")
    job_states = _named_entries(job_file, "enumType")["JobState"]
    state_values = _named_entries(job_states, "value")
    normalized_states = [
        name.removeprefix("JOB_STATE_").lower()
        for name in state_values
        if name != "JOB_STATE_UNSPECIFIED"
    ]
    assert normalized_states == job_contract["states"]
    job_runtime_source = _runtime_source_if_available(job_contract["runtime_source"])
    if job_runtime_source is not None:
        assert f"regexp.MustCompile(`{job_id_pattern}`)" in job_runtime_source
        job_projection_source = _flat_struct_source(job_runtime_source, "type Job struct {")
        for json_name in field_mappings:
            assert f'json:"{json_name}' in job_projection_source
        runtime_states = [
            value for value in job_contract["states"] if f'State = "{value}"' in job_runtime_source
        ]
        assert runtime_states == job_contract["states"]
    result_uri_source = _runtime_source_if_available(job_contract["result_uri_runtime_source"])
    if result_uri_source is not None:
        assert 'return "artifact://sha256/"' in result_uri_source

    artifact_contract = contract["artifact"]
    assert isinstance(artifact_contract, dict)
    begin_contract = artifact_contract["begin_upload"]
    assert isinstance(begin_contract, dict)
    begin = _proto_message(image, begin_contract["proto_message"])
    begin_fields = _named_entries(begin, "field")
    assert {name: begin_fields[name]["type"] for name in begin_fields} == begin_contract[
        "required_fields"
    ]
    artifact_runtime_source = _runtime_source_if_available(begin_contract["runtime_source"])
    if artifact_runtime_source is not None:
        begin_request_source = _flat_struct_source(artifact_runtime_source, "struct BeginRequest {")
        assert "session_id: String" in begin_request_source
    download_contract = artifact_contract["download"]
    assert isinstance(download_contract, dict)
    service = _proto_service(image, download_contract["proto_service"])
    method = _named_entries(service, "method")[download_contract["rpc"]]
    assert method["inputType"] == download_contract["request_type"]
    assert method["outputType"] == download_contract["response_type"]
    assert method.get("serverStreaming", False) is download_contract["server_streaming"]
    response = _proto_message(image, download_contract["response_type"].lstrip("."))
    response_field = _named_entries(response, "field")[download_contract["data_field"]]
    assert response_field["type"] == download_contract["data_type"]
    if artifact_runtime_source is not None:
        assert '(Method::GET, ["v1alpha1", "artifacts", digest])' in artifact_runtime_source


def test_release_manifest_is_development_only() -> None:
    schema = _document("schemas/release_manifest/release_manifest.schema.json")
    jsonschema.validate(_document("schemas/release_manifest/positive.json"), schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            _document("schemas/release_manifest/negative_mutable_reference.json"), schema
        )


def test_json_schema_and_openapi_baselines_are_current() -> None:
    schema_lock = _document("compatibility/baselines/json-schema.lock.json")
    assert isinstance(schema_lock, dict)
    expected = schema_lock["canonical_sha256"]
    assert isinstance(expected, dict)
    assert expected == {
        "model_manifest": _canonical_digest(
            _document("schemas/model_manifest/model_manifest.schema.json")
        ),
        "release_manifest": _canonical_digest(
            _document("schemas/release_manifest/release_manifest.schema.json")
        ),
    }
    openapi = yaml.safe_load((ROOT / "openapi/external-api.yaml").read_text(encoding="utf-8"))
    openapi_lock = _document("compatibility/baselines/openapi.lock.json")
    assert isinstance(openapi_lock, dict)
    assert openapi_lock["canonical_sha256"] == _canonical_digest(openapi)
    assert openapi["security"] == [{"bearerAuth": []}]


def test_protobuf_lint_and_breaking_baseline() -> None:
    buf = _require_buf()
    subprocess.run([buf, "lint"], cwd=REPOSITORY, check=True)
    subprocess.run(
        [buf, "breaking", "--against", "protocols/compatibility/baselines/protobuf.binpb"],
        cwd=REPOSITORY,
        check=True,
    )


@pytest.mark.network
def test_checked_in_generated_clients_have_no_drift(tmp_path: Path) -> None:
    buf = _require_buf()
    subprocess.run([buf, "generate", "--output", str(tmp_path)], cwd=REPOSITORY, check=True)
    expected_root = ROOT / "generated"
    actual_root = tmp_path / "protocols/generated"
    expected_files = sorted(
        path.relative_to(expected_root)
        for path in expected_root.rglob("*")
        if path.is_file() and path.suffix in {".go", ".py", ".rs"}
    )
    actual_files = sorted(
        path.relative_to(actual_root)
        for path in actual_root.rglob("*")
        if path.is_file() and path.suffix in {".go", ".py", ".rs"}
    )
    assert actual_files == expected_files
    for relative in expected_files:
        assert (actual_root / relative).read_bytes() == (expected_root / relative).read_bytes(), (
            relative
        )
