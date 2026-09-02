from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from deploy.policies.verify import violations

ROOT = Path(__file__).parents[1]
OVERLAY = ROOT / "gke" / "development"
RENDER_LOCK = ROOT / "rendered" / "development.sha256"


def _renderer() -> list[str]:
    if kubectl := shutil.which("kubectl"):
        return [kubectl, "kustomize"]
    if kustomize := shutil.which("kustomize"):
        return [kustomize, "build"]
    for candidate in (
        Path("/opt/homebrew/bin/kubectl"),
        Path("/usr/local/bin/kubectl"),
        Path("/usr/bin/kubectl"),
    ):
        if candidate.is_file():
            return [str(candidate), "kustomize"]
    raise AssertionError("kubectl or kustomize is required to verify the deployment render")


def _render() -> bytes:
    # Bazel runfiles are symlinks. Copying the two Kustomize trees into one
    # private root preserves Kustomize's load restrictions in every runner.
    with tempfile.TemporaryDirectory(prefix="mindclade-render-") as temporary:
        root = Path(temporary)
        shutil.copytree(ROOT / "base", root / "base")
        shutil.copytree(OVERLAY, root / "gke" / "development")
        completed = subprocess.run(
            [*_renderer(), str(root / "gke" / "development")],
            cwd=root,
            check=True,
            capture_output=True,
        )
        return completed.stdout.replace(b"\r\n", b"\n")


def _documents(rendered: bytes) -> list[dict[str, Any]]:
    return [
        document
        for document in yaml.safe_load_all(rendered.decode("utf-8"))
        if isinstance(document, dict)
    ]


def test_render_is_deterministic() -> None:
    first = _render()
    second = _render()
    assert first == second
    expected = RENDER_LOCK.read_text(encoding="utf-8").strip()
    assert expected.startswith("sha256:")
    assert "sha256:" + hashlib.sha256(first).hexdigest() == expected


def test_rendered_runtime_contracts() -> None:
    documents = _documents(_render())
    identities: set[tuple[str, str, str]] = set()
    for document in documents:
        metadata = document.get("metadata", {})
        identity = (
            str(document.get("kind", "")),
            str(metadata.get("namespace", "")),
            str(metadata.get("name", "")),
        )
        assert identity[0] and identity[2], identity
        assert identity not in identities, identity
        identities.add(identity)

    by_identity = {
        (str(document.get("kind", "")), str(document.get("metadata", {}).get("name", ""))): document
        for document in documents
    }
    control_plane = by_identity[("Deployment", "control-plane")]
    artifact_proxy = by_identity[("Deployment", "artifact-proxy")]
    assert control_plane["spec"]["replicas"] == 1
    assert artifact_proxy["spec"]["replicas"] == 1
    artifact_store = by_identity[("PersistentVolumeClaim", "artifact-store-development")]
    assert artifact_store["spec"]["storageClassName"] == "premium-rwo"
    artifact_volumes = artifact_proxy["spec"]["template"]["spec"]["volumes"]
    assert any(
        volume.get("persistentVolumeClaim", {}).get("claimName") == "artifact-store-development"
        for volume in artifact_volumes
    )

    assert not [document for document in documents if document.get("kind") == "JobSet"]
    assert not [
        document
        for document in documents
        if document.get("metadata", {}).get("labels", {}).get("kueue.x-k8s.io/queue-name")
    ]

    control_spec = control_plane["spec"]["template"]["spec"]
    assert control_spec["automountServiceAccountToken"] is False
    projected = [
        source
        for volume in control_spec["volumes"]
        for source in volume.get("projected", {}).get("sources", [])
        if "serviceAccountToken" in source
    ]
    assert len(projected) == 1
    assert projected[0]["serviceAccountToken"]["expirationSeconds"] == 3600
    assert ("Role", "control-plane-attempt-launcher") in by_identity
    role = by_identity[("Role", "control-plane-attempt-launcher")]
    assert {"create", "get", "list", "patch", "delete"} == set(role["rules"][0]["verbs"])

    monitors = [document for document in documents if document.get("kind") == "ServiceMonitor"]
    assert monitors
    assert all(
        endpoint.get("path") == "/metrics"
        for item in monitors
        for endpoint in item["spec"]["endpoints"]
    )
    services = [document for document in documents if document.get("kind") == "Service"]
    for monitor in monitors:
        selector = monitor["spec"]["selector"]["matchLabels"]
        matching = [
            service
            for service in services
            if all(
                service.get("metadata", {}).get("labels", {}).get(key) == value
                for key, value in selector.items()
            )
        ]
        assert matching, monitor["metadata"]["name"]
        ports = {port["name"] for service in matching for port in service["spec"]["ports"]}
        assert all(endpoint["port"] in ports for endpoint in monitor["spec"]["endpoints"])

    policies = [document for document in documents if document.get("kind") == "NetworkPolicy"]
    policy_names = {item["metadata"]["name"] for item in policies}
    assert {
        "default-deny",
        "edge-to-runtime-gateway",
        "monitoring-ingress",
        "worker-egress-to-artifact-proxy",
        "worker-egress-to-control-plane",
        "control-plane-egress-kubernetes-api",
        "control-plane-egress-to-artifact-proxy",
    } <= policy_names


def test_values_are_schema_valid_and_development_only() -> None:
    schema = json.loads((ROOT / "values.schema.json").read_text(encoding="utf-8"))
    values = json.loads((ROOT / "values.development.json").read_text(encoding="utf-8"))
    jsonschema.validate(values, schema)
    forbidden = dict(values)
    forbidden["stage"] = "production"
    with __import__("pytest").raises(jsonschema.ValidationError):
        jsonschema.validate(forbidden, schema)


def test_rendered_images_are_bound_to_values() -> None:
    values = json.loads((ROOT / "values.development.json").read_text(encoding="utf-8"))
    documents = _documents(_render())
    deployments = {
        document["metadata"]["name"]: document
        for document in documents
        if document.get("kind") == "Deployment"
    }
    images = values["images"]
    assert (
        deployments["control-plane"]["spec"]["template"]["spec"]["containers"][0]["image"]
        == images["controlPlane"]
    )
    assert (
        deployments["runtime-gateway"]["spec"]["template"]["spec"]["containers"][0]["image"]
        == images["runtimeGateway"]
    )
    assert (
        deployments["artifact-proxy"]["spec"]["template"]["spec"]["containers"][0]["image"]
        == images["artifactProxy"]
    )
    control_environment = {
        item["name"]: item.get("value")
        for item in deployments["control-plane"]["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    gateway_environment = {
        item["name"]: item.get("value")
        for item in deployments["runtime-gateway"]["spec"]["template"]["spec"]["containers"][0][
            "env"
        ]
    }
    assert control_environment["MINDCLADE_WORKER_IMAGE"] == images["inferenceWorker"]
    queue_seconds = int(control_environment["MINDCLADE_QUEUE_DEADLINE_SECONDS"])
    startup_seconds = int(control_environment["MINDCLADE_ATTEMPT_STARTUP_DEADLINE_SECONDS"])
    active_seconds = int(control_environment["MINDCLADE_ATTEMPT_ACTIVE_DEADLINE_SECONDS"])
    reconcile_seconds = int(control_environment["MINDCLADE_JOBSET_RECONCILE_INTERVAL_SECONDS"])
    capability_seconds = int(control_environment["MINDCLADE_STAGING_CAPABILITY_TTL_SECONDS"])
    launch_seconds = int(control_environment["MINDCLADE_KUBERNETES_LAUNCH_TIMEOUT_SECONDS"])
    verify_seconds = int(control_environment["MINDCLADE_ARTIFACT_VERIFY_TIMEOUT_SECONDS"])
    handler_seconds = int(control_environment["MINDCLADE_CONTROL_PLANE_HANDLER_TIMEOUT_SECONDS"])
    assert handler_seconds == int(
        gateway_environment["MINDCLADE_CONTROL_PLANE_HANDLER_TIMEOUT_SECONDS"]
    )
    assert handler_seconds >= 15 + max(5 * launch_seconds, verify_seconds) + 5
    assert capability_seconds >= (
        queue_seconds
        + startup_seconds
        + active_seconds
        + launch_seconds
        + 2 * reconcile_seconds
        + 60
    )


def test_all_yaml_documents_parse() -> None:
    for path in ROOT.rglob("*.yaml"):
        list(yaml.safe_load_all(path.read_text(encoding="utf-8")))


def test_static_workload_policy() -> None:
    assert violations(ROOT) == []
