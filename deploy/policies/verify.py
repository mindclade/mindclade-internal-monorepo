"""Fail-closed static policy checks for rendered Kubernetes resources."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

DIGEST_IMAGE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")


def _documents(root: Path):
    for path in sorted(root.rglob("*.yaml")):
        if path.name == "kustomization.yaml":
            continue
        for document in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if isinstance(document, dict) and "kind" in document:
                yield path, document


def violations(root: Path) -> list[str]:
    errors: list[str] = []
    for path, document in _documents(root):
        text = path.read_text(encoding="utf-8")
        credential_pattern = (
            r"(?i)(GOOGLE_APPLICATION_CREDENTIALS|"
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----|client[_-]?secret\s*:)"
        )
        if re.search(credential_pattern, text):
            errors.append(f"{path}: static credential material or configuration is forbidden")
        kind = document.get("kind")
        if kind == "Secret":
            errors.append(f"{path}: static Secret resources are forbidden")
        pod_spec = None
        if kind == "Deployment":
            pod_spec = document.get("spec", {}).get("template", {}).get("spec", {})
        elif kind == "JobSet":
            jobs = document.get("spec", {}).get("replicatedJobs", [])
            pod_specs = [
                job.get("template", {}).get("spec", {}).get("template", {}).get("spec", {})
                for job in jobs
            ]
            for candidate in pod_specs:
                errors.extend(_check_pod(path, candidate))
        if pod_spec is not None:
            errors.extend(_check_pod(path, pod_spec))
    return errors


def _check_pod(path: Path, spec: dict) -> list[str]:
    errors: list[str] = []
    if spec.get("serviceAccountName") in {None, "", "default"}:
        errors.append(f"{path}: workload must use a dedicated service account")
    if spec.get("automountServiceAccountToken") is not False:
        errors.append(f"{path}: ambient service-account token mounting is forbidden")
    if not spec.get("securityContext", {}).get("runAsNonRoot"):
        errors.append(f"{path}: pod must set runAsNonRoot")
    pod_security = spec.get("securityContext", {})
    for field in ("runAsUser", "runAsGroup", "fsGroup"):
        if not isinstance(pod_security.get(field), int) or pod_security[field] <= 0:
            errors.append(f"{path}: pod must set a positive {field}")
    projected_tokens = [
        source
        for volume in spec.get("volumes", [])
        for source in volume.get("projected", {}).get("sources", [])
        if "serviceAccountToken" in source
    ]
    if projected_tokens and spec.get("serviceAccountName") != "control-plane":
        errors.append(
            f"{path}: only the control plane may receive a projected Kubernetes API token"
        )
    if spec.get("serviceAccountName") == "control-plane" and len(projected_tokens) != 1:
        errors.append(
            f"{path}: control plane requires exactly one bounded projected Kubernetes API token"
        )
    for container in [*spec.get("initContainers", []), *spec.get("containers", [])]:
        image = container.get("image", "")
        if not DIGEST_IMAGE.fullmatch(image):
            errors.append(f"{path}: image is not digest pinned: {image}")
        security = container.get("securityContext", {})
        if security.get("allowPrivilegeEscalation") is not False:
            errors.append(f"{path}: container must disable privilege escalation")
        if security.get("readOnlyRootFilesystem") is not True:
            errors.append(f"{path}: container must use a read-only root filesystem")
        if security.get("runAsNonRoot") is not True:
            errors.append(f"{path}: container must set runAsNonRoot")
        for field in ("runAsUser", "runAsGroup"):
            if not isinstance(security.get(field), int) or security[field] <= 0:
                errors.append(f"{path}: container must set a positive {field}")
        if "ALL" not in security.get("capabilities", {}).get("drop", []):
            errors.append(f"{path}: container must drop ALL capabilities")
        if security.get("privileged") is True:
            errors.append(f"{path}: privileged containers are forbidden")
    return errors


def main() -> int:
    root = Path(__file__).parents[1]
    errors = violations(root)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("deployment policy: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
