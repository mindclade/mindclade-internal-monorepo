from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / ".buildkite" / "steps" / "release.py"
    spec = importlib.util.spec_from_file_location("release_evidence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_evidence_requires_the_qualified_two_device_profile(tmp_path: Path) -> None:
    release = _module()
    path = tmp_path / "gpu-evidence.json"
    evidence = {
        "accelerators": ["L4, 600.1, 23034", "L4, 600.1, 23034"],
        "bf16_supported": True,
        "compiled_cuda": "13.0",
        "cuda_available": True,
        "device_count": 2,
        "qualification_dtype": "bfloat16",
        "source_revision": "a" * 40,
        "torch_version": "2.13.0",
        "world_size": 2,
    }
    path.write_text(json.dumps(evidence), encoding="utf-8")
    release._validate_gpu_evidence(path, "a" * 40)

    evidence["device_count"] = 1
    path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(SystemExit, match="incomplete"):
        release._validate_gpu_evidence(path, "a" * 40)


def test_deployment_binding_is_explicit_when_development_refs_do_not_match() -> None:
    release = _module()
    images = {
        name: {"image_digest": f"sha256:{index:064x}"}
        for index, name in enumerate(sorted(release._IMAGE_NAMES), start=1)
    }
    matching = {
        deployment_key: f"registry.invalid/{name}@{images[name]['image_digest']}"
        for name, deployment_key in release._IMAGE_NAMES.items()
    }
    _, status = release._deployment_binding(images, matching)
    assert status == "matches-local-image-digests"

    matching["controlPlane"] = "registry.invalid/control-plane@sha256:" + "f" * 64
    _, status = release._deployment_binding(images, matching)
    assert status == "not-bound"


def test_image_evidence_schema_is_validated_before_artifacts(tmp_path: Path) -> None:
    release = _module()
    evidence = tmp_path / "artifact-proxy.evidence.json"
    evidence.write_text(json.dumps({"source_revision": "a" * 40}), encoding="utf-8")
    with pytest.raises(SystemExit, match="schema is incomplete"):
        release._validate_image_evidence(evidence, "a" * 40, object())


def test_release_pipeline_inherits_the_service_go_race_gate() -> None:
    root = Path(__file__).parents[1]
    justfile = (root / "justfile").read_text(encoding="utf-8")
    service_step = (root / ".buildkite/steps/service.py").read_text(encoding="utf-8")
    pipeline = (root / ".buildkite/pipeline.yml").read_text(encoding="utf-8")

    assert "service-check:\n    go test -race ./...\n" in justfile
    assert 'subprocess.run(["just", "service-check"], check=True)' in service_step
    assert "depends_on: [cpu, gpu, service]" in pipeline


def test_wheel_build_does_not_clear_independent_image_evidence() -> None:
    justfile = (Path(__file__).parents[1] / "justfile").read_text(encoding="utf-8")
    wheel_recipe = justfile.split("build-wheels:\n", 1)[1].split("\n\n", 1)[0]

    assert "--clear" not in wheel_recipe
    assert wheel_recipe.count("--out-dir dist") == 4
