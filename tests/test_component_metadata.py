from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def _load_validator():
    path = ROOT / "tools" / "repo" / "component_metadata.py"
    spec = importlib.util.spec_from_file_location("component_metadata", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator()
SCHEMA = ROOT / "tools" / "repo" / "component.schema.json"


def _release(mode: str, artifact: str | None, version: str | None) -> dict[str, object]:
    return {
        "mode": mode,
        "artifact": artifact,
        "version": version,
        "public": False,
        "controls": [],
    }


def _repository_manifest() -> dict[str, object]:
    return {
        "apiVersion": "components.mindclade.dev/v1alpha1",
        "kind": "Component",
        "metadata": {"name": "test-repository"},
        "spec": {
            "type": "repository",
            "owner": "@mindclade/platform-runtime",
            "maturity": "experimental",
            "visibility": "private",
            "dataClassification": "internal",
            "description": "Test repository root.",
            "dependencies": [],
            "release": _release("none", None, None),
        },
    }


def _service_manifest(name: str, dependencies: list[str]) -> dict[str, object]:
    return {
        "apiVersion": "components.mindclade.dev/v1alpha1",
        "kind": "Component",
        "metadata": {"name": name},
        "spec": {
            "type": "service",
            "owner": "@mindclade/platform-runtime",
            "maturity": "experimental",
            "visibility": "private",
            "dataClassification": "internal",
            "description": f"Test component {name}.",
            "dependencies": dependencies,
            "release": _release("oci-image", name, None),
        },
    }


def _write_manifest(root: Path, relative_path: str, manifest: object) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def _messages(root: Path) -> list[str]:
    return [
        issue.render() for issue in VALIDATOR.validate_repository(root=root, schema_path=SCHEMA)
    ]


def test_repository_component_metadata_is_valid() -> None:
    assert VALIDATOR.validate_repository(root=ROOT, schema_path=SCHEMA) == []


def test_schema_rejects_unknown_fields_and_wrong_release_mode(tmp_path: Path) -> None:
    root_manifest = _repository_manifest()
    root_spec = root_manifest["spec"]
    assert isinstance(root_spec, dict)
    root_spec["undeclared"] = True
    _write_manifest(tmp_path, "component.yaml", root_manifest)

    service = _service_manifest("api-service", [])
    service_spec = service["spec"]
    assert isinstance(service_spec, dict)
    service_spec["release"] = _release("none", None, None)
    _write_manifest(tmp_path, "services/api/component.yaml", service)

    messages = _messages(tmp_path)
    assert any(
        "Additional properties are not allowed ('undeclared' was unexpected)" in item
        for item in messages
    )
    assert any("'oci-image' was expected" in item for item in messages)


def test_validator_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "component.yaml", _repository_manifest())
    path = tmp_path / "service" / "component.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        "apiVersion: components.mindclade.dev/v1alpha1\nkind: Component\nkind: Component\n",
        encoding="utf-8",
    )

    assert any("found duplicate key 'kind'" in item for item in _messages(tmp_path))


def test_validator_rejects_nested_repository_components(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "component.yaml", _repository_manifest())
    nested_repository = _repository_manifest()
    nested_metadata = nested_repository["metadata"]
    assert isinstance(nested_metadata, dict)
    nested_metadata["name"] = "nested-repository"
    _write_manifest(tmp_path, "nested/component.yaml", nested_repository)

    assert any(
        "only the root component may have type 'repository'" in item for item in _messages(tmp_path)
    )


def test_validator_rejects_duplicate_names_and_dangling_dependencies(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "component.yaml", _repository_manifest())
    _write_manifest(
        tmp_path,
        "services/one/component.yaml",
        _service_manifest("shared-name", ["missing-component"]),
    )
    _write_manifest(
        tmp_path,
        "services/two/component.yaml",
        _service_manifest("shared-name", []),
    )

    messages = _messages(tmp_path)
    assert any("duplicate component name 'shared-name'" in item for item in messages)
    assert any("unknown component dependency 'missing-component'" in item for item in messages)


def test_validator_rejects_self_dependencies_and_cycles(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "component.yaml", _repository_manifest())
    _write_manifest(
        tmp_path,
        "services/one/component.yaml",
        _service_manifest("service-one", ["service-two"]),
    )
    _write_manifest(
        tmp_path,
        "services/two/component.yaml",
        _service_manifest("service-two", ["service-one"]),
    )
    self_dependent = deepcopy(_service_manifest("self-dependent", []))
    self_spec = self_dependent["spec"]
    assert isinstance(self_spec, dict)
    self_spec["dependencies"] = ["self-dependent"]
    _write_manifest(tmp_path, "services/self/component.yaml", self_dependent)

    messages = _messages(tmp_path)
    assert any("a component cannot depend on itself" in item for item in messages)
    assert any(
        "dependency cycle: service-one -> service-two -> service-one" in item for item in messages
    )
