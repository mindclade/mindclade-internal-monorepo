from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_policy():
    path = Path(__file__).parents[1] / "tools" / "repo" / "dependency_policy.py"
    spec = importlib.util.spec_from_file_location("dependency_policy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_repository_obeys_dependency_policy() -> None:
    policy = _load_policy()
    assert policy.find_violations() == []


def test_generated_protocol_imports_are_forbidden_for_every_execution_library(
    tmp_path: Path,
    monkeypatch,
) -> None:
    policy = _load_policy()
    roots: dict[str, Path] = {}
    for owner in ("models", "training", "inference"):
        root = tmp_path / owner
        root.mkdir()
        roots[owner] = root
        imports = "\n".join(
            f"import {namespace}" for namespace in sorted(policy.GENERATED_CLIENT_NAMESPACES)
        )
        if owner == "inference":
            imports += (
                "\nfrom mindclade.inference import InferenceRequest\n"
                "from mindclade.inference import v1alpha1\n"
            )
        (root / "generated_dependency.py").write_text(imports, encoding="utf-8")

    monkeypatch.setattr(policy, "PACKAGE_ROOTS", roots)
    violations = policy.find_violations()
    actual = {(item.owner, item.imported) for item in violations}
    expected = {
        (owner, namespace) for owner in roots for namespace in policy.GENERATED_CLIENT_NAMESPACES
    }
    assert expected <= actual
    assert ("inference", "mindclade.inference.v1alpha1") in actual
    assert ("inference", "mindclade.inference.InferenceRequest") not in actual
