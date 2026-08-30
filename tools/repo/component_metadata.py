"""Validate governed ``component.yaml`` manifests and their dependency graph."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonschema import ValidationError
from jsonschema.validators import validator_for
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = Path(__file__).with_name("component.schema.json")
IGNORED_DIRECTORIES = {
    ".direnv",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "dist",
    "node_modules",
    "target",
}


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: UniqueKeyLoader, node: MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True, order=True)
class ValidationIssue:
    """One deterministic validation failure."""

    path: str
    location: str
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.location}: {self.message}"


def discover_component_files(root: Path) -> list[Path]:
    """Return source-owned component manifests without following generated trees."""

    manifests: list[Path] = []
    for directory, child_directories, filenames in os.walk(root, followlinks=False):
        child_directories[:] = sorted(
            child
            for child in child_directories
            if child not in IGNORED_DIRECTORIES
            and not child.startswith("bazel-")
            and not (Path(directory) / child).is_symlink()
        )
        if "component.yaml" in filenames:
            manifests.append(Path(directory) / "component.yaml")
    return sorted(manifests)


def _json_path(parts: Sequence[object]) -> str:
    result = "$"
    for part in parts:
        if isinstance(part, int):
            result += f"[{part}]"
        else:
            result += f".{part}"
    return result


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _schema_issues(
    path: Path,
    root: Path,
    document: object,
    validator: Any,
) -> Iterator[ValidationIssue]:
    errors: list[ValidationError] = sorted(
        validator.iter_errors(document),
        key=lambda error: (
            tuple(str(part) for part in error.absolute_path),
            error.message,
        ),
    )
    relative_path = _relative_path(path, root)
    for error in errors:
        yield ValidationIssue(
            relative_path,
            _json_path(list(error.absolute_path)),
            error.message,
        )


def _load_document(path: Path, root: Path) -> tuple[object | None, list[ValidationIssue]]:
    try:
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        return None, [ValidationIssue(_relative_path(path, root), "$", str(error))]
    return document, []


def _component_name(document: Mapping[str, object]) -> str:
    metadata = document["metadata"]
    assert isinstance(metadata, Mapping)
    name = metadata["name"]
    assert isinstance(name, str)
    return name


def _component_dependencies(document: Mapping[str, object]) -> list[str]:
    spec = document["spec"]
    assert isinstance(spec, Mapping)
    dependencies = spec["dependencies"]
    assert isinstance(dependencies, list)
    assert all(isinstance(dependency, str) for dependency in dependencies)
    return dependencies


def _canonical_cycle(cycle: list[str]) -> tuple[str, ...]:
    nodes = cycle[:-1]
    rotations = [tuple(nodes[index:] + nodes[:index]) for index in range(len(nodes))]
    return min(rotations)


def _dependency_cycles(graph: Mapping[str, list[str]]) -> list[tuple[str, ...]]:
    state = {name: 0 for name in graph}
    stack: list[str] = []
    cycles: set[tuple[str, ...]] = set()

    def visit(name: str) -> None:
        state[name] = 1
        stack.append(name)
        for dependency in sorted(graph[name]):
            if dependency == name or dependency not in graph:
                continue
            if state[dependency] == 0:
                visit(dependency)
            elif state[dependency] == 1:
                start = stack.index(dependency)
                cycles.add(_canonical_cycle([*stack[start:], dependency]))
        stack.pop()
        state[name] = 2

    for name in sorted(graph):
        if state[name] == 0:
            visit(name)
    return sorted(cycles)


def validate_repository(
    root: Path = ROOT,
    schema_path: Path = SCHEMA_PATH,
) -> list[ValidationIssue]:
    """Validate every manifest below ``root`` and all cross-component references."""

    root = root.resolve()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator_type = validator_for(schema)
    validator_type.check_schema(schema)
    validator = validator_type(schema)

    paths = discover_component_files(root)
    if not paths:
        return [ValidationIssue("<repository>", "$", "no component.yaml manifests found")]

    issues: list[ValidationIssue] = []
    valid_documents: list[tuple[Path, Mapping[str, object]]] = []
    for path in paths:
        document, load_issues = _load_document(path, root)
        issues.extend(load_issues)
        if load_issues:
            continue
        schema_issues = list(_schema_issues(path, root, document, validator))
        issues.extend(schema_issues)
        if not schema_issues:
            assert isinstance(document, Mapping)
            valid_documents.append((path, document))

    root_manifest = root / "component.yaml"
    if root_manifest not in paths:
        issues.append(ValidationIssue("<repository>", "$", "root component.yaml is required"))

    by_name: dict[str, tuple[Path, Mapping[str, object]]] = {}
    for path, document in valid_documents:
        name = _component_name(document)
        if name in by_name:
            first_path = _relative_path(by_name[name][0], root)
            issues.append(
                ValidationIssue(
                    _relative_path(path, root),
                    "$.metadata.name",
                    f"duplicate component name {name!r}; first declared in {first_path}",
                )
            )
        else:
            by_name[name] = (path, document)

    root_entry = next((item for item in valid_documents if item[0] == root_manifest), None)
    if root_entry is not None:
        root_spec = root_entry[1]["spec"]
        assert isinstance(root_spec, Mapping)
        if root_spec["type"] != "repository":
            issues.append(
                ValidationIssue(
                    "component.yaml",
                    "$.spec.type",
                    "the root component must have type 'repository'",
                )
            )

    for path, document in valid_documents:
        if path == root_manifest:
            continue
        spec = document["spec"]
        assert isinstance(spec, Mapping)
        if spec["type"] == "repository":
            issues.append(
                ValidationIssue(
                    _relative_path(path, root),
                    "$.spec.type",
                    "only the root component may have type 'repository'",
                )
            )

    graph: dict[str, list[str]] = {}
    for name, (path, document) in sorted(by_name.items()):
        dependencies = _component_dependencies(document)
        graph[name] = dependencies
        for index, dependency in enumerate(dependencies):
            if dependency == name:
                issues.append(
                    ValidationIssue(
                        _relative_path(path, root),
                        f"$.spec.dependencies[{index}]",
                        "a component cannot depend on itself",
                    )
                )
            elif dependency not in by_name:
                issues.append(
                    ValidationIssue(
                        _relative_path(path, root),
                        f"$.spec.dependencies[{index}]",
                        f"unknown component dependency {dependency!r}",
                    )
                )

    for cycle in _dependency_cycles(graph):
        closed_cycle = (*cycle, cycle[0])
        issues.append(
            ValidationIssue(
                "<repository>",
                "$.spec.dependencies",
                f"dependency cycle: {' -> '.join(closed_cycle)}",
            )
        )

    return sorted(set(issues))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    issues = validate_repository(args.root, args.schema)
    if issues:
        print("\n".join(issue.render() for issue in issues), file=sys.stderr)
        return 1
    count = len(discover_component_files(args.root.resolve()))
    print(f"component metadata: ok ({count} manifests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
