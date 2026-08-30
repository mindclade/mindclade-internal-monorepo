"""Enforce the Python dependency law directly from imports.

The check is intentionally dependency-free so it can run before the workspace is
resolved. Imports are classified by their public ``mindclade`` package name.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOTS = {
    "models": ROOT / "models" / "src",
    "training": ROOT / "training" / "src",
    "inference": ROOT / "inference" / "src",
}
GENERATED_CLIENT_NAMESPACES = frozenset(
    {
        "mindclade.admin.v1alpha1",
        "mindclade.artifact.v1alpha1",
        "mindclade.common.v1alpha1",
        "mindclade.inference.v1alpha1",
        "mindclade.job.v1alpha1",
        "mindclade.model.v1alpha1",
    }
)
EXECUTION_FORBIDDEN = {
    "models": {"mindclade.training", "mindclade.inference", "services", "workers", "deploy"},
    "training": {"mindclade.inference", "services", "workers", "deploy"},
    "inference": {"mindclade.training", "services", "workers", "deploy"},
}
FORBIDDEN = {
    owner: prefixes | GENERATED_CLIENT_NAMESPACES for owner, prefixes in EXECUTION_FORBIDDEN.items()
}


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    owner: str
    imported: str

    def render(self) -> str:
        relative = self.path.relative_to(ROOT)
        return f"{relative}:{self.line}: {self.owner} must not import {self.imported}"


def _imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            # Include imported members so shorthand such as
            # ``from mindclade.inference import v1alpha1`` cannot evade the
            # generated-client boundary. Relative imports remain harmless
            # because their module lacks the public ``mindclade`` prefix.
            imports.extend(
                (
                    node.lineno,
                    node.module if alias.name == "*" else f"{node.module}.{alias.name}",
                )
                for alias in node.names
            )
    return imports


def find_violations() -> list[Violation]:
    violations: list[Violation] = []
    for owner, root in PACKAGE_ROOTS.items():
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            for line, imported in _imports(path):
                if any(
                    imported == prefix or imported.startswith(prefix + ".")
                    for prefix in FORBIDDEN[owner]
                ):
                    violations.append(Violation(path, line, owner, imported))
    return violations


def main() -> int:
    violations = find_violations()
    if violations:
        print("\n".join(item.render() for item in violations), file=sys.stderr)
        return 1
    print("dependency policy: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
