"""Project the root uv resolution into the format consumed by rules_python.

``uv.lock`` remains authoritative. This script performs no resolution and CI can
use ``--check`` to detect a stale generated projection.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / "tools" / "bazel" / "uv_export.lock"
HEADER = (
    "# GENERATED from uv.lock by tools/bazel/export_uv_lock.py; do not edit.\n"
    "# uv.lock is the sole Python dependency resolution.\n"
)


def render() -> str:
    """Export every workspace package and the root default development group."""
    result = subprocess.run(
        [
            "uv",
            "export",
            "--all-packages",
            "--frozen",
            "--no-emit-workspace",
            "--no-header",
            "--no-annotate",
        ],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    return HEADER + result.stdout


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        content = render()
    except FileNotFoundError:
        print("uv is not on PATH", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        print(error.stderr, file=sys.stderr)
        return 2

    if args.check:
        if not EXPORT.exists() or EXPORT.read_text(encoding="utf-8") != content:
            print("tools/bazel/uv_export.lock is stale", file=sys.stderr)
            return 1
        return 0

    EXPORT.write_text(content, encoding="utf-8")
    print(f"wrote {EXPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
