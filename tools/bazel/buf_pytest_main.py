"""Bazel pytest entry point that exposes the hermetic Buf CLI to a test."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


def _prepend_buf_to_path() -> None:
    runfiles = os.environ.get("RUNFILES_DIR")
    if runfiles is None:
        raise RuntimeError("RUNFILES_DIR is required for the Bazel Buf test")
    matches = list(Path(runfiles).glob("rules_buf*toolchains/buf"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one hermetic Buf binary, found {len(matches)}")
    current = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(part for part in (str(matches[0].parent), current) if part)


if __name__ == "__main__":
    _prepend_buf_to_path()
    raise SystemExit(pytest.main(sys.argv[1:]))
