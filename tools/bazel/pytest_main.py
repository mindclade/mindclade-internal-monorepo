"""Hermetic Bazel entry point for repository pytest suites."""

from __future__ import annotations

import sys

import pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main(sys.argv[1:]))
