"""Buildkite service and artifact-contract qualification entrypoint."""

from __future__ import annotations

import subprocess


def main() -> None:
    subprocess.run(["just", "service-check"], check=True)
    subprocess.run(["just", "render-deploy"], check=True)
    subprocess.run(["just", "bazel-test"], check=True)


if __name__ == "__main__":
    main()
