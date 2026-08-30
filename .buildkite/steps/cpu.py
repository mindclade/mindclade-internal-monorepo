"""Buildkite CPU qualification entrypoint."""

from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> None:
    subprocess.run(["uv", "sync", "--all-packages", "--locked"], check=True)
    subprocess.run(["just", "check"], check=True)
    subprocess.run(["just", "protocol-check"], check=True)
    subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "-m",
            "not gpu and not distributed and not nightly and not network",
        ],
        check=True,
    )
    subprocess.run(["just", "build-wheels"], check=True)
    wheels = sorted(Path("dist").glob("*.whl"))
    if len(wheels) != 4:
        raise RuntimeError(f"expected four wheels, found {len(wheels)}")
    subprocess.run(
        ["uv", "run", "check-wheel-contents", *(str(path) for path in wheels)],
        check=True,
    )
    subprocess.run(
        ["uv", "run", "python", "protocols/python/wheel_smoke.py", "--wheel-dir", "dist"],
        check=True,
    )


if __name__ == "__main__":
    main()
