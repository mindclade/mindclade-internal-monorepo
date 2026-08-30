"""Install built wheels offline and verify their shared namespace in isolation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

EXPECTED_DISTRIBUTIONS = {
    "mindclade_inference",
    "mindclade_models",
    "mindclade_protocols",
    "mindclade_training",
}


def _distribution_name(path: Path) -> str:
    return path.name.split("-", 1)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()
    root = Path(__file__).parents[2].resolve()
    wheel_dir = args.wheel_dir.resolve()
    wheels = sorted(wheel_dir.glob("*.whl"))
    names = {_distribution_name(path) for path in wheels}
    if names != EXPECTED_DISTRIBUTIONS:
        raise SystemExit(f"expected wheels {sorted(EXPECTED_DISTRIBUTIONS)}, found {sorted(names)}")

    with tempfile.TemporaryDirectory(prefix="mindclade-wheel-smoke-") as directory:
        temporary_root = Path(directory)
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment.pop("UV_PROJECT_ENVIRONMENT", None)
        environment.pop("VIRTUAL_ENV", None)
        environment.update({"PYTHONNOUSERSITE": "1", "UV_OFFLINE": "1"})
        constraints = temporary_root / "constraints.txt"
        subprocess.run(
            [
                "uv",
                "export",
                "--all-packages",
                "--no-dev",
                "--no-emit-project",
                "--no-emit-workspace",
                "--no-hashes",
                "--frozen",
                "--offline",
                "--output-file",
                str(constraints),
            ],
            check=True,
            cwd=root,
            env=environment,
        )
        virtual_environment = temporary_root / "venv"
        subprocess.run(
            [
                "uv",
                "venv",
                "--offline",
                "--python",
                sys.executable,
                str(virtual_environment),
            ],
            check=True,
            cwd=root,
            env=environment,
        )
        virtual_python = virtual_environment / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(virtual_python),
                "--offline",
                "--no-sources",
                "--strict",
                "--constraints",
                str(constraints),
                "--find-links",
                str(wheel_dir),
                *(str(wheel) for wheel in wheels),
            ],
            check=True,
            cwd=root,
            env=environment,
        )
        command = """
from importlib.metadata import version
from pathlib import Path
import site

import mindclade.inference
import mindclade.models
import mindclade.training
from mindclade.inference import InferenceRequest
from mindclade.inference.v1alpha1 import inference_pb2
from mindclade.job.v1alpha1 import job_pb2

expected = {
    "mindclade-inference": "0.1.0",
    "mindclade-models": "0.1.0",
    "mindclade-protocols": "0.1.0",
    "mindclade-training": "0.1.0",
}
assert {name: version(name) for name in expected} == expected
message = job_pb2.Job(seed=7, diffusion_steps=16, state=job_pb2.JOB_STATE_RUNNING)
decoded = job_pb2.Job.FromString(message.SerializeToString(deterministic=True))
assert decoded == message
assert inference_pb2.InferenceOptions(seed=7, diffusion_steps=16).diffusion_steps == 16
assert InferenceRequest.__module__.startswith("mindclade.inference.contracts.")
site_root = Path(site.getsitepackages()[0]).resolve()
for module in (mindclade.inference, mindclade.models, mindclade.training, job_pb2):
    assert site_root in Path(module.__file__).resolve().parents
"""
        subprocess.run(
            [str(virtual_python), "-I", "-c", command],
            env=environment,
            check=True,
            cwd=directory,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
