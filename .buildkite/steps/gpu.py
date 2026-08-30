"""Buildkite CUDA qualification and complete environment evidence."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def main() -> None:
    subprocess.run(["uv", "sync", "--all-packages", "--locked"], check=True)
    probe = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    environment = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            (
                "import json,torch; print(json.dumps({"
                "'torch_version':torch.__version__,"
                "'compiled_cuda':torch.version.cuda,"
                "'cuda_available':torch.cuda.is_available(),"
                "'device_count':torch.cuda.device_count(),"
                "'bf16_supported':torch.cuda.is_bf16_supported(),"
                "'qualification_dtype':'bfloat16'}))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    details = json.loads(environment.stdout)
    if not details["cuda_available"] or not details["bf16_supported"]:
        raise SystemExit("CUDA and bfloat16 support are required")
    if not str(details["compiled_cuda"]).startswith("13.0"):
        raise SystemExit(f"expected PyTorch CUDA 13.0 profile, got {details['compiled_cuda']}")
    subprocess.run(["uv", "run", "pytest", "-m", "gpu and not distributed"], check=True)
    subprocess.run(
        [
            "uv",
            "run",
            "torchrun",
            "--standalone",
            "--nproc-per-node=2",
            "-m",
            "pytest",
            "-q",
            "-m",
            "distributed",
        ],
        check=True,
    )
    evidence = {
        **details,
        "accelerators": probe.stdout.strip().splitlines(),
        "source_revision": os.environ.get("BUILDKITE_COMMIT", ""),
        "world_size": 2,
    }
    Path("gpu-evidence.json").write_text(
        json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
