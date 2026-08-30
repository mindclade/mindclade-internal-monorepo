# GPU qualification profile

The Q0 accelerator lane is pinned by `uv.lock` to PyTorch 2.13 and the CUDA 13.0
runtime family. The supported qualification shape is Linux x86-64 with an NVIDIA
driver compatible with CUDA 13, bfloat16-capable GPUs, and one GPU per eager worker.

`nix develop .#cuda` exposes the Linux CUDA development shell. Buildkite records the
GPU model, driver, PyTorch version, compiled CUDA version, bfloat16 support, dtype,
visible-device count, and distributed world size in `gpu-evidence.json`. A release
claim is invalid unless that evidence and the GPU-marked test results come from the
same immutable source revision.
