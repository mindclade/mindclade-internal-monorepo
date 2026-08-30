"""Command-line composition root for one scheduler-owned attempt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .completion import publish_completion
from .contracts import JobManifest
from .publication import publish_result_artifact
from .runner import MindcladeModelExecutor, WorkerRoots, execute_job
from .staging import stage_job
from .trust import TrustedKeyring


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mindclade-inference-worker")
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage")
    _add_common(stage)
    stage.add_argument("--artifact-proxy-url", required=True)
    run = commands.add_parser("run")
    _add_common(run)
    run.add_argument("--result-root", type=Path, required=True)
    run.add_argument("--control-plane-url", required=True)
    run.add_argument("--artifact-proxy-url", required=True)
    args = parser.parse_args(argv)
    value = json.loads(args.job_manifest.read_text(encoding="utf-8"))
    manifest = JobManifest.from_dict(value)
    trust = TrustedKeyring.from_file(args.trusted_keyring)
    if args.command == "stage":
        stage_job(
            manifest,
            trust=trust,
            artifact_root=args.artifact_root,
            artifact_proxy_url=args.artifact_proxy_url,
        )
        return 0
    roots = WorkerRoots(artifact_root=args.artifact_root, result_root=args.result_root)
    receipt = execute_job(manifest, MindcladeModelExecutor(trust), trust=trust, roots=roots)
    publish_result_artifact(
        args.control_plane_url,
        args.artifact_proxy_url,
        manifest,
        receipt,
        result_root=args.result_root,
    )
    publish_completion(args.control_plane_url, manifest, receipt)
    print(json.dumps(receipt.to_dict(), sort_keys=True))
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--job-manifest", type=Path, required=True)
    parser.add_argument("--trusted-keyring", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)


if __name__ == "__main__":
    raise SystemExit(main())
