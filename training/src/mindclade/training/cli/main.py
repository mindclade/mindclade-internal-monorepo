"""Dependency-light command line interface for reference training workflows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from mindclade.training.checkpointing import verify_checkpoint
from mindclade.training.providers import dcp_capability, fsdp2_capability
from mindclade.training.recipes import qualify_overfit, resolve_recipe, run_reference_recipe


def _print_json(value: object) -> None:
    print(json.dumps(value, allow_nan=False, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mindclade-training")
    subcommands = parser.add_subparsers(dest="command", required=True)

    inspect = subcommands.add_parser("inspect", help="resolve a recipe or inspect a checkpoint")
    inspect.add_argument("path", type=Path)
    inspect.add_argument("--checkpoint", action="store_true")

    capabilities = subcommands.add_parser(
        "capabilities", help="show local DCP/FSDP2 provider readiness"
    )
    del capabilities

    run = subcommands.add_parser("run", help="run a local single-process reference recipe")
    run.add_argument("recipe", type=Path)
    run.add_argument("--output", type=Path, required=True)

    resume = subcommands.add_parser("resume", help="resume a committed DCP checkpoint")
    resume.add_argument("checkpoint", type=Path)
    resume.add_argument("--recipe", type=Path, required=True)
    resume.add_argument("--output", type=Path, required=True)
    resume.add_argument("--allow-reshard", action="store_true")

    qualify = subcommands.add_parser(
        "qualify", help="run a recipe and apply the 10-step overfit reduction gate"
    )
    qualify.add_argument("recipe", type=Path)
    qualify.add_argument("--output", type=Path, required=True)
    qualify.add_argument("--window", type=int, default=10)
    qualify.add_argument("--required-ratio", type=float, default=0.90)
    return parser


def _inspect(path: Path, *, checkpoint: bool) -> int:
    if checkpoint:
        manifest = verify_checkpoint(path)
        _print_json({"manifest": manifest.to_dict(), "manifest_sha256": manifest.sha256})
    else:
        receipt = resolve_recipe(path)
        _print_json(
            {
                "recipe": receipt.resolved.to_dict(),
                "recipe_sha256": receipt.sha256,
                "source": str(receipt.source),
            }
        )
    return 0


def _run(recipe_path: Path, output: Path, resume: Path | None = None) -> int:
    recipe = resolve_recipe(recipe_path).resolved
    result = run_reference_recipe(recipe, output=output, resume_from=resume)
    _print_json(
        {
            "global_step": result.state.global_step,
            "last_loss": result.state.last_loss,
            "run_id": result.state.run_id,
            "status": result.state.status.value,
        }
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            return _inspect(args.path, checkpoint=args.checkpoint)
        if args.command == "capabilities":
            dcp = dcp_capability()
            fsdp = fsdp2_capability()
            _print_json(
                {
                    "dcp": {
                        "available": dcp.available,
                        "reason": dcp.reason,
                        "torch_version": dcp.torch_version,
                    },
                    "fsdp2": {
                        "api_available": fsdp.api_available,
                        "cuda_available": fsdp.cuda_available,
                        "distributed_initialized": fsdp.distributed_initialized,
                        "ready": fsdp.ready,
                        "reason": fsdp.reason,
                        "torch_version": fsdp.torch_version,
                    },
                }
            )
            return 0
        if args.command == "run":
            return _run(args.recipe, args.output)
        if args.command == "resume":
            recipe = resolve_recipe(args.recipe).resolved
            result = run_reference_recipe(
                recipe,
                output=args.output,
                resume_from=args.checkpoint,
                allow_reshard=args.allow_reshard,
            )
            _print_json(
                {
                    "global_step": result.state.global_step,
                    "last_loss": result.state.last_loss,
                    "run_id": result.state.run_id,
                    "status": result.state.status.value,
                }
            )
            return 0
        if args.command == "qualify":
            recipe = resolve_recipe(args.recipe).resolved
            result = run_reference_recipe(recipe, output=args.output)
            qualification = qualify_overfit(
                [record.metrics["loss"] for record in result.history],
                window=args.window,
                required_ratio=args.required_ratio,
            )
            _print_json(qualification.to_dict())
            return 0 if qualification.passed else 1
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
