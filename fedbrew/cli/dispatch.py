"""Single ``fedbrew`` console-script entry point.

Dispatches to each subcommand's own, unmodified ``main()`` -- this module
only decides which one runs, then hands it the remaining argv exactly as if
it had been invoked directly (``python -m <module> <remaining args>``). No
subcommand's flags are redefined here.
"""

from __future__ import annotations

import argparse
import sys
from importlib import import_module

#: Subcommand name -> (target module, one-line help for ``fedbrew --help``).
COMMANDS: dict[str, tuple[str, str]] = {
    "run": ("fedbrew.core.runner", "Run a fedbrew experiment."),
    "generate": ("fedbrew.data.generate", "Generate federated data manifests."),
    "report": ("fedbrew.cli.make_report", "Create fedbrew Markdown reports."),
    "inspect-data": (
        "fedbrew.cli.inspect_generated_data",
        "Validate and inspect one generated federated dataset.",
    ),
    "cleanup": (
        "fedbrew.cli.cleanup",
        "Remove runtime artifacts while preserving selected paths.",
    ),
    "check-hpc": (
        "fedbrew.cli.check_hpc_environment",
        "Print local runtime and path information for HPC jobs.",
    ),
    "prepare-llm": (
        "fedbrew.data.llm_assets.prepare",
        "Prepare a pinned Hugging Face causal-LM for offline use.",
    ),
    "prepare-oasst1": (
        "fedbrew.data.oasst1",
        "Prepare pinned OASST1 train/validation snapshots for offline use.",
    ),
    "eval-medmcqa": (
        "fedbrew.cli.eval_medmcqa_choice",
        "Multiple-choice accuracy on the held-out MedMCQA split, from a run checkpoint.",
    ),
    "eval-base-model": (
        "fedbrew.cli.eval_base_model",
        "Evaluate the frozen pretrained base model on a run's global test set.",
    ),
    "list-common-datasets": (
        "fedbrew.cli.list_common_datasets",
        "List top-level folders under $COMMON_DATASETS.",
    ),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fedbrew",
        description="fedbrew: a modular research-grade federated learning benchmark framework.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")
    for name, (_, help_text) in COMMANDS.items():
        subparsers.add_parser(name, help=help_text, add_help=False)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse only the subcommand name, then hand off the rest of argv untouched."""

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()

    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        raise SystemExit(0 if argv else 2)

    command, remainder = argv[0], argv[1:]
    if command not in COMMANDS:
        parser.error(
            f"argument command: invalid choice: {command!r} "
            f"(choose from {', '.join(repr(name) for name in COMMANDS)})"
        )

    target, _ = COMMANDS[command]
    module = import_module(target)

    original_argv = sys.argv
    try:
        sys.argv = [f"fedbrew {command}", *remainder]
        module.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
