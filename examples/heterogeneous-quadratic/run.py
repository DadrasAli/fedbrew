#!/usr/bin/env python
"""Run this example's arms through `fedbrew run`, then table what they produced.

Not a runner. Every arm is a shipped config under
``configs/examples/heterogeneous-quadratic/``, run by the ``fedbrew`` CLI as a
subprocess, against data written by ``fedbrew generate`` from
``data/configs/examples/heterogeneous-quadratic.yaml`` (the heterogeneous
quadratic's coupled centre). The table reads ``outputs/`` afterwards:
``run.json`` for the status, ``round_metrics.csv`` for the gap and
``grad_norm_sq``. FedAvg's closed-form exact floor per (alpha, K) is in the
manifest's ``reference``; a stochastic run sits above it by its noise floor.

Usage
-----
    python examples/heterogeneous-quadratic/run.py                  # all eight arms
    python examples/heterogeneous-quadratic/run.py --arm scaffold_k10
    python examples/heterogeneous-quadratic/run.py --table-only     # re-table outputs/
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parent.parent

#: The directory of arm configs, and the generator config that writes the
#: data each reads.
SETTING = "heterogeneous-quadratic"

#: The README's order: FedAvg at K = 1, 10, 100, its equal-sample minibatch
#: baselines, then SCAFFOLD.
ARM_ORDER = (
    "fedavg_k1",
    "fedavg_k10",
    "fedavg_k100",
    "minibatch_sgd_k10",
    "minibatch_sgd_k100",
    "scaffold_k1",
    "scaffold_k10",
    "scaffold_k100",
)


def config_paths(arm: str | None) -> list[Path]:
    """The arm configs to run, in the README's order."""

    directory = REPO_ROOT / "configs" / "examples" / SETTING
    # Not the family base (_base.yaml), which every arm extends and is not run.
    available = {
        path.stem: path for path in directory.glob("*.yaml") if not path.name.startswith("_")
    }
    if arm is not None:
        if arm not in available:
            raise SystemExit(f"unknown arm {arm!r}; {SETTING} has: {', '.join(sorted(available))}")
        return [available[arm]]
    ordered = [available[name] for name in ARM_ORDER if name in available]
    return ordered + [available[name] for name in sorted(set(available) - set(ARM_ORDER))]


def manifest_path() -> Path:
    """Where the generator config writes its manifest."""

    return REPO_ROOT / "data" / "generated" / "examples" / SETTING / "manifest.json"


def run_arm(config_path: Path, quiet: bool) -> None:
    """Run one arm through the fedbrew CLI, from the repository root.

    A subprocess rather than an import: the point of the migration is that
    these arms go through the same entry point every other config does, and
    calling `runner.run` in-process would quietly skip the console script,
    the argument parsing and the exit code.
    """

    command = [sys.executable, "-m", "fedbrew.cli.dispatch", "run", "--config", str(config_path)]
    if quiet:
        command.append("--quiet")
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(f"{config_path} exited {result.returncode}")


def read_rounds(output_dir: Path) -> list[dict[str, str]]:
    """Every row of an arm's round_metrics.csv."""

    path = output_dir / "round_metrics.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], column: str) -> float | None:
    value = row.get(column)
    if value in (None, "", "nan"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def summarise(output_dir: Path) -> dict[str, Any] | None:
    """The columns the README's table reports, read off one arm's artifacts.

    The gap and ``grad_norm_sq`` at the last round, and their medians over the
    last ten measured rounds: a stochastic arm's last value is one draw from
    its noise.
    """

    rounds = read_rounds(output_dir)
    if not rounds:
        return None
    run_record = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    measured = [row for row in rounds if _float(row, "central_test_optimality_gap") is not None]
    if not measured:
        return None
    gaps = [_float(row, "central_test_optimality_gap") for row in measured]
    norms = [value for row in measured if (value := _float(row, "grad_norm_sq")) is not None]
    return {
        "arm": output_dir.name,
        "status": run_record.get("status", "?"),
        "rounds": len(rounds),
        "gap": gaps[-1],
        "median_gap": _median(gaps[-10:]),
        "grad_norm_sq": norms[-1] if norms else None,
        "median_grad_norm_sq": _median(norms[-10:]) if norms else None,
    }


def _median(values: list[Any]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


COLUMNS = (
    ("arm", "arm", "{}"),
    ("gap", "gap @ final", "{:.2e}"),
    ("median_gap", "median gap, last 10 measured", "{:.2e}"),
    ("grad_norm_sq", "grad_norm_sq @ final", "{:.2e}"),
    ("median_grad_norm_sq", "median grad_norm_sq, last 10", "{:.2e}"),
    ("status", "status", "{}"),
)


def render(rows: list[dict[str, Any]]) -> str:
    """The comparison table, as the README's Markdown."""

    header = "| " + " | ".join(label for _, label, _ in COLUMNS) + " |"
    rule = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    lines = [header, rule]
    for row in rows:
        cells = []
        for key, _, form in COLUMNS:
            value = row.get(key)
            cells.append("—" if value is None else form.format(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Run the arms, then print the table built from outputs/."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", help="One arm, rather than the whole directory.")
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Skip the runs and table whatever outputs/ already holds.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress each run's surface.")
    args = parser.parse_args(argv)

    paths = config_paths(args.arm)
    if not args.table_only:
        manifest = manifest_path()
        if not manifest.is_file():
            raise SystemExit(
                f"{manifest} does not exist. Generate it first:\n"
                f"  fedbrew generate --config data/configs/examples/{SETTING}.yaml"
            )
        for path in paths:
            run_arm(path, args.quiet)

    rows = []
    for path in paths:
        summary = summarise(REPO_ROOT / "outputs" / "examples" / SETTING / path.stem)
        if summary is not None:
            rows.append(summary)
    if not rows:
        raise SystemExit("no arm produced a round_metrics.csv to table")
    print(render(rows))


if __name__ == "__main__":
    main()
