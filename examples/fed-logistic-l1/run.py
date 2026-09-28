#!/usr/bin/env python
"""Run one corpus's arms through `fedbrew run`, then table what they produced.

Not a runner. Every arm is a shipped config under
``configs/examples/<corpus>/``, one problem at one `lam` each, run by the
``fedbrew`` CLI as a subprocess, against the corpus written by ``fedbrew
generate`` from ``data/configs/examples/<corpus>.yaml``. Nothing here composes a config,
registers a component or touches the loop: it runs a directory of arms in
order, and tables their final round from ``outputs/`` afterwards.

Usage
-----
    python examples/fed-logistic-l1/run.py                    # the synthetic corpus
    python examples/fed-logistic-l1/run.py --corpus fed-logistic-l1-a9a \\
        --arm logistic-l2sq-lambda0.001
    python examples/fed-logistic-l1/run.py --table-only       # re-table outputs/
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parent.parent
DEFAULT_CORPUS = "fed-logistic-l1-synthetic"

#: The columns tabled, as (central_test_ column, label, format). A problem
#: without a certified optimum writes no gap and no distance to it.
COLUMNS = (
    ("loss", "F(x) @ final", "{:.6e}"),
    ("optimality_gap", "gap @ final", "{:.2e}"),
    ("distance_to_optimum", "‖x − x*‖", "{:.2e}"),
    ("distance_to_truth", "‖x − x_true‖", "{:.4f}"),
    ("support_size", "support size", "{:.0f}"),
    ("support_f1", "support F1", "{:.2f}"),
    ("exact_zeros", "exact zeros", "{:.0f}"),
)


def config_paths(corpus: str, arm: str | None) -> list[Path]:
    """The corpus's arm configs to run, sorted, or the one asked for."""

    directory = REPO_ROOT / "configs" / "examples" / corpus
    if not directory.is_dir():
        raise SystemExit(f"{directory} does not exist")
    available = {path.stem: path for path in sorted(directory.glob("*.yaml"))}
    if arm is None:
        return list(available.values())
    if arm not in available:
        raise SystemExit(f"unknown arm {arm!r}; {corpus} has: {', '.join(available)}")
    return [available[arm]]


def output_dir_of(config_path: Path) -> Path:
    """Where this arm writes, as the config itself says."""

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return REPO_ROOT / config["experiment"]["output_dir"]


def run_arm(config_path: Path, quiet: bool) -> None:
    """Run one arm through the fedbrew CLI, from the repository root."""

    command = [sys.executable, "-m", "fedbrew.cli.dispatch", "run", "--config", str(config_path)]
    if quiet:
        command.append("--quiet")
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(f"{config_path} exited {result.returncode}")


def summarise(arm: str, output_dir: Path) -> dict[str, Any] | None:
    """The final round's central columns, and the run's status."""

    path = output_dir / "round_metrics.csv"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        rounds = list(csv.DictReader(handle))
    if not rounds:
        return None
    record = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    final = rounds[-1]
    row: dict[str, Any] = {"arm": arm, "status": record.get("status", "?"), "rounds": len(rounds)}
    for column, _, _ in COLUMNS:
        value = final.get(f"central_test_{column}")
        row[column] = float(value) if value not in (None, "", "nan") else None
    return row


def render(rows: list[dict[str, Any]]) -> str:
    """The table, as Markdown."""

    labels = ["arm", "rounds", *(label for _, label, _ in COLUMNS), "status"]
    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join("---" for _ in labels) + " |"]
    for row in rows:
        cells = [row["arm"], str(row["rounds"])]
        for key, _, form in COLUMNS:
            cells.append("—" if row[key] is None else form.format(row[key]))
        cells.append(row["status"])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Run the arms, then print the table built from outputs/."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, help="The corpus's directory name.")
    parser.add_argument("--arm", help="One arm, rather than the whole directory.")
    parser.add_argument(
        "--table-only", action="store_true", help="Table whatever outputs/ already holds."
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress each run's surface.")
    args = parser.parse_args(argv)

    paths = config_paths(args.corpus, args.arm)
    if not args.table_only:
        manifest = REPO_ROOT / "data" / "generated" / "examples" / args.corpus / "manifest.json"
        if not manifest.is_file():
            raise SystemExit(
                f"{manifest} does not exist. Generate it first:\n"
                f"  fedbrew generate --config data/configs/examples/{args.corpus}.yaml"
            )
        for path in paths:
            run_arm(path, args.quiet)

    rows = [row for path in paths if (row := summarise(path.stem, output_dir_of(path)))]
    if not rows:
        raise SystemExit("no arm produced a round_metrics.csv to table")
    print(render(rows))


if __name__ == "__main__":
    main()
