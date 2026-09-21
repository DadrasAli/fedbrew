#!/usr/bin/env python
"""Run this example's arms through `fedbrew run`, then table what they produced.

Not a runner. Every arm here is a shipped config under
``configs/examples/simplex-lsq/``, run by the ``fedbrew`` CLI as a subprocess,
against data written by ``fedbrew generate`` from
``data/configs/examples/simplex-lsq.yaml``. Nothing in this file composes a
config, registers a component or touches the loop -- which is the whole point:
what the README reports has to be reproducible by someone who never opens this
script.

What it is for is the two jobs the CLI does not do. It runs a directory of
arms in order, and it builds the comparison table afterwards by reading
``outputs/`` -- ``run.json`` for the status and the final round,
``round_metrics.csv`` for the curve. A reader who prefers to do it by hand
runs the eight commands and reads the same files.

Usage
-----
    python examples/simplex-lsq/run.py                         # the base dials
    python examples/simplex-lsq/run.py --setting simplex-lsq-feasible
    python examples/simplex-lsq/run.py --arm scaffold
    python examples/simplex-lsq/run.py --table-only            # re-table outputs/
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

#: One directory of arm configs per dial setting, and the generator config
#: that writes the data each reads. The feasible control ships three arms,
#: one per family: its claim is that the constraint is responsible for the
#: whole base table, and "every arm is right" is a claim about more than one.
SETTINGS = ("simplex-lsq", "simplex-lsq-feasible")

#: The order the README's table uses.
ARM_ORDER = (
    "fedavg",
    "fedprox",
    "fedavgm",
    "fedadam",
    "fedyogi",
    "fedadagrad",
    "scaffold",
    "fedlalr",
)


def config_paths(setting: str, arm: str | None) -> list[Path]:
    """The arm configs to run, in the README's order."""

    directory = REPO_ROOT / "configs" / "examples" / setting
    if not directory.is_dir():
        raise SystemExit(f"no such setting: {setting} ({directory} does not exist)")
    available = {path.stem: path for path in directory.glob("*.yaml")}
    if arm is not None:
        if arm not in available:
            raise SystemExit(f"unknown arm {arm!r}; {setting} has: {', '.join(sorted(available))}")
        return [available[arm]]
    ordered = [available[name] for name in ARM_ORDER if name in available]
    return ordered + [available[name] for name in sorted(set(available) - set(ARM_ORDER))]


def manifest_for(setting: str) -> Path:
    """Where the generator config for this setting writes its manifest."""

    return REPO_ROOT / "data" / "generated" / "examples" / setting / "manifest.json"


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

    ``gap`` is ``central_test_optimality_gap``, and on this problem it is
    **negative**: the arms converge to the unconstrained optimum, which beats
    every feasible point. ``feasible_gap`` projects the iterate first and is
    the number a reader actually wants. Both are reported, in that order,
    because a table that showed only the first would say the opposite of what
    happened.
    """

    rounds = read_rounds(output_dir)
    if not rounds:
        return None
    run_record = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    gaps = [
        value for row in rounds if (value := _float(row, "central_test_optimality_gap")) is not None
    ]
    if not gaps:
        return None
    final = rounds[-1]
    return {
        "arm": output_dir.name,
        "status": run_record.get("status", "?"),
        "rounds": len(rounds),
        "gap": gaps[-1],
        "first_negative": _first_below(gaps, 0.0),
        "feasible_gap": _float(final, "central_test_feasible_gap"),
        "violation": _float(final, "central_test_constraint_violation"),
        "simplex_sum": _float(final, "central_test_simplex_sum"),
        "min_coordinate": _float(final, "central_test_min_coordinate"),
        "negative_mass": _float(final, "central_test_negative_mass"),
        "distance_to_optimum": _float(final, "central_test_distance_to_optimum"),
    }


def _first_below(values: list[float], threshold: float) -> int | None:
    """The first round whose gap is under ``threshold``, 1-indexed."""

    for index, value in enumerate(values, start=1):
        if value < threshold:
            return index
    return None


COLUMNS = (
    ("arm", "arm", "{}"),
    ("gap", "gap @ 150", "{:.4f}"),
    ("first_negative", "first round gap < 0", "{}"),
    ("feasible_gap", "feasible gap @ 150", "{:.1e}"),
    ("violation", "violation", "{:.4f}"),
    ("simplex_sum", "Σx", "{:.4f}"),
    ("min_coordinate", "min x_j", "{:.4f}"),
    ("negative_mass", "negative mass", "{:.4f}"),
    ("distance_to_optimum", "‖x − x*‖", "{:.4f}"),
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
    """Run a setting's arms, then print the table built from outputs/."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--setting",
        default=SETTINGS[0],
        choices=SETTINGS,
        help="Which dial setting's arm configs to run.",
    )
    parser.add_argument("--arm", help="One arm, rather than the whole directory.")
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Skip the runs and table whatever outputs/ already holds.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress each run's surface.")
    args = parser.parse_args(argv)

    paths = config_paths(args.setting, args.arm)
    if not args.table_only:
        manifest = manifest_for(args.setting)
        if not manifest.is_file():
            raise SystemExit(
                f"{manifest} does not exist. Generate it first:\n"
                f"  fedbrew generate --config data/configs/examples/{args.setting}.yaml"
            )
        for path in paths:
            run_arm(path, args.quiet)

    rows = []
    for path in paths:
        summary = summarise(REPO_ROOT / "outputs" / "examples" / args.setting / path.stem)
        if summary is not None:
            rows.append(summary)
    if not rows:
        raise SystemExit("no arm produced a round_metrics.csv to table")
    print(render(rows))


if __name__ == "__main__":
    main()
