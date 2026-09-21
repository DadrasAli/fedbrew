#!/usr/bin/env python
"""Run this example's arms through `fedbrew run`, then table what they produced.

Not a runner. Every arm here is a shipped config under
``configs/examples/pl-1d/``, run by the ``fedbrew`` CLI as a subprocess,
against data written by ``fedbrew generate`` from
``data/configs/examples/pl-1d.yaml``. Nothing in this file composes a config,
registers a component or touches the loop -- which is the whole point: what
the README reports has to be reproducible by someone who never opens this
script.

What it is for is the two jobs the CLI does not do. It runs a directory of
arms in order, and it builds the comparison table afterwards by reading
``outputs/`` -- ``run.json`` for the status and the final round,
``round_metrics.csv`` for the curve. A reader who prefers to do it by hand
runs the seven commands and reads the same files.

Usage
-----
    python examples/pl-1d/run.py                  # all seven arms
    python examples/pl-1d/run.py --arm scaffold
    python examples/pl-1d/run.py --table-only     # re-table outputs/
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
SETTING = "pl-1d"

#: The order the README's table uses: the mechanism each arm brings, roughly
#: in increasing strength, with the control variate last but one.
ARM_ORDER = (
    "fedavg",
    "fedavgm",
    "fedadam",
    "fedyogi",
    "fedadagrad",
    "scaffold",
    "fedlalr",
)


def config_paths(arm: str | None) -> list[Path]:
    """The arm configs to run, in the README's order."""

    directory = REPO_ROOT / "configs" / "examples" / SETTING
    available = {path.stem: path for path in directory.glob("*.yaml")}
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

    ``gap`` is ``central_test_loss``, which for this problem *is* `F(x) - F*`
    for the aggregated iterate, because `F* = 0` exactly. The median over the
    last twenty rounds is here because the final-round value of a
    floor-limited arm is a question about the phase of an oscillation, and a
    table that reports only the last round hides that.
    """

    rounds = read_rounds(output_dir)
    if not rounds:
        return None
    run_record = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    gaps = [value for row in rounds if (value := _float(row, "central_test_loss")) is not None]
    if not gaps:
        return None
    tail = sorted(gaps[-20:])
    median = tail[len(tail) // 2] if len(tail) % 2 else sum(tail[len(tail) // 2 - 1 :][:2]) / 2
    final = rounds[-1]
    return {
        "arm": output_dir.name,
        "status": run_record.get("status", "?"),
        "rounds": len(rounds),
        "gap": gaps[-1],
        "median_tail": median,
        "best": min(gaps),
        "first_below_1e6": _first_below(gaps, 1e-6),
        "first_below_1e12": _first_below(gaps, 1e-12),
        "fit_distance_to_optimum": _float(final, "fit_distance_to_optimum"),
    }


def _first_below(values: list[float], threshold: float) -> int | None:
    """The first round whose gap is under ``threshold``, 1-indexed."""

    for index, value in enumerate(values, start=1):
        if value < threshold:
            return index
    return None


COLUMNS = (
    ("arm", "arm", "{}"),
    ("gap", "gap @ final", "{:.2e}"),
    ("median_tail", "median gap, last 20", "{:.2e}"),
    ("best", "best gap", "{:.2e}"),
    ("first_below_1e6", "first < 1e-6", "{}"),
    ("first_below_1e12", "first < 1e-12", "{}"),
    ("fit_distance_to_optimum", "fit dist to x* @ final", "{:.2e}"),
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
