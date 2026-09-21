#!/usr/bin/env python
"""Run this example's arms through `fedbrew run`, then table what they produced.

Not a runner. Every arm here is a shipped config under
``configs/examples/nonconvex-simplex/``, run by the ``fedbrew`` CLI as a subprocess,
against data written by ``fedbrew generate`` from
``data/configs/examples/nonconvex-simplex.yaml``. Nothing in this file composes a
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
    python examples/nonconvex-simplex/run.py                         # the base dials
    python examples/nonconvex-simplex/run.py --arm scaffold
    python examples/nonconvex-simplex/run.py --table-only            # re-table outputs/
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

#: One directory of arm configs, and the generator config that writes the data
#: it reads. There is no control setting: the obvious one -- remove the decoy
#: -- is refused by `ProblemSpec.__post_init__`, which requires the star to
#: out-rank the clique spectrally. The README says so.
SETTINGS = ("nonconvex-simplex",)

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

    ``gap`` is ``central_test_optimality_gap``, and on this problem it heads to
    `-inf`: there is no unconstrained minimum to converge to. ``feasible_gap``
    projects the iterate first, and it goes *up*, to `+0.4` -- a large iterate
    projects onto its single largest coordinate, which is the star's hub,
    which spans no edge. Post-hoc projection is not a rescue here, and that
    column is what says so.
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
    feasible = [
        value for row in rounds if (value := _float(row, "central_test_feasible_gap")) is not None
    ]
    best_round = min(range(len(feasible)), key=feasible.__getitem__) if feasible else None
    return {
        "arm": output_dir.name,
        "status": run_record.get("status", "?"),
        "rounds": len(rounds),
        "loss": _float(final, "central_test_loss"),
        "gap": gaps[-1],
        "feasible_gap": _float(final, "central_test_feasible_gap"),
        "iterate_norm": _float(final, "central_test_iterate_norm"),
        "simplex_sum": _float(final, "central_test_simplex_sum"),
        "mass_on_clique": _float(final, "central_test_mass_on_clique"),
        # The best feasible point the run ever visits, and when. It is round 2
        # on most arms: before the divergence takes over, after which every
        # arm gets monotonically worse forever.
        "best_feasible_gap": feasible[best_round] if best_round is not None else None,
        "best_feasible_round": None if best_round is None else best_round + 1,
    }


def _first_below(values: list[float], threshold: float) -> int | None:
    """The first round whose gap is under ``threshold``, 1-indexed."""

    for index, value in enumerate(values, start=1):
        if value < threshold:
            return index
    return None


COLUMNS = (
    ("arm", "arm", "{}"),
    ("loss", "loss @ 150", "{:.2e}"),
    ("gap", "gap @ 150", "{:.2e}"),
    ("feasible_gap", "feasible gap @ 150", "{:+.4f}"),
    ("iterate_norm", "‖x‖ @ 150", "{:.2e}"),
    ("simplex_sum", "Σx", "{:.2e}"),
    ("mass_on_clique", "mass on clique", "{:.4f}"),
    ("best_feasible_gap", "best feasible gap", "{:.4f}"),
    ("best_feasible_round", "at round", "{}"),
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
    parser.add_argument("--arm", help="One arm, rather than the whole directory.")
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Skip the runs and table whatever outputs/ already holds.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress each run's surface.")
    args = parser.parse_args(argv)

    paths = config_paths(SETTINGS[0], args.arm)
    if not args.table_only:
        manifest = manifest_for(SETTINGS[0])
        if not manifest.is_file():
            raise SystemExit(
                f"{manifest} does not exist. Generate it first:\n"
                f"  fedbrew generate --config data/configs/examples/{SETTINGS[0]}.yaml"
            )
        for path in paths:
            run_arm(path, args.quiet)

    rows = []
    for path in paths:
        summary = summarise(REPO_ROOT / "outputs" / "examples" / SETTINGS[0] / path.stem)
        if summary is not None:
            rows.append(summary)
    if not rows:
        raise SystemExit("no arm produced a round_metrics.csv to table")
    print(render(rows))


if __name__ == "__main__":
    main()
