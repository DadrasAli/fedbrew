#!/usr/bin/env python
"""Run this example's arms through `fedbrew run`, then table what they produced.

Not a runner. Every arm here is a shipped config under
``configs/examples/fed-lasso/``, run by the ``fedbrew`` CLI as a subprocess,
against data written by ``fedbrew generate`` from
``data/configs/examples/fed-lasso.yaml``. Nothing in this file composes a
config, registers a component or touches the loop -- which is the whole point:
what the README reports has to be reproducible by someone who never opens this
script.

What it is for is the two jobs the CLI does not do. It runs a directory of
arms in order, and it builds the comparison table afterwards by reading
``outputs/`` -- ``run.json`` for the status and the final round,
``round_metrics.csv`` for the curve. A reader who prefers to do it by hand
runs the nine commands and reads the same files.

Usage
-----
    python examples/fed-lasso/run.py                         # the base dials
    python examples/fed-lasso/run.py --setting fed-lasso-l2
    python examples/fed-lasso/run.py --setting fed-lasso-smooth
    python examples/fed-lasso/run.py --arm fedavg_decay
    python examples/fed-lasso/run.py --table-only            # re-table outputs/
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
#: that writes the data each reads. The null control ships one arm: its
#: claim is that the floor is the penalty's, and FedAvg reaching exactly 0.0
#: at lam = 0 is the whole of it. The smooth control ships all nine, because
#: its claim is about every arm rather than about one.
SETTINGS = ("fed-lasso", "fed-lasso-smooth", "fed-lasso-l2")

#: The order the README's table uses.
ARM_ORDER = (
    "fedavg",
    "fedavg_decay",
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

    ``gap`` is ``central_test_optimality_gap`` and *not* ``central_test_loss``:
    `F*` is 0.164 here rather than 0, so unlike the other examples the loss and
    the gap are different numbers and only the gap says how well the run did.
    The median over the last fifteen rounds is here because the final-round
    value of a floor-limited arm is a question about the phase of an
    oscillation, and a table that reports only the last round hides that.
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
    tail = sorted(gaps[-15:])
    median = tail[len(tail) // 2] if len(tail) % 2 else sum(tail[len(tail) // 2 - 1 :][:2]) / 2
    final = rounds[-1]
    return {
        "arm": output_dir.name,
        "status": run_record.get("status", "?"),
        "rounds": len(rounds),
        "gap": gaps[-1],
        "median_tail": median,
        "best": min(gaps),
        "first_below_1e3": _first_below(gaps, 1e-3),
        "distance_to_optimum": _float(final, "central_test_distance_to_optimum"),
        "distance_to_truth": _float(final, "central_test_distance_to_truth"),
        "support_size": _float(final, "central_test_support_size"),
        "support_f1": _float(final, "central_test_support_f1"),
        "exact_zeros": _float(final, "central_test_exact_zeros"),
        # Per selected client per round: SCAFFOLD moves two model-shaped
        # states per direction and FedLALR three, against FedAvg's one.
        "params_per_round": _float(final, "communicated_parameters"),
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
    ("median_tail", "median gap, last 15", "{:.2e}"),
    ("best", "best gap", "{:.2e}"),
    ("first_below_1e3", "first < 1e-3", "{}"),
    ("distance_to_optimum", "‖x − x*‖", "{:.2e}"),
    ("distance_to_truth", "‖x − x_true‖", "{:.4f}"),
    ("support_size", "support size", "{:.0f}"),
    ("support_f1", "support F1", "{:.2f}"),
    ("exact_zeros", "exact zeros", "{:.0f}"),
    ("params_per_round", "params/round", "{:.0f}"),
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
