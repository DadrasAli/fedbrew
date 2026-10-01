"""``fedbrew tune --config X``: choose X's dials by its ``tuning`` section.

Runs the section's method (``grid_and_edge`` or ``pilot``) over the dials it
names, through ``fedbrew sweep``, scores the runs from their CSVs, extends the
grid past an edge until the pick is interior, and writes the evidence and the
config with the chosen values (``fedbrew/core/tuning.py``).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from fedbrew.core.refusal import RunRefused

if TYPE_CHECKING:
    from fedbrew.core.tuning import Tuner


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fedbrew tune",
        description=(
            "Choose a config's dials by its tuning section: run the grid with fedbrew sweep, "
            "score it, extend it past an edge until the pick is interior, and write the "
            "evidence and the config with the chosen values."
        ),
    )
    parser.add_argument("--config", required=True, help="the run config with a tuning section")
    parser.add_argument(
        "--out",
        default=None,
        help="the tune's directory (default: tuning.output_dir, else <output_dir>-tuning)",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="write the first grid's configs and print how they group, and run nothing",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    from fedbrew.core.config import load_config, standalone_config_mapping
    from fedbrew.core.paths import resolve_output_dir
    from fedbrew.core.tuning import TuneError, Tuner

    path = Path(args.config)
    try:
        config = load_config(path)
        if config.tuning.method is None:
            raise RunRefused(f"{path} has no tuning section with a method: nothing to tune")
        out = Path(
            args.out
            or config.tuning.output_dir
            or f"{resolve_output_dir(config.experiment.output_dir)}-tuning"
        )
        tuner = Tuner(path, standalone_config_mapping(path), config.tuning, config, out)
        if args.plan:
            _plan(tuner)
            return
        selection = tuner.tune()
        tuner.write(selection)
    except (RunRefused, TuneError) as error:
        print(f"fedbrew tune: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    chosen = selection.selected
    settings = selection.settings
    sign = 1.0 if settings.direction == "min" else -1.0
    print(
        f"fedbrew tune: {settings.method.name} on {settings.metric}: picked {chosen.label} "
        f"(score {sign * chosen.aggregate:.6g}, "
        f"{'interior' if selection.interior else 'on an edge'}"
        f"; {len(selection.candidates)} candidates in {len(selection.steps)} steps)"
    )
    others = [c.label for c in selection.tied if c is not chosen]
    if others:
        print(f"fedbrew tune: tied with it: {', '.join(others)}")
    print(f"fedbrew tune: stopped: {selection.stop['reason']}")
    print(
        f"fedbrew tune: wrote {out}/selection.json, evidence.csv, analysis/ and {selection.config}"
    )


def _plan(tuner: Tuner) -> None:
    from fedbrew.core.sweep import _print_plan, plan

    candidates = tuner.plan_step(0)
    seeds = tuner.settings.seeds
    paths = []
    for candidate in candidates:
        for seed in seeds:
            mapping = tuner.mapping_for(candidate.point, candidate.label, seed)
            path, _ = tuner._write_config(candidate.label, seed, mapping)
            paths.append(path)
    print(
        f"fedbrew tune: {len(candidates)} candidates x {len(seeds)} seeds in the first grid; "
        "the grid grows past an edge as the scores ask"
    )
    _print_plan(plan(paths))


if __name__ == "__main__":
    main()
