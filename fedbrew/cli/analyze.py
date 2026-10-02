"""``fedbrew analyze``: summaries of finished runs, from their round_metrics.csv.

Per run the last iterate, the best so far, the mean of log10 of a metric and the
running mean over the iterates; across the seeds of a group of runs the median,
minimum, maximum, the mean with its upward and downward RMS deviations, and
chosen quantiles of each, per round and at the end
(``fedbrew/core/analysis.py``). Reads the runs' files and writes four tables
into ``--out``; it runs no experiment and imports no model.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from fedbrew.core.analysis import (
    DEFAULT_METRICS,
    DEFAULT_QUANTILES,
    OUTPUT_FILES,
    STATISTICS,
    Analysis,
    AnalysisError,
    analyze,
    quantile_name,
    write_tables,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fedbrew analyze",
        description=(
            "Summarize finished runs from their round_metrics.csv: per run the last iterate, "
            "the best so far, the mean of log10 of a metric and the running mean; across the "
            "seeds of each config the median, min, max, the mean with its upward and downward "
            "RMS deviations, and quantiles, per round and final."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help=(
            "run directories, directories holding runs (searched below), or the config files "
            "of a sweep (their output directories)"
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        metavar="COLUMN",
        help=(
            "round columns to analyze (a bare name may leave out central_test_); default: "
            + ", ".join(DEFAULT_METRICS)
            + ", those a run wrote"
        ),
    )
    parser.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=list(DEFAULT_QUANTILES),
        metavar="Q",
        help="quantiles in [0, 1] written beside the median, min and max (default: 0.25 0.75)",
    )
    parser.add_argument(
        "--direction",
        action="append",
        default=[],
        metavar="COLUMN=min|max",
        help="which side of a metric is better, where the default (max for accuracy and F1, "
        "min for everything else) is wrong; repeatable",
    )
    parser.add_argument(
        "--out",
        default="analysis",
        help=f"directory the tables are written to: {', '.join(OUTPUT_FILES)} (default: analysis)",
    )
    return parser.parse_args(argv)


def _directions(items: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        name, separator, direction = item.partition("=")
        if not separator or not name:
            raise AnalysisError(f"--direction takes COLUMN=min or COLUMN=max, got {item!r}")
        out[name] = direction
    return out


def summary_lines(result: Analysis) -> list[str]:
    """One line per group and metric: the median over seeds of each per-run statistic."""

    lines = []
    rows = {(r["group"], r["metric"], r["statistic"]): r for r in result.group_rows()}
    for group in result.groups:
        for metric in result.metrics:
            if (group, metric, STATISTICS[0]) not in rows:
                continue
            n = int(rows[(group, metric, STATISTICS[0])]["n"])
            cells = "  ".join(
                f"{name} {rows[(group, metric, name)]['median']:.6g}" for name in STATISTICS
            )
            lines.append(f"  {group}  {metric}  n={n}  median over seeds of: {cells}")
    return lines


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        result = analyze(
            args.paths,
            metrics=args.metrics,
            quantiles=args.quantiles,
            directions=_directions(args.direction),
        )
    except AnalysisError as error:
        print(f"fedbrew analyze: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    paths = write_tables(result, args.out)
    runs = len({s.run for s in result.runs})
    print(
        f"fedbrew analyze: {runs} runs in {len(result.groups)} groups, "
        f"metrics {', '.join(result.metrics)}, quantiles "
        f"{', '.join(quantile_name(q) for q in result.quantiles)}"
    )
    for line in summary_lines(result):
        print(line)
    for note in result.notes:
        print(f"fedbrew analyze: note: {note}", file=sys.stderr)
    for warning in result.warnings:
        print(f"fedbrew analyze: warning: {warning}", file=sys.stderr)
    print("fedbrew analyze: wrote " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
