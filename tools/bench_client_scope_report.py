"""Summarise the client_scope=all vs selected benchmark.

Reads the per-round history from round_metrics.csv -- the only round-level
artifact `fedbrew run` writes -- and reports the phase breakdown plus the
all-vs-selected speedup at each local_iterations.

Each run directory under <bench_root> is named li<local_iterations>_<scope>,
e.g. li4_selected. The prefix was le, for the key's old name local_epochs;
directories named that way are not read.

Usage: python tools/bench_client_scope_report.py <bench_root> [--warmup 2]
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from fedbrew.core.artifacts import load_round_metrics_csv
from fedbrew.core.state import MetricRecord

PHASES = ("fit", "aggregate", "client_eval", "global_eval", "checkpoint")


def load_rounds(run_dir: Path, warmup: int) -> list[MetricRecord]:
    """Return completed rounds with timing data, with the warmup rounds dropped."""
    records = [record for record in load_round_metrics_csv(run_dir) if record.timings]
    return records[warmup:]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def fmt(seconds: float) -> str:
    if seconds != seconds:  # NaN
        return "  n/a"
    if seconds < 60:
        return f"{seconds:6.1f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bench_root", type=Path)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    runs: dict[tuple[str, str], dict] = {}
    for run_dir in sorted(args.bench_root.glob("li*")):
        parts = run_dir.name.split("_", 1)
        li, scope = (parts[0], parts[1] if len(parts) > 1 else "-")
        li = li.removeprefix("li")
        records = load_rounds(run_dir, args.warmup)
        if not records:
            print(f"!! no timed rounds in {run_dir}")
            continue
        runs[(li, scope)] = {
            "n": len(records),
            "total": mean([r.timings.total for r in records]),
            **{p: mean([getattr(r.timings, p) for r in records]) for p in PHASES},
            "acc": [r.metrics.get("central_test_accuracy") for r in records],
        }

    if not runs:
        raise SystemExit("no runs found")

    header = f"{'run':>16} {'rounds':>6} {'total':>8} " + " ".join(f"{p:>11}" for p in PHASES)
    print(header)
    print("-" * len(header))
    for (li, scope), r in sorted(runs.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])):
        row = f"{f'li{li}/{scope}':>16} {r['n']:>6} {fmt(r['total']):>8} "
        row += " ".join(f"{fmt(r[p]):>11}" for p in PHASES)
        print(row)

    print("\nSpeedup (all -> selected):")
    print(f"{'local_iterations':>16} {'all':>9} {'selected':>10} {'speedup':>9} {'500 rounds':>22}")
    for li in sorted({k[0] for k in runs}, key=int):
        a, s = runs.get((li, "all")), runs.get((li, "selected"))
        if not a or not s:
            continue
        speedup = a["total"] / s["total"]
        proj = f"{a['total'] * 500 / 3600:.1f}h -> {s['total'] * 500 / 3600:.1f}h"
        print(f"{li:>16} {fmt(a['total']):>9} {fmt(s['total']):>10} {speedup:>8.2f}x {proj:>22}")

    print("\nclient_eval share of round:")
    for (li, scope), r in sorted(runs.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])):
        share = 100.0 * r["client_eval"] / r["total"]
        print(f"{f'li{li}/{scope}':>16} {share:5.1f}%")

    # client_scope only changes what is measured, never what is trained, so the
    # two scopes must produce identical global_test accuracy at the same seed.
    print("\nTrajectory check (central_test_accuracy, all vs selected):")
    for li in sorted({k[0] for k in runs}, key=int):
        a, s = runs.get((li, "all")), runs.get((li, "selected"))
        if not a or not s:
            continue
        if len(a["acc"]) != len(s["acc"]):
            print(
                f"{f'li{li}':>16} round count differs: all={len(a['acc'])}, "
                f"selected={len(s['acc'])}"
            )
            continue
        pairs = [
            (x, y)
            for x, y in zip(a["acc"], s["acc"], strict=True)
            if isinstance(x, (int, float)) and isinstance(y, (int, float))
        ]
        if not pairs:
            print(f"{f'li{li}':>16} no comparable accuracies")
            continue
        worst = max(abs(x - y) for x, y in pairs)
        verdict = "IDENTICAL" if worst < 1e-6 else f"DIVERGES (max |diff| = {worst:.2e})"
        print(f"{f'li{li}':>16} {verdict} over {len(pairs)} rounds")


if __name__ == "__main__":
    main()
