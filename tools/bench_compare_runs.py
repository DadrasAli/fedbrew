"""Compare two client_scope benchmark roots: timings and metric equivalence.

Used to verify an optimization changed only wall clock. Any difference in
central_test_accuracy between the two roots means the change altered results and
must be rejected, however good the speedup looks. Run directories are named
li<local_iterations>_<scope>, as for tools/bench_client_scope_report.py; the old
le prefix is not read.

Usage: python tools/bench_compare_runs.py <before_root> <after_root> [--warmup 2]
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from fedbrew.core.artifacts import load_round_metrics_csv

PHASES = ("fit", "client_eval", "global_eval")


def load(run_dir: Path, warmup: int):
    records = [record for record in load_round_metrics_csv(run_dir) if record.timings]
    if not records:
        return None
    kept = records[warmup:]
    return {
        "total": statistics.fmean([r.timings.total for r in kept]),
        **{p: statistics.fmean([getattr(r.timings, p) for r in kept]) for p in PHASES},
        # Accuracy comparison uses EVERY round including warmup: correctness
        # does not get a grace period.
        "acc": [r.metrics.get("central_test_accuracy") for r in records],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    names = sorted(
        {d.name for d in args.before.glob("li*_*")} & {d.name for d in args.after.glob("li*_*")},
        key=lambda n: (int(n.split("_")[0].removeprefix("li")), n),
    )
    if not names:
        raise SystemExit("no run directories in common")

    header = (
        f"{'run':>16} {'total before':>13} {'total after':>12} {'speedup':>8} "
        f"{'client_eval before':>19} {'client_eval after':>18} {'speedup':>8}"
    )
    print(header)
    print("-" * len(header))
    for name in names:
        b, a = load(args.before / name, args.warmup), load(args.after / name, args.warmup)
        if not b or not a:
            print(f"{name:>16} missing data")
            continue
        print(
            f"{name:>16} {b['total']:12.1f}s {a['total']:11.1f}s "
            f"{b['total'] / a['total']:7.2f}x {b['client_eval']:18.1f}s "
            f"{a['client_eval']:17.1f}s {b['client_eval'] / max(a['client_eval'], 1e-9):7.2f}x"
        )

    print("\nCorrectness - central_test_accuracy must be identical:")
    all_ok = True
    for name in names:
        b, a = load(args.before / name, args.warmup), load(args.after / name, args.warmup)
        if not b or not a:
            continue
        if len(b["acc"]) != len(a["acc"]):
            print(f"{name:>16} round count differs: before={len(b['acc'])}, after={len(a['acc'])}")
            all_ok = False
            continue
        pairs = [
            (x, y)
            for x, y in zip(b["acc"], a["acc"], strict=True)
            if isinstance(x, (int, float)) and isinstance(y, (int, float))
        ]
        worst = max((abs(x - y) for x, y in pairs), default=None)
        if worst is None:
            print(f"{name:>16} no comparable rounds")
            all_ok = False
        elif worst == 0.0:
            print(f"{name:>16} IDENTICAL (exact) over {len(pairs)} rounds")
        else:
            print(f"{name:>16} DIFFERS - max |diff| = {worst:.3e} over {len(pairs)} rounds")
            all_ok = False

    print("\nVERDICT:", "results unchanged" if all_ok else "RESULTS CHANGED - reject")


if __name__ == "__main__":
    main()
