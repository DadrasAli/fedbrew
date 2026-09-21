"""Profile a few FL rounds to attribute the per-client client_eval overhead.

client_eval carries per-client time that is not compute, which shows up when it
is differenced against global_eval, the same examples pushed through one
contiguous tensor. This runs the real runner under cProfile so the cost can be
attributed to actual functions instead of guessed at.

Note: cProfile inflates Python-call-heavy code, which is exactly what is being
measured here. Use this for ATTRIBUTION (what dominates), not for absolute
timings - confirm any fix with a plain timed run.

Usage: python tools/profile_client_eval.py --config <cfg> --rounds 3 [runner args]
"""

from __future__ import annotations

import cProfile
import pstats
import sys
from pathlib import Path

from fedbrew.core.runner import main as runner_main


def main() -> None:
    argv = sys.argv[1:]
    out = Path("profile_client_eval.prof")
    if "--profile-out" in argv:
        index = argv.index("--profile-out")
        out = Path(argv[index + 1])
        del argv[index : index + 2]

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        runner_main(argv)
    finally:
        profiler.disable()
        profiler.dump_stats(str(out))

    stats = pstats.Stats(str(out))
    print("\n" + "=" * 78)
    print("TOP 35 BY CUMULATIVE TIME")
    print("=" * 78)
    stats.sort_stats("cumulative").print_stats(35)

    print("\n" + "=" * 78)
    print("TOP 35 BY SELF (TOTTIME)")
    print("=" * 78)
    stats.sort_stats("tottime").print_stats(35)

    # The eval path specifically: everything reachable from the per-client
    # evaluate() call, plus the pool churn that _release_client causes.
    for pattern in (
        "evaluate",
        "clone_model_state",
        "load_federated_model_state",
        "load_state_dict",
        "release_client",
        "build_dataloader",
        "get_client_data",
        "build_model",
    ):
        print("\n" + "=" * 78)
        print(f"CALLERS OF: {pattern}")
        print("=" * 78)
        stats.print_callers(pattern)


if __name__ == "__main__":
    main()
