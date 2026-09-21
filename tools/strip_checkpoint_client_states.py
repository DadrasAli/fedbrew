"""Rewrite best.pt checkpoints without their per-client state.

best.pt answers "which model scored highest". latest.pt and round_*.pt answer
"how do I continue" -- and find_latest_checkpoint only ever reads those two, so
best.pt never needs client state. For SCAFFOLD it carries one model-sized
control variate per client, which on FEMNIST's 3597 writers makes a 40 GB file
out of an 11 MB model.

This rewrites in place via a temporary file, so an interrupted run leaves the
original intact. Reports what it would do with --dry-run.

Usage: python tools/strip_checkpoint_client_states.py outputs/femnist [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument(
        "--name",
        default="best.pt",
        help=(
            "which checkpoint to rewrite. latest.pt and round_*.pt are what a resume "
            "reads, and for SCAFFOLD a resume needs the client state this drops -- "
            "loop._refuse_a_half_restored_resume refuses such a checkpoint rather than "
            "restoring half of it"
        ),
    )
    parser.add_argument(
        "--min-gb", type=float, default=1.0, help="skip files smaller than this; they carry no bulk"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = sorted(
        path
        for root in args.roots
        for path in root.rglob(f"checkpoints/{args.name}")
        if path.stat().st_size >= args.min_gb * 1e9
    )
    if not targets:
        print("nothing to do")
        return

    print(f"{len(targets)} candidate {args.name} files >= {args.min_gb} GB\n")
    freed = 0.0
    for path in targets:
        before = path.stat().st_size
        if args.dry_run:
            print(f"  would rewrite {before / 1e9:7.2f} GB  {path}")
            continue

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "client_states" not in payload:
            print(f"  skip (no client_states) {before / 1e9:7.2f} GB  {path}")
            del payload
            continue

        stripped = {k: v for k, v in payload.items() if k != "client_states"}
        del payload
        # Write beside the original and rename, so an interrupted run never
        # leaves a half-written checkpoint where a valid one used to be.
        temp = path.with_suffix(".pt.stripping")
        torch.save(stripped, temp)
        del stripped
        temp.replace(path)

        after = path.stat().st_size
        freed += before - after
        print(f"  {before / 1e9:7.2f} GB -> {after / 1e6:7.1f} MB   {path}")

    if not args.dry_run:
        print(f"\nfreed {freed / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
