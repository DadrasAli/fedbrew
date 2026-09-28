#!/usr/bin/env python
"""Certify convex problems on a corpus and write their optima into the table.

The optima table (``examples/fed-logistic-l1/optima.json``) holds, for each
convex problem posed on a corpus, the certified `x*` and `F*`, keyed by the
corpus's content digest, the loss, the penalty and `lam`. A run on a convex
problem looks its `F*` up there and is refused without one. This script is how
an entry is made: it rebuilds the corpus from its generator config -- the rows
``fedbrew generate`` writes, bit for bit, on the same machine -- solves each
problem asked for, and writes the entry.

Usage
-----
    python examples/fed-logistic-l1/certify.py \\
        --config data/configs/examples/fed-logistic-l1-a9a.yaml \\
        --problem logistic l1 0.03 --problem logistic l2sq 0.001
    python examples/fed-logistic-l1/certify.py --config ... --problem ... --table other.json

Run from the repository root, where a LIBSVM corpus's ``source.path`` points.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parent.parent


def main(argv: list[str] | None = None) -> None:
    """Certify each problem asked for and write its entry."""

    sys.path.insert(0, str(REPO_ROOT))
    from fedbrew.core import extensions

    problem = extensions._import_file(EXAMPLE_DIR / "problem.py")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="The corpus's generator config.")
    parser.add_argument(
        "--problem",
        nargs=3,
        action="append",
        required=True,
        metavar=("LOSS", "PENALTY", "LAM"),
        help="A convex problem to certify on the corpus; repeatable.",
    )
    parser.add_argument("--table", default=str(problem.OPTIMA_TABLE), help="The table to write.")
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    corpus = problem._spec_from_config(config)
    for loss, penalty, lam in args.problem:
        spec = dataclasses.replace(corpus, loss=loss, penalty=penalty, penalty_strength=float(lam))
        started = time.time()
        entry = problem.certify(spec)
        problem.write_optimum(Path(args.table), entry)
        print(
            f"{spec.corpus} {loss}+{penalty} lam={float(lam)!r}: F* = {entry['f_star']!r}, "
            f"KKT {entry['kkt_residual']:.2e}, digest {entry['digest'][:12]}, "
            f"{time.time() - started:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
