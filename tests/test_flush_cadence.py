"""runtime.flush_every: the run's files are written every N rounds, and nothing else changes.

Every round wrote its CSV rows, run.json and latest.pt, each fsynced: about
7 ms an fsync on /proj's NFS against 0.7 on /tmp, five a round
(measured on 2026-09-26). flush_every N writes them on rounds N, 2N, ... and the
final one instead; between flushes the rows wait in memory and the numbered
and best checkpoints wait staged. What is pinned here:

- the cadence changes no number: every CSV cell but the timings, run.json's
  final metrics and every checkpoint's model are the same at N = 1, 3 and 4;
- after every flush, and on every round between, the CSVs are at or past every
  visible checkpoint, and the flushes land on rounds N, 2N, ... and the last;
- a SIGKILL at a random point of a random round -- inside any write, or inside
  a client's fit -- followed by --resume-latest ends with the CSVs and the
  model of the uninterrupted run, bit for bit;
- a round a divergence detector stops the run on is flushed, whatever N is;
- a round aggregation refuses between flushes still leaves the rounds before
  it on disk, with the checkpoints they staged;
- flush_every is a positive integer, and a resume may change it.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import torch
import yaml

from fedbrew.core import loop
from fedbrew.core.checkpointing import get_checkpoint_round_id
from fedbrew.core.config import ReportingConfig
from fedbrew.core.refusal import RunRefused
from fedbrew.core.runner import run
from tests.test_non_finite_aggregation import _EVALUATION, _BlowsUpOnRound, _Client, _Dataset
from tests.test_reproducibility import TIMING, _config

ROUNDS = 10
_REPO = Path(__file__).resolve().parents[1]


def _write(root: Path, name: str, flush_every: int, rounds: int = ROUNDS, **extra: Any) -> Path:
    config = _config(root / name, rounds=rounds, checkpoint=True)
    config["runtime"]["flush_every"] = flush_every
    config["runtime"]["checkpointing"].update({"save_best": True, "keep_last": None})
    config.setdefault("reporting", {})["per_client_csv"] = True
    config.update(extra)
    path = root / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _run(path: Path, *, resume_latest: bool = False) -> Path:
    args = argparse.Namespace(resume_latest=resume_latest) if resume_latest else None
    run(path, args=args)
    return Path(yaml.safe_load(path.read_text())["experiment"]["output_dir"])


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {key: value for key, value in row.items() if key not in TIMING}
            for row in csv.DictReader(handle)
        ]


def _model_bits(path: Path) -> dict[str, bytes]:
    state = torch.load(path, weights_only=False)["model_state"]
    return {
        key: bytes(value.detach().reshape(-1).view(torch.uint8).tolist())
        for key, value in state.items()
    }


def _outcome(output_dir: Path) -> dict[str, Any]:
    """Everything a run computes, without its timings."""

    checkpoints = output_dir / "checkpoints"
    return {
        "csvs": {
            name: _csv_rows(output_dir / name)
            for name in ("round_metrics.csv", "client_metrics.csv", "client_update_metrics.csv")
        },
        "models": {path.name: _model_bits(path) for path in sorted(checkpoints.glob("*.pt"))},
        "final_metrics": yaml.safe_load((output_dir / "run.json").read_text())["results"][
            "final_metrics"
        ],
    }


def _csv_round_ids(output_dir: Path) -> list[int]:
    path = output_dir / "round_metrics.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [int(row["round_id"]) for row in csv.DictReader(handle)]


def _visible_checkpoint_rounds(output_dir: Path) -> dict[str, int]:
    checkpoints = output_dir / "checkpoints"
    if not checkpoints.is_dir():
        return {}
    return {
        path.name: get_checkpoint_round_id(torch.load(path, weights_only=False))
        for path in checkpoints.glob("*.pt")
    }


class _TempRoot(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)


class TheCadenceChangesNoNumberTest(_TempRoot):
    def test_every_output_is_the_same_at_one_three_and_four(self) -> None:
        outcomes = {n: _outcome(_run(_write(self.root, f"n{n}", n))) for n in (1, 3, 4)}
        self.assertEqual(len(outcomes[1]["csvs"]["round_metrics.csv"]), ROUNDS)
        self.assertIn("best.pt", outcomes[1]["models"])
        self.assertIn(f"round_{ROUNDS:03d}.pt", outcomes[1]["models"])
        for n in (3, 4):
            with self.subTest(flush_every=n):
                self.assertEqual(outcomes[n], outcomes[1])


class NoCheckpointIsAheadOfTheCsvTest(_TempRoot):
    def test_after_every_flush_and_on_every_round(self) -> None:
        output_dir = self.root / "n3"
        commits: list[int] = []
        real_commit = loop._commit_checkpoints
        real_update = loop._update_checkpoints

        def check(when: str) -> None:
            written = _csv_round_ids(output_dir)
            self.assertEqual(written, list(range(1, len(written) + 1)), when)
            for name, round_id in _visible_checkpoint_rounds(output_dir).items():
                self.assertLessEqual(round_id, len(written), f"{name} {when}")

        def commit(*args: Any, **kwargs: Any) -> None:
            real_commit(*args, **kwargs)
            commits.append(len(_csv_round_ids(output_dir)))
            check("after a flush")

        def update(*args: Any, **kwargs: Any) -> Any:
            check("between flushes")
            return real_update(*args, **kwargs)

        with (
            mock.patch.object(loop, "_commit_checkpoints", commit),
            mock.patch.object(loop, "_update_checkpoints", update),
        ):
            _run(_write(self.root, "n3", 3))

        self.assertEqual(commits, [3, 6, 9, 10])
        self.assertEqual(list((output_dir / "checkpoints").glob("*.tmp")), [])


#: Run in a child, which SIGKILLs itself at the K-th event: an fsync -- inside
#: any of the writes -- or a client fit.
_KILLED_CHILD = textwrap.dedent(
    """
    import os, signal, sys
    from pathlib import Path

    from fedbrew.core import loop
    from fedbrew.core.runner import run

    target, kind = int(sys.argv[2]), sys.argv[3]
    seen = 0

    def tick():
        global seen
        seen += 1
        if seen == target:
            os.kill(os.getpid(), signal.SIGKILL)

    if kind == "fsync":
        real_fsync = os.fsync

        def fsync(fd):
            real_fsync(fd)
            tick()

        os.fsync = fsync
    else:
        real_fit = loop._fit_client

        def fit(client, request):
            tick()
            return real_fit(client, request)

        loop._fit_client = fit
    run(Path(sys.argv[1]))
    """
)


class ASigkillAnywhereResumesToTheSameRunTest(_TempRoot):
    def _events(self, flush_every: int) -> dict[str, int]:
        """How many fsyncs and fits an uninterrupted run makes."""

        counts = {"fsync": 0, "fit": 0}
        real_fsync, real_fit = os.fsync, loop._fit_client

        def fsync(fd: int) -> None:
            counts["fsync"] += 1
            real_fsync(fd)

        def fit(client: Any, request: Any) -> Any:
            counts["fit"] += 1
            return real_fit(client, request)

        with (
            mock.patch("os.fsync", fsync),
            mock.patch.object(loop, "_fit_client", fit),
        ):
            _run(_write(self.root, f"count{flush_every}", flush_every))
        return counts

    def test_random_kills_under_each_cadence(self) -> None:
        reference = _outcome(_run(_write(self.root, "reference", 1)))
        rng = random.Random(20260926)
        environment = {**os.environ, "PYTHONPATH": str(_REPO)}
        resumed = 0
        for flush_every in (1, 3):
            events = self._events(flush_every)
            for trial in range(3):
                kind = rng.choice(["fsync", "fsync", "fit"])
                target = rng.randint(1, events[kind] - 1)
                name = f"killed{flush_every}-{trial}"
                path = _write(self.root, name, flush_every)
                with self.subTest(flush_every=flush_every, kind=kind, target=target):
                    child = subprocess.run(
                        [sys.executable, "-c", _KILLED_CHILD, str(path), str(target), kind],
                        cwd=_REPO,
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    self.assertEqual(child.returncode, -9, child.stderr[-2000:])
                    # Killed before any checkpoint was visible, --resume-latest
                    # has nothing to take and refuses; the run is started again.
                    resumable = bool(_visible_checkpoint_rounds(path.parent / name))
                    resumed += resumable
                    output_dir = _run(path, resume_latest=resumable)
                    self.assertEqual(_outcome(output_dir), reference)
        self.assertGreaterEqual(resumed, 4, "too few kills left a checkpoint to resume")


class AStoppingRoundIsFlushedTest(_TempRoot):
    def test_a_divergence_stop_between_flushes(self) -> None:
        path = _write(
            self.root,
            "stopped",
            4,
            divergence={"metric": "fit_loss", "non_finite": True, "blowup_absolute": 1e-12},
        )
        output_dir = _run(path)
        self.assertEqual(_csv_round_ids(output_dir), [1])
        self.assertEqual(_visible_checkpoint_rounds(output_dir)["latest.pt"], 1)
        run_json = yaml.safe_load((output_dir / "run.json").read_text())
        self.assertEqual(run_json["status"], "diverged")


class ARefusedRoundBetweenFlushesTest(_TempRoot):
    def test_the_rounds_before_it_are_written(self) -> None:
        state = loop.run_fl_loop(
            server=_BlowsUpOnRound(3),
            client={"client_0": _Client()},
            dataset=_Dataset(),
            global_rounds=5,
            output_dir=self.root,
            checkpointing={"enabled": True, "save_every_round": True, "keep_last": None},
            evaluation=_EVALUATION,
            reporting=ReportingConfig(per_client_csv=False),
            flush_every=4,
        )
        self.assertEqual(state.status, "diverged")
        self.assertEqual(_csv_round_ids(self.root), [1, 2])
        self.assertEqual(
            sorted(_visible_checkpoint_rounds(self.root).items()),
            [("round_001.pt", 1), ("round_002.pt", 2)],
        )
        self.assertEqual(list((self.root / "checkpoints").glob("*.tmp")), [])


class TheSettingTest(_TempRoot):
    def test_only_a_positive_integer(self) -> None:
        for bad in (0, -1, 1.5, True, "2"):
            with self.subTest(value=bad):
                path = _write(self.root, "bad", 1)
                raw = yaml.safe_load(path.read_text())
                raw["runtime"]["flush_every"] = bad
                path.write_text(yaml.safe_dump(raw), encoding="utf-8")
                with self.assertRaises(RunRefused) as refused:
                    run(path)
                self.assertIn("runtime.flush_every", str(refused.exception))

    def test_a_resume_may_change_it(self) -> None:
        reference = _outcome(_run(_write(self.root, "whole", 1)))
        _run(_write(self.root, "part", 3, rounds=4))
        resumed = _run(_write(self.root, "part", 4), resume_latest=True)
        self.assertEqual(_outcome(resumed), reference)


if __name__ == "__main__":
    unittest.main()
