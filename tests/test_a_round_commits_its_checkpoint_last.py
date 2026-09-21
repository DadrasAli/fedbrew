"""A round's checkpoint becomes visible only after its CSV row and run.json. POST-F24.

A resume replays round_metrics.csv up to the checkpoint's round, so a
checkpoint must never be on disk for a round the CSV does not hold. The loop
wrote each round's checkpoints first, then that round's CSV rows, then
run.json; a kill between them left the checkpoint one round ahead of the
history, and no resume could continue it. Late in a long run the full CSV
rewrite is most of each round, so a time limit lands there almost every time:
on 2026-09-20 every one of ten stopped SCAFFOLD points had `latest.pt` at round
N beside a `round_metrics.csv` ending at N - 1, and a cancelled job on
2026-09-21 left its two points the same way.

The loop now stages the round's checkpoints where it always wrote them -- so
`checkpoint_sec` still times the write -- and commits them after run.json. What
is pinned here:

- while the CSV rows and run.json are written, the round's checkpoints exist
  only as staged ".tmp" files;
- a kill inside either write leaves the previous round's checkpoint beside a
  history that reaches it, and resuming from it completes the run with every
  round recorded once;
- `keep_last` still counts the round's own checkpoint.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from fedbrew.core import loop
from fedbrew.core.artifacts import round_metrics_gap
from fedbrew.core.checkpointing import (
    find_latest_checkpoint,
    get_checkpoint_round_id,
    load_checkpoint,
)
from tests.test_resume_metrics_continuity import _round_ids, _run

#: Every checkpoint the loop writes each round.
_EVERY_ROUND = {"enabled": True, "save_every_round": True, "save_last": True, "keep_last": None}


class _Killed(Exception):
    """Stands in for the signal: raised where the process would have died."""


def _visible(output_dir: Path) -> list[str]:
    return sorted(p.name for p in (output_dir / "checkpoints").iterdir())


def _latest_round(output_dir: Path) -> int:
    return get_checkpoint_round_id(load_checkpoint(find_latest_checkpoint(output_dir)))


class WhileTheRoundIsRecordedItsCheckpointIsStagedTest(unittest.TestCase):
    """Observed from inside the two writes, round by round."""

    def test_neither_write_can_see_this_rounds_checkpoint(self) -> None:
        seen: dict[str, list[tuple[int, list[str]]]] = {"csv": [], "run.json": []}
        real_flush = loop.flush_round_artifacts

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)

            def flush(history: Any, *args: Any, **kwargs: Any) -> None:
                seen["csv"].append((history[-1].round_id, _visible(output_dir)))
                real_flush(history, *args, **kwargs)

            def write_run_json(state: Any) -> None:
                seen["run.json"].append((state.metrics_history[-1].round_id, _visible(output_dir)))

            with mock.patch.object(loop, "flush_round_artifacts", flush):
                _run(output_dir, 3, on_round_flush=write_run_json, checkpointing=_EVERY_ROUND)

            final = _visible(output_dir)

        for write, observations in seen.items():
            for round_id, names in observations:
                with self.subTest(write=write, round=round_id):
                    self.assertNotIn(f"round_{round_id:03d}.pt", names)
                    self.assertIn(f"round_{round_id:03d}.pt.tmp", names)
                    self.assertIn("latest.pt.tmp", names)
        self.assertEqual(final, ["latest.pt", "round_001.pt", "round_002.pt", "round_003.pt"])


class AKillInsideEitherWriteIsResumableTest(unittest.TestCase):
    def _kill_then_resume(self, where: str) -> tuple[int, list[int], list[int]]:
        """Kill round 3 inside `where`; return the checkpoint round, the CSV, and after."""

        real_flush = loop.flush_round_artifacts

        def flush(history: Any, *args: Any, **kwargs: Any) -> None:
            if where == "csv" and history[-1].round_id == 3:
                raise _Killed
            real_flush(history, *args, **kwargs)

        def write_run_json(state: Any) -> None:
            if where == "run.json" and state.metrics_history[-1].round_id == 3:
                raise _Killed

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with (
                mock.patch.object(loop, "flush_round_artifacts", flush),
                self.assertRaises(_Killed),
            ):
                _run(output_dir, 5, on_round_flush=write_run_json, checkpointing=_EVERY_ROUND)

            checkpoint_round = _latest_round(output_dir)
            before = _round_ids(output_dir)
            self.assertIsNone(
                round_metrics_gap(output_dir, checkpoint_round),
                "the checkpoint is ahead of the history it would be resumed onto",
            )
            _run(
                output_dir,
                5,
                resume_from=find_latest_checkpoint(output_dir),
                checkpointing=_EVERY_ROUND,
            )
            return checkpoint_round, before, _round_ids(output_dir)

    def test_a_kill_inside_the_csv_write(self) -> None:
        checkpoint_round, before, after = self._kill_then_resume("csv")
        self.assertEqual((checkpoint_round, before), (2, [1, 2]))
        self.assertEqual(after, [1, 2, 3, 4, 5])

    def test_a_kill_inside_the_run_json_write(self) -> None:
        """The CSV holds round 3 and the checkpoint does not; the row is recomputed."""

        checkpoint_round, before, after = self._kill_then_resume("run.json")
        self.assertEqual((checkpoint_round, before), (2, [1, 2, 3]))
        self.assertEqual(after, [1, 2, 3, 4, 5])


class KeepLastCountsTheRoundsOwnCheckpointTest(unittest.TestCase):
    def test_the_newest_are_kept(self) -> None:
        policy = {**_EVERY_ROUND, "keep_last": 2}
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            _run(output_dir, 4, checkpointing=policy)
            self.assertEqual(_visible(output_dir), ["latest.pt", "round_003.pt", "round_004.pt"])


if __name__ == "__main__":
    unittest.main()
