"""A resume that cannot be taken is refused, and the directory is left as it was. POST-F25.

A resume replays round_metrics.csv up to the checkpoint's round. When the CSV
does not reach it, the loop used to print one line, delete the attempt --
checkpoints, the CSVs, run.json -- and start again from round 1. In a batch job
nobody reads that line: a 10,000-round run stopped at round 9690 by a time
limit came back as a fresh run, the rounds it had already done were gone, and
so was the evidence of why. On 2026-09-21 that is what `--resume-latest` did to
a copy of a stopped SCAFFOLD point, and what it would have done to all ten of
them (POST-F24 is why they were in that state).

Now the resume is refused before anything is written. What is pinned here:

- every file in the directory is byte-for-byte what it was, and no file is
  added -- not a run.json, not a temp file, not a checkpoint;
- the refusal names the checkpoint, its round and what the CSV is missing, and
  the two ways forward;
- a resume the history does back still runs, so the refusal is not a blanket
  one.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from fedbrew.core.refusal import RunRefused
from tests.test_resume_metrics_continuity import _drop_rounds, _round_ids, _run

_EVERY_ROUND = {"enabled": True, "save_every_round": True, "save_last": True, "keep_last": None}


def _snapshot(directory: Path) -> dict[str, str]:
    """Every file under `directory`, by relative path, with its SHA-256."""

    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


class ACheckpointAheadOfItsHistoryIsRefusedTest(unittest.TestCase):
    """The state POST-F24 left ten SCAFFOLD points in: latest.pt at N, the CSV at N - 1."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)
        _run(self.out, 4, checkpointing=_EVERY_ROUND)
        _drop_rounds(self.out, {4})
        # What a kill inside a staged write leaves. A refused resume must not
        # sweep it either: the sweep belongs to a run that is going ahead.
        (self.out / "checkpoints" / "round_005.pt.tmp").write_bytes(b"killed")
        self.before = _snapshot(self.out)

    def _resume(self) -> str:
        with self.assertRaises(RunRefused) as caught:
            _run(
                self.out,
                6,
                resume_from=self.out / "checkpoints" / "latest.pt",
                checkpointing=_EVERY_ROUND,
            )
        return str(caught.exception)

    def test_nothing_on_disk_changes(self) -> None:
        self._resume()
        self.assertEqual(_snapshot(self.out), self.before)

    def test_the_attempt_it_refused_is_still_there_to_decide_from(self) -> None:
        self._resume()
        self.assertEqual(_round_ids(self.out), [1, 2, 3])
        self.assertTrue((self.out / "checkpoints" / "latest.pt").is_file())
        self.assertTrue((self.out / "checkpoints" / "round_004.pt").is_file())

    def test_the_refusal_says_what_and_where(self) -> None:
        message = self._resume()
        self.assertIn("latest.pt", message)
        self.assertIn("round 4", message)
        self.assertIn("round_metrics.csv is missing 1 of rounds 1-4", message)
        self.assertIn("nothing in", message)

    def test_it_names_both_ways_forward(self) -> None:
        message = self._resume()
        self.assertIn("--resume-from", message)
        self.assertIn("move the directory aside", message)

    def test_an_earlier_checkpoint_the_history_reaches_still_resumes(self) -> None:
        """The first way forward, taken: round_003.pt is backed by rows 1-3."""

        _run(
            self.out,
            6,
            resume_from=self.out / "checkpoints" / "round_003.pt",
            checkpointing=_EVERY_ROUND,
        )
        self.assertEqual(_round_ids(self.out), [1, 2, 3, 4, 5, 6])


class ACheckpointWithNoHistoryIsRefusedTest(unittest.TestCase):
    """`--resume-from` pointed beside an empty directory: nothing to replay."""

    def test_the_directory_stays_empty(self) -> None:
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as target:
            _run(Path(source), 2, checkpointing=_EVERY_ROUND)
            empty = Path(target)
            with self.assertRaises(RunRefused) as caught:
                _run(
                    empty,
                    4,
                    resume_from=Path(source) / "checkpoints" / "latest.pt",
                    checkpointing=_EVERY_ROUND,
                )
            self.assertIn("round_metrics.csv is missing", str(caught.exception))
            self.assertEqual(list(empty.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
