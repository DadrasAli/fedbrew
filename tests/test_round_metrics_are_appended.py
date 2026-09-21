"""round_metrics.csv is appended to each round, not rewritten.

The loop flushed round_metrics.csv every round by rewriting the whole file, so
round r wrote the rows of every round before it and the bytes written grew
with the square of the round count. A 10,000-round fed-logistic-l1 point
took 2707 s, 2165 s of them outside its rounds; appended, 919 s. The per-client CSVs
had already stopped doing this (tests/test_client_csv_append.py); this file
now works the same way. What is pinned here:

- the appended file is byte-for-byte the file a full rewrite writes, including
  when a metric first appears mid-run and widens the header;
- a steady-state round appends its row, and the bytes written track the file,
  not the run length -- one rewrite per attempt, the first flush;
- a row an interrupted append cut short is dropped on read, including a cut
  inside the last field, which leaves every field present and the value
  shortened;
- a kill inside the append leaves a run a resume continues, with every round
  recorded once.
"""

from __future__ import annotations

import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from fedbrew.core import artifacts
from fedbrew.core.artifacts import (
    flush_client_csvs,
    flush_round_metrics_csv,
    load_client_update_metrics_csv,
    load_round_metrics_csv,
    round_metrics_gap,
    save_round_metrics_csv,
)
from fedbrew.core.checkpointing import (
    find_latest_checkpoint,
    get_checkpoint_round_id,
    load_checkpoint,
)
from fedbrew.core.state import (
    ClientEvaluationHistory,
    ClientMetricRecord,
    ClientUpdateHistory,
    MetricRecord,
    RoundTimings,
)
from tests.test_resume_metrics_continuity import _round_ids, _run

_EVERY_ROUND = {"enabled": True, "save_every_round": True, "save_last": True, "keep_last": None}
_DROPPED = "dropping the incomplete last row"


def _record(
    round_id: int, extra: dict[str, float] | None = None, checkpoint: float = 0.25
) -> MetricRecord:
    return MetricRecord(
        round_id=round_id,
        metrics={"loss": 1.0 / round_id, "accuracy": 0.5, **(extra or {})},
        num_clients=2,
        num_examples=4,
        timings=RoundTimings(
            total=0.5,
            fit=0.125,
            aggregate=0.0625,
            client_eval=0.1,
            global_eval=0.1,
            checkpoint=checkpoint,
        ),
    )


def _flush_rounds(
    root: Path, records: list[MetricRecord], cursor: dict[str, Any] | None = None
) -> None:
    """Flush after each record, as the loop does."""

    cursor = {} if cursor is None else cursor
    history: list[MetricRecord] = []
    for record in records:
        history.append(record)
        flush_round_metrics_csv(history, root, cursor)


def _rewritten(records: list[MetricRecord]) -> bytes:
    with tempfile.TemporaryDirectory() as directory:
        return save_round_metrics_csv(records, directory).read_bytes()


@pytest.mark.fast
class AppendedIsWhatARewriteWritesTest(unittest.TestCase):
    def test_byte_for_byte_with_scheduled_and_late_metrics(self) -> None:
        """test_loss on rounds 3 and 6 only, as an every-3 schedule leaves it;
        optimality_gap first at round 4, so the header widens mid-run."""

        records = [
            _record(1),
            _record(2),
            _record(3, {"test_loss": 0.75}),
            _record(4, {"optimality_gap": 0.5}),
            _record(5, {"optimality_gap": 0.25}),
            _record(6, {"optimality_gap": 0.125, "test_loss": 0.5}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _flush_rounds(root, records)
            self.assertEqual((root / "round_metrics.csv").read_bytes(), _rewritten(records))

    def test_a_metric_that_appears_late_gets_its_own_column(self) -> None:
        """Appended under the round-1 header, its value would land in a timing column."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _flush_rounds(root, [_record(1), _record(2), _record(3, {"central_gap": 0.5})])
            loaded = load_round_metrics_csv(root)
        self.assertEqual(
            [record.metrics.get("central_gap") for record in loaded], [None, None, 0.5]
        )
        self.assertEqual([record.timings.checkpoint for record in loaded], [0.25, 0.25, 0.25])


@pytest.mark.fast
class ASteadyRoundAppendsTest(unittest.TestCase):
    def test_an_earlier_row_is_not_rewritten(self) -> None:
        """A rewrite would restore an edit to round 1's row; an append leaves it."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cursor: dict[str, Any] = {}
            history = [_record(1)]
            flush_round_metrics_csv(history, root, cursor)
            path = root / "round_metrics.csv"
            path.write_text(
                path.read_text(encoding="utf-8").replace("0.5,", "SENTINEL,", 1), encoding="utf-8"
            )

            history.append(_record(2))
            flush_round_metrics_csv(history, root, cursor)
            text = path.read_text(encoding="utf-8")
        self.assertIn("SENTINEL", text)
        self.assertEqual(len(text.splitlines()), 1 + 2)

    def test_bytes_written_track_the_file_not_the_run_length(self) -> None:
        """A rewrite each round writes ~rounds/2 times the file; 200 rounds make it ~100x."""

        written = 0
        real_open = Path.open

        def counting_open(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            handle = real_open(self, mode, *args, **kwargs)
            if "w" in mode or "a" in mode:
                real_write = handle.write

                def write(payload: str) -> int:
                    nonlocal written
                    written += len(payload)
                    return real_write(payload)

                handle.write = write
            return handle

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(Path, "open", counting_open),
        ):
            root = Path(directory)
            _flush_rounds(root, [_record(round_id) for round_id in range(1, 201)])
            on_disk = (root / "round_metrics.csv").stat().st_size
        self.assertLess(written, on_disk * 1.1)

    def test_a_shorter_history_rewrites(self) -> None:
        """A replaced history must not be appended onto the rows already there."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cursor: dict[str, Any] = {}
            _flush_rounds(root, [_record(round_id) for round_id in range(1, 6)], cursor)
            flush_round_metrics_csv([_record(1), _record(2)], root, cursor)
            self.assertEqual(_round_ids(root), [1, 2])


def _cut(path: Path, characters: int) -> None:
    """Remove the last `characters` characters, as a kill inside the append would."""

    path.write_bytes(path.read_bytes()[:-characters])


@pytest.mark.fast
class ARowCutShortIsDroppedTest(unittest.TestCase):
    def _written(self, root: Path, last_checkpoint: float = 0.25) -> None:
        _flush_rounds(root, [_record(1), _record(2), _record(3, checkpoint=last_checkpoint)])

    def _load(self, root: Path) -> tuple[list[MetricRecord], str]:
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            records = load_round_metrics_csv(root)
        return records, printed.getvalue()

    def test_a_row_cut_mid_way(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._written(root)
            _cut(root / "round_metrics.csv", 20)
            records, printed = self._load(root)
        self.assertEqual([record.round_id for record in records], [1, 2])
        self.assertEqual(printed.count(_DROPPED), 1)

    def test_a_cut_just_after_the_last_comma(self) -> None:
        """Every field present, the last one empty: the old reader made it a 0 s checkpoint."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._written(root)
            _cut(root / "round_metrics.csv", len("0.25\r\n"))
            self.assertTrue((root / "round_metrics.csv").read_bytes().endswith(b","))
            records, _ = self._load(root)
        self.assertEqual([record.round_id for record in records], [1, 2])

    def test_a_cut_inside_the_last_field_that_still_parses(self) -> None:
        """0.0123 cut to 0.01: every field present, a wrong value read silently."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._written(root, last_checkpoint=0.0123)
            _cut(root / "round_metrics.csv", len("23\r\n"))
            records, _ = self._load(root)
        self.assertEqual([record.round_id for record in records], [1, 2])

    def test_the_gap_check_reads_it_the_same_way_and_says_nothing(self) -> None:
        """The resume's gap check reads the file before the replay does; one warning, not two."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._written(root)
            _cut(root / "round_metrics.csv", 20)
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                backed = round_metrics_gap(root, 2)
                torn = round_metrics_gap(root, 3)
        self.assertIsNone(backed)
        self.assertEqual(torn, "round_metrics.csv is missing 1 of rounds 1-3 (3)")
        self.assertEqual(printed.getvalue(), "")

    def test_a_short_row_before_the_last_is_refused(self) -> None:
        """The old reader parsed it as a round of zeros."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._written(root)
            path = root / "round_metrics.csv"
            lines = path.read_text(encoding="utf-8").splitlines()
            lines[2] = lines[2].split(",")[0]
            path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                load_round_metrics_csv(root)
        self.assertIn("round_metrics.csv:3 has fewer fields", str(caught.exception))

    def test_the_per_client_update_file_reads_a_cut_last_field_the_same_way(self) -> None:
        """Its last column is a metric, so the cut value was a wrong metric or a refused resume."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            updates = ClientUpdateHistory()
            for round_id in (1, 2, 3):
                updates.append(
                    ClientMetricRecord(
                        round_id=round_id,
                        client_id="c0",
                        phase="fit",
                        num_examples=7,
                        metrics={"fit_loss": 5e-05},
                    )
                )
            flush_client_csvs(ClientEvaluationHistory(), updates, root, {})
            _cut(root / "client_update_metrics.csv", len("05\r\n"))
            self.assertTrue((root / "client_update_metrics.csv").read_bytes().endswith(b",5e-"))
            with contextlib.redirect_stdout(io.StringIO()):
                records = load_client_update_metrics_csv(root)
        self.assertEqual([record.round_id for record in records], [1, 2])


class _Killed(Exception):
    """Stands in for the signal: raised where the process would have died."""


class AKillInsideTheAppendIsResumableTest(unittest.TestCase):
    def test_every_round_is_recorded_once(self) -> None:
        real_append = artifacts._append_csv_rows

        def torn_append(path: Path, fieldnames: Any, rows: Any) -> None:
            rows = list(rows)
            if path.name == "round_metrics.csv" and rows[-1]["round_id"] == 3:
                buffer = io.StringIO()
                csv.DictWriter(buffer, fieldnames=list(fieldnames)).writerows(rows)
                payload = buffer.getvalue()
                with path.open("a", encoding="utf-8", newline="") as file:
                    file.write(payload[: len(payload) // 2])
                raise _Killed
            real_append(path, fieldnames, rows)

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with (
                mock.patch.object(artifacts, "_append_csv_rows", torn_append),
                self.assertRaises(_Killed),
            ):
                _run(output_dir, 5, checkpointing=_EVERY_ROUND)
            path = output_dir / "round_metrics.csv"
            self.assertFalse(path.read_bytes().endswith(b"\n"), "the append was not cut")
            self.assertEqual(
                get_checkpoint_round_id(load_checkpoint(find_latest_checkpoint(output_dir))), 2
            )

            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                _run(
                    output_dir,
                    5,
                    resume_from=find_latest_checkpoint(output_dir),
                    checkpointing=_EVERY_ROUND,
                )

            self.assertEqual(_round_ids(output_dir), [1, 2, 3, 4, 5])
            self.assertEqual(path.read_bytes(), _rewritten(load_round_metrics_csv(output_dir)))
        self.assertEqual(printed.getvalue().count(_DROPPED), 1)


class OneRewriteAnAttemptTest(unittest.TestCase):
    """The first flush of each attempt rewrites; every later round appends."""

    def _rewrites(self, output_dir: Path, rounds: int, resume_from: Path | None = None) -> int:
        real_save = artifacts.save_round_metrics_csv
        calls = 0

        def counting_save(*args: Any, **kwargs: Any) -> Path:
            nonlocal calls
            calls += 1
            return real_save(*args, **kwargs)

        with mock.patch.object(artifacts, "save_round_metrics_csv", counting_save):
            _run(output_dir, rounds, resume_from=resume_from, checkpointing=_EVERY_ROUND)
        return calls

    def test_a_fresh_run_and_a_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            self.assertEqual(self._rewrites(output_dir, 6), 1)
            # The resume's rewrite is what drops rounds 4-6 before they are redone.
            resumed = self._rewrites(
                output_dir, 8, resume_from=output_dir / "checkpoints" / "round_003.pt"
            )
            self.assertEqual(resumed, 1)
            self.assertEqual(_round_ids(output_dir), list(range(1, 9)))


if __name__ == "__main__":
    unittest.main()
