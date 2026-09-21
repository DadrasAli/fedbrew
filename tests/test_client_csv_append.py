"""The per-client CSVs must be current every round without rewriting the run.

flush_round_artifacts rewrote both per-client CSVs in full on every round that
"wrote a checkpoint". The gate was there to bound the cost -- 1.8M rows over a
500-round FEMNIST run at clients "all", every earlier round's rows rewritten
each round -- on the premise that writing a checkpoint is a sparse,
interval-controlled event. It is not: save_last writes latest.pt every round,
independently of interval, and every shipped training config sets it. So the
gate was true every round and per_client_csv: true bought exactly the write
volume the gate existed to prevent.

Throttling it instead would lose data. A resume rewinds to latest.pt, which
save_last writes every round, and the resumed run rebuilds its per-client
history out of these files, keeping rows before the checkpoint's round. Any
round the CSV lagged behind would lose its rows permanently. So the files have
to be current every round, and each round's rows are appended rather than the
whole run rewritten.

Rewriting in full makes the bytes written grow with the square of the round
count, because round r rewrites the rows of every round before it. Appending
writes each row once, so the bytes written are the size of the data.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import pytest

from fedbrew.core.artifacts import (
    flush_client_csvs,
    load_client_metrics_csv,
    load_client_update_metrics_csv,
    save_client_metrics_csv,
    save_client_update_metrics_csv,
)
from fedbrew.core.state import (
    ClientEvaluationHistory,
    ClientEvaluationRecord,
    ClientMetricRecord,
    ClientUpdateHistory,
)

pytestmark = pytest.mark.fast


def _evaluation(round_id: int, client_id: str) -> ClientEvaluationRecord:
    return ClientEvaluationRecord(
        round_id=round_id,
        client_id=client_id,
        participated=True,
        train_num_examples=10,
        test_num_examples=3,
        global_model_train_loss=1.0,
        global_model_train_accuracy=0.5,
        global_model_test_loss=1.25,
        global_model_test_accuracy=0.75,
    )


def _update(
    round_id: int,
    client_id: str,
    metrics: dict[str, float] | None = None,
) -> ClientMetricRecord:
    return ClientMetricRecord(
        round_id=round_id,
        client_id=client_id,
        phase="fit",
        num_examples=7,
        metrics={"fit_loss": 1.0} if metrics is None else metrics,
    )


class AppendedOutputTest(unittest.TestCase):
    """Appending has to produce exactly what the full rewrite produced."""

    def _round(
        self,
        evaluations: ClientEvaluationHistory,
        updates: ClientUpdateHistory,
        round_id: int,
    ) -> None:
        for index in range(3):
            evaluations.append(_evaluation(round_id, f"c{index}"))
            updates.append(_update(round_id, f"c{index}"))

    def test_incremental_and_full_writes_agree_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            incremental, whole = Path(first), Path(second)
            evaluations = ClientEvaluationHistory()
            updates = ClientUpdateHistory()
            cursor: dict[str, object] = {}
            for round_id in (1, 2, 3, 4):
                self._round(evaluations, updates, round_id)
                flush_client_csvs(evaluations, updates, incremental, cursor)

            save_client_metrics_csv(evaluations, whole)
            save_client_update_metrics_csv(updates, whole)

            for name in ("client_metrics.csv", "client_update_metrics.csv"):
                with self.subTest(name=name):
                    self.assertEqual(
                        (incremental / name).read_text(encoding="utf-8"),
                        (whole / name).read_text(encoding="utf-8"),
                    )

    def test_later_rounds_are_appended_not_rewritten(self) -> None:
        """A rewrite would repair an edit made to an earlier row; an append
        leaves it, which is what proves the run is not being rewritten."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluations = ClientEvaluationHistory()
            updates = ClientUpdateHistory()
            cursor: dict[str, object] = {}
            self._round(evaluations, updates, 1)
            flush_client_csvs(evaluations, updates, root, cursor)

            path = root / "client_metrics.csv"
            path.write_text(
                path.read_text(encoding="utf-8").replace("c0", "SENTINEL", 1),
                encoding="utf-8",
            )

            self._round(evaluations, updates, 2)
            flush_client_csvs(evaluations, updates, root, cursor)

            text = path.read_text(encoding="utf-8")
        self.assertIn("SENTINEL", text)
        self.assertEqual(text.count("\n"), 1 + 6)  # header + two rounds


class FullRewriteFallbackTest(unittest.TestCase):
    """Appending is only safe while the file it appends to still matches."""

    def test_a_new_metric_name_rewrites_the_file(self) -> None:
        """The update CSV's columns come from the history, so a metric that
        first appears at round 3 widens the schema; appending under the old
        header would put its values in the wrong columns."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            updates = ClientUpdateHistory()
            evaluations = ClientEvaluationHistory()
            cursor: dict[str, object] = {}
            updates.append(_update(1, "c0"))
            flush_client_csvs(evaluations, updates, root, cursor)
            updates.append(_update(2, "c0", {"fit_loss": 1.0, "grad_norm": 2.0}))
            flush_client_csvs(evaluations, updates, root, cursor)

            with (root / "client_update_metrics.csv").open(
                "r", encoding="utf-8", newline=""
            ) as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 2)
        self.assertIn("grad_norm", rows[0])
        self.assertEqual(rows[0]["grad_norm"], "")
        self.assertEqual(rows[1]["grad_norm"], "2.0")

    def test_a_shorter_history_rewrites_the_file(self) -> None:
        """A resume replaces the history object; the cursor must not survive
        it and append onto rows that are already there."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluations = ClientEvaluationHistory()
            updates = ClientUpdateHistory()
            cursor: dict[str, object] = {}
            for round_id in (1, 2, 3):
                evaluations.append(_evaluation(round_id, "c0"))
                updates.append(_update(round_id, "c0"))
            flush_client_csvs(evaluations, updates, root, cursor)

            replaced_evaluations = ClientEvaluationHistory()
            replaced_updates = ClientUpdateHistory()
            replaced_evaluations.append(_evaluation(1, "c0"))
            replaced_updates.append(_update(1, "c0"))
            flush_client_csvs(replaced_evaluations, replaced_updates, root, cursor)

            self.assertEqual(len(load_client_metrics_csv(root)), 1)
            self.assertEqual(len(load_client_update_metrics_csv(root)), 1)

    def test_a_missing_file_rewrites_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluations = ClientEvaluationHistory()
            updates = ClientUpdateHistory()
            cursor: dict[str, object] = {}
            for round_id in (1, 2):
                evaluations.append(_evaluation(round_id, "c0"))
                updates.append(_update(round_id, "c0"))
            flush_client_csvs(evaluations, updates, root, cursor)

            (root / "client_metrics.csv").unlink()
            flush_client_csvs(evaluations, updates, root, cursor)

            self.assertEqual(len(load_client_metrics_csv(root)), 2)


class TruncatedTailTest(unittest.TestCase):
    """An interrupted append can only leave a short last line."""

    def _write(self, root: Path) -> tuple[ClientEvaluationHistory, ClientUpdateHistory]:
        evaluations = ClientEvaluationHistory()
        updates = ClientUpdateHistory()
        cursor: dict[str, object] = {}
        for round_id in (1, 2, 3):
            evaluations.append(_evaluation(round_id, "c0"))
            updates.append(_update(round_id, "c0"))
        flush_client_csvs(evaluations, updates, root, cursor)
        return evaluations, updates

    def _tear_last_row(self, path: Path) -> None:
        """Cut the final row a few characters in, as a killed write would."""

        lines = path.read_text(encoding="utf-8").splitlines()
        torn = lines[-1].split(",")[0]
        path.write_text("\n".join(lines[:-1] + [torn]), encoding="utf-8")

    def test_a_half_written_last_row_costs_only_that_row(self) -> None:
        """Raising would refuse the whole resume, for a row the resumed run redoes."""

        for name, load in (
            ("client_metrics.csv", load_client_metrics_csv),
            ("client_update_metrics.csv", load_client_update_metrics_csv),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self._write(root)
                self._tear_last_row(root / name)
                records = load(root)
                self.assertEqual([record.round_id for record in records], [1, 2])

    def test_a_torn_row_is_not_read_as_a_record_of_zeros(self) -> None:
        """csv.DictReader fills a short row's columns with None, and every
        coercion here is `or 0` or str(), so it used to parse cleanly into a
        row of zeros for client "None" -- silently wrong rather than loud."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root)
            self._tear_last_row(root / "client_metrics.csv")
            records = load_client_metrics_csv(root)
        self.assertEqual([record.client_id for record in records], ["c0", "c0"])
        self.assertTrue(all(record.train_num_examples == 10 for record in records))

    def test_a_short_row_before_the_last_raises(self) -> None:
        """Only the final row can be a torn write."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root)
            path = root / "client_metrics.csv"
            lines = path.read_text(encoding="utf-8").splitlines()
            lines[1] = lines[1].split(",")[0]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            with self.assertRaises(ValueError) as caught:
                load_client_metrics_csv(root)
        message = str(caught.exception)
        self.assertIn("client_metrics.csv:2", message)
        self.assertIn("fewer fields", message)

    def test_an_earlier_bad_row_still_raises(self) -> None:
        """Only the last line can be a torn write; anything else is real
        corruption and must not be swallowed."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root)
            path = root / "client_metrics.csv"
            lines = path.read_text(encoding="utf-8").splitlines()
            lines[1] = lines[1].replace("1,c0", "not-a-round,c0", 1)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            with self.assertRaises(ValueError) as caught:
                load_client_metrics_csv(root)
        self.assertIn("client_metrics.csv:2", str(caught.exception))

    def test_an_intact_file_loads_every_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root)
            self.assertEqual(len(load_client_metrics_csv(root)), 3)
            self.assertEqual(len(load_client_update_metrics_csv(root)), 3)


class WriteVolumeTest(unittest.TestCase):
    """The property the whole change is for."""

    def test_write_volume_tracks_the_data_not_the_run_length(self) -> None:
        rounds = 12
        clients = 20
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluations = ClientEvaluationHistory()
            updates = ClientUpdateHistory()
            cursor: dict[str, object] = {}
            written = 0
            real_open = Path.open

            def counting_open(self, mode="r", *args, **kwargs):  # type: ignore[no-untyped-def]
                handle = real_open(self, mode, *args, **kwargs)
                if "w" in mode or "a" in mode:
                    real_write = handle.write

                    def write(payload):  # type: ignore[no-untyped-def]
                        nonlocal written
                        written += len(payload)
                        return real_write(payload)

                    handle.write = write
                return handle

            Path.open = counting_open  # type: ignore[method-assign]
            try:
                for round_id in range(1, rounds + 1):
                    for index in range(clients):
                        evaluations.append(_evaluation(round_id, f"c{index}"))
                        updates.append(_update(round_id, f"c{index}"))
                    flush_client_csvs(evaluations, updates, root, cursor)
            finally:
                Path.open = real_open  # type: ignore[method-assign]

            on_disk = sum(path.stat().st_size for path in root.glob("*.csv"))

        # A full rewrite each round writes the history once per round, so
        # ~rounds/2 times the data. Appending writes it once.
        self.assertLess(written, on_disk * 1.1)


if __name__ == "__main__":
    unittest.main()
