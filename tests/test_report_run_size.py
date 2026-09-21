"""fedbrew report's Run Size lines must read keys that run.json actually carries.

Two of them did not. make_report.py asked scale for "total_client_updates" and
"total_client_evaluation_records"; artifacts.py writes "total_client_fits" and
"total_client_evaluations". _first(None, "") renders an empty string, so the
report printed those two labels with nothing after them -- for every run, with
no error. Neither name appears in any run.json a run writes.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pytest

from fedbrew.cli.make_report import SCALE_LINES, _scale_lines, make_run_report
from fedbrew.core.artifacts import save_run_json
from fedbrew.core.config import load_config
from fedbrew.core.state import (
    ClientEvaluationRecord,
    ClientMetricRecord,
    MetricRecord,
)

pytestmark = pytest.mark.fast


def _history(rounds: int) -> list[MetricRecord]:
    return [
        MetricRecord(round_id=r, metrics={"fit_loss": 1.0}, num_clients=2, num_examples=8)
        for r in range(1, rounds + 1)
    ]


def _client_history(rounds: int) -> list[ClientEvaluationRecord]:
    """Post-aggregation evaluation records: what total_client_evaluations counts."""

    return [
        ClientEvaluationRecord(
            round_id=r,
            client_id=f"client_{index}",
            participated=True,
            train_num_examples=4,
            test_num_examples=2,
            global_model_train_loss=1.0,
            global_model_train_accuracy=0.5,
            global_model_test_loss=1.0,
            global_model_test_accuracy=0.5,
        )
        for r in range(1, rounds + 1)
        for index in range(2)
    ]


def _client_updates(rounds: int) -> list[ClientMetricRecord]:
    """Per-client fit diagnostics: what total_client_update_metric_records counts."""

    return [
        ClientMetricRecord(
            round_id=r,
            client_id=f"client_{index}",
            phase="fit",
            num_examples=4,
            metrics={"fit_loss": 1.0},
        )
        for r in range(1, rounds + 1)
        for index in range(2)
    ]


class ScaleKeysExistTests(unittest.TestCase):
    """The table and the writer have to name the same keys."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = Path(self._tmp.name)
        save_run_json(
            _history(3),
            self.tmpdir,
            load_config("configs/dev/smoke.yaml"),
            _client_history(3),
            run_metadata={"run_id": "r", "status": "completed"},
            client_update_history=_client_updates(3),
        )
        self.run = json.loads((self.tmpdir / "run.json").read_text(encoding="utf-8"))

    def test_every_key_the_report_prints_is_written(self) -> None:
        written = set(self.run["scale"])
        for key, label in SCALE_LINES:
            with self.subTest(key=key, label=label):
                self.assertIn(key, written)

    def test_the_report_shows_the_values_not_blanks(self) -> None:
        make_run_report(self.tmpdir)
        report = (self.tmpdir / "report.md").read_text(encoding="utf-8")
        for key, label in SCALE_LINES:
            with self.subTest(key=key):
                self.assertIn(f"- {label}: {self.run['scale'][key]}", report)
                # A label followed by nothing is the defect being fixed.
                self.assertNotIn(f"- {label}: \n", report)

    def test_the_counts_are_the_ones_the_run_actually_had(self) -> None:
        # Not just non-blank: right. Six eval records over three rounds.
        self.assertEqual(self.run["scale"]["total_client_evaluations"], 6)
        make_run_report(self.tmpdir)
        report = (self.tmpdir / "report.md").read_text(encoding="utf-8")
        self.assertIn("- Client evaluation records: 6", report)


class MissingKeyTests(unittest.TestCase):
    def test_an_absent_key_says_so_rather_than_rendering_empty(self) -> None:
        lines = _scale_lines({})
        self.assertEqual(len(lines), len(SCALE_LINES))
        for line in lines:
            with self.subTest(line=line):
                self.assertTrue(line.endswith(": not recorded"))

    def test_a_zero_is_printed_as_zero(self) -> None:
        # 0 is a measurement; "not recorded" is not. _first() treated them alike.
        key, label = SCALE_LINES[0]
        self.assertIn(f"- {label}: 0", _scale_lines({key: 0}))


if __name__ == "__main__":
    unittest.main()
