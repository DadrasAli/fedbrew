"""report.md must not call a killed run's last round its final result.

fedbrew report read results.final_metrics -- the last row of the history -- and
printed it under "## Final Metrics" with nothing beside it. For a diverged run
that row is the round the run blew up at, and a run still in flight carries a
populated final_metrics too. status, termination and final_round
were all two keys away in the same JSON object and none was printed.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pytest

from fedbrew.cli.make_report import make_run_report

pytestmark = pytest.mark.fast


def _run_json(**overrides: Any) -> dict[str, Any]:
    run = {
        "run_id": "r0",
        "status": "completed",
        "num_rounds": 500,
        "first_round": 1,
        "final_round": 500,
        "config": {
            "experiment": {"name": "fedavg_femnist", "seed": 42},
            "server": {"strategy": "fedavg", "global_rounds": 500},
            "client": {"update_rule": "fedavg"},
            "data": {"name": "manifest_dataset"},
            "model": {"name": "femnist_resnet18"},
        },
        "scale": {"total_client_updates": 18000},
        "results": {"final_metrics": {"fit_loss": 0.42, "val_accuracy_avg": 0.81}},
    }
    run.update(overrides)
    return run


class ReportStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.run_dir = Path(self._tmp.name)

    def _report(self, **overrides: Any) -> str:
        (self.run_dir / "run.json").write_text(json.dumps(_run_json(**overrides)), encoding="utf-8")
        return make_run_report(self.run_dir).read_text(encoding="utf-8")

    def test_a_completed_run_still_says_final_metrics(self) -> None:
        report = self._report()
        self.assertIn("## Final Metrics", report)
        self.assertIn("- Status: completed", report)
        self.assertNotIn("Last Evaluated", report)
        self.assertNotIn("- Stopped:", report)

    def test_a_diverged_run_does_not_call_them_final(self) -> None:
        report = self._report(
            status="diverged",
            num_rounds=1,
            final_round=1,
            termination={
                "detector": "blowup_absolute",
                "round_id": 1,
                "metric": "fit_loss",
                "value": 1525775323.67,
                "reason": "fit_loss reached 1.52578e+09 at round 1, above the "
                "absolute ceiling of 41.3",
            },
        )
        self.assertNotIn("## Final Metrics", report)
        self.assertIn("## Last Evaluated Metrics (run diverged)", report)

    def test_the_reason_the_run_stopped_is_printed(self) -> None:
        report = self._report(
            status="diverged",
            termination={"reason": "fit_loss reached 1.5e+09 at round 1"},
        )
        self.assertIn("- Stopped: fit_loss reached 1.5e+09 at round 1", report)

    def test_a_termination_without_a_reason_still_names_the_detector(self) -> None:
        report = self._report(
            status="stalled",
            termination={"detector": "patience", "round_id": 120},
        )
        self.assertIn("- Stopped: patience at round 120", report)

    def test_a_run_still_in_flight_is_named_as_such(self) -> None:
        report = self._report(status="running", num_rounds=177, final_round=177)
        self.assertIn("## Last Evaluated Metrics (run running)", report)
        self.assertIn("- Status: running", report)

    def test_a_missing_status_is_not_assumed_to_be_success(self) -> None:
        run = _run_json()
        del run["status"]
        (self.run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
        report = make_run_report(self.run_dir).read_text(encoding="utf-8")
        self.assertIn("## Last Evaluated Metrics (run status unknown)", report)
        self.assertIn("- Status: unknown", report)

    def test_a_short_run_is_measured_against_the_configured_total(self) -> None:
        """ "Rounds: 1" alone reads as a one-round run, not one that died."""

        report = self._report(status="diverged", num_rounds=1, final_round=1)
        self.assertIn("- Rounds: 1 of 500 configured", report)

    def test_a_full_run_does_not_repeat_the_configured_total(self) -> None:
        self.assertIn("- Rounds: 500\n", self._report())

    def test_the_metrics_themselves_are_unchanged(self) -> None:
        """Only the heading is a claim; the numbers are reported either way."""

        for status in ("completed", "diverged", "running"):
            with self.subTest(status=status):
                report = self._report(status=status)
                self.assertIn("| fit_loss | 0.42 |", report)
                self.assertIn("| val_accuracy_avg | 0.81 |", report)


if __name__ == "__main__":
    unittest.main()
