"""A task may say which of its declared metrics a given run reports, per pass.

``TaskAdapter.METRICS`` is one mapping per task, and the plan header lists
``fit_<name>`` and ``central_test_<name>`` for each. A task whose columns
depend on the problem its model block poses -- an optimality gap only where
F* is certified -- or whose central pass measures what a client's cannot
registers a function of the config that narrows them
(``registry.tasks.register(..., reported=...)``, ``ReportedMetrics``). Pinned
here: the header follows it on both passes, None leaves every declared name,
and a name the task does not declare is refused.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import pytest

from fedbrew.core import registry
from fedbrew.core.config import load_config, task_reported_metrics
from fedbrew.core.logging import _planned_metric_names
from fedbrew.tasks.base import ReportedMetrics

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "configs" / "dev" / "synthetic.yaml"


@pytest.mark.fast
class ARunsReportedMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        registry.register_builtin_components()
        self.config = load_config(CONFIG)

    def _narrowed(self, reported: ReportedMetrics | None) -> mock._patch:
        narrowings = {self.config.task.name: lambda config: reported}
        return mock.patch.dict(registry.tasks._reported, narrowings)

    def test_without_a_narrowing_every_declared_name_on_both(self) -> None:
        declared = tuple(registry.tasks.metrics(self.config.task.name) or ())
        self.assertEqual(
            task_reported_metrics(self.config), ReportedMetrics(client=declared, central=declared)
        )
        with self._narrowed(None):
            self.assertEqual(task_reported_metrics(self.config).central, declared)

    def test_the_header_follows_each_pass(self) -> None:
        planned = set(_planned_metric_names(self.config))
        self.assertLessEqual({"fit_accuracy", "central_test_accuracy"}, planned)
        with self._narrowed(ReportedMetrics(client=("loss",), central=("loss", "accuracy"))):
            narrowed = set(_planned_metric_names(self.config))
        # fit_accuracy and the client splits' accuracy aggregates go; the
        # central pass keeps its column.
        dropped = planned - narrowed
        self.assertIn("fit_accuracy", dropped)
        self.assertIn("test_accuracy_avg", dropped)
        self.assertTrue(all("accuracy" in name for name in dropped), dropped)
        self.assertIn("central_test_accuracy", narrowed)

    def test_an_undeclared_name_is_refused(self) -> None:
        with self._narrowed(ReportedMetrics(client=("loss", "gap"), central=("loss",))):
            with self.assertRaisesRegex(ValueError, r"client passes report \['gap'\]"):
                task_reported_metrics(self.config)


if __name__ == "__main__":
    unittest.main()
