"""A column's gloss says what the run's task measures, not what classification does.

The plan header glossed every ``loss`` as a cross-entropy -- ``fit_loss``,
``fit_total_loss``, the client aggregates -- whatever the task was, so a
quadratic's objective read as a cross-entropy beside its own column. A task
now says what each of its metrics measures (``TaskAdapter.METRIC_GLOSSES``,
registered with ``glosses=``), and the header composes from that. Pinned here:

- a task's glosses reach ``fit_<m>``, ``central_test_<m>``, ``fit_total_loss``
  and the client aggregates;
- a task that declares nothing reads as classification, and those five
  columns' fixed text is the same composition, so the two cannot disagree;
- every shipped task declares a gloss for every metric it reports.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import pytest

from fedbrew.core import registry
from fedbrew.core.metrics import FIXED_METRIC_GLOSSES, METRIC_BASE_GLOSSES, metric_gloss

REPO = Path(__file__).resolve().parent.parent
QUADRATIC = {"loss": "client objective ½xᵀAx − b_iᵀx", "optimality_gap": "gap F(x) − F*"}


@pytest.mark.fast
class TheTaskSaysWhatItsMetricsAreTest(unittest.TestCase):
    def test_the_fit_and_central_columns(self) -> None:
        self.assertEqual(
            metric_gloss("fit_loss", metric_glosses=QUADRATIC),
            "Example-weighted mean client objective ½xᵀAx − b_iᵀx of selected clients' "
            "post-fit local models on their train sets.",
        )
        self.assertEqual(
            metric_gloss("central_test_optimality_gap", metric_glosses=QUADRATIC),
            "Gap F(x) − F* of the global model on the complete global test set.",
        )
        self.assertIn("client objective", metric_gloss("fit_total_loss", metric_glosses=QUADRATIC))

    def test_the_client_aggregates(self) -> None:
        gloss = metric_gloss("test_loss_std", metric_glosses=QUADRATIC)
        self.assertTrue(gloss.startswith("Client objective ½xᵀAx − b_iᵀx on client test data"))
        self.assertNotIn("ross-entropy", gloss)

    def test_no_task_is_classification_and_the_fixed_text_agrees(self) -> None:
        for name in (
            "fit_loss",
            "fit_accuracy",
            "fit_total_loss",
            "central_test_loss",
            "central_test_accuracy",
        ):
            with self.subTest(column=name):
                self.assertEqual(metric_gloss(name), FIXED_METRIC_GLOSSES[name])
                self.assertEqual(
                    metric_gloss(name, metric_glosses=METRIC_BASE_GLOSSES),
                    FIXED_METRIC_GLOSSES[name],
                )

    def test_the_built_in_tasks_declare_every_metric(self) -> None:
        registry.register_builtin_components()
        for task in ("classification", "causal_lm"):
            with self.subTest(task=task):
                self.assertEqual(
                    set(registry.tasks.glosses(task) or {}), set(registry.tasks.metrics(task) or {})
                )
        self.assertIn("token", registry.tasks.glosses("causal_lm")["loss"])

    def test_the_example_tasks_declare_every_metric(self) -> None:
        """Read from the source, so no example is imported (each one self-checks at import)."""

        checked = 0
        for problem in sorted((REPO / "examples").glob("*/problem.py")):
            for node in ast.walk(ast.parse(problem.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.ClassDef):
                    continue
                declared = {
                    target.id: ast.literal_eval(statement.value)
                    for statement in node.body
                    if isinstance(statement, ast.Assign)
                    for target in statement.targets
                    if isinstance(target, ast.Name) and target.id in {"METRICS", "METRIC_GLOSSES"}
                }
                if "METRICS" not in declared:
                    continue
                checked += 1
                with self.subTest(task=f"{problem.parent.name}.{node.name}"):
                    self.assertEqual(
                        set(declared.get("METRIC_GLOSSES", {})), set(declared["METRICS"])
                    )
                    self.assertIn("glosses=", problem.read_text(encoding="utf-8"))
        self.assertEqual(checked, 5)


if __name__ == "__main__":
    unittest.main()
