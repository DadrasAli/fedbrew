"""`evaluate_model` is optional, and until now nothing said so anywhere.

`eval_step` is per batch and every task must have it. `evaluate_model` scores a
whole dataset in one call and only the central test set needs it, so it is not
on the `TaskAdapter` contract -- twelve task doubles in this suite implement the
five abstract methods and stop, which is the shape that makes it optional.

The absence of a statement is what this guards. Three callers read it three
ways: both servers looked it up with `getattr(self.task, "evaluate_model",
None)` and dropped every `central_test_*` column when it was missing,
`fedbrew/cli/eval_base_model.py` called it outright and would have raised
`AttributeError` on a task that lacked it, and `docs/12` §3.5 listed the five
methods a new task implements without naming it -- so a task written from the
chapter loses its central-test metrics silently and correctly, by the letter of
the documentation.

Nothing here changes what a run produces. The degrade to `{}` is pinned rather
than removed: it is what lets a minimal double work.
"""

from __future__ import annotations

import unittest
from typing import Any

import pytest
import torch
from torch import nn

from fedbrew.core import registry
from fedbrew.tasks.base import SupportsDatasetEvaluation, TaskAdapter
from fedbrew.tasks.causal_lm.torch_causal_lm import TorchCausalLMTask
from fedbrew.tasks.classification.torch_classification import TorchClassificationTask

pytestmark = pytest.mark.fast

#: The five the chapter lists, which is what a minimal task implements.
ABSTRACT_METHODS = ("build_model", "build_dataloader", "train_step", "eval_step", "compute_metrics")


class _MinimalTask(TaskAdapter):
    """Exactly the abstract contract: the shape twelve doubles here have."""

    def build_model(self, config: Any) -> Any:
        torch.manual_seed(0)
        return nn.Linear(1, 1)

    def build_dataloader(self, data: Any, config: Any) -> Any:
        return []

    def train_step(self, model: Any, batch: Any, optimizer: Any = None) -> dict[str, float]:
        return {}

    def eval_step(self, model: Any, batch: Any) -> dict[str, float]:
        return {}

    def compute_metrics(self, outputs: Any) -> dict[str, float]:
        return {}


class TheContractStillHasFiveAbstractMethodsTest(unittest.TestCase):
    def test_the_abstract_set_is_what_the_chapter_lists(self) -> None:
        """Making the hook abstract would break every minimal double at once."""

        self.assertEqual(sorted(TaskAdapter.__abstractmethods__), sorted(ABSTRACT_METHODS))

    def test_a_minimal_task_can_still_be_constructed(self) -> None:
        _MinimalTask()


class TheCapabilityIsDeclaredTest(unittest.TestCase):
    def test_both_registered_tasks_have_it(self) -> None:
        registry.register_builtin_components()
        self.assertEqual(sorted(registry.tasks.builtin()), ["causal_lm", "classification"])
        for task in (TorchClassificationTask, TorchCausalLMTask):
            with self.subTest(task=task.__name__):
                self.assertTrue(issubclass(task, SupportsDatasetEvaluation))

    def test_a_minimal_task_does_not(self) -> None:
        """Anti-vacuity: a protocol nothing fails admits everything."""

        self.assertNotIsInstance(_MinimalTask(), SupportsDatasetEvaluation)


class TheServersDegradeRatherThanFailTest(unittest.TestCase):
    """Pinned, not changed: it is the escape hatch a minimal double uses."""

    def _server(self, module_name: str, class_name: str) -> Any:
        """A server far enough constructed to reach the capability check.

        `__new__` rather than the constructor: the point is the one branch, and
        a real server needs a config, a strategy and a registry to exist.
        """

        import importlib

        task = _MinimalTask()
        model = task.build_model({})
        server_class = getattr(importlib.import_module(module_name), class_name)
        server = server_class.__new__(server_class)
        server.task = task
        server.model_config = {}
        server._model_state = task.get_federated_model_state(model)
        server._model_state_scope = "full"
        server._model_state_metadata = task.federated_model_state_metadata(model)
        return server

    def test_fedavg_returns_no_central_metrics(self) -> None:
        server = self._server("fedbrew.servers.fedavg", "FedAvgServer")
        self.assertEqual(server.evaluate_global(object()), {})

    def test_scaffold_returns_no_central_metrics(self) -> None:
        server = self._server("fedbrew.servers.scaffold", "ScaffoldServer")
        self.assertEqual(server.evaluate_global(object()), {})


class TheChapterNamesTheHookTest(unittest.TestCase):
    """A task written from §6 used to lose central_test_* and be told nothing."""

    def test_the_task_section_says_it_exists_and_what_omitting_it_costs(self) -> None:
        from pathlib import Path

        from docs_sections import section_of

        chapter = (Path(__file__).resolve().parent.parent / "docs" / "12-extending.md").read_text(
            encoding="utf-8"
        )
        section = section_of(chapter, "## 6. A new task or dataset backend")
        self.assertIn("evaluate_model", section)
        self.assertIn("central_test_", section)
        self.assertIn("SupportsDatasetEvaluation", section)


if __name__ == "__main__":
    unittest.main()
