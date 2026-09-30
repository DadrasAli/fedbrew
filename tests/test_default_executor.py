"""What a run that states no executor and no gradient form runs, and what it records.

``runtime.performance.executor`` left out asks for ``batched``, which runs
wherever the task and rule can be batched and the sequential executor where
they cannot -- recorded as the default, with the reason, not as a fallback,
since nobody asked for batched. ``gradient_form`` left out is the task's closed
form wherever it gives one, under either executor, and autograd otherwise.
Held here:

- run.json's ``reproducibility.executor`` for each of those choices, stated
  and left out, and the plan header's rows for them;
- a linear example run as written trains batched on its closed form, and a
  task without one on the form it declares;
- a model the executor cannot batch runs sequentially by default, with the
  reason and no fallback; stated ``batched``, the same run is a fallback;
- ``executor: sequential`` stated is the reference, and takes the closed form
  unless ``gradient_form: autograd`` asks for autograd;
- the config's refusals: a batched gradient form beside a stated sequential
  executor, and nothing refused when the executor is left out.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pytest
import yaml

from fedbrew.core.config import load_config
from fedbrew.core.console import AMBER
from fedbrew.core.logging import _executor_rows
from fedbrew.core.refusal import RunRefused
from fedbrew.core.runner import run
from tests.test_batched_executor_tolerance import classification_config, example_config


def _record(output: Path) -> dict[str, Any]:
    return json.loads((output / "run.json").read_text())["reproducibility"]["executor"]


class ARunAsWrittenTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self._count = 0

    def tearDown(self) -> None:
        self._directory.cleanup()

    def run_as(self, config: dict[str, Any], **performance: Any) -> Path:
        """The config run with exactly the ``runtime.performance`` keys given, and no others."""

        self._count += 1
        config = copy.deepcopy(config)
        output = self.root / f"run{self._count}"
        config["experiment"]["output_dir"] = str(output)
        config["schedule"]["rounds"] = 2
        stated = config["runtime"].setdefault("performance", {})
        for key in ("executor", "gradient_form"):
            stated.pop(key, None)
        stated.update(performance)
        path = self.root / f"run{self._count}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        run(path, args=None)
        return output

    def test_a_linear_example_trains_batched_on_its_closed_form(self) -> None:
        record = _record(self.run_as(example_config("fed-lasso")))
        self.assertEqual(record["used"], "batched")
        self.assertIs(record["default"], True)
        self.assertEqual(record["gradient_form"], {"used": "closed_form", "default": True})

    def test_a_task_without_a_closed_form_trains_on_its_declared_form(self) -> None:
        record = _record(self.run_as(classification_config()))
        self.assertEqual((record["used"], record["default"]), ("batched", True))
        self.assertEqual(record["gradient_form"], {"used": "vmap_grad", "default": True})

    def test_a_model_it_cannot_batch_runs_sequentially_with_the_reason(self) -> None:
        config = classification_config()
        config["model"]["dropout"] = 0.3
        record = _record(self.run_as(config))
        self.assertEqual((record["used"], record["default"]), ("sequential", True))
        self.assertIn("dropout at p = 0.3", record["reason"])
        self.assertNotIn("fallback", record)
        self.assertEqual(record["gradient_form"], {"used": "autograd", "default": True})
        stated = _record(self.run_as(config, executor="batched"))
        self.assertIs(stated["default"], False)
        self.assertIn("dropout at p = 0.3", stated["fallback"])
        self.assertNotIn("reason", stated)

    def test_the_sequential_reference_stated(self) -> None:
        config = example_config("fed-lasso")
        record = _record(self.run_as(config, executor="sequential"))
        self.assertEqual(
            record,
            {
                "used": "sequential",
                "default": False,
                "gradient_form": {"used": "closed_form", "default": True},
            },
        )
        autograd = _record(self.run_as(config, executor="sequential", gradient_form="autograd"))
        self.assertEqual(autograd["gradient_form"], {"used": "autograd", "default": False})
        batched = _record(self.run_as(config, executor="batched", gradient_form="autograd"))
        # fed-lasso declares its autograd form fastest summed (batched_gradient).
        self.assertEqual(batched["gradient_form"], {"used": "summed", "default": False})

    def test_a_batched_form_on_a_run_that_is_sequential_by_default(self) -> None:
        config = classification_config()
        config["model"]["dropout"] = 0.3
        record = _record(self.run_as(config, gradient_form="summed"))
        self.assertEqual(record["used"], "sequential")
        form = record["gradient_form"]
        self.assertEqual((form["used"], form["default"]), ("autograd", False))
        self.assertIn("the run is sequential", form["fallback"])


class ThePlanHeaderSaysWhichTest(unittest.TestCase):
    @pytest.mark.fast
    def test_the_rows(self) -> None:
        def rows(record: dict[str, Any]) -> list[tuple[str, str, bool]]:
            return [(row.label, row.value, row.tone == AMBER) for row in _executor_rows(record)]

        closed = {"used": "closed_form", "default": True}
        self.assertEqual(
            rows({"used": "batched", "default": True, "gradient_form": closed}),
            [
                ("Executor", "batched (default)", False),
                ("Gradient", "closed_form (default)", False),
            ],
        )
        self.assertEqual(
            rows(
                {
                    "used": "sequential",
                    "default": True,
                    "reason": "the model has frozen parameters",
                    "gradient_form": {"used": "autograd", "default": True},
                }
            ),
            [
                ("Executor", "sequential (default): the model has frozen parameters", False),
                ("Gradient", "autograd (default)", False),
            ],
        )
        self.assertEqual(
            rows(
                {
                    "used": "sequential",
                    "default": False,
                    "gradient_form": {"used": "autograd", "default": False},
                }
            ),
            [("Executor", "sequential", False), ("Gradient", "autograd", False)],
        )
        fallback = rows(
            {
                "used": "sequential",
                "default": False,
                "fallback": "the model has frozen parameters",
                "gradient_form": {"used": "autograd", "default": True},
            }
        )
        self.assertEqual(fallback[0][:1], ("Executor",))
        self.assertTrue(fallback[0][2])


class TheConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def _load(self, **performance: Any) -> Any:
        config = example_config("fed-lasso")
        config["experiment"]["output_dir"] = str(self.root / "unused")
        config["runtime"]["performance"] = dict(performance)
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return load_config(path)

    def test_what_is_accepted_and_refused(self) -> None:
        for performance in (
            {},
            {"compile": "on"},
            {"gradient_form": "vmap_grad"},
            {"gradient_form": "summed"},
            {"executor": "sequential", "gradient_form": "autograd"},
            {"executor": "sequential", "gradient_form": "closed_form"},
            {"executor": "batched", "gradient_form": "autograd"},
        ):
            with self.subTest(**performance):
                self._load(**performance)
        for form in ("vmap_grad", "summed"):
            with self.subTest(refused=form):
                with self.assertRaisesRegex(RunRefused, f"gradient_form {form} is a mode of"):
                    self._load(executor="sequential", gradient_form=form)
        with self.assertRaisesRegex(RunRefused, "compile is a mode of"):
            self._load(executor="sequential", compile="on")


if __name__ == "__main__":
    unittest.main()
