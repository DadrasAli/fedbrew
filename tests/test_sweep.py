"""``fedbrew sweep`` groups what differs only in numeric hyperparameters, and runs the rest alone.

Configs group when they are equal but for the run's name and ``VARIABLE``
(``fedbrew/core/sweep.py``): a group of two or more runs as one child process,
every other config as ``fedbrew run --config`` would. Chapter 11 §10.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

from fedbrew.core.sweep import _worst, main, plan
from tests.test_batched_executor_tolerance import example_config

REPO_ROOT = Path(__file__).resolve().parent.parent


class SweepRuns(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.base = example_config("fed-lasso")
        self.base["runtime"].setdefault("performance", {})["executor"] = "batched"

    def tearDown(self) -> None:
        self._directory.cleanup()

    def write(self, name: str, **sections: dict[str, Any]) -> str:
        config = copy.deepcopy(self.base)
        for section, values in sections.items():
            config[section].update(values)
        if "output_dir" not in sections.get("experiment", {}):
            config["experiment"]["output_dir"] = str(self.root / "out" / name)
        path = self.root / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return str(path)


class ThePlanTest(SweepRuns):
    def test_what_groups_and_what_runs_alone(self) -> None:
        base = self.write("base")
        faster = self.write("faster", client={"learning_rate": 0.1})
        clipped = self.write("clipped", client={"max_grad_norm": 0.05, "momentum": 0.5})
        seeded = self.write("seeded", experiment={"seed": 7})
        longer = self.write("longer", client={"local_iterations": 5})
        sequential = self.write("sequential", runtime={"performance": {"executor": "sequential"}})
        same_place = self.write(
            "same_place",
            client={"learning_rate": 0.2},
            experiment={"output_dir": str(self.root / "out" / "base")},
        )
        planned = plan([base, faster, clipped, seeded, longer, sequential, same_place])
        groups = [child for child in planned if child.alone is None]
        self.assertEqual([child.configs for child in groups], [[base, faster, clipped]])
        self.assertEqual(
            groups[0].varies,
            ["client.extra.max_grad_norm", "client.extra.momentum", "client.learning_rate"],
        )
        alone = {child.configs[0]: child.alone for child in planned if child.alone is not None}
        self.assertEqual(set(alone), {seeded, longer, sequential, same_place})
        self.assertEqual(alone[sequential], "runtime.performance.executor is not batched")

    def test_a_config_that_does_not_load_runs_alone(self) -> None:
        broken = self.root / "broken.yaml"
        broken.write_text("experiment: [", encoding="utf-8")
        (child,) = plan([str(broken)])
        self.assertTrue(child.alone and child.alone.startswith("does not load"))

    def test_plan_prints_and_runs_nothing(self) -> None:
        paths = [self.write("a"), self.write("b", client={"learning_rate": 0.1})]
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            main(["--plan", *paths])
        self.assertIn("group of 2, varies client.learning_rate", printed.getvalue())
        self.assertFalse((self.root / "out").exists())

    def test_run_group_refuses_what_is_not_one_group(self) -> None:
        paths = [self.write("a"), self.write("b", experiment={"seed": 7})]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main(["--run-group", *paths])
        self.assertEqual(raised.exception.code, 2)


class TheSweepRunsTest(SweepRuns):
    def test_a_group_and_a_config_alone(self) -> None:
        grouped = [self.write("a"), self.write("b", client={"learning_rate": 0.1})]
        alone = self.write("c", runtime={"performance": {"executor": "sequential"}})
        environment = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
        finished = subprocess.run(
            [sys.executable, "-m", "fedbrew.cli.dispatch", "sweep", *grouped, alone],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(finished.returncode, 0, finished.stdout[-2000:] + finished.stderr[-2000:])
        records = {
            name: json.loads((self.root / "out" / name / "run.json").read_text())
            for name in ("a", "b", "c")
        }
        for name in ("a", "b"):
            self.assertEqual(records[name]["status"], "completed")
            self.assertEqual(records[name]["reproducibility"]["group"]["size"], 2)
        self.assertEqual(records["c"]["status"], "completed")
        self.assertNotIn("group", records["c"]["reproducibility"])


class TheExitStatusTest(unittest.TestCase):
    def test_crashed_over_refused_over_ended(self) -> None:
        self.assertEqual(_worst([0, 0]), 0)
        self.assertEqual(_worst([0, 2]), 2)
        self.assertEqual(_worst([2, 1]), 1)
        self.assertEqual(_worst([0, 137]), 1)


if __name__ == "__main__":
    unittest.main()
