"""``runtime.performance.compile: on`` compiles a bucket's local loop, within the tolerance.

A bucket's whole round of steps is one compiled call (``local_loop``): its
batches gathered, the rule's step vmapped, a pass's combination, every step,
so dynamo's guards and wrappers run once a round. Resident runs take it too.

``torch.compile`` fuses the step's operations, which rounds them in another
order and nothing else, so a compiled run is held where the batched executor
is held against the sequential one (chapter 11 §9): ``1e-12`` relative per
model tensor and cell in float64, ``1e-4`` for the float32 MLP. Measured over
four rounds (2026-09-28): fed-lasso 9.6e-18 of the model and 4.2e-16 in a
cell, the MLP 2.9e-7 and 1.6e-6.

A step that does not compile -- no C++ compiler for inductor, an operation
dynamo does not trace, the recompilation limit -- runs eagerly, as does
every later step; the run says so on stderr and in run.json, and is the
reference run.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import yaml

from fedbrew.core.batched_executor import StepContext
from fedbrew.core.config import load_config
from fedbrew.core.refusal import RunRefused
from tests.test_batched_executor_tolerance import (
    FLOAT32_TOLERANCE,
    ExecutorRuns,
    classification_rule_config,
    example_config,
    rule_arms,
    rule_config,
)


def _working_compiler() -> str | None:
    """A C++ compiler inductor can use: ``$CXX``, ``g++`` or the system's, whichever runs."""

    for candidate in (os.environ.get("CXX"), "g++", "/usr/bin/g++"):
        path = shutil.which(candidate) if candidate else None
        if path is None:
            continue
        try:
            version = subprocess.run(
                [path, "--version"], capture_output=True, text=True, timeout=60
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if version.returncode == 0 and "Free Software Foundation" in version.stdout:
            return path
    return None


def executor_record(output: Path) -> dict[str, Any]:
    return json.loads((output / "run.json").read_text())["reproducibility"]["executor"]


class CompiledRuns(ExecutorRuns):
    def compiled_and_reference(self, config: dict[str, Any]) -> tuple[Path, Path]:
        from torch._dynamo.utils import counters

        counters.clear()
        compiled = self.run_config(config, "batched", compile=True)
        #: Whether dynamo made a graph of the step: a step run eagerly
        #: would pass every tolerance below.
        self.graphs = counters["stats"]["unique_graphs"]
        return compiled, self.run_config(config, "batched")


class CompiledStepTest(CompiledRuns):
    @classmethod
    def setUpClass(cls) -> None:
        compiler = _working_compiler()
        if compiler is None:
            raise unittest.SkipTest("no C++ compiler inductor can run")
        import torch._inductor.config

        torch._inductor.config.cpp.cxx = (compiler,)

    def test_the_linear_examples_within_the_executor_tolerance(self) -> None:
        arms = dict(rule_arms())
        for label, config in (
            ("fed-lasso fedavg", example_config("fed-lasso")),
            *(
                (f"fed-lasso-l2 {label}", rule_config(arms[label]))
                for label in (
                    "local_sgd/sequential_epoch",
                    "local_sgd/full_gradient",
                    "local_adamw/sequential_epoch",
                    "scaffold/full_gradient",
                )
            ),
        ):
            with self.subTest(run=label):
                compiled, reference = self.compiled_and_reference(config)
                self.assertEqual(executor_record(compiled)["compile"], {"used": "on"})
                self.assertGreater(self.graphs, 0)
                self.assertAgree(compiled, reference)

    def test_the_mlp_within_the_float32_bound(self) -> None:
        compiled, reference = self.compiled_and_reference(
            classification_rule_config({"update_rule": "local_sgd"})
        )
        self.assertEqual(executor_record(compiled)["compile"], {"used": "on"})
        self.assertGreater(self.graphs, 0)
        self.assertAgree(compiled, reference, tolerance=FLOAT32_TOLERANCE)


class TheLoopIsCompiledWholeTest(CompiledRuns):
    """One compiled call a bucket a round, resident or not, in each gradient form."""

    @classmethod
    def setUpClass(cls) -> None:
        CompiledStepTest.setUpClass()

    def test_one_call_a_round_within_the_executor_tolerance(self) -> None:
        from tests.test_closed_form_gradients import CASES, _config

        loop = StepContext.loop
        hetero = CASES["heterogeneous quadratic"]
        logistic = CASES["l1-regularized logistic regression"]
        for label, (arm, setting), form in (
            ("heterogeneous-quadratic, vmap_grad", hetero, None),
            ("heterogeneous-quadratic, closed_form", hetero, "closed_form"),
            ("fed-logistic-l1, closed_form", logistic, "closed_form"),
        ):
            config = _config(arm, setting, self.root)
            config["schedule"]["rounds"] = 4
            config["runtime"]["quiet"] = True
            config["runtime"]["checkpointing"].update(
                enabled=True, save_last=True, save_every_round=True, keep_last=None
            )
            forms = {} if form is None else {"gradient_form": form}
            with self.subTest(run=label):
                calls: list[int] = []

                def counted(
                    context: StepContext, *args: Any, calls: list[int] = calls, loop: Any = loop
                ) -> Any:
                    calls.append(1)
                    return loop(context, *args)

                with mock.patch.object(StepContext, "loop", counted):
                    compiled = self.run_config(config, "batched", compile=True, **forms)
                reference = self.run_config(config, "batched", **forms)
                record = executor_record(compiled)
                self.assertEqual(record["compile"], {"used": "on"})
                self.assertEqual(record["rounds"], {"used": "resident"})
                # One bucket a round: every round's steps in one call.
                self.assertEqual(len(calls), 4)
                self.assertAgree(compiled, reference)


class AStepThatDoesNotCompileRunsEagerlyTest(CompiledRuns):
    def test_the_reference_bit_for_bit_and_the_record(self) -> None:
        def no_compiler(self: StepContext) -> Any:
            def fail(*args: Any) -> Any:
                raise RuntimeError("no C++ compiler found")

            return fail

        config = classification_rule_config({"update_rule": "local_sgd"})
        with (
            mock.patch.object(StepContext, "_compiler", no_compiler),
            mock.patch.object(StepContext, "_loop_compiler", no_compiler),
        ):
            compiled, reference = self.compiled_and_reference(config)
        self.assertAgree(compiled, reference, exact=True)
        self.assertEqual(
            executor_record(compiled)["compile"],
            {"used": "off", "fallback": "RuntimeError: no C++ compiler found"},
        )


class TheRecordNamesTheWrappedCauseTest(unittest.TestCase):
    def test_a_wrapped_error_is_recorded_with_its_cause(self) -> None:
        record: dict[str, Any] = {"used": "batched"}
        context = StepContext(True, "reference", record)
        try:
            try:
                raise OSError("gcc: command not found")
            except OSError as cause:
                raise RuntimeError("backend='inductor' raised:\nlong trace") from cause
        except RuntimeError as error:
            with mock.patch("sys.stderr"):
                context._fail(error)
        self.assertEqual(
            record["compile"]["fallback"],
            "RuntimeError: backend='inductor' raised: OSError: gcc: command not found",
        )


class ConfigTest(unittest.TestCase):
    def _load(self, **performance: Any) -> None:
        config = copy.deepcopy(example_config("fed-lasso"))
        config["runtime"].setdefault("performance", {}).update(performance)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            load_config(path)

    def test_compile_needs_the_batched_executor(self) -> None:
        with self.assertRaisesRegex(RunRefused, "executor: batched"):
            self._load(compile=True)
        self._load(compile=False)

    def test_on_and_off_as_yaml_reads_them_or_as_words(self) -> None:
        for value in (True, False, "on", "off"):
            with self.subTest(value=value):
                self._load(executor="batched", compile=value)
        with self.assertRaisesRegex(RunRefused, "compile must be on or off"):
            self._load(executor="batched", compile="yes")


if __name__ == "__main__":
    unittest.main()
