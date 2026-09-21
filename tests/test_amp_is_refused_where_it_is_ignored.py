"""`runtime.use_amp: true` on a task with no AMP path is refused, not ignored.

`TorchCausalLMTask.__init__` takes no `use_amp` and its `train_step` has no
`autocast` or `GradScaler`. The factory passed the flag only to the
classification task, spelled as `config.task.name == "classification"`, so an
LLM config setting `use_amp: true` was accepted by `validate_config`, passed
preflight without a word, echoed into `run.json`, and ran in fp32 throughout.

The numbers stayed right -- fp32 is the reference -- which is exactly what kept
it quiet. What was wrong is the record: a `run.json` saying the run used mixed
precision when it did not. It also left two guards reading as protection they
do not give, since `getattr(task, "_scaler", None)` in `delta_sgd` and
`fedlalr` can never find a scaler on a task that has none.

Refused rather than implemented. Adding autocast to the causal-LM task is a
feature and would move every LLM number; every shipped LLM config already sets
`use_amp: false`, so refusing costs nothing today -- `TheShippedConfigsTest`
checks that rather than trusting it.

`AMP_AWARE_TASKS` is the one home for which tasks understand the flag. The
factory reads it to decide what to pass *and* what to refuse, so the two cannot
drift; before, the second half did not exist and the first was an inline
string comparison. An extension task is never in the set, which is right:
`EXTENSION_TASK_KEYS` does not carry `use_amp` either, so an out-of-tree
adapter cannot receive it and must not be told a run honoured it.

Two places say it, deliberately. `factory._refuse_amp_a_task_cannot_honour`
raises, because that is the gate an ordinary `fedbrew run` passes through;
`validation._validate_task` reports it as a preflight error, because
`--validate-only` is where a reader looks before spending a GPU hour. The four
existing AMP refusals are per algorithm and live in preflight for the same
reason. See FINDINGS.csv P03-F07.
"""

from __future__ import annotations

import copy
import glob
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from fedbrew.core import factory as factory_module
from fedbrew.core.config import load_config
from fedbrew.core.factory import (
    AMP_AWARE_TASKS,
    EXTENSION_TASK_KEYS,
    _refuse_amp_a_task_cannot_honour,
)
from fedbrew.core.registry import register_builtin_components, tasks
from fedbrew.core.validation import validate_full_config

REPO_ROOT = Path(__file__).resolve().parent.parent
LLM_CONFIG = "configs/dev/tiny_causal_lm.yaml"


def _amp_issues(config: object) -> list[tuple[str, str]]:
    return [
        (issue.severity, issue.code)
        for issue in validate_full_config(config).issues
        if "amp" in issue.code
    ]


@pytest.mark.fast
class TheSetIsTheOneHomeTest(unittest.TestCase):
    def test_only_classification_implements_it(self) -> None:
        self.assertEqual(AMP_AWARE_TASKS, {"classification"})

    def test_every_name_in_it_is_a_registered_task(self) -> None:
        register_builtin_components()
        for name in sorted(AMP_AWARE_TASKS):
            with self.subTest(task=name):
                self.assertTrue(tasks.exists(name))

    def test_a_registered_task_outside_it_really_takes_no_use_amp(self) -> None:
        """Otherwise the refusal fires on a task that would have honoured it."""

        import inspect

        from fedbrew.tasks.causal_lm.torch_causal_lm import TorchCausalLMTask

        self.assertNotIn("causal_lm", AMP_AWARE_TASKS)
        self.assertNotIn("use_amp", inspect.signature(TorchCausalLMTask).parameters)
        source = inspect.getsource(TorchCausalLMTask)
        for absent in ("autocast", "GradScaler"):
            with self.subTest(absent=absent):
                self.assertNotIn(absent, source)

    def test_the_classification_task_does_take_it(self) -> None:
        import inspect

        from fedbrew.tasks.classification.torch_classification import (
            TorchClassificationTask,
        )

        self.assertIn("use_amp", inspect.signature(TorchClassificationTask).parameters)

    def test_an_extension_task_cannot_receive_it_either(self) -> None:
        self.assertNotIn("use_amp", EXTENSION_TASK_KEYS)


@pytest.mark.fast
class TheFactoryRefusesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(LLM_CONFIG)

    def test_the_shipped_setting_is_accepted(self) -> None:
        self.assertFalse(self.config.runtime.use_amp)
        _refuse_amp_a_task_cannot_honour(self.config)

    def test_turning_it_on_is_refused(self) -> None:
        config = copy.deepcopy(self.config)
        config.runtime.use_amp = True
        with self.assertRaises(ValueError) as caught:
            _refuse_amp_a_task_cannot_honour(config)
        message = str(caught.exception)
        self.assertIn("causal_lm", message)
        self.assertIn("classification", message)
        self.assertIn("fp32", message)

    def test_a_classification_config_is_untouched(self) -> None:
        config = load_config("configs/mnist/fedavg.yaml")
        self.assertEqual(config.task.name, "classification")
        for use_amp in (True, False):
            with self.subTest(use_amp=use_amp):
                candidate = copy.deepcopy(config)
                candidate.runtime.use_amp = use_amp
                _refuse_amp_a_task_cannot_honour(candidate)

    def test_an_unknown_task_with_amp_off_is_left_to_the_task_check(self) -> None:
        config = copy.deepcopy(self.config)
        config.task.name = "not_a_task"
        _refuse_amp_a_task_cannot_honour(config)


@pytest.mark.fast
class TheFactoryActuallyCallsItTest(unittest.TestCase):
    """The wiring, not the predicate.

    Deleting the one call from `_build_task` left every other test in this
    module passing: they exercise `_refuse_amp_a_task_cannot_honour` directly,
    and a refusal nothing calls refuses nothing. Found by the deliberate error
    that was supposed to be routine.

    Driven through `build_components` on a synthetic-classification config --
    which needs no manifest on disk -- with `AMP_AWARE_TASKS` emptied, so the
    refusal fires on the one task that is cheap to build.
    """

    def _config(self, directory: Path, use_amp: bool) -> Any:
        path = directory / "amp.yaml"
        path.write_text(
            textwrap.dedent(f"""
            experiment:
              seed: 1
              output_dir: {directory / "run"}
            server:
              strategy: fedavg
              participation_rate: 1
              metrics: [fit_loss]
            client:
              update_rule: local_sgd
              batch_size: 4
              learning_rate: 0.05
              learning_rate_schedule: constant
              min_learning_rate: 0.0
              momentum: 0.0
              weight_decay: 0.0
              nesterov: false
              metrics: [fit_loss]
            data:
              num_clients: 2
              samples_per_client: 8
              input_dim: 4
              num_classes: 2
            model:
              name: mlp
              input_dim: 4
              hidden_dim: 8
              num_classes: 2
            runtime:
              deterministic: true
              device: cpu
              use_amp: {str(use_amp).lower()}
            evaluation:
              train: {{every: 1, clients: all}}
            defaults:
              global_rounds: 1
              local_iterations: 1
            """).lstrip(),
            encoding="utf-8",
        )
        return load_config(path)

    def test_build_components_raises_when_the_task_cannot_honour_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory), use_amp=True)
            with mock.patch.object(factory_module, "AMP_AWARE_TASKS", set()):
                with self.assertRaises(ValueError) as caught:
                    factory_module.build_components(config)
            self.assertIn("mixed-precision", str(caught.exception))

    def test_build_components_is_untouched_when_it_can(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory), use_amp=True)
            self.assertIn(config.task.name, AMP_AWARE_TASKS)
            factory_module.build_components(config)

    def test_amp_off_builds_either_way(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory), use_amp=False)
            with mock.patch.object(factory_module, "AMP_AWARE_TASKS", set()):
                factory_module.build_components(config)


class PreflightSaysItTooTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(LLM_CONFIG)

    @pytest.mark.fast
    def test_it_is_an_error_not_a_warning(self) -> None:
        config = copy.deepcopy(self.config)
        config.runtime.use_amp = True
        self.assertEqual(_amp_issues(config), [("error", "task.amp_unsupported")])

    @pytest.mark.fast
    def test_the_shipped_config_raises_none(self) -> None:
        self.assertEqual(_amp_issues(self.config), [])

    def test_a_classification_config_raises_none_either_way(self) -> None:
        config = load_config("configs/mnist/fedavg.yaml")
        for use_amp in (True, False):
            with self.subTest(use_amp=use_amp):
                candidate = copy.deepcopy(config)
                candidate.runtime.use_amp = use_amp
                self.assertEqual(_amp_issues(candidate), [])

    @pytest.mark.fast
    def test_an_unknown_task_reports_that_and_not_this(self) -> None:
        """One finding per config error; the amp check waits its turn."""

        config = copy.deepcopy(self.config)
        config.task.name = "not_a_task"
        config.runtime.use_amp = True
        codes = [issue.code for issue in validate_full_config(config).issues]
        self.assertIn("task.name_unknown", codes)
        self.assertNotIn("task.amp_unsupported", codes)


class TheShippedConfigsTest(unittest.TestCase):
    """Why refusing is free: nothing that ships asks for it."""

    def test_no_shipped_config_sets_amp_on_a_task_that_ignores_it(self) -> None:
        offenders = []
        checked = 0
        for path in sorted(glob.glob("configs/**/*.yaml", recursive=True)):
            try:
                config = load_config(path)
            except Exception:
                continue
            checked += 1
            if config.runtime.use_amp and config.task.name not in AMP_AWARE_TASKS:
                offenders.append(path)
        self.assertGreater(checked, 0, "no config loaded; the sweep found nothing")
        self.assertEqual(offenders, [])

    def test_at_least_one_shipped_config_is_on_the_ignoring_task(self) -> None:
        """Otherwise the sweep above passes because the case does not exist."""

        found = [
            path
            for path in sorted(glob.glob("configs/**/*.yaml", recursive=True))
            if _task_name(path) not in (None, *AMP_AWARE_TASKS)
        ]
        self.assertTrue(found, "no shipped config uses a task without an AMP path")


def _task_name(path: str) -> str | None:
    try:
        return load_config(path).task.name
    except Exception:
        return None


if __name__ == "__main__":
    unittest.main()
