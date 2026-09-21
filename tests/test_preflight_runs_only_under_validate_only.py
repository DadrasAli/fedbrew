"""Which checks an ordinary run makes, and which only `--validate-only` makes.

Every `fedbrew run` loads the config through `validate_config`, which stops
at the first problem and refuses the run, and then builds every component,
whose builders refuse what they cannot build. The ten-area preflight —
`validation.run_checks` over `CHECKS`, reporting every error, warning and note
in one pass — runs only under `--validate-only`. An ordinary run does not call
it, so a warning or note only preflight raises (a non-empty output directory,
an unchecked extension pairing, the aggregation-weighting notice) is never
printed by a training run. Chapter 04 §1 says so; the audit (03-final-release-
gate, B03) found the paper implying the opposite.

Both halves are asserted here through `runner.main`, the function the console
script calls, so a change that starts or stops calling either on either path
fails here.
"""

from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from unittest import mock

import yaml

from fedbrew.core import config as config_module
from fedbrew.core import runner, validation

TEN_AREAS = (
    "components",
    "experiment",
    "server",
    "client",
    "task",
    "data",
    "model",
    "evaluation",
    "runtime",
    "algorithm compatibility",
)


def _calls(argv_tail: list[str]) -> dict[str, int]:
    """How often each entry point ran for one `fedbrew run` invocation."""

    counts = {"validate_config": 0, "run_checks": 0}
    validate_config = config_module.validate_config
    run_checks = validation.run_checks

    def count_validate(config: Any) -> None:
        counts["validate_config"] += 1
        validate_config(config)

    def count_run_checks(config: Any, on_check: Any = None) -> Any:
        counts["run_checks"] += 1
        return run_checks(config, on_check)

    with tempfile.TemporaryDirectory() as directory:
        raw = yaml.safe_load(Path("configs/dev/smoke.yaml").read_text(encoding="utf-8"))
        raw["experiment"]["output_dir"] = str(Path(directory) / "run")
        raw["runtime"]["checkpointing"] = {"enabled": False}
        path = Path(directory) / "run.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with (
            mock.patch.object(config_module, "validate_config", count_validate),
            mock.patch.object(validation, "run_checks", count_run_checks),
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()),
        ):
            runner.main(["--config", str(path), "--quiet", *argv_tail])
        wrote = (Path(directory) / "run" / "round_metrics.csv").exists()
    counts["trained"] = int(wrote)
    return counts


class TheFullPreflightRunsOnlyUnderValidateOnlyTest(unittest.TestCase):
    def test_the_ten_areas_are_the_ones_the_docs_name(self) -> None:
        self.assertEqual(validation.CHECK_NAMES, TEN_AREAS)
        chapter = Path("docs/04-configuration.md").read_text(encoding="utf-8")
        for area in TEN_AREAS:
            self.assertIn(f"`{area}`", chapter)

    def test_an_ordinary_run_validates_the_config_and_skips_the_preflight(self) -> None:
        counts = _calls([])
        self.assertEqual(counts["trained"], 1)
        self.assertGreaterEqual(counts["validate_config"], 1)
        self.assertEqual(counts["run_checks"], 0)

    def test_validate_only_runs_the_preflight_and_does_not_train(self) -> None:
        counts = _calls(["--validate-only"])
        self.assertEqual(counts["trained"], 0)
        self.assertGreaterEqual(counts["validate_config"], 1)
        self.assertEqual(counts["run_checks"], 1)


if __name__ == "__main__":
    unittest.main()
