"""`--validate-only` streams its checks, then settles into the plan.

Three properties, and each of them is a decision that could reasonably have
gone the other way.

**Every check runs, including after one fails.** The run path's
`validate_config` stops at the first problem; preflight exists so a reader can
fix everything in one pass rather than discovering the next error after the
next launch. A rail that stopped executing at the first ✗ would quietly turn
preflight into the thing it was built to replace.

**A failing check withholds the plan.** The plan is what a clean preflight has
proved. Printed under an error it reads as "here is what will happen", when
nothing will.

**Errors appear twice, warnings once.** Each error is inline at the check that
raised it *and* in the verdict, because a rail is read while it runs and a
verdict is read when it stops -- a reader who watched ten lines scroll past
should not scroll back to find what to fix. Warnings stay inline only: they are
advisory, and repeating them would bury the errors among them.
"""

from __future__ import annotations

import io
import unittest
import unittest.mock
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from fedbrew.core import runner
from fedbrew.core.config import (
    ClientConfig,
    DataConfig,
    ExperimentConfig,
    FullConfig,
    ModelConfig,
    RuntimeConfig,
    ServerConfig,
    TaskConfig,
)
from fedbrew.core.console import DONE, FAIL, RAIL, WARN, build_surface
from fedbrew.core.validation import (
    CHECK_NAMES,
    CHECKS,
    ValidationIssue,
    ValidationReport,
    print_validation_verdict,
    run_checks,
    stream_checks,
    validate_full_config,
)

pytestmark = pytest.mark.fast

SMOKE_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "dev" / "smoke.yaml"


def _config(
    *,
    output_dir: str = "outputs/rail-test",
    data_path: str | None = None,
    model: str = "mlp",
) -> FullConfig:
    return FullConfig(
        experiment=ExperimentConfig(seed=1, output_dir=output_dir, name="rail-test"),
        server=ServerConfig(strategy="fedavg", global_rounds=5, participation_rate=1.0, metrics=[]),
        client=ClientConfig(
            update_rule="local_sgd",
            local_iterations=1,
            batch_size=8,
            metrics=[],
            learning_rate=0.1,
        ),
        task=TaskConfig(name="classification"),
        data=DataConfig(name="manifest_dataset", path=data_path)
        if data_path
        else DataConfig(name="synthetic_classification", num_clients=4, input_dim=6, num_classes=3),
        model=ModelConfig(name=model, input_dim=6, hidden_dim=8, num_classes=3),
        runtime=RuntimeConfig(device="cpu", use_amp=False),
    )


def _failing_config(output_dir: str) -> FullConfig:
    """A config that loads but fails two different checks.

    A manifest path that does not exist (the data check) and an `mlp` with no
    dimensions (the model check), so the "later checks still run" property has
    something to be true about.
    """

    config = _config(output_dir=output_dir, data_path="data/generated/definitely-not-here.json")
    config.model = ModelConfig(name="mlp")
    return config


def _stream(config: FullConfig) -> tuple[str, ValidationReport]:
    buffer = io.StringIO()
    surface = build_surface(file=buffer)
    report = stream_checks(config, "rail-test.yaml", surface.rail(["load config", *CHECK_NAMES]))
    return buffer.getvalue(), report


class TheCheckSequenceTest(unittest.TestCase):
    def test_the_hook_fires_once_per_check_in_order(self) -> None:
        seen: list[str] = []
        run_checks(_config(), on_check=lambda name, _: seen.append(name))
        self.assertEqual(seen, list(CHECK_NAMES))

    def test_each_hook_call_carries_only_that_check_s_findings(self) -> None:
        """Attribution is the whole point: a finding shown under the wrong
        check sends the reader to the wrong config block."""

        with TemporaryDirectory() as scratch:
            by_check: dict[str, list[str]] = {}
            run_checks(
                _failing_config(scratch),
                on_check=lambda name, found: by_check.__setitem__(
                    name, [issue.code for issue in found]
                ),
            )
        self.assertEqual(by_check["data"], ["data.manifest_missing"])
        self.assertTrue(all(code.startswith("model.") for code in by_check["model"]))
        self.assertEqual(by_check["server"], [])

    def test_the_sequence_is_what_validate_full_config_reports(self) -> None:
        """The hook must not change what preflight finds -- every existing
        caller of validate_full_config reads the same report it always did."""

        with TemporaryDirectory() as scratch:
            config = _failing_config(scratch)
            streamed = [issue.code for issue in run_checks(config)]
            reported = [issue.code for issue in validate_full_config(config).issues]
        self.assertEqual(streamed, reported)

    def test_every_check_has_a_name_and_a_callable(self) -> None:
        self.assertEqual(len(CHECKS), len(CHECK_NAMES))
        for name, check in CHECKS:
            self.assertTrue(name and callable(check), f"malformed check entry: {name!r}")


class TheRailTest(unittest.TestCase):
    def test_a_clean_check_is_its_marker_and_its_name(self) -> None:
        """No value. "ok" ten times is ten lines of noise around the two lines
        that carry something."""

        rendered, report = _stream(_config())
        self.assertEqual(report.num_errors, 0)
        for name in CHECK_NAMES:
            self.assertIn(f"{RAIL} {DONE} {name}", rendered)

    def test_every_check_still_runs_after_one_fails(self) -> None:
        with TemporaryDirectory() as scratch:
            rendered, report = _stream(_failing_config(scratch))

        self.assertIn(f"{RAIL} {FAIL} data", rendered)
        for name in CHECK_NAMES:
            self.assertIn(name, rendered, f"the rail stopped before {name}")
        # The checks after the first failure are where the second failure was
        # found, which is the entire argument for running them.
        self.assertGreater(report.num_errors, 1)
        self.assertLess(rendered.index("data"), rendered.index("algorithm compatibility"))

    def test_a_failing_check_settles_with_a_count_and_lists_its_findings(self) -> None:
        with TemporaryDirectory() as scratch:
            rendered, _ = _stream(_failing_config(scratch))

        self.assertIn("1 error", rendered)
        self.assertIn("data.manifest_missing", rendered)
        # The hint lands under the finding, indented past it.
        self.assertIn("Run fedbrew generate", rendered)

    def test_a_warning_settles_amber_without_failing_the_rail(self) -> None:
        with TemporaryDirectory() as directory:
            (Path(directory) / "run.json").write_text("{}", encoding="utf-8")
            rendered, report = _stream(_config(output_dir=directory))

        self.assertIn(f"{RAIL} {WARN} experiment", rendered)
        self.assertIn("1 warning", rendered)
        self.assertEqual(report.num_errors, 0)

    def test_the_findings_are_indented_under_the_check_that_raised_them(self) -> None:
        with TemporaryDirectory() as scratch:
            rendered, _ = _stream(_failing_config(scratch))

        finding = next(line for line in rendered.splitlines() if "data.manifest_missing" in line)
        self.assertFalse(finding.startswith(RAIL), "a finding is not a rail stage")
        self.assertTrue(finding.startswith("  "), f"finding is not indented: {finding!r}")


class TheVerdictTest(unittest.TestCase):
    def _verdict(self, issues: list[ValidationIssue]) -> str:
        buffer = io.StringIO()
        print_validation_verdict(
            ValidationReport(config_path="x.yaml", issues=issues),
            build_surface(file=buffer),
        )
        return buffer.getvalue()

    def test_it_collects_every_error(self) -> None:
        rendered = self._verdict(
            [
                ValidationIssue("error", "data.manifest_missing", "no manifest"),
                ValidationIssue("error", "model.input_dim_invalid", "no input_dim"),
            ]
        )
        self.assertIn("FIX ERRORS BEFORE RUNNING", rendered)
        self.assertIn("data.manifest_missing", rendered)
        self.assertIn("model.input_dim_invalid", rendered)
        self.assertIn("2 errors", rendered)

    def test_it_does_not_repeat_the_warnings(self) -> None:
        """They are advisory, they are attached to the check that knows why,
        and repeating them here would bury the errors among them."""

        rendered = self._verdict(
            [
                ValidationIssue("error", "model.input_dim_invalid", "no input_dim"),
                ValidationIssue("warning", "experiment.output_dir_not_empty", "already has files"),
            ]
        )
        self.assertIn("model.input_dim_invalid", rendered)
        self.assertNotIn("experiment.output_dir_not_empty", rendered)
        self.assertIn("1 error  1 warning", rendered)

    def test_a_clean_config_is_ready_to_run(self) -> None:
        rendered = self._verdict([])
        self.assertIn("READY TO RUN", rendered)
        self.assertIn("0 errors  0 warnings", rendered)

    def test_warnings_alone_are_ready_but_reviewable(self) -> None:
        rendered = self._verdict(
            [ValidationIssue("warning", "experiment.output_dir_not_empty", "already has files")]
        )
        self.assertIn("READY", rendered)
        self.assertNotIn("FIX ERRORS", rendered)


class ThePlanIsTheRailsResultTest(unittest.TestCase):
    """`_run_preflight` end to end: rail, then plan, then verdict."""

    def _preflight(self, *argv: str) -> tuple[bool, str]:
        buffer = io.StringIO()
        args = runner.parse_args(["--validate-only", *argv])
        with unittest.mock.patch("sys.stdout", buffer):
            failed = runner._run_preflight(args)
        return failed, buffer.getvalue()

    def test_a_clean_preflight_prints_the_plan_and_succeeds(self) -> None:
        failed, rendered = self._preflight("--config", str(SMOKE_CONFIG))

        self.assertFalse(failed)
        self.assertIn("PREFLIGHT", rendered)
        self.assertIn("EXPERIMENT PLAN", rendered)
        self.assertIn("READY", rendered)
        self.assertLess(rendered.index("PREFLIGHT"), rendered.index("EXPERIMENT PLAN"))
        self.assertLess(rendered.index("EXPERIMENT PLAN"), rendered.index("READY"))

    def test_a_config_that_will_not_load_fails_at_the_first_stage(self) -> None:
        failed, rendered = self._preflight("--config", "configs/dev/no-such-file.yaml")

        self.assertTrue(failed)
        self.assertIn(f"{RAIL} {FAIL} load config", rendered)
        self.assertIn("FIX ERRORS BEFORE RUNNING", rendered)
        self.assertNotIn("EXPERIMENT PLAN", rendered)
        # The checks never ran, and the rail says so by not claiming them.
        self.assertNotIn("algorithm compatibility", rendered)

    def test_no_plan_is_claimed_when_a_check_fails(self) -> None:
        import yaml

        # Parsed and re-dumped rather than string-replaced: the point of the
        # test is a config that *loads* and then fails a check, and a
        # search-and-replace that silently matched nothing would leave a valid
        # config and a test that passes for the wrong reason.
        with TemporaryDirectory() as scratch:
            document = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
            document["data"]["name"] = "manifest_dataset"
            document["data"]["path"] = f"{scratch}/no-such-manifest.json"
            broken = Path(scratch) / "broken.yaml"
            broken.write_text(yaml.safe_dump(document), encoding="utf-8")
            failed, rendered = self._preflight("--config", str(broken))

        self.assertTrue(failed)
        self.assertNotIn("EXPERIMENT PLAN", rendered)
        self.assertIn("FIX ERRORS BEFORE RUNNING", rendered)

    def test_quiet_prints_nothing_at_all(self) -> None:
        failed, rendered = self._preflight("--config", str(SMOKE_CONFIG), "--quiet")

        self.assertFalse(failed)
        self.assertEqual(rendered, "")


if __name__ == "__main__":
    unittest.main()
