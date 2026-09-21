"""The three `fedbrew` lines in README.md's Quickstart section, actually run.

tests/test_readme_is_an_overview.py checks the README's shape and that its
install line agrees with the Install section. It cannot tell you the commands
work. This runs them -- the short-name form exactly as the README shows it,
never the `--config <path>` form the short names resolve to.

Marked ``quickstart`` and excluded from the default suite by addopts in
pyproject.toml, the same way tests/test_docs_quickstart_e2e.py is: it trains a
model. Not folded into that file -- this guards the README's own, separate
three-line quickstart and its own dataset, not chapter 03's four-command
walkthrough of the dev fixture, and it is intentionally outside the
chapter-guard registry in tests/test_docs_testing.py: it is not a chapter
guard, it is named ``test_readme_*`` rather than ``test_docs_*`` on purpose.

Nothing here touches the network. The synthetic generator builds its data from
a seeded linear teacher, which is the reason the README uses it here too.

This test runs into the config's own `output_dir`
(outputs/synthetic/fedavg/), not a scratch directory: that path is designed
to be short, stable and overwritable -- configs/synthetic/fedavg.yaml says so
in its own comment -- so exercising the literal README line here, with no
`--output-dir` override, is the point rather than a risk.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import pytest

from fedbrew.core.artifacts import _RUN_ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parent.parent


def _clear_run_artifacts(output_dir: Path) -> None:
    """Remove what a run writes into `output_dir`, and nothing else in it.

    The package deleted a refused attempt's artifacts through a function of
    its own until POST-F25 made that resume a refusal; this is the same walk
    over `_RUN_ARTIFACTS`, kept here for the one caller left.
    """

    for name in _RUN_ARTIFACTS:
        path = output_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


#: The README tells a reader to run `fedbrew <subcommand>`. That needs the
#: console script installed, which a checkout may not have, so the equivalent
#: module form is used here -- the same entry point, as chapter 03 says.
ENTRY = (sys.executable, "-m", "fedbrew.cli.dispatch")

GENERATOR_NAME = "synthetic"
EXPERIMENT_NAME = "synthetic/fedavg"
MANIFEST = "data/generated/synthetic/manifest.json"
EXPERIMENT_CONFIG = "configs/synthetic/fedavg.yaml"
OUTPUT_DIR = "outputs/synthetic/fedavg"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    """Run one quickstart step from the repository root."""

    completed = subprocess.run(
        [*ENTRY, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"`fedbrew {' '.join(args)}` exited {completed.returncode}\n"
            f"--- stdout ---\n{completed.stdout[-3000:]}\n"
            f"--- stderr ---\n{completed.stderr[-3000:]}"
        )
    return completed


@pytest.mark.quickstart
@unittest.skipUnless(
    (REPO_ROOT / "data" / "configs" / "synthetic.yaml").is_file(),
    "README quickstart generator config is absent",
)
@unittest.skipUnless(
    (REPO_ROOT / EXPERIMENT_CONFIG).is_file(),
    "README quickstart experiment config is absent",
)
class ReadmeQuickstartEndToEndTest(unittest.TestCase):
    """One test, because the three lines are a pipeline: each needs the last."""

    def test_the_three_lines_run_and_produce_the_documented_artifacts(self) -> None:
        # Line 1 (`pip install -e .`) is not run here: it is exercised by the
        # environment this suite already runs in, not by a subprocess.

        # Line 2 -- generate, by short name. Writes into data/generated/,
        # which is gitignored, and is idempotent, so re-running does not
        # accumulate.
        generated = _run("generate", GENERATOR_NAME)
        self.assertIn("Clients", generated.stdout)
        self.assertTrue((REPO_ROOT / MANIFEST).is_file(), "no manifest written")

        # Line 3 -- run, by short name. No --rounds override: the config's own
        # defaults.global_rounds is what the README line actually runs.
        #
        # The run's own artifacts are cleared first, through the function the
        # package uses for exactly that, so what is asserted below is what this
        # invocation wrote. The output path is fixed and reused by design, and
        # a `client_metrics.csv` left by some earlier run under a different
        # configuration would otherwise read as one this run had produced. It
        # also makes the test start where a README reader starts: an empty
        # output directory.
        output_dir = REPO_ROOT / OUTPUT_DIR
        if output_dir.is_dir():
            _clear_run_artifacts(output_dir)
        trained = _run("run", EXPERIMENT_NAME)
        self.assertIn("EXPERIMENT COMPLETE", trained.stdout)
        self._assert_artifacts(output_dir)

        # The README's fourth and fifth lines: `ls` the fixed output path and
        # summarise it. Not asserted as a subprocess `ls`, since the point is
        # the path and the report, not the shell builtin.
        #
        # Removed first, so that finding it afterwards is evidence this
        # invocation wrote it. The output path is fixed and reused, so a
        # report.md left by an earlier invocation would satisfy the assertion
        # below without `report` having produced anything.
        report = output_dir / "report.md"
        report.unlink(missing_ok=True)
        reported = _run("report", "--run-dir", OUTPUT_DIR)
        self.assertIn("report.md", reported.stdout)
        self.assertTrue(report.is_file())

    def _assert_artifacts(self, output_dir: Path) -> None:
        """Which of the run's own artifacts a default configuration produces.

        Scoped to `_RUN_ARTIFACTS` -- what the package counts as belonging to
        a run -- rather than to everything in the directory. The
        directory is a fixed, reused path by design (see the module docstring),
        so it holds things this run did not write: `report.md`, written by the
        `report` step at the end of this very test, which made the assertion
        fail on every invocation after the first. `report.md` is deliberately
        outside `_RUN_ARTIFACTS`, so the intersection is the honest subject,
        and the caller
        clears the other half of it before running.

        The claim is unchanged and still exact within that subject: a default
        run writes these three of the five, and neither per-client CSV.
        """

        written = {path.name for path in output_dir.iterdir()}
        self.assertEqual(
            written & set(_RUN_ARTIFACTS),
            {"round_metrics.csv", "run.json", "checkpoints"},
            "a default run writes two metric files and a checkpoint directory",
        )

        run = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(run["status"], "completed")
        self.assertIsNone(run["termination"])
        self.assertTrue(run["results"]["final_metrics"])
        # json_safe turns a non-finite metric into null, so a None here means
        # the run diverged rather than that the key is missing.
        self.assertNotIn(
            None,
            run["results"]["final_metrics"].values(),
            "a metric came back non-finite; the quickstart should not diverge",
        )


if __name__ == "__main__":
    unittest.main()
