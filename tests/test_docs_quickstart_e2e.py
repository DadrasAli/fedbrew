"""The four commands in docs/03-quickstart.md, actually run.

tests/test_docs_quickstart.py checks that the chapter names real commands,
configs and artifacts. It cannot tell you they work. This runs them.

Marked ``quickstart`` and excluded from the default suite by addopts in
pyproject.toml: it trains a model and costs about a minute, against a default
suite of two. CI runs it on every push, which is where a quickstart that
stopped working should be caught -- the reader who finds it otherwise is the
one least able to tell whether the fault is theirs.

Nothing here touches the network. The synthetic generator builds its data from
a seeded linear teacher, which is the reason the chapter uses it.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The chapter tells a reader to run `fedbrew <subcommand>`. That needs the
#: console script installed, which a checkout may not have, so the equivalent
#: module form is used here -- the same entry point, as the chapter says.
ENTRY = (sys.executable, "-m", "fedbrew.cli.dispatch")

GENERATOR_CONFIG = "data/configs/synthetic_label_skew.yaml"
MANIFEST = "data/generated/synthetic_label_skew/manifest.json"
EXPERIMENT_CONFIG = "configs/dev/synthetic_label_skew.yaml"


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
    (REPO_ROOT / GENERATOR_CONFIG).is_file(),
    "quickstart generator config is absent",
)
class QuickstartEndToEndTest(unittest.TestCase):
    """One test, because the four steps are a pipeline: each needs the last."""

    def test_the_four_steps_run_and_produce_the_documented_artifacts(self) -> None:
        # Step 1 -- generate. Writes into data/generated/, which is gitignored,
        # and is idempotent, so re-running does not accumulate.
        generated = _run("generate", "--config", GENERATOR_CONFIG)
        self.assertIn("Clients", generated.stdout)
        self.assertTrue((REPO_ROOT / MANIFEST).is_file(), "no manifest written")

        # Step 2 -- inspect. The chapter shows this reporting a valid dataset.
        inspected = _run("inspect-data", MANIFEST)
        self.assertIn("VALID DATASET", inspected.stdout)

        # Step 3 -- validate without training. The verdict line depends on
        # repository state: a clean checkout reports VALID EXPERIMENT / READY
        # TO RUN, while one whose outputs/ already holds this run warns about
        # the non-empty directory and reports READY -- REVIEW WARNINGS. Both
        # are a pass; what must hold is zero errors and a READY verdict.
        validated = _run("run", "--config", EXPERIMENT_CONFIG, "--validate-only")
        self.assertIn("0 errors", validated.stdout)
        self.assertIn("READY", validated.stdout)
        self.assertIn("PREFLIGHT", validated.stdout)

        # The plan header a passing preflight settles into carries the column
        # count the chapter's plan block shows. Checked against what a fresh
        # run prints, since that is the number a reader compares.
        printed = re.search(r"(?m)^Columns +(\d+ of \d+ listed)", validated.stdout)
        self.assertIsNotNone(printed, "the plan printed no Columns row")
        chapter = (REPO_ROOT / "docs" / "03-quickstart.md").read_text(encoding="utf-8")
        self.assertRegex(chapter, rf"(?m)^Columns +{re.escape(printed.group(1))};")

        # Step 4 -- train. Into a scratch directory rather than the config's
        # own output_dir, so running the suite never overwrites a real run.
        # One level down: a run appends to runs_index.jsonl in its output
        # directory's parent, which for the scratch directory itself was the
        # system temp directory, shared by every checkout and every user.
        with tempfile.TemporaryDirectory() as scratch:
            output_dir = Path(scratch) / "run"
            _run(
                "run",
                "--config",
                EXPERIMENT_CONFIG,
                "--rounds",
                "2",
                "--output-dir",
                str(output_dir),
            )
            self._assert_artifacts(output_dir)

    def _assert_artifacts(self, output_dir: Path) -> None:
        """Exactly what docs/03 section 3 says a default run leaves behind."""

        from fedbrew.core.artifacts import _ROUND_TIMING_FIELDS

        written = {path.name for path in output_dir.iterdir()}
        self.assertEqual(
            written,
            {"round_metrics.csv", "run.json", "checkpoints"},
            "a default run writes two metric files and a checkpoint directory",
        )

        header = (
            (output_dir / "round_metrics.csv")
            .read_text(encoding="utf-8")
            .splitlines()[0]
            .split(",")
        )
        self.assertEqual(header[:3], ["round_id", "num_clients", "num_examples"])
        self.assertEqual(
            header[-len(_ROUND_TIMING_FIELDS) :],
            list(_ROUND_TIMING_FIELDS),
            "timing columns are appended after the metric columns",
        )

        from test_docs_quickstart import _expected_round_metrics_columns

        expected_total, _ = _expected_round_metrics_columns()
        self.assertEqual(
            len(header),
            expected_total,
            "drifted from the count tests/test_docs_quickstart.py derives from "
            "source and checks against the chapter's own prose",
        )

        rows = (output_dir / "round_metrics.csv").read_text(encoding="utf-8")
        self.assertEqual(len(rows.strip().splitlines()), 3, "header plus two rounds")

        run = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(run["status"], "completed")
        self.assertIsNone(run["termination"])
        self.assertEqual(run["num_rounds"], 2)
        self.assertTrue(run["results"]["final_metrics"])
        # json_safe turns a non-finite metric into null, so a None here means
        # the run diverged rather than that the key is missing.
        self.assertNotIn(
            None,
            run["results"]["final_metrics"].values(),
            "a metric came back non-finite; the quickstart should not diverge",
        )

        checkpoints = run["artifacts"]["checkpoints"]
        self.assertEqual(checkpoints["latest_checkpoint"], "checkpoints/latest.pt")
        self.assertEqual(checkpoints["best_checkpoint"], "checkpoints/best.pt")
        for name in ("latest.pt", "best.pt"):
            self.assertTrue((output_dir / "checkpoints" / name).is_file())

        seeding = run["reproducibility"]["seeding"]
        self.assertIn("seed", seeding)
        self.assertIn("git_commit", run["reproducibility"]["code_state"])


if __name__ == "__main__":
    unittest.main()
