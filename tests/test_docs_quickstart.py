"""docs/03-quickstart.md must name real commands, configs and artifacts.

The quickstart is the first thing a new reader runs, so its errors are the
most expensive: someone who cannot get past step 1 has no way to tell whether
the problem is the tool, their machine, or the instructions.

This is the cheap half of the check -- that every command, flag, config path
and artifact name in the chapter corresponds to something real. It runs in the
default suite. The expensive half, which actually executes the four steps, is
tests/test_docs_quickstart_e2e.py, marked `quickstart` and excluded from the
default suite because it trains.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import pytest

from fedbrew.cli.dispatch import COMMANDS

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER = REPO_ROOT / "docs" / "03-quickstart.md"

#: The four steps the chapter is built around, in order.
EXPECTED_STEPS = (
    "generate",
    "inspect-data",
    "run",
)


def _chapter_text() -> str:
    return CHAPTER.read_text(encoding="utf-8")


def _flowed(text: str) -> str:
    """Collapse whitespace, so a wrapped sentence still matches as one."""

    return " ".join(text.split())


class ChapterShapeTest(unittest.TestCase):
    def test_present_and_agent_facing(self) -> None:
        self.assertTrue(CHAPTER.is_file())
        self.assertIn("\n## For agents\n", _chapter_text())


class CommandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _chapter_text()

    def test_the_four_steps_use_real_subcommands(self) -> None:
        used = set(re.findall(r"\bfedbrew ([a-z][a-z0-9-]*)", self.text))
        self.assertTrue(used, "the chapter runs no commands")
        unknown = sorted(used - set(COMMANDS))
        self.assertEqual(unknown, [], f"not real subcommands: {unknown}")
        for step in EXPECTED_STEPS:
            with self.subTest(step=step):
                self.assertIn(step, used)

    def test_the_module_form_is_the_dispatch_entry_point(self) -> None:
        """The chapter offers `python -m ...` for an uninstalled checkout."""

        self.assertIn("python -m fedbrew.cli.dispatch", self.text)
        module = REPO_ROOT / "fedbrew" / "cli" / "dispatch.py"
        self.assertTrue(module.is_file())
        self.assertIn(
            'if __name__ == "__main__":',
            module.read_text(encoding="utf-8"),
            "dispatch.py must be runnable with -m for the chapter's fallback",
        )


class NamedPathTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _chapter_text()

    def test_every_named_config_exists(self) -> None:
        cited = set(re.findall(r"((?:data/)?configs/[\w/]+\.yaml)", self.text))
        self.assertGreaterEqual(len(cited), 3)
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
        self.assertEqual(missing, [])

    def test_cited_modules_and_tests_exist(self) -> None:
        for pattern, label in (
            (r"`(fedbrew/[\w/]+\.py)", "module"),
            (r"`(tests/test_\w+\.py)`", "test"),
        ):
            cited = set(re.findall(pattern, self.text))
            with self.subTest(kind=label):
                self.assertTrue(cited)
                missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
                self.assertEqual(missing, [])

    def test_the_generated_and_output_paths_are_gitignored(self) -> None:
        """The chapter writes into the checkout; those paths must not be tracked."""

        ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        for path in ("data/generated/", "outputs/"):
            with self.subTest(path=path):
                self.assertIn(path, ignored)
                self.assertIn(path.rstrip("/"), self.text)


def _expected_round_metrics_columns() -> tuple[int, int]:
    """(total, metric) column counts a default run's round_metrics.csv has.

    3 identity + per-split metrics * 3 splits + central_test + fit + timing.
    The one place this arithmetic lives: this file's static check and
    test_docs_quickstart_e2e.py's trained-and-measured check both call it,
    so they cannot independently drift from each other the way the e2e
    guard's hardcoded ``49`` just did after {split}_num_clients columns
    were added.
    """

    from fedbrew.core.artifacts import _ROUND_TIMING_FIELDS
    from fedbrew.core.config import ClientStatisticsConfig, client_metric_names

    per_split = len(client_metric_names("test", ClientStatisticsConfig()))
    metric_columns = per_split * 3 + 2 + 2  # 3 splits, central_test, fit
    identity = 3
    total = identity + metric_columns + len(_ROUND_TIMING_FIELDS)
    return total, metric_columns


class ArtifactClaimTest(unittest.TestCase):
    """The chapter says a default run writes two metric files. Check it."""

    def test_a_default_run_writes_two_metric_artifacts(self) -> None:
        import yaml

        from fedbrew.core.runner import _artifact_file_names

        config_path = REPO_ROOT / "configs" / "dev" / "synthetic_label_skew.yaml"
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        statistics = loaded.get("client_statistics") or {}
        self.assertFalse(
            statistics.get("per_client_csv", False),
            "the quickstart config now writes per-client CSVs; the chapter says it does not",
        )

        from test_docs_metric_names import _minimal_config

        config = _minimal_config()
        config.client_statistics.per_client_csv = False
        self.assertEqual(
            _artifact_file_names(config),
            ["round_metrics.csv", "run.json"],
        )
        text = _chapter_text()
        self.assertIn("**Two metric files, not four.**", text)

    def test_the_stated_column_arithmetic_is_self_consistent(self) -> None:
        """52 columns = 3 identity + 43 metric + 6 timing."""

        from fedbrew.core.artifacts import _ROUND_TIMING_FIELDS

        text = _chapter_text()
        total, metric_columns = _expected_round_metrics_columns()

        self.assertEqual(metric_columns, 43)
        self.assertEqual(total, 52)
        self.assertIn(f"{total} columns", text)
        self.assertIn(f"{metric_columns} metric columns", text)
        self.assertIn(f"{len(_ROUND_TIMING_FIELDS)} timing columns", text)


#: Enough to spell any count the plan block's prose gives in words.
_NUMBER_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
).split()


class PlanColumnCountTest(unittest.TestCase):
    """The plan block's column counts are the ones the plan computes.

    Section 3's column arithmetic had a guard and was kept current. The plan
    block's ``Columns`` line had none: it went on showing 45 after the command
    had started printing 43, beside a "twelve more" that never added up to the
    fifteen it elided from. Every number here comes from the rows the plan
    header prints, built from the config the chapter runs.
    """

    def setUp(self) -> None:
        from fedbrew.core.config import load_config
        from fedbrew.core.logging import _metrics_rows

        config = load_config(str(REPO_ROOT / "configs" / "dev" / "synthetic_label_skew.yaml"))
        rows = {row.label: row.value for row in _metrics_rows(config, verbose=False)}
        self.columns = rows["Columns"]
        counts = re.fullmatch(r"(\d+) of (\d+) listed; --verbose lists them all", self.columns)
        self.assertIsNotNone(counts, f"the plan's Columns row changed shape: {self.columns!r}")
        self.listed, self.total = (int(group) for group in counts.groups())
        self.text = _chapter_text()

    def test_the_plan_block_shows_the_printed_row(self) -> None:
        self.assertRegex(self.text, rf"(?m)^Columns +{re.escape(self.columns)}$")

    def test_the_prose_gives_the_same_counts(self) -> None:
        self.assertIn(
            f"lists all {self.total} instead of the {_NUMBER_WORDS[self.listed]} shown",
            _flowed(self.text),
        )

    def test_the_elided_columns_add_up_to_the_listed_count(self) -> None:
        """The block glosses a few columns and says how many more it left out."""

        block = re.search(
            r"(?m)^Columns .*\n((?:.*\n)*?)\.\.\. +\((\w+) more columns, each glossed\)$",
            self.text,
        )
        self.assertIsNotNone(block, "the plan block no longer elides its column list")
        shown = sum(1 for line in block.group(1).splitlines() if not line.startswith(" "))
        self.assertEqual(block.group(2), _NUMBER_WORDS[self.listed - shown])


class ForeignSeedClaimTest(unittest.TestCase):
    """The chapter's last failure mode names a real guard."""

    def test_a_foreign_seed_is_refused(self) -> None:
        from fedbrew.core.runner import _refuse_a_foreign_seed

        self.assertTrue(callable(_refuse_a_foreign_seed))
        self.assertIn(
            "refuses a config whose seed differs",
            _flowed(_chapter_text()),
            "the chapter must warn that a second seed into one output_dir is "
            "refused; _refuse_a_foreign_seed is what does it",
        )


if __name__ == "__main__":
    unittest.main()
