"""docs/11-performance-and-cost.md must match the keys and tools it describes.

A performance chapter carries two kinds of claim, and only one is checkable.

The settings are data: which keys exist, what gates them, what they default
to. Those are diffed against the config allow-lists, because a chapter that
tells someone to set `persistent_workers` without saying it is unreachable at
`num_workers: 0` has cost them an afternoon.

The measured numbers are not. They are labelled and dated in the chapter, and
what is checked here is that the ones taken from code comments still match
those comments, and that the tools said to reproduce the rest still exist.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import pytest
from docs_sections import section_of, table_after

from fedbrew.core.config import _KNOWN_EXTRA_KEYS

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER = REPO_ROOT / "docs" / "11-performance-and-cost.md"


def _chapter_text() -> str:
    return CHAPTER.read_text(encoding="utf-8")


def _says(fragment: str) -> bool:
    return fragment in " ".join(_chapter_text().split())


class ChapterShapeTest(unittest.TestCase):
    def test_present_and_agent_facing(self) -> None:
        self.assertTrue(CHAPTER.is_file())
        self.assertIn("\n## For agents\n", _chapter_text())


class RoundTimingTableTest(unittest.TestCase):
    """Section 1's table against the fields the runner actually writes.

    Added because nothing checked it in this direction: deleting `fit_sec`'s
    row went unnoticed, with the name gone from the chapter entirely. Section 1
    is where a reader looks to find out where a round's time goes, so a field
    the code records and the table omits is a gap exactly when someone is
    trying to account for a slow round.
    """

    def test_every_timing_field_has_a_row(self) -> None:
        from fedbrew.core.artifacts import _ROUND_TIMING_FIELDS

        table = table_after(_chapter_text(), "## 1. Where a round's time goes")
        self.assertTrue(_ROUND_TIMING_FIELDS, "no timing fields declared; the import is broken")
        for field in _ROUND_TIMING_FIELDS:
            with self.subTest(field=field):
                self.assertIn(f"`{field}`", table)

    def test_the_table_invents_no_field(self) -> None:
        import re

        from fedbrew.core.artifacts import _ROUND_TIMING_FIELDS

        table = table_after(_chapter_text(), "## 1. Where a round's time goes")
        listed = set(re.findall(r"^\| `(\w+_sec)` \|", table, flags=re.MULTILINE))
        self.assertTrue(listed, "no timing rows parsed; the scope or the pattern is wrong")
        self.assertEqual(
            listed - set(_ROUND_TIMING_FIELDS),
            set(),
            "section 1 names timing fields the runner does not write",
        )


class TunableKeyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _chapter_text()

    def test_every_dataloader_key_is_documented_with_its_gate(self) -> None:
        # Scoped to 4.3's table. Unscoped this could not fail: deleting the
        # whole pin_memory row left the guard green, because the name also
        # appears in 4.6 and in the For-agents tables. The test's own name
        # promises the gate is documented, and only the row carries that.
        table = table_after(self.text, "### 4.3 `runtime.performance.dataloader`")
        keys = _KNOWN_EXTRA_KEYS["runtime.performance.dataloader"]
        self.assertTrue(keys, "no dataloader keys declared; the import is broken")
        for key in keys:
            with self.subTest(key=key):
                self.assertIn(f"`{key}`", table)
        # The two that do nothing at num_workers 0 must say so; that is the
        # document-only finding this section exists to carry.
        self.assertTrue(_says("only read when `num_workers > 0`"))
        self.assertTrue(_says("unreachable\nas shipped".replace("\n", " ")))

    def test_every_throughput_key_is_documented_here(self) -> None:
        """Against the code's classification, not a copy of it.

        This file used to keep its own NUMERICS_KEYS literal and chapter 10
        kept a third one in prose, so the three could drift apart and the only
        thing that noticed would be a reader. The split now lives beside the
        keys it classifies.
        """

        from fedbrew.core.config import THROUGHPUT_ONLY_PERFORMANCE_KEYS

        # The dataloader keys have their own section and their own guard; this
        # is about the settings directly under runtime.performance.
        keys = THROUGHPUT_ONLY_PERFORMANCE_KEYS & set(_KNOWN_EXTRA_KEYS["runtime.performance"])
        self.assertTrue(keys, "no throughput keys classified; the import is broken")
        for key in keys:
            with self.subTest(key=key):
                self.assertIn(f"`{key}`", self.text)

    def test_the_numerics_keys_are_documented_in_chapter_ten(self) -> None:
        from fedbrew.core.config import NUMERICS_PERFORMANCE_KEYS

        chapter_ten = (REPO_ROOT / "docs" / "10-reproducibility.md").read_text(encoding="utf-8")
        self.assertTrue(NUMERICS_PERFORMANCE_KEYS, "no numerics keys classified")
        for key in NUMERICS_PERFORMANCE_KEYS:
            with self.subTest(key=key):
                self.assertIn(f"`{key}`", chapter_ten)

    def test_this_chapter_never_names_a_numerics_key(self) -> None:
        """The direction that keeps the two chapters from contradicting.

        Chapter 10's table omitted torch_num_threads while section 4.4 here
        documented it as a throughput setting -- an inconsistency between two
        chapters that nothing could catch, because neither was checked against
        the code. This is the other half: a key that changes the numbers must
        not turn up in this chapter's cost-tuning sections at all, because a
        reader tuning throughput would take it as safe to change.
        """

        from fedbrew.core.config import NUMERICS_PERFORMANCE_KEYS

        self.assertTrue(NUMERICS_PERFORMANCE_KEYS, "no numerics keys classified")
        section = section_of(self.text, "## 4. The settings that change cost")
        for key in NUMERICS_PERFORMANCE_KEYS:
            with self.subTest(key=key):
                self.assertNotIn(
                    key,
                    section,
                    f"{key} changes the numbers, so it is chapter 10's; naming it "
                    "among the settings that change cost invites a reader to "
                    "treat it as free",
                )

    def test_every_staging_key_is_documented(self) -> None:
        table = table_after(self.text, "## 5. Data staging")
        keys = _KNOWN_EXTRA_KEYS["runtime.data_staging"]
        self.assertTrue(keys, "no staging keys declared; the import is broken")
        for key in keys:
            with self.subTest(key=key):
                self.assertIn(f"`{key}`", table)
        self.assertTrue(_says("the only way to point staging at a specific"))

    def test_the_client_scopes_match_the_parser(self) -> None:
        from fedbrew.core.config import parse_evaluation_client_scope

        table = table_after(self.text, "### 4.1 `evaluation.*.clients` — the largest lever")
        for scope in ("all", "participating"):
            with self.subTest(scope=scope):
                mode, size = parse_evaluation_client_scope(scope)
                self.assertEqual((mode, size), (scope, None))
                self.assertIn(f"`{scope}`", table)
        for scope in ("sample", "resample"):
            with self.subTest(scope=scope):
                mode, size = parse_evaluation_client_scope(f"{scope}:5")
                self.assertEqual((mode, size), (scope, 5))
                self.assertIn(f"`{scope}:<N>`", table)


class MeasuredNumberTest(unittest.TestCase):
    """Numbers taken from code comments must still match those comments."""

    def test_the_per_client_csv_volume_matches_the_source(self) -> None:
        source = (REPO_ROOT / "fedbrew" / "core" / "config.py").read_text(encoding="utf-8")
        self.assertIn("1.8M rows", source)
        self.assertTrue(_says("1.8M rows over 500 rounds"))

    def test_the_old_write_cost_is_explained_from_the_source(self) -> None:
        """The old write volume took the rewriting code to measure, and it is gone.

        What stays checkable is the mechanism: the source records why the
        rewrite ran every round and what that did to the bytes written, and the
        chapter says the same.
        """

        source = " ".join(
            (REPO_ROOT / "fedbrew" / "core" / "artifacts.py").read_text(encoding="utf-8").split()
        )
        self.assertIn("save_last makes that true every round", source)
        self.assertIn("grow with the square of the round count", source)
        self.assertTrue(_says("which `save_last` makes every round"))
        self.assertTrue(_says("grew with the square of the round count"))

    def test_measured_numbers_are_labelled_as_such(self) -> None:
        self.assertTrue(_says("Measured numbers in this chapter are labelled"))
        self.assertTrue(_says("not guarded by tests"))


class ToolsTest(unittest.TestCase):
    """Every script the chapter offers for remeasuring must exist."""

    def test_every_named_tool_exists(self) -> None:
        named = set(re.findall(r"`(tools/[\w.]+\.py)`", _chapter_text()))
        self.assertGreaterEqual(len(named), 5)
        missing = sorted(path for path in named if not (REPO_ROOT / path).is_file())
        self.assertEqual(missing, [])

    def test_every_tool_is_named(self) -> None:
        """The other direction: a script nobody is told about."""

        present = {f"tools/{path.name}" for path in (REPO_ROOT / "tools").glob("*.py")}
        named = set(re.findall(r"`(tools/[\w.]+\.py)`", _chapter_text()))
        self.assertEqual(
            sorted(present - named),
            [],
            "tools/ holds scripts this chapter does not mention",
        )


class CitedPathsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _chapter_text()

    def test_cited_modules_exist(self) -> None:
        cited = set(re.findall(r"`(fedbrew/[\w/]+\.py)", self.text))
        self.assertGreaterEqual(len(cited), 6)
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
        self.assertEqual(missing, [])

    def test_cited_tests_exist(self) -> None:
        cited = set(re.findall(r"`(tests/test_\w+\.py)`", self.text))
        self.assertGreaterEqual(len(cited), 8)
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
