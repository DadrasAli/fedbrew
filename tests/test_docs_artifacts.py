"""docs/09-artifacts.md must match what the writers actually produce.

The artifact layer is where a documentation error is discovered latest. A
reader believes the schema described here, writes an analysis script against
it, and finds out weeks later that the column they keyed on is named something
else -- or that the file they expected was never written because a switch they
did not know about is off by default.

So the file lists and the fixed schemas are diffed against the writers, and the
run.json key set is diffed against a run.json this test produces, rather than
against a list kept here.
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

import pytest
from docs_sections import fenced_block_after, table_after

from fedbrew.core.artifacts import (
    _CLIENT_EVALUATION_FIELDS,
    _ROUND_TIMING_FIELDS,
    DEFAULT_ARTIFACT_FILES,
)

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER = REPO_ROOT / "docs" / "09-artifacts.md"


def _chapter_text() -> str:
    return CHAPTER.read_text(encoding="utf-8")


def _flowed() -> str:
    """The chapter with whitespace collapsed, so a wrapped sentence matches."""

    return " ".join(_chapter_text().split())


def _says(fragment: str) -> bool:
    return fragment in _flowed()


#: A chapter may spell a small count instead of writing a digit, so a check on
#: one has to accept both forms -- but only for the *true* value. The previous
#: form of the count checks below normalised the other way,
#: `text.replace("Nineteen", str(len(run_json)))`, which rewrote whichever word
#: the chapter happened to use into the number under test. That cannot fail: the
#: chapter said "Nineteen" while run.json carried twenty keys, the substitution
#: produced "20 top-level keys", and the assertion passed.
_NUMBER_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty "
    "twenty-one twenty-two twenty-three twenty-four twenty-five"
).split()


def _states_count(text: str, value: int, noun: str) -> bool:
    """True when `text` states `value` immediately before `noun`."""

    forms = {str(value)}
    if value < len(_NUMBER_WORDS):
        word = _NUMBER_WORDS[value]
        forms |= {word, word.capitalize()}
    return any(f"{form} {noun}" in text for form in forms)


class ChapterShapeTest(unittest.TestCase):
    def test_present_and_agent_facing(self) -> None:
        self.assertTrue(CHAPTER.is_file())
        self.assertIn("\n## For agents\n", _chapter_text())


class FileListTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _chapter_text()

    def test_every_default_artifact_file_is_documented(self) -> None:
        # Scoped to section 1's tree. Unscoped this could not fail: every one
        # of these names appears in the prose, in section 7 and in the
        # `For agents` tables, so deleting round_metrics.csv from the listing
        # a reader actually reads left the guard green.
        listing = fenced_block_after(self.text, "## 1. The files")
        self.assertTrue(DEFAULT_ARTIFACT_FILES, "no artifact files declared; the import is broken")
        for name in DEFAULT_ARTIFACT_FILES:
            with self.subTest(artifact=name):
                self.assertIn(name, listing)

    def test_the_gating_matches_the_runner(self) -> None:
        """Section 1's claim about which files a default run writes."""

        from test_docs_metric_names import _minimal_config

        from fedbrew.core.runner import _artifact_file_names

        config = _minimal_config()
        config.client_statistics.per_client_csv = False
        default = _artifact_file_names(config)
        self.assertEqual(default, ["round_metrics.csv", "run.json"])
        self.assertTrue(_says("A default run writes **two**"))

        config.client_statistics.per_client_csv = True
        self.assertEqual(len(_artifact_file_names(config)), len(DEFAULT_ARTIFACT_FILES))


class FixedSchemaTest(unittest.TestCase):
    def test_the_two_fixed_schemas_are_pointed_at(self) -> None:
        """Chapter 08 owns the columns; 09 must point there, not restate them."""

        self.assertTrue(_says("Chapter 08 documents every column"))
        # The line references must still resolve to the definitions.
        source = (REPO_ROOT / "fedbrew" / "core" / "artifacts.py").read_text(encoding="utf-8")
        self.assertIn("_CLIENT_EVALUATION_FIELDS = [", source)
        self.assertIn("_ROUND_TIMING_FIELDS = {", source)
        self.assertEqual(len(_CLIENT_EVALUATION_FIELDS), 13)
        self.assertEqual(len(_ROUND_TIMING_FIELDS), 6)


class RunJsonSchemaTest(unittest.TestCase):
    """The key table must be the keys save_run_json actually writes."""

    @classmethod
    def setUpClass(cls) -> None:
        from fedbrew.core.artifacts import save_run_json
        from fedbrew.core.state import MetricRecord, RoundTimings

        cls._scratch = tempfile.TemporaryDirectory()
        history = [
            MetricRecord(
                round_id=1,
                metrics={"fit_loss": 0.5, "fit_accuracy": 0.25},
                num_clients=2,
                num_examples=20,
                timings=RoundTimings(
                    fit=0.1,
                    aggregate=0.01,
                    client_eval=0.02,
                    global_eval=0.0,
                    checkpoint=0.0,
                    total=0.2,
                ),
            )
        ]
        from test_docs_metric_names import _minimal_config

        path = save_run_json(
            history,
            cls._scratch.name,
            _minimal_config(),
            run_metadata={"run_id": "docs-test", "status": "completed"},
        )
        # Not `cls.run`: TestCase.run is the method unittest calls to
        # execute the test, and shadowing it makes every test in the
        # class fail with "'dict' object is not callable".
        cls.run_json = json.loads(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._scratch.cleanup()

    def _key_table(self) -> str:
        """Section 3's table only.

        Scoped, not a whole-file search: every key name also appears in the
        invariants and the tests table, so `attempts` deleted from the table
        would still be "found" somewhere in the chapter and the check would
        pass while the table was wrong.

        This file found that trap first and fixed it here only, and the shape
        then spread to test_every_default_artifact_file_is_documented above and
        to four guards in other chapters. The scoping now comes from
        tests/docs_sections.py so there is one of it.
        """

        return table_after(_chapter_text(), "| Key | Content |")

    def test_every_top_level_key_has_a_row(self) -> None:
        table = self._key_table()
        missing = sorted(key for key in self.run_json if f"`{key}`" not in table)
        self.assertEqual(
            missing,
            [],
            "run.json carries top-level keys section 3's table omits",
        )

    def test_the_stated_key_count_matches(self) -> None:
        self.assertTrue(
            _states_count(_chapter_text(), len(self.run_json), "top-level keys"),
            f"the chapter must say {len(self.run_json)} top-level keys, which is "
            "what save_run_json writes",
        )

    def test_results_holds_the_final_round(self) -> None:
        self.assertIn("final_metrics", self.run_json["results"])
        self.assertTrue(_says("The **last** round's complete metrics"))

    def test_sort_keys_is_off(self) -> None:
        """The chapter says the section order is the organisation."""

        source = (REPO_ROOT / "fedbrew" / "core" / "artifacts.py").read_text(encoding="utf-8")
        self.assertIn("sort_keys is off on purpose", source)
        self.assertTrue(_says("`sort_keys` is off"))


class AtomicityClaimTest(unittest.TestCase):
    """Section 2's two write strategies, and why they differ."""

    def setUp(self) -> None:
        self.source = (REPO_ROOT / "fedbrew" / "core" / "artifacts.py").read_text(encoding="utf-8")

    def test_round_metrics_is_replaced_atomically(self) -> None:
        self.assertIn("os.replace(temp_path, path)", self.source)
        self.assertTrue(_says("rewritten in full every round, atomically"))

    def test_the_per_client_files_are_appended(self) -> None:
        self.assertIn("def _append_csv_rows", self.source)
        self.assertTrue(_says("appended"))
        self.assertTrue(_says("not atomic"))

    def test_the_foreign_seed_refusal_exists(self) -> None:
        runner = (REPO_ROOT / "fedbrew" / "core" / "runner.py").read_text(encoding="utf-8")
        self.assertIn("def _refuse_a_foreign_seed", runner)
        self.assertTrue(_says("A run refuses a foreign seed"))


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
