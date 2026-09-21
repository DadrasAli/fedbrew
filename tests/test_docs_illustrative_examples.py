"""docs/15-illustrative-examples.md against `examples/`, in both directions.

The chapter makes four kinds of claim, and each is checked against the thing
that would make it false rather than against itself:

- **§3's table is the example set.** An example the chapter omits and a chapter
  row naming no example both fail. The verdict column -- whether any shipped
  algorithm solves the problem -- is diffed against `examples/README.md`, which
  is the index and the authority.
- **§4's rules are properties of the tree.** The one rule that names a
  mechanism, "assert the ground truth in `problem.py`", is checked by parsing
  every `problem.py` for a `_self_check` definition *and* a module-level call
  to it. A rule stating a discipline nothing in the tree follows is worse than
  no rule.
- **§6 is a claim about what is missing.** Both unbuilt items are asserted
  absent, so building either fails here until the chapter is rewritten. This is
  the direction a "what is not built" section normally rots in: the feature
  lands and the section stays.
- **Every path the chapter names exists.**

Scoped through `docs_sections`, for the reason that module documents: a check
that searches the whole chapter for a table's contents cannot fail, because
every name in a table also appears in the prose and in `## For agents`.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

import pytest
from docs_sections import section_of, table_after

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER = REPO_ROOT / "docs" / "15-illustrative-examples.md"
EXAMPLES = REPO_ROOT / "examples"
INDEX = EXAMPLES / "README.md"

#: The anchor above each of the two tables this guard reads.
CHAPTER_TABLE_ANCHOR = "it and guarded against it in both directions."
INDEX_TABLE_ANCHOR = "## The examples"

#: Files every example ships. `run.py` is a convenience over the two CLI
#: commands; the other three are the example.
REQUIRED_FILES = ("problem.py", "run.py", "README.md")

#: The section an example that no shipped arm can solve must carry.
UNSOLVABLE_SECTION = "## Which shipped algorithms can solve it"


def _chapter() -> str:
    return CHAPTER.read_text(encoding="utf-8")


def _example_directories() -> set[str]:
    return {
        path.name
        for path in EXAMPLES.iterdir()
        if path.is_dir() and not path.name.startswith((".", "__"))
    }


def _rows(table: str) -> list[list[str]]:
    rows = []
    for line in table.splitlines():
        if not line.startswith("|") or set(line) <= set("| -:"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0].lower() in {"example", ""}:
            continue
        rows.append(cells)
    return rows


def _named_example(cell: str) -> str:
    """The example name out of a table cell, whether or not it is a link."""

    linked = re.search(r"\[`([^`]+)`\]", cell)
    if linked:
        return linked.group(1)
    bare = re.search(r"`([^`]+)`", cell)
    assert bare is not None, f"no example name in {cell!r}"
    return bare.group(1)


def _solvable(verdict: str) -> bool:
    """False when the verdict says no shipped algorithm solves the problem."""

    return not verdict.lstrip().startswith("**None")


def _chapter_rows() -> dict[str, list[str]]:
    table = table_after(_chapter(), CHAPTER_TABLE_ANCHOR)
    return {_named_example(row[0]): row for row in _rows(table)}


def _index_rows() -> dict[str, list[str]]:
    table = table_after(INDEX.read_text(encoding="utf-8"), INDEX_TABLE_ANCHOR)
    return {_named_example(row[0]): row for row in _rows(table)}


class ChapterShapeTest(unittest.TestCase):
    def test_present_and_agent_facing(self) -> None:
        self.assertTrue(CHAPTER.is_file())
        self.assertIn("\n## For agents\n", _chapter())


class TheTableIsTheExampleSetTest(unittest.TestCase):
    def test_the_chapter_names_exactly_the_examples_that_exist(self) -> None:
        self.assertEqual(sorted(_chapter_rows()), sorted(_example_directories()))

    def test_the_index_names_exactly_the_examples_that_exist(self) -> None:
        # If this fails the chapter is not wrong; its source is.
        self.assertEqual(sorted(_index_rows()), sorted(_example_directories()))

    def test_every_named_example_ships_the_files_an_example_ships(self) -> None:
        for name in sorted(_chapter_rows()):
            for filename in REQUIRED_FILES:
                with self.subTest(example=name, file=filename):
                    self.assertTrue((EXAMPLES / name / filename).is_file())

    def test_the_verdict_column_agrees_with_the_index(self) -> None:
        chapter, index = _chapter_rows(), _index_rows()
        for name in sorted(chapter):
            with self.subTest(example=name):
                self.assertEqual(
                    _solvable(chapter[name][-1]),
                    _solvable(index[name][-1]),
                    f"{name}: the chapter and examples/README.md disagree about "
                    "whether any shipped algorithm solves it",
                )

    def test_the_chapter_counts_the_examples_it_lists(self) -> None:
        stated = re.search(r"^## 3\. The (\w+) that ship$", _chapter(), flags=re.MULTILINE)
        self.assertIsNotNone(stated, "§3's heading has been reworded")
        assert stated is not None
        words = "zero one two three four five six seven eight nine ten".split()
        self.assertEqual(words.index(stated.group(1)), len(_chapter_rows()))


class AnUnsolvableExampleSaysSoTest(unittest.TestCase):
    def test_each_one_carries_its_own_section(self) -> None:
        unsolvable = [name for name, row in _chapter_rows().items() if not _solvable(row[-1])]
        self.assertTrue(unsolvable, "the chapter claims every example is solvable")
        for name in sorted(unsolvable):
            with self.subTest(example=name):
                readme = (EXAMPLES / name / "README.md").read_text(encoding="utf-8")
                headings = [line.rstrip() for line in readme.splitlines() if line.startswith("## ")]
                self.assertIn(
                    UNSOLVABLE_SECTION,
                    headings,
                    f"{name} is a failure demonstration and its README has no "
                    f"'{UNSOLVABLE_SECTION}' section; it has {headings}",
                )

    def test_the_chapter_states_how_many_there_are(self) -> None:
        unsolvable = [name for name, row in _chapter_rows().items() if not _solvable(row[-1])]
        words = "zero one two three four five six seven eight nine ten".split()
        # The sentence wraps; compare with line breaks flattened.
        self.assertIn(
            f"**{words[len(unsolvable)].capitalize()} of the five pose problems "
            "no shipped algorithm can solve.**",
            " ".join(_chapter().split()),
        )


class TheGroundTruthRuleIsFollowedTest(unittest.TestCase):
    """§4's one mechanical rule, checked against every `problem.py`."""

    def test_every_problem_module_defines_and_calls_a_self_check(self) -> None:
        for name in sorted(_example_directories()):
            with self.subTest(example=name):
                tree = ast.parse((EXAMPLES / name / "problem.py").read_text(encoding="utf-8"))
                defined = any(
                    isinstance(node, ast.FunctionDef) and node.name == "_self_check"
                    for node in tree.body
                )
                called = any(
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "_self_check"
                    for node in tree.body
                )
                self.assertTrue(defined, f"{name}/problem.py defines no _self_check")
                self.assertTrue(
                    called,
                    f"{name}/problem.py defines _self_check but never calls it at "
                    "module scope, so the ground truth is not asserted at import",
                )

    def test_a_self_check_actually_raises(self) -> None:
        """A `_self_check` with no failure path would satisfy the rule and check nothing."""

        for name in sorted(_example_directories()):
            with self.subTest(example=name):
                tree = ast.parse((EXAMPLES / name / "problem.py").read_text(encoding="utf-8"))
                function = next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_self_check"
                )
                raises = [node for node in ast.walk(function) if isinstance(node, ast.Raise)]
                self.assertTrue(raises, f"{name}/problem.py's _self_check cannot fail")

    def test_the_rule_is_stated_in_the_chapter(self) -> None:
        rules = section_of(_chapter(), "## 4. The rules a new example must follow")
        self.assertIn("`_self_check()` that runs at import", rules)


class WhatIsNotBuiltIsStillNotBuiltTest(unittest.TestCase):
    """§6 rots in one direction: the feature lands and the section stays."""

    def test_no_example_ships_a_committed_best_yaml(self) -> None:
        found = sorted(
            str(path.relative_to(REPO_ROOT)) for path in REPO_ROOT.glob("**/examples/**/best*.y*ml")
        )
        self.assertEqual(
            found,
            [],
            f"docs/15 §6 says the tune/run split is not built; these files say otherwise: {found}",
        )

    def test_no_example_config_carries_a_status_field(self) -> None:
        carrying = []
        directories = (
            REPO_ROOT / "configs" / "examples",
            REPO_ROOT / "data" / "configs" / "examples",
        )
        for directory in directories:
            for path in sorted(directory.rglob("*.yaml")):
                text = path.read_text(encoding="utf-8")
                if re.search(r"^\s*status\s*:", text, flags=re.MULTILINE):
                    carrying.append(str(path.relative_to(REPO_ROOT)))
        self.assertEqual(
            carrying,
            [],
            "docs/15 §6 says the comparison/failure-demo status field is not "
            f"built; these configs carry one: {carrying}",
        )

    def test_the_section_names_both(self) -> None:
        body = section_of(_chapter(), "## 6. What is not built, and why")
        self.assertIn("best.yaml", body)
        self.assertIn("`status` field", body)


class EveryPathTheChapterNamesExistsTest(unittest.TestCase):
    def test_paths_resolve(self) -> None:
        text = _chapter()
        # Repo-relative paths written in backticks: a directory, or a file with
        # a suffix. `<name>` placeholders are skipped.
        candidates = {
            match
            for match in re.findall(r"`((?:examples|configs|data|docs|tests)/[^`\s]*)`", text)
            if "<" not in match
        }
        self.assertTrue(candidates, "the chapter names no paths; the pattern is wrong")
        for candidate in sorted(candidates):
            with self.subTest(path=candidate):
                self.assertTrue(
                    (REPO_ROOT / candidate).exists(),
                    f"{candidate} is named by docs/15 and does not exist",
                )


if __name__ == "__main__":
    unittest.main()
