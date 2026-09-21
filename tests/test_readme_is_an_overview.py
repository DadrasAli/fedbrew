"""README.md must stay an overview, not become a second reference.

The previous documentation set drifted two CLI generations stale while still
reading as authoritative. The mechanism was not neglect: it was having two
places that described the same thing, so a fix applied to one left the other
saying something false with equal confidence. The reproducibility section was
the last instance -- README.md and docs/10 both carried it, and only the
chapter was guarded.

So this checks the shape of README.md rather than its wording:

- it stays short, because length is how a pointer becomes a reference;
- it links to the index, which is where a reader is supposed to end up;
- it carries no heading a chapter owns; and
- it states no default, key table or column name, which are the three kinds of
  fact that go stale silently.

None of this stops the README explaining the project. It stops the README
being the second place a reader could look up a config key.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"

#: An overview of a project this size fits in this. The previous README was
#: 822 lines, which is a reference pretending to be a front page.
MAX_LINES = 140

#: Headings that belong to a chapter. A README section with any of these names
#: is a second copy of material a chapter already owns and guards.
CHAPTER_OWNED_HEADINGS = frozenset(
    {
        "algorithms and datasets",
        "adding an algorithm or a dataset",
        "architecture",
        "configuration reference",
        "data and partitioning",
        "environment variables",
        "extending",
        "inside run.json",
        "metrics",
        "models and tasks",
        "performance",
        "project layout",
        "reproducibility",
        "reproducibility and determinism",
        "testing",
        "the cli",
        "what a run writes",
    }
)


def _declared_extras() -> set[str]:
    """The extras pyproject defines.

    Read from the file rather than kept as a literal here. The previous form
    was a frozenset naming the extras of the day, referenced by no test, so it
    still said `femnist` after `vision` replaced it -- a stale allowlist that
    could not fail.
    """

    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    start = text.index("[project.optional-dependencies]")
    end = text.find("\n[", start + 1)
    block = text[start : end if end > 0 else len(text)]
    return set(re.findall(r"^(\w+)\s*=\s*\[", block, flags=re.MULTILINE))


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def _headings() -> list[str]:
    return re.findall(r"^#{2,3} (.+)$", _readme(), flags=re.MULTILINE)


class ShapeTest(unittest.TestCase):
    def test_it_stays_short(self) -> None:
        lines = len(_readme().splitlines())
        self.assertLessEqual(
            lines,
            MAX_LINES,
            f"README.md is {lines} lines. An overview that grows past "
            f"{MAX_LINES} is becoming a reference; move the section into the "
            "chapter that owns it and link to it instead.",
        )

    def test_every_extra_it_names_is_a_real_one(self) -> None:
        """The README lists the extras; the list must not outlive them."""

        declared = _declared_extras()
        self.assertTrue(declared, "no extras parsed from pyproject")
        named = set(re.findall(r"`(\w+)`", _readme())) & (declared | {"femnist", "vision"})
        self.assertTrue(named, "the README names no extra at all")
        self.assertEqual(
            named - declared,
            set(),
            f"README.md names extras pyproject does not define: {sorted(named - declared)}",
        )

    def test_it_points_at_the_index(self) -> None:
        self.assertIn("docs/00-index.md", _readme())

    def test_every_chapter_it_links_to_exists(self) -> None:
        linked = set(re.findall(r"\((docs/[\w./-]+\.md)\)", _readme()))
        self.assertTrue(linked, "the README must link into docs/")
        missing = sorted(path for path in linked if not (REPO_ROOT / path).is_file())
        self.assertEqual(
            missing,
            [],
            f"README.md links to chapters that do not exist: {missing}",
        )


#: What the project cannot do, as (owning chapter, phrases that chapter must
#: contain). The README used to list these under a heading of its own, and
#: three of the five were stated there and nowhere else: a limitation whose
#: only statement is on the front page is a disclaimer, not documentation. Each
#: is now stated by the chapter that owns the subject, and the README carries
#: none of them, so there is one place to keep each true.
LIMITATIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "FEMNIST's test split is per-writer": (
        "docs/05-data-and-partitioning.md",
        ("per-writer", "seen writers", "unseen writers"),
    ),
    "The two per-client histories are held in memory": (
        "docs/11-performance-and-cost.md",
        ("held in memory for the whole run",),
    ),
    "OpenImage is not supported end to end": (
        "docs/05-data-and-partitioning.md",
        (
            "OpenImage is not supported end to end",
            "none of the eight generators reads OpenImage",
            "fails at load with a missing manifest",
        ),
    ),
    "Simulation only": (
        "docs/01-architecture.md",
        (
            "It simulates federated learning and does not deploy it",
            "one device per run",
            "There is no sweep runner",
            "no plotting",
        ),
    ),
    "A research harness, not a deployment framework": (
        "docs/01-architecture.md",
        (
            "no network layer, no client daemon and no secure aggregation",
            "Python objects the server calls in sequence",
            "its cost is measured and reported",
            "not incurred",
        ),
    ),
    "The schema is not stable across versions": (
        "docs/04-configuration.md",
        ("The schema is strict and not stable", "changed in meaning between versions"),
    ),
}


def _flattened(path: Path) -> str:
    """Hard wrapping removed, so a phrase spanning a line break still matches."""

    return " ".join(path.read_text(encoding="utf-8").split())


def _chapters() -> dict[str, str]:
    return {path.name: _flattened(path) for path in sorted((REPO_ROOT / "docs").glob("*.md"))}


class LimitationOwnershipTest(unittest.TestCase):
    """Each limitation is stated by the chapter that owns it, and not on the front page."""

    def test_every_limitation_is_stated_by_its_chapter(self) -> None:
        chapters = _chapters()
        for key, (chapter, phrases) in sorted(LIMITATIONS.items()):
            text = chapters[Path(chapter).name]
            for phrase in phrases:
                with self.subTest(limitation=key, phrase=phrase):
                    self.assertIn(phrase, text, f"{chapter} must state: {key}")

    def test_the_readme_carries_no_limitations_section(self) -> None:
        """A second list would be the front-page disclaimer this replaced."""

        offenders = [heading for heading in _headings() if "limitation" in heading.lower()]
        self.assertEqual(offenders, [], "README.md lists limitations; a chapter owns each one")

    def test_the_stability_note_names_the_version_pyproject_declares(self) -> None:
        """The note is about one release; a version bump has to revisit it."""

        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        version = re.search(r'^version = "([^"]+)"$', text, flags=re.MULTILINE)
        assert version is not None
        self.assertIn(f"version {version.group(1)}", _chapters()["04-configuration.md"])

    def test_the_chapter_that_owns_the_split_does_not_contradict_it(self) -> None:
        """It did: two sentences and an invariant said the opposite."""

        chapter = (REPO_ROOT / "docs" / "05-data-and-partitioning.md").read_text(encoding="utf-8")
        for sentence in (
            "The client test split comes from the official test set, not from the",
            "**The client test split comes from the global test set.**",
        ):
            with self.subTest(sentence=sentence):
                self.assertNotIn(sentence, chapter)


class NoSecondReferenceTest(unittest.TestCase):
    def test_no_section_duplicates_a_chapter(self) -> None:
        offenders = sorted(
            heading for heading in _headings() if heading.strip().lower() in CHAPTER_OWNED_HEADINGS
        )
        self.assertEqual(
            offenders,
            [],
            "README.md has sections a chapter already owns and guards: "
            f"{offenders}. Two places describing one thing is how the previous "
            "docs/ went stale.",
        )

    def test_it_does_not_restate_the_config_surface(self) -> None:
        """A dotted config key in the README is a reference entry."""

        from fedbrew.core.config import _KNOWN_EXTRA_KEYS

        sections = {name for name in _KNOWN_EXTRA_KEYS if "." not in name}
        dotted = re.findall(r"`((?:\w+\.)+\w+)`", _readme())
        offenders = sorted({key for key in dotted if key.split(".")[0] in sections})
        self.assertEqual(
            offenders,
            [],
            f"README.md names config keys: {offenders}. Chapter 04 owns the "
            "config surface and its tables are checked against the code.",
        )

    def test_it_does_not_restate_metric_column_names(self) -> None:
        """An aggregate column name in the README is a chapter-08 fact."""

        from fedbrew.core.config import ClientStatisticsConfig, client_metric_names

        columns = set()
        for split in ("train", "val", "test"):
            columns |= client_metric_names(split, ClientStatisticsConfig())
        offenders = sorted(name for name in columns if name in _readme())
        self.assertEqual(
            offenders,
            [],
            f"README.md names metric columns: {offenders}. Chapter 08 owns "
            "them and diffs its tables against the emitter.",
        )


class StillUsefulTest(unittest.TestCase):
    """Shortening it must not empty it."""

    def test_it_still_says_what_fedbrew_is(self) -> None:
        text = _readme()
        for expected in ("federated", "benchmark", "single-process simulator"):
            with self.subTest(phrase=expected):
                self.assertIn(expected, text)

    def test_it_still_carries_a_runnable_quickstart(self) -> None:
        from fedbrew.cli.dispatch import COMMANDS

        # Ruff's arguments are paths, and two of them are the package
        # directory and the examples directory -- adjacent, so the pattern
        # below reads the second as a subcommand of the first. Neither is an
        # invocation. Skipped by line, the same way and for the same reason as
        # tests/test_cli_commands_exist.py, which scans the whole tree for
        # this and hits the identical command in four other files.
        prose = "\n".join(
            line for line in _readme().splitlines() if not re.match(r"^[\s$>]*ruff\b", line)
        )
        used = set(re.findall(r"\bfedbrew ([a-z][a-z0-9-]*)", prose))
        self.assertTrue(
            {"generate", "run"} <= used,
            "the README's quickstart must still show generate and run",
        )
        self.assertEqual(sorted(used - set(COMMANDS)), [])

    def test_the_install_line_matches_the_install_section(self) -> None:
        """The quickstart's install command must be the Install section's, verbatim.

        Two independent lines saying how to install is how they drift --
        which is exactly the failure this module's docstring describes for
        the reproducibility section. Comparing them beats restating either:
        whichever one someone edits, the other has to follow or this fails.

        When this repository is published to PyPI, both fenced blocks below
        become `pip install fedbrew` on the same day -- this test does not
        care which string it is, only that the two agree, so it will not
        block that change or need editing for it.
        """

        def _install_line(heading: str) -> str:
            section = re.search(
                rf"^## {heading}\n(.*?)(?=\n## |\Z)", _readme(), flags=re.DOTALL | re.MULTILINE
            )
            self.assertIsNotNone(section, f"README.md has no '## {heading}' section")
            assert section is not None
            lines = [
                line for line in section.group(1).splitlines() if line.startswith("pip install")
            ]
            self.assertEqual(
                len(lines),
                1,
                f"expected exactly one install line under '## {heading}', found {lines}",
            )
            return lines[0]

        self.assertEqual(
            _install_line("Install"),
            _install_line("Quickstart"),
            "the Install section and the Quickstart section give different install commands",
        )

    def test_it_still_carries_the_licence_and_citation(self) -> None:
        text = _readme()
        self.assertIn("MIT", text)
        self.assertIn("CITATION.cff", text)
        self.assertTrue((REPO_ROOT / "CITATION.cff").is_file())
        self.assertTrue((REPO_ROOT / "LICENSE").is_file())


if __name__ == "__main__":
    unittest.main()
