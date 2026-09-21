"""Counts the documentation states about the tree must be the tree's counts.

docs/00-index.md's "For agents" table inventories the repository -- how many
run configs, how many test modules -- and README.md says how many chapters
there are. None of those had a guard, and the test-module count drifted from
82 to 106 without anything noticing, which is exactly the failure the index
itself describes: a claim no test protects reads as authoritative for as long
as it takes someone to count.

configs/README.md goes one step further and lists every run config by name,
with an arm count beside each `examples/` directory. That listing had no guard
either, and it drifted the same way: it named seven of FEMNIST's ten configs
and left out an `examples/` directory of nine arms.

They are cheap to check because each one is a directory listing. Kept in one
module rather than spread across the chapter guards, because the thing being
protected is the same in every case -- a number in prose against a number on
disk -- and a reader fixing one wants to see the others.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import pytest
from docs_sections import fenced_block_after

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX = REPO_ROOT / "docs" / "00-index.md"
README = REPO_ROOT / "README.md"
CONFIGS_README = REPO_ROOT / "configs" / "README.md"

#: The sentence that introduces configs/README.md's listing.
CONFIGS_LISTING_ANCHOR = "One directory per dataset; one file per method."

#: Directories under configs/ that the listing leaves to the README's table of
#: the other directories, because they hold no experiments.
NOT_IN_CONFIGS_LISTING = frozenset({"dev", "llm_assets"})

#: One part of an `examples/` note: "(7 arms)", "(8 arms each)", "(1 arm)".
_ARMS = re.compile(r"(\d+) arms?( each)?")

#: A chapter may spell a small count instead of writing a digit.
_NUMBER_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
).split()


def _states_count(text: str, value: int, noun: str) -> bool:
    """True when `text` states `value` immediately before `noun`."""

    forms = {str(value)}
    if value < len(_NUMBER_WORDS):
        word = _NUMBER_WORDS[value]
        forms |= {word, word.capitalize()}
    return any(f"{form} {noun}" in text for form in forms)


def _test_modules() -> list[Path]:
    return sorted((REPO_ROOT / "tests").glob("test_*.py"))


def _run_configs() -> list[Path]:
    """Configs that carry a `runtime` block, which is what makes one a run config."""

    return sorted(
        path for path in (REPO_ROOT / "configs").rglob("*.yaml") if "llm_assets" not in path.parts
    )


def _asset_configs() -> list[Path]:
    return sorted((REPO_ROOT / "configs" / "llm_assets").glob("*.yaml"))


def _configs_listing() -> dict[str, list[tuple[list[str], str]]]:
    """configs/README.md's listing as {directory: [(entries, note), ...]}, one pair per line.

    A line that starts in the first column opens a directory; an indented line
    continues the one above it. A trailing parenthetical is the line's note.
    """

    block = fenced_block_after(CONFIGS_README.read_text(encoding="utf-8"), CONFIGS_LISTING_ANCHOR)
    listing: dict[str, list[tuple[list[str], str]]] = {}
    directory = ""
    for line in block.splitlines():
        note = re.search(r"\(([^)]*)\)\s*$", line)
        tokens = (line[: note.start()] if note else line).split()
        if not line.startswith(" "):
            directory = tokens.pop(0).rstrip("/")
            listing[directory] = []
        if not directory:
            raise AssertionError(f"configs/README.md's listing opens with a continuation: {line!r}")
        listing[directory].append((tokens, note.group(1) if note else ""))
    return listing


def _arm_counts(entries: list[str], note: str) -> list[tuple[str, int]]:
    """Each directory on one listing line, with the arm count its note gives it.

    The note's later parts belong to the line's last directories, one each; its
    first part covers every directory before those, and has to say "each" when
    that is more than one. A note that cannot be read that way raises rather
    than guessing.
    """

    parts = _ARMS.findall(note)
    leading = len(entries) - (len(parts) - 1)
    if not parts or leading < 1 or (leading > 1 and not parts[0][1]):
        raise AssertionError(f"cannot read arm counts for {entries} from ({note})")
    counts = [int(parts[0][0])] * leading + [int(count) for count, _ in parts[1:]]
    return [(entry.rstrip("/"), count) for entry, count in zip(entries, counts, strict=True)]


class IndexInventoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = INDEX.read_text(encoding="utf-8")

    def test_the_test_module_count_is_the_directory_listing(self) -> None:
        count = len(_test_modules())
        self.assertGreater(count, 0, "no test modules found; the glob is wrong")
        self.assertTrue(
            _states_count(self.text, count, "test modules"),
            f"docs/00-index.md must say {count} test modules; tests/ holds that many",
        )

    def test_the_run_config_count_is_the_directory_listing(self) -> None:
        count = len(_run_configs())
        self.assertGreater(count, 0)
        self.assertTrue(
            _states_count(self.text, count, "run configs"),
            f"docs/00-index.md must say {count} run configs",
        )

    def test_every_counted_run_config_really_carries_a_runtime_block(self) -> None:
        """The parenthetical in that row is the definition, so check it holds."""

        without = sorted(
            path.relative_to(REPO_ROOT).as_posix()
            for path in _run_configs()
            if not re.search(r"^runtime:", path.read_text(encoding="utf-8"), flags=re.MULTILINE)
        )
        self.assertEqual(
            without,
            [],
            "docs/00-index.md defines a run config as one carrying a `runtime` "
            f"block; these are counted as run configs and have none: {without}",
        )

    def test_the_asset_config_count_is_the_directory_listing(self) -> None:
        count = len(_asset_configs())
        self.assertGreater(count, 0)
        self.assertTrue(
            _states_count(self.text, count, "asset-preparation configs"),
            f"docs/00-index.md must say {count} asset-preparation configs",
        )


class ReadmeChapterCountTest(unittest.TestCase):
    def test_the_chapter_count_is_the_directory_listing(self) -> None:
        chapters = sorted((REPO_ROOT / "docs").glob("[0-9][0-9]-*.md"))
        self.assertGreater(len(chapters), 0)
        self.assertTrue(
            _states_count(README.read_text(encoding="utf-8"), len(chapters), "chapters"),
            f"README.md must say {len(chapters)} chapters; docs/ holds that many",
        )


class ConfigsReadmeListingTest(unittest.TestCase):
    """configs/README.md names every run config, and the names are the directory's.

    Checked by name rather than by count: a listing with the right number of
    entries and one of them wrong reads just as authoritative, and a reader
    looking for a config's name is reading the names.
    """

    def setUp(self) -> None:
        self.listing = _configs_listing()

    def test_every_directory_under_configs_is_in_the_listing(self) -> None:
        on_disk = sorted(
            path.name
            for path in (REPO_ROOT / "configs").iterdir()
            if path.is_dir() and path.name not in NOT_IN_CONFIGS_LISTING
        )
        self.assertGreater(len(on_disk), 2, "configs/ walk found almost nothing")
        self.assertEqual(sorted(self.listing), on_disk)

    def test_each_dataset_line_names_exactly_the_configs_in_its_directory(self) -> None:
        for directory, lines in self.listing.items():
            if directory == "examples":
                continue
            with self.subTest(directory=directory):
                listed = sorted(entry for entries, _ in lines for entry in entries)
                on_disk = sorted(
                    path.stem for path in (REPO_ROOT / "configs" / directory).glob("*.yaml")
                )
                self.assertGreater(len(on_disk), 0)
                self.assertEqual(listed, on_disk, f"configs/README.md's {directory}/ line")

    def test_each_example_directory_is_listed_with_its_arm_count(self) -> None:
        self.assertIn("examples", self.listing)
        listed = sorted(
            pair
            for entries, note in self.listing["examples"]
            for pair in _arm_counts(entries, note)
        )
        on_disk = sorted(
            (path.name, len(list(path.glob("*.yaml"))))
            for path in (REPO_ROOT / "configs" / "examples").iterdir()
            if path.is_dir()
        )
        self.assertGreater(len(on_disk), 2)
        self.assertEqual(listed, on_disk, "configs/README.md examples/ lines as (directory, arms)")


if __name__ == "__main__":
    unittest.main()
