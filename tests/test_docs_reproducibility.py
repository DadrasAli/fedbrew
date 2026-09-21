"""docs/10-reproducibility.md must match what seeding and setup actually do.

The reproducibility chapter is the one whose errors are hardest to notice: a
reader follows it, gets numbers, and has no way to tell that the guarantee they
relied on was never true. So the claims that can be executed are executed here
rather than described.

The headline claim -- that matmul_precision changes numerics, stated in the
section a reader checking reproducibility would open -- is guarded by
tests/test_matmul_precision.py, which was retargeted from README.md to this
chapter when the material moved. It is not duplicated here.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import pytest
from docs_sections import table_after

from fedbrew.core.config import MATMUL_PRECISIONS
from fedbrew.core.seeding import dataloader_seed, derive_seed

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER = REPO_ROOT / "docs" / "10-reproducibility.md"


def _chapter_text() -> str:
    return CHAPTER.read_text(encoding="utf-8")


def _says(fragment: str) -> bool:
    """Whitespace-insensitive, so a wrapped sentence still matches."""

    return fragment in " ".join(_chapter_text().split())


def _throughput_table_keys(chapter: str) -> set[str]:
    """The Setting column of the throughput-only table, as bare key names.

    The key column only. Reading the whole row also picks up `DataLoader` out
    of the prose beside it, which is the same "matched in the wrong place"
    mistake this table's guard used to make.
    """

    table = table_after(chapter, "**These settings are throughput-only")
    keys: set[str] = set()
    for row in table.splitlines()[2:]:
        if not row.strip().startswith("|"):
            continue
        cell = row.strip("| ").split("|")[0]
        keys |= {token.strip("`,").rsplit(".", 1)[-1] for token in cell.split() if "`" in token}
    keys.discard("")
    return keys


class ChapterShapeTest(unittest.TestCase):
    def test_present_and_agent_facing(self) -> None:
        self.assertTrue(CHAPTER.is_file())
        self.assertIn("\n## For agents\n", _chapter_text())

    def test_it_carries_the_headings_the_matmul_guard_anchors_on(self) -> None:
        """test_matmul_precision.py locates its claim by these two headings."""

        text = _chapter_text()
        for heading in ("## Reproducibility", "**do** change numerics"):
            with self.subTest(heading=heading):
                self.assertEqual(
                    text.count(heading),
                    1,
                    f"{heading!r} must appear exactly once for the matmul "
                    "guard to locate the right section",
                )


class SeedDerivationTest(unittest.TestCase):
    """Section 'What is guaranteed' makes three executable claims."""

    def test_the_derivation_hashes_rather_than_adds(self) -> None:
        """The claim that neighbouring seeds are not one shifted stream.

        If derive_seed added, derive_seed(42, "round", 2) and
        derive_seed(43, "round", 1) would collide for some pairing. Hashing
        makes that a coincidence rather than a guarantee, and the chapter's
        warning about seeds 42/43/44 depends on it.
        """

        shifted = {
            derive_seed(base, "round", offset) for base, offset in ((42, 2), (43, 1), (44, 0))
        }
        self.assertEqual(len(shifted), 3, "neighbouring seeds collided")
        self.assertTrue(_says("hashes rather than adds"))

    def test_a_stream_is_keyed_by_client_and_round_not_by_order(self) -> None:
        first = dataloader_seed(7, 3, "client_0", "fit")
        again = dataloader_seed(7, 3, "client_0", "fit")
        other_client = dataloader_seed(7, 3, "client_1", "fit")
        other_round = dataloader_seed(7, 4, "client_0", "fit")
        other_phase = dataloader_seed(7, 3, "client_0", "eval")

        self.assertEqual(first, again, "the same key must give the same stream")
        self.assertNotEqual(first, other_client)
        self.assertNotEqual(first, other_round)
        self.assertNotEqual(first, other_phase)
        self.assertTrue(_says("not on how many clients were selected"))

    def test_strict_determinism_is_the_default(self) -> None:
        import inspect

        from fedbrew.core.runtime_setup import seed_everything

        signature = inspect.signature(seed_everything)
        self.assertIs(
            signature.parameters["warn_only"].default,
            False,
            "the chapter says deterministic_warn_only defaults to false",
        )
        self.assertTrue(_says("`deterministic_warn_only` defaults to\n`false`".replace("\n", " ")))


class MatmulClaimTest(unittest.TestCase):
    def test_every_precision_value_is_documented(self) -> None:
        text = _chapter_text()
        for value in MATMUL_PRECISIONS:
            with self.subTest(precision=value):
                self.assertIn(f"`{value}`", text)

    def test_the_absent_default_is_stated(self) -> None:
        """The single most expensive thing to get wrong in this chapter."""

        self.assertTrue(_says('Absent means `highest`, not "unset"'))

    def test_the_value_check_still_exists(self) -> None:
        source = (REPO_ROOT / "fedbrew" / "core" / "config.py").read_text(encoding="utf-8")
        self.assertIn("matmul_precision must be one of", source)


class CublasClaimTest(unittest.TestCase):
    def test_it_is_set_with_setdefault(self) -> None:
        """Never over an existing value, which the chapter states twice."""

        source = (REPO_ROOT / "fedbrew" / "core" / "runtime_setup.py").read_text(encoding="utf-8")
        self.assertIn("os.environ.setdefault(", source)
        self.assertIn(":4096:8", source)
        self.assertTrue(_says("only if not already set"))


class ThroughputOnlyTest(unittest.TestCase):
    """The settings the chapter promises do not change results."""

    def test_the_table_is_the_code_s_throughput_only_set(self) -> None:
        """Both directions, against the classification the code now carries.

        This held a hardcoded list of five names and never opened the chapter,
        so it could not notice the table losing a row -- deleting
        shard_cache_bytes' row went unnoticed. It could not be made
        bidirectional either, because "throughput-only" existed as a
        classification in this table and nowhere else: there was no authority
        to diff against, which is the condition every guard fixed in this sweep
        grew out of. THROUGHPUT_ONLY_PERFORMANCE_KEYS is that authority.
        """

        from fedbrew.core.config import THROUGHPUT_ONLY_PERFORMANCE_KEYS

        listed = _throughput_table_keys(_chapter_text())
        self.assertTrue(listed, "no settings parsed from the throughput-only table")
        self.assertEqual(
            THROUGHPUT_ONLY_PERFORMANCE_KEYS - listed,
            set(),
            "the table omits throughput-only keys: "
            f"{sorted(THROUGHPUT_ONLY_PERFORMANCE_KEYS - listed)}",
        )
        self.assertEqual(
            listed - THROUGHPUT_ONLY_PERFORMANCE_KEYS,
            set(),
            "the table calls settings throughput-only that the code does not: "
            f"{sorted(listed - THROUGHPUT_ONLY_PERFORMANCE_KEYS)}",
        )

    def test_the_two_classifications_do_not_overlap(self) -> None:
        """Guards the guard: an empty numerics set would make the split vacuous."""

        from fedbrew.core.config import (
            NUMERICS_PERFORMANCE_KEYS,
            THROUGHPUT_ONLY_PERFORMANCE_KEYS,
        )

        self.assertTrue(NUMERICS_PERFORMANCE_KEYS)
        self.assertTrue(THROUGHPUT_ONLY_PERFORMANCE_KEYS)
        self.assertEqual(NUMERICS_PERFORMANCE_KEYS & THROUGHPUT_ONLY_PERFORMANCE_KEYS, frozenset())
        for key in NUMERICS_PERFORMANCE_KEYS:
            with self.subTest(key=key):
                self.assertIn(f"`{key}`", _chapter_text())


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
