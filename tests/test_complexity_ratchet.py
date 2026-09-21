"""The complexity ceiling can fall on its own and can only rise deliberately.

Nothing in the tree observed cyclomatic complexity, so the number of functions
above the conventional threshold of 10 could grow with no one able to notice.
`C901` is in
`[tool.ruff.lint] select` now, pinned at 22 -- what the two worst functions in
the tree measured when it was adopted -- so `ruff check` failed nothing that
existed and fails a function that goes past the ceiling.

A ratchet needs a direction, and ruff can only enforce one of them. It checks
that no function exceeds the configured limit; it cannot check that the limit
was not simply raised to make a failure go away, which is the whole failure
mode -- the cheapest response to a C901 is one digit in `pyproject.toml`. So
the other direction is here: the configured limit may never exceed the value
adopted with the rule. Lowering it costs nothing and needs no edit here.
Raising it fails this test, and the fix is to change `ADOPTED_CEILING` with a
sentence saying why, the same way `NON_COMMAND_WORDS` is a review moment rather
than an escape hatch in `tests/test_cli_commands_exist.py`.

Seventeen functions sit above 10 and are deliberately not split. `logging.py` is
the second-largest module in the package and stays whole: splitting it because
a metric says so is refactoring to the metric rather than to a problem. What
was missing was not smaller functions, it was a number anyone could see.

Two of the tests below measure the tree, and measuring needs ruff. ruff comes
with the `dev` extra, so an environment installed without extras -- CI's
`core-only` job is one -- has none, and there those two skip and say why.
Before they did, they failed on being unable to look: `python -m ruff` without
ruff exits 1 and prints nothing to stdout, and an empty measurement cannot be
told apart from a tree with nothing above 10.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import unittest
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CHAPTER = REPO_ROOT / "docs" / "13-testing.md"

#: The value C901 was adopted at, and an upper bound rather than the expected
#: value: a refactor that lowers the real worst case should be free to lower
#: the config without editing a test. Only raising it has to be argued for.
ADOPTED_CEILING = 22

#: The conventional threshold, and not what is enforced. Kept so the gap
#: between "conventional" and "what this tree actually does" is stated rather
#: than implied by a bare 22.
CONVENTIONAL_THRESHOLD = 10

#: ruff is in the `dev` extra, not a core dependency. `OptionalExtraCoverageTest`
#: in tests/test_docs_testing.py reads this `find_spec` and fails unless some CI
#: job installs the extra that declares ruff, so the skip it gates cannot become
#: a skip in every job without that test saying so.
RUFF_INSTALLED = importlib.util.find_spec("ruff") is not None
NO_RUFF = (
    "ruff is not installed (pip install -e '.[dev]'), so the tree's complexity cannot "
    "be measured here; `ruff check` enforces the ceiling itself wherever ruff is installed"
)

#: The files that say how many functions sit above CONVENTIONAL_THRESHOLD. Each
#: says it once, and what it says has to be ruff's count.
COUNT_STATEMENTS = (PYPROJECT, CHAPTER, Path(__file__).resolve())

#: Prose starting a sentence spells a count rather than writing digits.
_NUMBER_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
).split()

#: `[\s#]+` between the words, because pyproject.toml wraps the sentence across
#: comment lines.
_COUNT_STATEMENT = re.compile(
    r"\b(\d+|" + "|".join(_NUMBER_WORDS) + r")[\s#]+functions[\s#]+sit[\s#]+above\b",
    re.IGNORECASE,
)


def _configured_ceiling() -> int:
    section = re.search(
        r"^\[tool\.ruff\.lint\.mccabe\]$(.*?)(?=^\[)",
        PYPROJECT.read_text(encoding="utf-8"),
        re.MULTILINE | re.DOTALL,
    )
    assert section is not None, "pyproject declares no [tool.ruff.lint.mccabe] section"
    found = re.search(r"^max-complexity\s*=\s*(\d+)$", section.group(1), re.MULTILINE)
    assert found is not None, "the mccabe section declares no max-complexity"
    return int(found.group(1))


def _selected_rules() -> list[str]:
    found = re.search(
        r"^\[tool\.ruff\.lint\]$\s*^select\s*=\s*\[([^\]]*)\]",
        PYPROJECT.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert found is not None, "pyproject declares no lint select list"
    return re.findall(r'"([^"]+)"', found.group(1))


def _complexities(limit: int) -> dict[str, int]:
    """Every function ruff reports above `limit`, by `path:line:name`.

    Raises when ruff ran without producing a measurement. The exit code alone
    cannot tell: ruff exits 1 whenever it reports a function, which at the
    conventional threshold is the normal case, and `python -m ruff` also exits
    1 when ruff fails to start. So a run measured something when it exits 0,
    having nothing to report, or exits 1 with findings that parse. Anything
    else is ruff failing, and its stderr is the message.
    """

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--select",
            "C901",
            "--config",
            f"lint.mccabe.max-complexity={limit}",
            "--output-format",
            "concise",
            "tests",
            "tools",
            "fedbrew",
            "examples",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    pattern = re.compile(r"^(\S+?):(\d+):\d+: C901 `(.+?)` is too complex \((\d+) > \d+\)$")
    found: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        match = pattern.match(line)
        if match:
            found[f"{match.group(1)}:{match.group(2)}:{match.group(3)}"] = int(match.group(4))
    if completed.returncode not in (0, 1) or (completed.returncode == 1 and not found):
        output = completed.stderr.strip() or completed.stdout.strip() or "(no output)"
        raise AssertionError(
            f"ruff is installed but exited {completed.returncode} without a measurement "
            f"this test can read:\n{output}"
        )
    return found


def _stated_counts(text: str) -> list[int]:
    """Every '<count> functions sit above' in `text`, as integers."""

    return [
        int(word) if word.isdigit() else _NUMBER_WORDS.index(word.lower())
        for word in _COUNT_STATEMENT.findall(text)
    ]


@pytest.mark.fast
class TheRuleIsActuallyOnTest(unittest.TestCase):
    def test_c901_is_selected(self) -> None:
        """Without this the ceiling below is a number nothing reads."""

        self.assertIn("C901", _selected_rules())

    def test_the_ceiling_is_declared(self) -> None:
        self.assertGreater(_configured_ceiling(), 0)


@pytest.mark.fast
class TheCeilingOnlyFallsTest(unittest.TestCase):
    def test_it_is_not_above_the_value_it_was_adopted_at(self) -> None:
        configured = _configured_ceiling()
        self.assertLessEqual(
            configured,
            ADOPTED_CEILING,
            f"max-complexity is {configured}, above the {ADOPTED_CEILING} C901 was "
            "adopted at. Raising the ceiling is how a complexity gate stops "
            "meaning anything; simplify the function, or change ADOPTED_CEILING "
            "here with the reason.",
        )


@unittest.skipUnless(RUFF_INSTALLED, NO_RUFF)
class TheCeilingIsNotSlackTest(unittest.TestCase):
    """A limit far above everything in the tree enforces nothing.

    ruff passing proves no function exceeds the limit. It cannot notice a limit
    of 200, which would pass forever. So the measured worst case is compared
    against the configured one: they are allowed to differ, but not by enough
    for the rule to have stopped biting.
    """

    #: How far the configured ceiling may sit above the real worst case before
    #: the rule is decorative. One step, so a single function may be simplified
    #: without immediately failing this.
    SLACK = 1

    def test_the_configured_ceiling_tracks_the_real_worst_case(self) -> None:
        above = _complexities(CONVENTIONAL_THRESHOLD)
        self.assertTrue(
            above,
            f"ruff measured no function above {CONVENTIONAL_THRESHOLD}, so the ceiling "
            f"of {_configured_ceiling()} is far above the worst case. Lower it.",
        )
        worst = max(above.values())
        configured = _configured_ceiling()
        self.assertGreaterEqual(configured, worst, "the tree already violates its own ceiling")
        self.assertLessEqual(
            configured - worst,
            self.SLACK,
            f"max-complexity is {configured} and the worst function measures {worst}. "
            "Lower the ceiling to the worst case; a ceiling nothing approaches is "
            "a rule that has stopped applying.",
        )


@unittest.skipUnless(RUFF_INSTALLED, NO_RUFF)
class TheStatedCountIsRuffsTest(unittest.TestCase):
    """How many functions are above the threshold is a number prose states.

    pyproject.toml, chapter 13 and this module's docstring each stated it, and
    all three still said sixteen when ruff counted 19: the count moved and
    nothing read the prose. So each file states the count exactly once -- a
    rephrasing that drops it fails instead of escaping the check -- and the
    count it states is ruff's.
    """

    def test_each_file_states_ruffs_count_once(self) -> None:
        measured = len(_complexities(CONVENTIONAL_THRESHOLD))
        for path in COUNT_STATEMENTS:
            name = path.relative_to(REPO_ROOT).as_posix()
            with self.subTest(file=name):
                stated = _stated_counts(path.read_text(encoding="utf-8"))
                self.assertEqual(
                    len(stated),
                    1,
                    f"{name} states the count above {CONVENTIONAL_THRESHOLD} {len(stated)} "
                    "times; state it once, as '<count> functions sit above'",
                )
                self.assertEqual(
                    stated[0],
                    measured,
                    f"{name} states {stated[0]} functions above {CONVENTIONAL_THRESHOLD}; "
                    f"ruff counts {measured}",
                )


@pytest.mark.fast
class TheChapterStatesTheNumberTest(unittest.TestCase):
    """Raising the ceiling has to move prose too, not just one digit."""

    def test_chapter_13_names_the_configured_ceiling(self) -> None:
        # `assertIn` on a chapter dumps the chapter into the failure message.
        text = CHAPTER.read_text(encoding="utf-8")
        ceiling = str(_configured_ceiling())
        self.assertTrue("C901" in text, "chapter 13 does not mention C901")
        self.assertTrue(ceiling in text, f"chapter 13 does not state the ceiling {ceiling}")


if __name__ == "__main__":
    unittest.main()
