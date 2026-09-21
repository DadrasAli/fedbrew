"""Every ``fedbrew ...`` invocation this repository ships must be runnable.

The eleven ``fedbrew-*`` console scripts were collapsed into a single
``fedbrew`` entry point with subcommands. Three separate places went on
shipping the old spelling afterwards -- config comments, two READMEs, and ten
error messages inside ``fedbrew/`` that told a user to run a command that did
not exist -- because nothing checked. A stale command in an error message is
worse than a stale comment: it is read exactly when someone is already stuck.

This is the check. It walks the shipped tree rather than a hand-kept list, so
a new file cannot quietly reintroduce the defect.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import pytest

from fedbrew.cli.dispatch import COMMANDS

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Text formats that can carry a command a human will copy and run.
SCANNED_SUFFIXES = (".py", ".sh", ".md", ".yaml", ".yml")

#: The subset of SCANNED_SUFFIXES the repository actually uses. Split out so
#: the coverage check below stays honest: ".yml" is scanned in case a file
#: ever arrives spelled that way, and asserting it is present would fail.
PRESENT_SUFFIXES = (".py", ".sh", ".md", ".yaml")

#: Directory names never descended into. Generated trees (``outputs``,
#: ``data/generated``) hold copies of configs whose comments this test already
#: checked at their source, and local-only notes (``AUDIT``) are not shipped.
SKIPPED_DIRS = frozenset(
    {
        ".git",
        ".hypothesis",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "AUDIT",
        "__pycache__",
        "build",
        "dist",
        "generated",
        "logs_and_errs",
        "outputs",
        "raw",
        "venv",
    }
)

#: Words that follow "fedbrew" in prose rather than starting a command line.
#: Deliberately tiny and explicit: a new entry here is a claim that the text is
#: prose, and making that claim is the review moment this test exists to force.
NON_COMMAND_WORDS = frozenset(
    {
        # "Run a fedbrew experiment." -- the CLI's own description string.
        "experiment",
        # "CONDA_ENV_NAME=fedbrew bash SLURMs/..." -- an environment value,
        # not an invocation.
        "bash",
    }
)

#: ``fedbrew`` followed by a subcommand-shaped token. Both separators are
#: matched: a space is the current form, a hyphen is the console-script form
#: that no longer resolves to anything.
_INVOCATION = re.compile(r"\bfedbrew([ -])([a-z][a-z0-9-]*)")

#: The spelling from before the project was named fedbrew: ``fl``, a hyphen,
#: then the command. The pattern above keys on ``fedbrew`` and never reads it,
#: which is how chapter 07 shipped a pre-rename validate command -- one with no
#: subcommand counterpart at all -- and a test docstring a pre-rename spelling
#: of ``eval-medmcqa``, both past every check in this file.
_PRE_RENAME_INVOCATION = re.compile(r"\bfl(-)([a-z][a-z0-9-]*)")

#: Words that follow the pre-rename prefix without being a command. The same
#: claim NON_COMMAND_WORDS makes, held to the same staleness check.
PRE_RENAME_NON_COMMAND_WORDS = frozenset(
    {
        # FINDINGS.md's note on the rename names the old repository, whose
        # name is that prefix followed by this word.
        "codebase",
    }
)

#: An HTML tag: ``<``, then a letter or ``/`` -- a real tag never starts with
#: anything else -- up to the next ``>``. Anchored on the letter/slash rather
#: than bare ``<[^>]*>`` so a stray comparison in a .py/.sh line, ``x <
#: fedbrew_threshold``, is never mistaken for a tag: the space after ``<``
#: cannot match ``[a-zA-Z/]``.
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")

#: An attribute's quoted value inside a tag -- ``src="..."`` or ``alt='...'``.
_ATTRIBUTE_VALUE = re.compile(r"=(\".*?\"|'.*?')")

#: A line invoking the linter. Anchored at the start, after optional prompt or
#: indentation, so it matches the command itself and not a sentence about it.
_LINT_COMMAND = re.compile(r"^[\s$>]*ruff\b")

# Built rather than written out, so this file does not itself contain the
# literal the hyphen check forbids.
_OLD_PREFIX = "fedbrew" + "-"
_PRE_RENAME_PREFIX = "fl" + "-"

#: The gate command, assembled the same way and for the same reason: the
#: package directory followed by the next linted directory is precisely the
#: false positive under test, so writing it out would trip the scan here.
_GATE_LINE = "ruff check tests tools " + "fedbrew" + " examples"

#: A word that is not and will not become a subcommand.
_NOT_A_SUBCOMMAND = "fedbrew" + " nonesuch"


def _without_html_attribute_values(line: str) -> str:
    """``line`` with every HTML tag's attribute values blanked out.

    A ``src`` or ``alt`` value is markup, not prose a human would copy and
    run. README.md's own logo tag is the motivating case: its image
    filename hyphenates the project name the way a retired console-script
    invocation did, and its alt text repeats the bare project name next to
    it -- both inside one ``<img ...>`` tag, neither of them a command line.

    Scoped to inside an actual ``<...>`` tag, not every ``word="value"`` on
    the line, so a shell variable assignment shaped the same way -- prose
    the scan must still read -- is not swept up by the same rule.
    """

    return _HTML_TAG.sub(lambda match: _ATTRIBUTE_VALUE.sub("=", match.group(0)), line)


def _is_a_lint_command(line: str) -> bool:
    """Whether ``line`` invokes ruff, whose arguments are paths.

    The gate command names the package directory and then the next directory
    to lint, which reads to this scan as the project name followed by a
    subcommand-shaped word. It is neither: both are paths being linted, and
    nothing on such a line is an invocation of this project's CLI at all.

    Whole-line rather than argument-by-argument, because such a line cannot
    carry an invocation in the first place, and because the alternative --
    deciding which of ruff's arguments are paths -- is a second parser to keep
    right. Anchored at the start, so prose *about* the command is still read.
    """

    return bool(_LINT_COMMAND.match(line))


def _shipped_files() -> list[Path]:
    found = []
    for path in REPO_ROOT.rglob("*"):
        if path.suffix not in SCANNED_SUFFIXES or not path.is_file():
            continue
        if SKIPPED_DIRS.intersection(path.relative_to(REPO_ROOT).parts):
            continue
        found.append(path)
    return sorted(found)


def _invocations(
    skip: frozenset[Path] = frozenset(),
    pattern: re.Pattern[str] = _INVOCATION,
) -> list[tuple[Path, int, str, str]]:
    """Every (file, line number, separator, subcommand) ``pattern`` finds in the shipped tree."""

    hits = []
    for path in _shipped_files():
        if path in skip:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            if _is_a_lint_command(line):
                continue
            scanned = _without_html_attribute_values(line)
            for separator, word in pattern.findall(scanned):
                hits.append((path, number, separator, word))
    return hits


class ShippedCommandsTests(unittest.TestCase):
    def test_the_scan_reaches_the_tree_it_claims_to_check(self) -> None:
        """A scan-based guard that silently matches nothing always passes."""

        files = _shipped_files()
        self.assertGreater(len(files), 100, "file walk found almost nothing")

        scanned = {path.suffix for path in files}
        for suffix in PRESENT_SUFFIXES:
            with self.subTest(suffix=suffix):
                self.assertIn(suffix, scanned)

        # The package itself, the configs, and the example submit scripts all
        # have to be inside the walk for the check below to mean anything.
        parents = {path.relative_to(REPO_ROOT).parts[0] for path in files}
        for expected in ("fedbrew", "configs", "SLURMs", "tests"):
            with self.subTest(directory=expected):
                self.assertIn(expected, parents)

        self.assertGreater(len(_invocations()), 20, "no invocations found")

    def test_no_file_ships_the_retired_console_script_spelling(self) -> None:
        offenders = [
            f"{path.relative_to(REPO_ROOT)}:{number}  {_OLD_PREFIX}{word}"
            for path, number, separator, word in _invocations()
            if separator == "-"
        ]
        self.assertEqual(
            offenders,
            [],
            f"{_OLD_PREFIX}* console scripts no longer exist; write the subcommand form instead",
        )

    def test_every_invocation_names_a_subcommand_dispatch_knows(self) -> None:
        known = set(COMMANDS) | NON_COMMAND_WORDS
        offenders = [
            f"{path.relative_to(REPO_ROOT)}:{number}  fedbrew {word}"
            for path, number, _, word in _invocations()
            if word not in known
        ]
        self.assertEqual(
            offenders,
            [],
            f"unknown subcommand; dispatch knows {sorted(COMMANDS)}",
        )

    def test_the_prose_allowlist_does_not_mask_a_real_subcommand(self) -> None:
        """Allowlisting a genuine command name would silence a genuine bug."""

        self.assertEqual(NON_COMMAND_WORDS & set(COMMANDS), set())

    def test_every_allowlisted_word_still_appears_in_the_tree(self) -> None:
        """A word kept after its text is gone is a stale exemption.

        This file is excluded from its own scan here, and only here: the
        comments beside NON_COMMAND_WORDS quote the very phrases being looked
        for, so counting them would let the allowlist justify itself.
        """

        seen = {word for _, _, _, word in _invocations(skip=frozenset({Path(__file__).resolve()}))}
        for word in sorted(NON_COMMAND_WORDS):
            with self.subTest(word=word):
                self.assertIn(word, seen)


class PreRenameCommandsTest(unittest.TestCase):
    """A console script from before the rename resolves to nothing either.

    Every check above reads only text that begins ``fedbrew``, so a command
    spelled the way it was before the project had that name passed all of
    them without being looked at.
    """

    def test_no_file_ships_the_pre_rename_command_spelling(self) -> None:
        offenders = [
            f"{path.relative_to(REPO_ROOT)}:{number}  {_PRE_RENAME_PREFIX}{word}"
            for path, number, _, word in _invocations(pattern=_PRE_RENAME_INVOCATION)
            if word not in PRE_RENAME_NON_COMMAND_WORDS
        ]
        self.assertEqual(
            offenders,
            [],
            f"{_PRE_RENAME_PREFIX}* console scripts do not exist; write the subcommand form",
        )

    def test_the_pattern_reads_a_pre_rename_command(self) -> None:
        line = f"Then run `{_PRE_RENAME_PREFIX}report` on the run directory."
        self.assertEqual(_PRE_RENAME_INVOCATION.findall(line), [("-", "report")])

    def test_the_allowlist_does_not_mask_a_real_subcommand(self) -> None:
        self.assertEqual(PRE_RENAME_NON_COMMAND_WORDS & set(COMMANDS), set())

    def test_every_allowlisted_word_still_appears_in_the_tree(self) -> None:
        """Excludes this file, for the reason the check above it does."""

        seen = {
            word
            for _, _, _, word in _invocations(
                skip=frozenset({Path(__file__).resolve()}), pattern=_PRE_RENAME_INVOCATION
            )
        }
        for word in sorted(PRE_RENAME_NON_COMMAND_WORDS):
            with self.subTest(word=word):
                self.assertIn(word, seen)


class HTMLAttributeValuesAreNotProseTest(unittest.TestCase):
    """A ``src`` or ``alt`` value is markup, not something a human copies and
    runs. ``NON_COMMAND_WORDS`` is the wrong tool for this: it exempts a
    *word* everywhere in the tree, not a *location* on one line, and
    exempting the image's own word that way would also wave through a
    genuine stale invocation using the same word anywhere else in the
    repository."""

    def test_a_path_containing_the_word_fedbrew_does_not_trip_it(self) -> None:
        line = '<img src="assets/fedbrew-logo.svg" alt="fedbrew" width="360">'
        self.assertEqual(_INVOCATION.findall(_without_html_attribute_values(line)), [])

    def test_an_invocation_outside_a_tag_is_still_caught(self) -> None:
        """The exemption is scoped to inside a tag, not to the whole line."""

        line = "See the docs, then run <code>fedbrew run --config x.yaml</code>."
        self.assertEqual(
            _INVOCATION.findall(_without_html_attribute_values(line)),
            [(" ", "run")],
        )

    def test_a_shell_assignment_is_not_mistaken_for_a_tag(self) -> None:
        """No ``<`` or ``>`` on the line, so nothing here is a tag at all --
        an equals sign and a quote are not enough on their own.

        Built rather than written out, like ``_OLD_PREFIX`` itself, so this
        line does not trip the very check it exists to exercise.
        """

        line = f'CMD="{_OLD_PREFIX}run --config x.yaml"'
        self.assertEqual(
            _INVOCATION.findall(_without_html_attribute_values(line)),
            [("-", "run")],
        )


class LintPathArgumentsAreNotInvocationsTest(unittest.TestCase):
    """In the gate command the package directory is a path being linted, and
    the word after it is the next directory. Same reasoning as the tag above:
    the exemption is a *location* -- ruff's argument list -- and not a word
    waved through everywhere, which is what ``NON_COMMAND_WORDS`` would have
    done to every occurrence of ``examples`` in the tree."""

    def test_the_gate_command_does_not_trip_it(self) -> None:
        self.assertTrue(_is_a_lint_command(_GATE_LINE))
        self.assertTrue(_is_a_lint_command(f"          {_GATE_LINE}"))
        self.assertNotEqual(_INVOCATION.findall(_GATE_LINE), [])

    def test_prose_about_the_command_is_still_read(self) -> None:
        """Anchored at the start, so a sentence mentioning ruff is not skipped."""

        line = f"Before running ruff, run {_NOT_A_SUBCOMMAND} to check."
        self.assertFalse(_is_a_lint_command(line))
        self.assertEqual(_INVOCATION.findall(line), [(" ", "nonesuch")])

    def test_another_tools_line_is_not_skipped(self) -> None:
        """Scoped to ruff, not to every command line that has arguments."""

        self.assertFalse(_is_a_lint_command(f"python -m pytest && {_NOT_A_SUBCOMMAND}"))

    def test_the_shipped_gate_lines_are_the_reason_it_exists(self) -> None:
        """The exemption earns itself: real lines in the tree need it.

        Mirrors the check below ``NON_COMMAND_WORDS`` -- an exemption nothing
        in the tree exercises is an exemption that should be deleted.
        """

        needed = [
            f"{path.relative_to(REPO_ROOT)}:{number}"
            for path in _shipped_files()
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if _is_a_lint_command(line) and _INVOCATION.search(line)
        ]
        self.assertNotEqual(needed, [])


if __name__ == "__main__":
    unittest.main()
