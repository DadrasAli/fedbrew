"""Every ``fedbrew <subcommand> --flag`` this repository ships must be real.

The sibling guard, tests/test_cli_commands_exist.py, checks the subcommand.
This checks what follows it. The two defects are the same defect: a flag that
argparse does not define fails the same way a retired console script does --
at the moment someone copies the line and runs it -- and the documentation set
will accumulate far more flags than commands, so the flag surface is where the
drift will happen next.

Flags are read from the parsers themselves, not from a list kept here. Six of
the eleven subcommands build their parser in ``parse_args`` and five build it
inside ``main``, so neither name is enough on its own; instead
``ArgumentParser.parse_args`` is intercepted, which is the one point every
argparse CLI reaches after its parser is fully constructed and before it does
any work. A subcommand that never reaches argparse -- the three that take a
positional or nothing at all -- declares no flags, so documenting one for it
fails here, which is correct.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import importlib
import io
import re
import sys
import unittest
from pathlib import Path

import pytest

from fedbrew.cli.dispatch import COMMANDS

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Kept in step with tests/test_cli_commands_exist.py; see its rationale for
#: each entry.
SCANNED_SUFFIXES = (".py", ".sh", ".md", ".yaml", ".yml")
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

#: ``fedbrew`` followed by a subcommand-shaped token.
_INVOCATION = re.compile(r"\bfedbrew ([a-z][a-z0-9-]*)")

#: A long option. The lookbehind keeps it from matching inside a word or
#: splitting an em-dash-joined token, and ``--`` alone is not a flag.
_FLAG = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")


class _ParserReached(Exception):
    """Carries the option strings of a fully built parser."""

    def __init__(self, options: frozenset[str]) -> None:
        super().__init__("parser reached")
        self.options = options


@functools.cache
def _declared_flags(command: str) -> frozenset[str]:
    """Every option string the subcommand's argparse parser defines.

    Cached: importing ``fedbrew.core.runner`` pulls in torch, and the checks
    below ask for every subcommand's flags more than once.
    """

    target, _ = COMMANDS[command]
    module = importlib.import_module(target)

    def capture(self: argparse.ArgumentParser, *args: object, **kwargs: object) -> None:
        options: set[str] = set()
        for action in self._actions:
            options.update(action.option_strings)
        raise _ParserReached(frozenset(options))

    real_parse_args = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = capture  # type: ignore[method-assign]
    original_argv = sys.argv
    sys.argv = [f"fedbrew {command}"]
    try:
        # Both entry shapes are tried, and neither runs the command: parse_args
        # is where the interception fires, and it fires before main() gets an
        # answer to act on.
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            entry = getattr(module, "parse_args", None)
            if entry is not None:
                entry([])
            else:
                module.main()
        return frozenset()
    except _ParserReached as reached:
        return reached.options
    except SystemExit:
        # No argparse parser on this path: the subcommand takes a positional
        # or no arguments, and printed its usage instead.
        return frozenset()
    finally:
        argparse.ArgumentParser.parse_args = real_parse_args  # type: ignore[method-assign]
        sys.argv = original_argv


@functools.lru_cache(maxsize=1)
def _shipped_files() -> tuple[Path, ...]:
    found = []
    for path in REPO_ROOT.rglob("*"):
        if path.suffix not in SCANNED_SUFFIXES or not path.is_file():
            continue
        if SKIPPED_DIRS.intersection(path.relative_to(REPO_ROOT).parts):
            continue
        found.append(path)
    return tuple(sorted(found))


@functools.lru_cache(maxsize=1)
def _flag_uses() -> tuple[tuple[Path, int, str, str], ...]:
    """Every (file, line, subcommand, flag) the shipped tree writes.

    A flag belongs to the nearest ``fedbrew <subcommand>`` to its left, so two
    invocations piped together on one line are attributed separately. Shell
    line continuations are joined first, because a wrapped command puts its
    flags on later lines than its subcommand.
    """

    uses = []
    for path in _shipped_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        # Join continuations, then count lines on the joined text so the
        # reported number still points at the subcommand.
        for number, line in enumerate(text.replace("\\\n", " ").splitlines(), start=1):
            invocations = list(_INVOCATION.finditer(line))
            for index, match in enumerate(invocations):
                command = match.group(1)
                if command not in COMMANDS:
                    continue
                end = invocations[index + 1].start() if index + 1 < len(invocations) else len(line)
                for flag in _FLAG.finditer(line[match.end() : end]):
                    uses.append((path, number, command, flag.group(1)))
    return tuple(uses)


class DeclaredFlagsTests(unittest.TestCase):
    def test_every_subcommand_is_introspectable(self) -> None:
        """The capture must reach a real parser, or report none deliberately."""

        without_parser = {command for command in COMMANDS if not _declared_flags(command)}
        # These three take a positional or nothing; every other subcommand must
        # expose flags, or the capture silently stopped working and this whole
        # guard would pass by matching nothing.
        self.assertEqual(
            without_parser,
            {"inspect-data", "check-hpc", "list-common-datasets"},
            "the set of flagless subcommands changed; if a subcommand gained "
            "or lost its argparse parser, update this expectation deliberately",
        )

    def test_run_declares_the_flags_the_docs_lean_on(self) -> None:
        """--validate-only is quoted by more than one chapter."""

        flags = _declared_flags("run")
        for expected in ("--config", "--validate-only", "--resume-latest", "--tag"):
            with self.subTest(flag=expected):
                self.assertIn(expected, flags)


class ShippedFlagsTests(unittest.TestCase):
    def test_the_scan_reaches_the_tree_it_claims_to_check(self) -> None:
        """A scan-based guard that silently matches nothing always passes."""

        uses = _flag_uses()
        self.assertGreater(len(uses), 15, "no flag uses found; the scan is broken")
        self.assertGreater(
            len({path for path, _, _, _ in uses}),
            3,
            "flag uses found in too few files for the scan to be checking much",
        )

    def test_every_documented_flag_is_one_argparse_defines(self) -> None:
        declared = {command: _declared_flags(command) for command in COMMANDS}
        offenders = [
            f"{path.relative_to(REPO_ROOT)}:{number}  fedbrew {command} {flag}"
            for path, number, command, flag in _flag_uses()
            if flag not in declared[command]
        ]
        self.assertEqual(
            offenders,
            [],
            "these flags are written down but no argparse parser defines them; "
            "a documented flag that does not exist fails exactly where a "
            "documented command that does not exist fails",
        )


if __name__ == "__main__":
    unittest.main()
