"""The lint command covers every tracked ``.py``, and says the same thing everywhere.

Two failures, one shape. ``ruff check tests tools fedbrew`` named three of the
four directories holding Python, so ``examples/`` was unlinted from the commit
that created it -- not by a decision, but because the path list is a literal in
a shell step and adding a directory does not touch it. And that literal is
copied into five other files: CI, `CONTRIBUTING.md`, `README.md` and two
chapters all quote the gate, so widening it in one place leaves contributors
running a weaker check than CI and finding out on push.

So both directions are checked here. Forward: every tracked ``.py`` is under a
path CI lints. Backward: every copy of the command in the tree names the same
paths CI does. Neither is a list kept in this file -- the paths are read out of
the workflow, and the tree is read from ``git ls-files`` -- because a list here
would be a seventh copy of the thing that drifted.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tests.yml"

#: Kept in step with tests/test_cli_commands_exist.py; see its rationale.
SCANNED_SUFFIXES = (".md", ".yml", ".yaml", ".sh")
SKIPPED_DIRS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "AUDIT",
        "outputs",
        "logs_and_errs",
        "data",
    }
)

_RUFF_INVOCATION = re.compile(r"ruff (?:format --check|check) ([^\n`]*)")


def _paths_named_by(invocation: str) -> tuple[str, ...]:
    return tuple(word for word in invocation.split() if not word.startswith("-"))


def _workflow_lint_paths() -> tuple[str, ...]:
    text = WORKFLOW.read_text(encoding="utf-8")
    found = {_paths_named_by(match) for match in _RUFF_INVOCATION.findall(text)}
    if len(found) != 1:
        raise AssertionError(f"the workflow's two ruff halves disagree: {sorted(found)}")
    return found.pop()


def _scanned_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in SCANNED_SUFFIXES:
            continue
        if SKIPPED_DIRS.intersection(path.relative_to(REPO_ROOT).parts):
            continue
        files.append(path)
    return files


#: Skipped wherever they appear, because they are caches and build products
#: and can sit at any depth.
_CACHE_DIRS = frozenset({".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"})

#: Skipped only as a top-level directory. `data` is the datasets tree; matching
#: it at any depth would also skip `fedbrew/data/`, which is 29 tracked modules
#: and the largest subpackage in the tree. SKIPPED_DIRS above is matched against
#: every part, which is safe for the suffixes it scans and would not be here --
#: test_the_walk_covers_everything_the_listing_does is what caught that.
_TOP_LEVEL_ONLY = frozenset(
    {"AUDIT", "outputs", "logs_and_errs", "data", ".venv", "venv", "build", "dist"}
)


def _python_files_by_walking() -> list[str]:
    """Every `.py` in the tree, for when there is no repository to ask.

    A release archive has no `.git`, and "tracked" has no meaning in one --
    but every file in it was tracked, which is what makes the walk the right
    answer there rather than a weaker one. In a checkout the walk is a
    superset of the listing, and the test below pins that direction so this
    path cannot quietly start missing files.
    """

    files: list[str] = []
    for path in REPO_ROOT.rglob("*.py"):
        relative = path.relative_to(REPO_ROOT)
        if _CACHE_DIRS.intersection(relative.parts):
            continue
        if relative.parts[0] in _TOP_LEVEL_ONLY:
            continue
        if any(part.endswith(".egg-info") for part in relative.parts):
            continue
        files.append(str(relative))
    return files


def _tracked_python_files() -> list[str]:
    """The tracked `.py` files, or every `.py` when there is no repository.

    This used to be `git ls-files` with `check=True` and nothing else, so the
    whole module failed from a release archive with a CalledProcessError --
    one of six tests in the suite that assumed the caller was standing in a
    checkout. The claim being guarded ("no Python in this tree escapes the
    lint command") is true of an archive too, and is worth checking there.
    """

    try:
        listed = subprocess.run(
            ["git", "ls-files", "*.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return _python_files_by_walking()
    return [line for line in listed.stdout.splitlines() if line]


@pytest.mark.fast
class TheCommandIsWrittenIdenticallyEverywhereTest(unittest.TestCase):
    def test_every_copy_names_the_paths_ci_lints(self) -> None:
        expected = _workflow_lint_paths()
        wrong: list[str] = []
        for path in _scanned_files():
            for invocation in _RUFF_INVOCATION.findall(path.read_text(encoding="utf-8")):
                named = _paths_named_by(invocation)
                if named and named != expected:
                    wrong.append(f"{path.relative_to(REPO_ROOT)}: {' '.join(named)}")
        self.assertEqual(wrong, [], f"lint command differs from CI's {' '.join(expected)}: {wrong}")

    def test_the_scan_finds_the_copies_it_is_meant_to_guard(self) -> None:
        """Guards the guard: a regex matching nothing would pass silently."""

        quoting = [
            str(path.relative_to(REPO_ROOT))
            for path in _scanned_files()
            if _RUFF_INVOCATION.search(path.read_text(encoding="utf-8"))
        ]
        self.assertIn("CONTRIBUTING.md", quoting)
        self.assertIn("README.md", quoting)
        self.assertIn("docs/13-testing.md", quoting)
        self.assertGreaterEqual(len(quoting), 5)


class EveryTrackedPythonFileIsLintedTest(unittest.TestCase):
    def test_no_tracked_module_falls_outside_the_linted_paths(self) -> None:
        """The direction that missed ``examples/``.

        A new top-level directory of Python is invisible to a literal path
        list, and stays invisible: nothing fails, the files are simply never
        checked.
        """

        roots = _workflow_lint_paths()
        unlinted = [
            name
            for name in _tracked_python_files()
            if not any(name == root or name.startswith(f"{root}/") for root in roots)
        ]
        self.assertEqual(unlinted, [], f"tracked but not linted by {' '.join(roots)}: {unlinted}")

    def test_the_enumeration_finds_the_tree(self) -> None:
        """Guards the guard: an empty listing would satisfy the check above."""

        found = _tracked_python_files()
        self.assertGreater(len(found), 100, "the file enumeration returned almost nothing")
        self.assertIn("fedbrew/core/loop.py", found)

    def test_the_walk_covers_everything_the_listing_does(self) -> None:
        """The fallback is only safe if it is a superset, so check it here.

        Nothing exercises the walk in CI, which runs in a checkout. Comparing
        the two where both are available is what keeps the archive path honest
        without a second CI job to run it in.
        """

        listed = subprocess.run(
            ["git", "ls-files", "*.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if listed.returncode != 0:  # pragma: no cover - the archive has no listing
            self.skipTest("no repository to compare the walk against")
        missing = sorted(set(listed.stdout.split()) - set(_python_files_by_walking()))
        self.assertEqual(missing, [], f"tracked but not found by the walk: {missing}")

    @pytest.mark.fast
    def test_every_linted_path_exists(self) -> None:
        for root in _workflow_lint_paths():
            self.assertTrue((REPO_ROOT / root).is_dir(), f"lint path does not exist: {root}")

    @pytest.mark.fast
    def test_examples_is_among_them(self) -> None:
        """The specific gap this file was added for, pinned by name."""

        self.assertIn("examples", _workflow_lint_paths())


if __name__ == "__main__":
    unittest.main()
