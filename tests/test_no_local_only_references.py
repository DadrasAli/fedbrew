"""Nothing in the shipped tree may cite a path inside AUDIT/.

`AUDIT/` is gitignored. It exists on the machine the audits were run on and
nowhere else, so a citation to it is a pointer to a file the reader cannot
open -- and the reports describe the pre-fix state anyway, with several of
their findings resolved by deleting the option entirely rather than fixing it.
A reader who follows one is worse off than a reader given nothing, because the
citation promised an explanation.

This started as a docs-only rule and was enforced only over `docs/`. The tree
carried 34 citations outside it: 2 in `fedbrew/`, 31 in `tests/` and one in a
shipped generator config, several of them in the exact modules the chapters
send a reader to read. Chapter 14 names the same defect for commit messages,
for the same reason -- the reference outlives the reader's ability to resolve
it -- so the rule is enforced here over everything a clone contains.

What replaces a citation is a self-contained sentence saying what the defect
was. That is strictly more useful than the identifier even to someone holding
the reports, and it survives the reports being deleted.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Directories that are not part of a clone, or not ours to police.
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

SCANNED_SUFFIXES = (".py", ".sh", ".md", ".yaml", ".yml", ".toml", ".cff", ".csv")

#: A filename character after the slash. Naming the directory itself is
#: allowed and necessary -- docs/00-index.md warns agents off it, this module
#: explains itself, and .gitignore has to list it -- so only a path *into* it
#: is an offence.
_CITATION = re.compile(r"AUDIT/[\w.-]")

#: Files whose subject is the rule itself.
_ALLOWED = frozenset({"tests/test_no_local_only_references.py", ".gitignore"})


def _scanned_files() -> list[Path]:
    found = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in SCANNED_SUFFIXES:
            continue
        if SKIPPED_DIRS & set(path.relative_to(REPO_ROOT).parts):
            continue
        found.append(path)
    return sorted(found)


class NoLocalOnlyReferencesTest(unittest.TestCase):
    def test_the_scan_reaches_the_tree_it_claims_to_check(self) -> None:
        """Guards the guard: a scan that finds nothing would pass silently."""

        scanned = {path.relative_to(REPO_ROOT).as_posix() for path in _scanned_files()}
        for required in (
            "README.md",
            "pyproject.toml",
            "docs/00-index.md",
            "fedbrew/data/generate.py",
            "tests/test_regression_baseline.py",
            "data/configs/femnist_natural.yaml",
            "SLURMs/example_sweep.sh",
            "FINDINGS.csv",
        ):
            with self.subTest(path=required):
                self.assertIn(required, scanned)

    def test_nothing_in_the_tree_cites_an_audit_path(self) -> None:
        offenders = []
        for path in _scanned_files():
            relative = path.relative_to(REPO_ROOT).as_posix()
            if relative in _ALLOWED:
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if _CITATION.search(line):
                    offenders.append(f"{relative}:{number}  {line.strip()}")
        self.assertEqual(
            offenders,
            [],
            "these cite a path inside AUDIT/, which is gitignored and absent "
            "from a fresh clone. Replace each with a sentence saying what the "
            f"defect was; the identifier resolves to nothing: {offenders}",
        )


if __name__ == "__main__":
    unittest.main()
