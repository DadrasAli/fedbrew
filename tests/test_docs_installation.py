"""docs/02-installation.md must match pyproject.toml and the dispatch table.

An installation chapter is the first thing a new reader follows and the one
whose errors cost most: a wrong Python floor or a missing extra sends them to
debug an environment rather than the code. It is also unusually easy to keep
correct mechanically, because everything it claims is declared in
pyproject.toml or in fedbrew/cli/dispatch.py.

Only the declarations are checked here. Whether `pip install -e .` succeeds on
a given machine is not something a unit test can answer, and the chapter's
four verification steps exist so a reader can answer it themselves.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

import pytest
from docs_sections import section_of, table_after

from fedbrew.cli.dispatch import COMMANDS

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER = REPO_ROOT / "docs" / "02-installation.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _chapter_text() -> str:
    return CHAPTER.read_text(encoding="utf-8")


def _project() -> dict:
    """The three pieces of [project] this chapter documents.

    Hand-parsed rather than read with tomllib, which is 3.11+ while
    pyproject declares requires-python >=3.10 -- a guard for the
    installation chapter has to run on the floor that chapter states.
    Only the two list shapes actually present in this file are handled,
    and each is asserted non-empty by its caller, so a format change
    surfaces as a failure rather than as an empty pass.
    """

    text = PYPROJECT.read_text(encoding="utf-8")

    floor = re.search(r'^requires-python\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    assert floor is not None, "pyproject has no requires-python"

    def _list(name: str, source: str) -> list[str]:
        match = re.search(rf"^{re.escape(name)}\s*=\s*(\[.*?^\])", source, re.M | re.S)
        if match is None:
            return []
        # ast.literal_eval over the bracketed block, with comment lines
        # stripped: they are the only non-literal content pyproject uses.
        body = "\n".join(
            line for line in match.group(1).splitlines() if not line.strip().startswith("#")
        )
        return list(ast.literal_eval(body))

    optional_start = text.index("[project.optional-dependencies]")
    optional_end = text.find("\n[", optional_start + 1)
    optional_text = text[optional_start : optional_end if optional_end > 0 else len(text)]
    extras = {
        name: _list(name, optional_text)
        for name in re.findall(r"^(\w+)\s*=\s*\[", optional_text, flags=re.MULTILINE)
    }

    return {
        "requires-python": floor.group(1),
        "dependencies": _list("dependencies", text[:optional_start]),
        "optional-dependencies": extras,
    }


def _requirement_name(specifier: str) -> str:
    """ "huggingface-hub>=0.34,<1" -> "huggingface-hub"."""

    return re.split(r"[<>=!\[;\s]", specifier, maxsplit=1)[0]


#: A chapter may spell a small count instead of writing a digit, so a check on
#: one has to accept both forms -- but only for the *true* value. The previous
#: form of the count checks below normalised the other way,
#: `text.replace("Nineteen", str(len(run_json)))`, which rewrote whichever word
#: the chapter happened to use into the number under test. That cannot fail: the
#: chapter said "Nineteen" while run.json carried twenty keys, the substitution
#: produced "20 top-level keys", and the assertion passed.
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


class ChapterShapeTest(unittest.TestCase):
    def test_present_and_agent_facing(self) -> None:
        self.assertTrue(CHAPTER.is_file())
        self.assertIn("\n## For agents\n", _chapter_text())


class RequirementsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _chapter_text()
        self.project = _project()

    def test_the_python_floor_is_the_declared_one(self) -> None:
        declared = self.project["requires-python"]
        self.assertIn(
            f"`{declared}`",
            self.text,
            f"pyproject requires-python is {declared!r}; the chapter says something else",
        )

    def test_every_core_dependency_is_named(self) -> None:
        # Scoped to section 1's *table*, not to section 1: `torch` appears a
        # dozen times in this chapter and twice in that section's own prose, so
        # neither the whole chapter nor the whole section can notice the
        # requirement row losing it. Only the table can.
        requirements = table_after(self.text, "## 1. Requirements")
        for specifier in self.project["dependencies"]:
            name = _requirement_name(specifier)
            with self.subTest(dependency=name):
                self.assertIn(f"`{name}`", requirements)

    def test_every_extra_is_documented_with_its_contents(self) -> None:
        extras = self.project["optional-dependencies"]
        self.assertTrue(extras, "no extras parsed from pyproject; the parse is broken")
        table = section_of(self.text, "### 2.1 Extras")
        for extra, specifiers in extras.items():
            with self.subTest(extra=extra):
                self.assertIn(
                    f'pip install -e ".[{extra}]"',
                    table,
                    f"the chapter must show how to install the {extra} extra",
                )
                for specifier in specifiers:
                    name = _requirement_name(specifier)
                    self.assertIn(
                        f"`{name}`",
                        table,
                        f"{name} is in the {extra} extra but section 2.1 does not name it",
                    )

    def test_no_extra_is_invented(self) -> None:
        offered = set(re.findall(r'pip install -e "\.\[([\w,]+)\]"', self.text))
        named: set[str] = set()
        for group in offered:
            named |= set(group.split(","))
        # Anti-vacuity, the same measure tests/test_cli_commands_exist.py takes:
        # without it a chapter that stopped showing install commands at all
        # would satisfy this check by offering nothing.
        self.assertTrue(
            named,
            "the chapter offers no extras at all; this check would pass vacuously",
        )
        self.assertEqual(
            named - set(_project()["optional-dependencies"]),
            set(),
            "the chapter offers an extra pyproject does not define",
        )


class SubcommandTableTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _chapter_text()

    def test_every_subcommand_has_a_row(self) -> None:
        for command in COMMANDS:
            with self.subTest(command=command):
                # re.MULTILINE, not assertRegex, which cannot pass flags: ^
                # would otherwise anchor to the whole file.
                self.assertIsNotNone(
                    re.search(
                        rf"^\| `{re.escape(command)}` \| ",
                        self.text,
                        flags=re.MULTILINE,
                    ),
                    f"{command} has no row in the command table",
                )

    def test_the_stated_count_is_right(self) -> None:
        self.assertTrue(
            _states_count(self.text, len(COMMANDS), "subcommands"),
            f"the chapter must say {len(COMMANDS)} subcommands, which is len(COMMANDS)",
        )


class EnvironmentVariableTest(unittest.TestCase):
    """Every FL_* and offline variable the code reads must have a row."""

    def test_every_variable_the_package_reads_has_a_row(self) -> None:
        """The direction the chapter's table was missing.

        test_documented_variables_are_read_somewhere below checks chapter ->
        code: nothing documented is unread. Nothing checked code -> chapter, so
        deleting FL_CACHE_ROOT's row went unnoticed even with the name gone
        from the chapter entirely. docs/00 says a guard diffs both ways.
        """

        table = table_after(_chapter_text(), "## 3. Environment variables")
        package = "\n".join(
            path.read_text(encoding="utf-8") for path in (REPO_ROOT / "fedbrew").rglob("*.py")
        )
        # Every FL_* literal, not only the direct environ[...] forms: the
        # resolvers pass these names to a helper, so a narrower pattern found
        # one of the four and this check passed while FL_CACHE_ROOT's row was
        # deleted.
        read = set(re.findall(r"""["'](FL_\w+)["']""", package))
        self.assertTrue(read, "no FL_* variable found in the package; the scan is broken")
        for name in sorted(read):
            with self.subTest(variable=name):
                self.assertIn(
                    f"`{name}`",
                    table,
                    f"{name} is read by the package but section 3's table omits it",
                )

    def test_documented_variables_are_read_somewhere(self) -> None:
        text = _chapter_text()
        documented = set(re.findall(r"^\| `([A-Z_]+)` \| ", text, flags=re.MULTILINE))
        self.assertTrue(documented, "no environment-variable rows found")

        sources = "\n".join(
            path.read_text(encoding="utf-8") for path in (REPO_ROOT / "fedbrew").rglob("*.py")
        )
        slurm = REPO_ROOT / "SLURMs"
        if slurm.is_dir():
            sources += "\n".join(
                path.read_text(encoding="utf-8") for path in slurm.rglob("*") if path.is_file()
            )
        unread = sorted(name for name in documented if name not in sources)
        self.assertEqual(
            unread,
            [],
            "the chapter documents variables nothing reads",
        )

    def test_cublas_is_the_only_one_fedbrew_sets(self) -> None:
        """The chapter's central claim about environment variables."""

        package = "\n".join(
            path.read_text(encoding="utf-8") for path in (REPO_ROOT / "fedbrew").rglob("*.py")
        )
        assignments = set(re.findall(r'environ\.setdefault\(\s*[\w"]*?["\']?(\w+)', package))
        assignments |= set(re.findall(r'environ\[["\'](\w+)["\']\]\s*=', package))
        # setdefault is called with the module constant, so resolve it.
        from fedbrew.core.runtime_setup import _CUBLAS_WORKSPACE_CONFIG

        self.assertEqual(_CUBLAS_WORKSPACE_CONFIG, "CUBLAS_WORKSPACE_CONFIG")
        self.assertIn("CUBLAS_WORKSPACE_CONFIG", _chapter_text())
        stray = assignments - {"_CUBLAS_WORKSPACE_CONFIG", "CUBLAS_WORKSPACE_CONFIG"}
        self.assertEqual(
            stray,
            set(),
            "something in fedbrew/ now writes an environment variable other "
            "than CUBLAS_WORKSPACE_CONFIG; the chapter says it is the only one",
        )


class CitedPathsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _chapter_text()

    def test_cited_modules_exist(self) -> None:
        cited = set(re.findall(r"`(fedbrew/[\w/]+\.py)", self.text))
        self.assertTrue(cited)
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
        self.assertEqual(missing, [])

    def test_cited_directories_exist(self) -> None:
        cited = set(re.findall(r"`([\w]+/[\w/]*)`", self.text))
        directories = [name for name in cited if name.endswith("/")]
        missing = sorted(path for path in directories if not (REPO_ROOT / path).is_dir())
        self.assertEqual(missing, [])

    def test_cited_tests_exist(self) -> None:
        cited = set(re.findall(r"`(tests/test_\w+\.py)`", self.text))
        self.assertTrue(cited)
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
        self.assertEqual(missing, [])

    def test_cited_configs_exist(self) -> None:
        cited = set(re.findall(r"(configs/[\w/]+\.yaml)", self.text))
        self.assertTrue(cited)
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
