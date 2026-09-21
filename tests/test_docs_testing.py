"""docs/13-testing.md must name every guard, and every guard must be named.

This is the test that closes the loop. Fifteen chapters each have a guard;
this one guards the list of them, in both directions:

- a chapter with no companion guard fails here, because an unguarded chapter is
  how the previous documentation set drifted; and
- a `test_docs_*.py` file the chapter does not list fails here, because a guard
  nobody knows about is one nobody maintains.

The exemption list is explicit and small. An entry in it is a claim that a file
is deliberately not a per-chapter guard, and making that claim is the review
moment this test exists to force -- the same construction as
NON_COMMAND_WORDS in tests/test_cli_commands_exist.py.
"""

from __future__ import annotations

import ast
import fnmatch
import re
import unittest
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER = REPO_ROOT / "docs" / "13-testing.md"
DOCS = REPO_ROOT / "docs"
TESTS = REPO_ROOT / "tests"

#: The chapter spells small counts as words, so a check against a table's
#: length has to read them back.
_NUMBER_WORDS = {
    word: value
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}

#: Guards that protect documentation without belonging to one chapter. Each is
#: listed in the chapter's second table rather than its first.
CROSS_CUTTING_GUARDS = frozenset(
    {
        "test_cli_commands_exist.py",
        "test_cli_flags_exist.py",
        "test_docs_inventory_counts.py",
        "test_docs_references_resolve.py",
        "test_lint_covers_the_tree.py",
        "test_no_local_only_references.py",
        "test_optional_dependency_imports.py",
        "test_readme_is_an_overview.py",
        "test_matmul_precision.py",
        "test_divergence_metric_reachable.py",
        "test_metric_filter_scope.py",
        "test_shipped_config_explicitness.py",
    }
)

#: Chapters with no per-chapter guard of their own, and why. 00 is the index:
#: it lists the chapters, and every chapter's existence is already checked by
#: the chapter-to-guard mapping below, so a guard for it would only restate
#: that. Its own paths are checked by test_docs_references_resolve.py and the
#: counts in its inventory table by test_docs_inventory_counts.py.
CHAPTERS_WITHOUT_A_GUARD = frozenset({"00-index.md"})

#: chapter number -> the guard that owns it.
CHAPTER_GUARDS = {
    "01": "test_docs_architecture.py",
    "02": "test_docs_installation.py",
    "03": "test_docs_quickstart.py",
    "04": "test_docs_config_keys.py",
    "05": "test_docs_data_keys.py",
    "06": "test_docs_model_keys.py",
    "07": "test_docs_algorithms.py",
    "08": "test_docs_metric_names.py",
    "09": "test_docs_artifacts.py",
    "10": "test_docs_reproducibility.py",
    "11": "test_docs_performance.py",
    "12": "test_docs_extending.py",
    "13": "test_docs_testing.py",
    "14": "test_docs_working_on_fedbrew.py",
    "15": "test_docs_illustrative_examples.py",
}


def _chapter_text() -> str:
    return CHAPTER.read_text(encoding="utf-8")


def _chapters() -> list[str]:
    return sorted(path.name for path in DOCS.glob("*.md"))


def _doc_guards() -> set[str]:
    return {path.name for path in TESTS.glob("test_docs_*.py")}


class ChapterShapeTest(unittest.TestCase):
    def test_present_and_agent_facing(self) -> None:
        self.assertTrue(CHAPTER.is_file())
        self.assertIn("\n## For agents\n", _chapter_text())


class EveryChapterHasAGuardTest(unittest.TestCase):
    def test_the_documentation_set_is_complete(self) -> None:
        """00 through 15, no gaps."""

        expected = {f"{number:02d}" for number in range(16)}
        present = {name[:2] for name in _chapters()}
        self.assertEqual(
            present,
            expected,
            "the chapter set has a gap or an extra; the index's table and this "
            "mapping both assume 00 through 15",
        )

    def test_every_chapter_has_a_guard_that_exists(self) -> None:
        for chapter in _chapters():
            number = chapter[:2]
            if chapter in CHAPTERS_WITHOUT_A_GUARD:
                continue
            with self.subTest(chapter=chapter):
                guard = CHAPTER_GUARDS.get(number)
                self.assertIsNotNone(
                    guard,
                    f"{chapter} has no guard in the mapping. An unguarded "
                    "chapter is how the previous docs/ went stale.",
                )
                assert guard is not None
                self.assertTrue(
                    (TESTS / guard).is_file(),
                    f"{chapter}'s guard {guard} does not exist",
                )

    def test_the_chapter_names_each_guard_beside_its_chapter(self) -> None:
        text = _chapter_text()
        for number, guard in CHAPTER_GUARDS.items():
            with self.subTest(chapter=number):
                self.assertRegex(
                    text,
                    rf"`tests/{re.escape(guard)}`.*\| {number} \|",
                    f"the guard table must put {guard} against chapter {number}",
                )


class EveryGuardIsNamedTest(unittest.TestCase):
    def test_no_guard_is_unlisted(self) -> None:
        """The other direction: a guard nobody knows about."""

        known = set(CHAPTER_GUARDS.values()) | CROSS_CUTTING_GUARDS
        # The e2e guard is a second guard for chapter 03 rather than a chapter's
        # own, so it is listed in the chapter but not in CHAPTER_GUARDS.
        known.add("test_docs_quickstart_e2e.py")
        unlisted = sorted(_doc_guards() - known)
        self.assertEqual(
            unlisted,
            [],
            f"these documentation guards are not accounted for: {unlisted}. "
            "Add each to CHAPTER_GUARDS or to CROSS_CUTTING_GUARDS, and to "
            "the chapter's tables.",
        )

    def test_every_guard_appears_in_the_chapter(self) -> None:
        text = _chapter_text()
        for guard in sorted(_doc_guards() | CROSS_CUTTING_GUARDS):
            with self.subTest(guard=guard):
                self.assertIn(
                    f"`tests/{guard}`",
                    text,
                    f"{guard} exists but the chapter does not list it",
                )

    def test_the_cross_cutting_guards_exist(self) -> None:
        for guard in sorted(CROSS_CUTTING_GUARDS):
            with self.subTest(guard=guard):
                self.assertTrue((TESTS / guard).is_file())

    def test_the_chapter_counts_the_cross_cutting_table_it_prints(self) -> None:
        """The sentence introducing that table says how many rows it has.

        It said "Nine" against seventeen rows, because a row is added by
        someone reading the table and the sentence is above it. Cheap to
        check: the number is the length of the table directly below.
        """

        from docs_sections import table_after

        anchor = "guard documentation without belonging to one chapter:"
        rows = [
            row
            for row in table_after(_chapter_text(), anchor).splitlines()
            if row.startswith("| `tests/")
        ]
        stated = re.search(rf"^(\w+) more {re.escape(anchor)}", _chapter_text(), flags=re.MULTILINE)
        self.assertIsNotNone(stated, "the sentence introducing that table has been reworded")
        assert stated is not None
        self.assertEqual(
            _NUMBER_WORDS.get(stated.group(1).lower()),
            len(rows),
            f"the chapter says {stated.group(1)!r} guards are cross-cutting and "
            f"then lists {len(rows)}",
        )


class MarkerTest(unittest.TestCase):
    """The quickstart marker: registered, excluded, and run by CI."""

    def setUp(self) -> None:
        self.pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    def test_it_is_registered_and_excluded_by_default(self) -> None:
        self.assertIn("addopts = \"-m 'not quickstart'\"", self.pyproject)
        self.assertIn("quickstart:", self.pyproject)
        self.assertIn("addopts", _chapter_text())

    def test_testpaths_is_set_and_no_module_relies_on_it(self) -> None:
        """Both halves of the rule section 1 now states."""

        self.assertIn('testpaths = ["tests"]', self.pyproject)
        offenders = sorted(
            str(path.relative_to(REPO_ROOT))
            for path in (REPO_ROOT / "fedbrew").rglob("*.py")
            if fnmatch.fnmatch(path.name, "test_*.py")
        )
        self.assertEqual(
            offenders,
            [],
            f"production modules named like tests: {offenders}",
        )
        self.assertIn("matches pytest's discovery", _chapter_text())

    def test_ci_runs_the_suite_and_the_quickstart_marker(self) -> None:
        workflow = REPO_ROOT / ".github" / "workflows" / "tests.yml"
        self.assertTrue(workflow.is_file(), "the chapter says CI runs on push")
        text = workflow.read_text(encoding="utf-8")
        self.assertIn("pytest -q", text)
        self.assertIn("pytest -m quickstart -q", text)
        self.assertIn(".github/workflows/tests.yml", _chapter_text())

    def _matrix_versions(self) -> list[str]:
        import yaml

        workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
        )
        return list(workflow["jobs"]["tests"]["strategy"]["matrix"]["python-version"])

    def test_ci_tests_the_declared_floor_and_something_above_it(self) -> None:
        """`requires-python` has no upper bound, so both ends need a job.

        The floor is the half that was already covered, and the reason is in
        the workflow: a tree that only ever runs on the newest Python accepts
        syntax its own floor cannot parse and nothing says so. The other half
        was missing entirely -- every version above 3.10 was claimed by
        `requires-python` and none was tested.

        Both ends are checked here because either can be dropped with one edit
        to a list, and neither disappearance fails anything else: removing the
        floor leaves a green matrix testing versions the package does not
        promise, and removing the ceiling leaves a green matrix testing one.
        """

        versions = self._matrix_versions()
        self.assertGreaterEqual(
            len(versions),
            2,
            "the tests job runs one interpreter; requires-python promises a range",
        )

        floor = re.search(
            r'^requires-python\s*=\s*">=\s*([0-9]+\.[0-9]+)"',
            (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        self.assertIsNotNone(floor, "pyproject declares no requires-python floor")
        assert floor is not None
        self.assertIn(
            floor.group(1),
            versions,
            f"pyproject's floor is {floor.group(1)} and no CI job runs it",
        )

        def _key(version: str) -> tuple[int, ...]:
            return tuple(int(part) for part in version.split("."))

        below = [version for version in versions if _key(version) < _key(floor.group(1))]
        self.assertEqual(below, [], f"the matrix runs versions below the declared floor: {below}")
        self.assertTrue(
            [version for version in versions if _key(version) > _key(floor.group(1))],
            "every version in the matrix is the floor; nothing tests above it",
        )

    def test_the_chapter_names_every_interpreter_ci_runs(self) -> None:
        """Changing the matrix has to move prose, not just a YAML list.

        Read off the workflow rather than from a list here, so bumping the
        upper version fails the chapter instead of passing against a second
        copy of the old one. Scoped to the section that makes the claim: both
        numbers appear only there today, and a whole-file search would stop
        meaning anything the moment one appeared anywhere else -- §5.3 of the
        chapter this test guards is about exactly that.
        """

        from docs_sections import section_of

        section = " ".join(section_of(_chapter_text(), "## 2.1 The install-shape jobs").split())
        for version in self._matrix_versions():
            self.assertTrue(
                version in section,
                f"chapter 13 §2.1 does not name Python {version}, which CI runs",
            )

    def test_ci_runs_both_halves_of_the_style_gate(self) -> None:
        """The formatter was adopted; the chapter and CI must agree it was."""

        workflow = (REPO_ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
        # Comment lines are excluded on purpose: the workflow explains in a
        # comment why it does not run the formatter, and a naive substring
        # search reads that explanation as the thing it forbids.
        commands = "\n".join(
            line for line in workflow.splitlines() if not line.strip().startswith("#")
        )
        self.assertIn("ruff check", commands)
        self.assertIn("ruff format --check", commands)
        self.assertIn(
            "`ruff format --check` gate",
            " ".join(_chapter_text().split()),
            "the chapter must say the formatter gates, now that it does",
        )


class FastMarkTest(unittest.TestCase):
    """The `fast` marker: registered, and section 2.2 names the work it refuses.

    The authority is `HEAVY_WORK` in tests/conftest.py, the table the guard
    reads when it fails a `fast` test. The chapter's table is diffed against it
    both ways: a kind of work the guard counts and the chapter omits is a gap,
    and one the chapter names and the guard does not count is a promise the
    guard does not keep.
    """

    def test_it_is_registered(self) -> None:
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertRegex(pyproject, r'(?m)^\s*"fast: ')

    def test_the_table_names_exactly_the_work_the_guard_refuses(self) -> None:
        import conftest
        from docs_sections import section_of, table_after

        section = section_of(_chapter_text(), "## 2.2 The `fast` mark and the two gates")
        rows = table_after(section, "| What a fast test must not do |").splitlines()[2:]
        named = {row.split("|")[1].strip() for row in rows}
        self.assertEqual(named, set(conftest.HEAVY_WORK.values()))


class SuiteCompositionTest(unittest.TestCase):
    """The chapter describes the suite's shape, not its size.

    An exact module count was tried and removed: it churns on every added
    test, which is a maintenance tax on the documentation for a number a
    reader can get from `ls`. What is worth pinning is that the three kinds of
    test the chapter names all still exist.
    """

    def test_each_named_example_test_exists(self) -> None:
        # Both spellings: the tables use `tests/test_x.py`, the prose in
        # section 3 uses the bare `test_x.py`.
        named = {
            name.rsplit("/", 1)[-1]
            for name in re.findall(r"`((?:tests/)?test_\w+\.py)`", _chapter_text())
        }
        self.assertGreaterEqual(len(named), 10)
        missing = sorted(name for name in named if not (TESTS / name).is_file())
        self.assertEqual(missing, [], f"the chapter names tests that do not exist: {missing}")

    def test_the_documentation_guards_are_a_real_group(self) -> None:
        self.assertGreaterEqual(
            len(_doc_guards()),
            13,
            "one guard per chapter, plus the quickstart end-to-end one",
        )


if __name__ == "__main__":
    unittest.main()


class OptionalExtraCoverageTest(unittest.TestCase):
    """Every extra a test is gated on must be installed by some CI job.

    This is the guard for a defect class the documentation sweep could not
    see, because it lives in the gap between the suite and its environment: a
    `skipUnless(find_spec("transformers"))` reads as a pass on every machine
    without the extra, and a suite summary counts it beside the passes. No CI
    job installed the `llm` extra, so the 21 tests behind it had never run
    anywhere, and eight had gone stale -- against `server.name`, against
    `FedAvgServer.get_state`, against a `writer` config section -- while the
    suite stayed green.

    The mapping from a gate to an extra is read out of pyproject rather than
    kept here, so a new extra with gated tests is covered without editing this
    file. A gate naming a package no extra declares (`tokenizers`, `pyarrow`
    -- both transitive) is not evidence of anything and is passed over; in
    every such gate a sibling name that *is* declared appears alongside it, so
    the coverage is not lost.
    """

    def test_every_gated_extra_is_installed_by_a_job(self) -> None:
        gates = _dependency_gate_packages()
        self.assertTrue(gates, "the scan found no dependency gates at all")

        extras = _declared_extras()
        installed = _extras_installed_by_ci()
        uncovered = sorted(
            f"{package} (in extra {extra!r})"
            for package, extra in _extras_for(gates, extras).items()
            if extra not in installed
        )
        self.assertEqual(
            uncovered,
            [],
            "tests are gated on an extra no CI job installs, so they never run: "
            + ", ".join(uncovered),
        )

    def test_the_llm_extra_is_one_of_them(self) -> None:
        """Named directly, because it is the one this guard was written for."""

        self.assertIn("llm", _extras_installed_by_ci())
        self.assertIn("llm", _declared_extras())

    def test_the_scan_would_notice_an_uninstalled_extra(self) -> None:
        """Guards the guard: the mapping must be able to report a miss."""

        extras = _declared_extras()
        self.assertIn("vision", extras, "the fixture for this check is gone")
        self.assertNotIn(
            "vision",
            _extras_installed_by_ci(),
            "vision is now installed by a job; pick another uncovered extra here",
        )
        mapped = _extras_for({"torchvision"}, extras)
        self.assertEqual(mapped, {"torchvision": "vision"})


def _dependency_gate_packages() -> set[str]:
    """Every string literal reaching `find_spec` anywhere under `tests/`.

    Including the comprehension form, where the argument is the loop variable
    and the names are in the tuple it walks.
    """

    packages: set[str] = set()
    for path in sorted(TESTS.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        loop_names = {
            generator.target.id: [
                element.value
                for element in getattr(generator.iter, "elts", [])
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            ]
            for node in ast.walk(tree)
            if isinstance(node, ast.GeneratorExp | ast.ListComp)
            for generator in node.generators
            if isinstance(generator.target, ast.Name)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else None
            if name is None and isinstance(node.func, ast.Name):
                name = node.func.id
            if name != "find_spec" or not node.args:
                continue
            argument = node.args[0]
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                packages.add(argument.value)
            elif isinstance(argument, ast.Name):
                packages.update(loop_names.get(argument.id, []))
    return packages


def _declared_extras() -> dict[str, set[str]]:
    """Extra name -> the distribution names it requires, from pyproject."""

    section = re.search(
        r"^\[project\.optional-dependencies\]$(.*?)^\[",
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        re.MULTILINE | re.DOTALL,
    )
    assert section is not None, "pyproject declares no optional dependencies"

    extras: dict[str, set[str]] = {}
    current: str | None = None
    for line in section.group(1).splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        opening = re.match(r"^([A-Za-z0-9_-]+)\s*=\s*\[", stripped)
        if opening is not None:
            current = opening.group(1)
            extras[current] = set()
            stripped = stripped[opening.end() :]
        if current is None:
            continue
        for requirement in re.findall(r'"([^"]+)"', stripped):
            extras[current].add(re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0].lower())
        if stripped.endswith("]"):
            current = None
    return extras


def _extras_installed_by_ci() -> set[str]:
    """Extras named inside a `pip install -e ".[...]"` in the workflow."""

    workflow = (REPO_ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    commands = "\n".join(line for line in workflow.splitlines() if not line.strip().startswith("#"))
    installed: set[str] = set()
    for group in re.findall(r'pip install -e "\.\[([^\]]+)\]"', commands):
        installed.update(part.strip() for part in group.split(","))
    return installed


def _extras_for(packages: set[str], extras: dict[str, set[str]]) -> dict[str, str]:
    """Gate package -> the extra that declares it, for those an extra declares.

    `fedbrew[vision]`-style aliases resolve to the extra they alias, so the
    `femnist` alias does not read as separate coverage.
    """

    mapping: dict[str, str] = {}
    for package in sorted(packages):
        for extra, requirements in sorted(extras.items()):
            if package.lower() in requirements:
                mapping[package] = extra
                break
    return mapping
