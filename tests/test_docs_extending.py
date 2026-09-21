"""docs/12-extending.md must name extension points that exist, and mean them.

An extending chapter is a set of instructions, and the way it fails is by
telling someone to edit a function that has been renamed or a constant that has
moved. That reader then either gives up or invents a different place to put
their change, which is worse.

So every function, constant and module the chapter tells a reader to touch is
checked to exist. Two things beyond that, both added when the chapter was
rewritten around the out-of-tree hook:

**Every excerpt is a verbatim quote.** The chapter shows ``register()`` and two
config blocks as the thing a reader should write, labelled with the file they
come from. A shown-but-stale excerpt is worse than no excerpt, because it is
copied. Any fenced block whose info string carries a repository path must be a
character-for-character substring of that file, which also means the chapter
can only show contiguous code -- an elided excerpt fails here rather than
misleading someone quietly.

**The worked case actually runs.** The chapter claims an out-of-tree component
reaches the CLI with no edit to the package. That is a claim about two shipped
commands, so both are run against the configs the chapter names -- each from
its own cold interpreter, for the reason `WorkedCaseRunsTest` gives. It costs
about eight seconds and trains nothing.

What still cannot be checked is whether the order of the steps is right; that
is prose, and the invariants each step protects are guarded by the tests the
chapter's own table names.
"""

from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER = REPO_ROOT / "docs" / "12-extending.md"

#: (symbol, module) the chapter instructs a reader to edit or subclass.
#: Deliberately explicit: a symbol added here is a claim that the chapter
#: names it, and making that claim is the review moment this test forces.
EXTENSION_POINTS = (
    ("ClientUpdate", "fedbrew/clients/base.py"),
    ("ServerStrategy", "fedbrew/servers/base.py"),
    ("TaskAdapter", "fedbrew/tasks/base.py"),
    ("FederatedDataset", "fedbrew/data/dataset.py"),
    ("register_builtin_components", "fedbrew/core/registry.py"),
    ("MODEL_TASKS", "fedbrew/core/registry.py"),
    ("registering_from", "fedbrew/core/registry.py"),
    ("BUILTIN", "fedbrew/core/registry.py"),
    ("GeneratorRegistry", "fedbrew/core/registry.py"),
    ("load_extensions", "fedbrew/core/extensions.py"),
    ("LoadedExtension", "fedbrew/core/extensions.py"),
    ("REGISTER_HOOK", "fedbrew/core/extensions.py"),
    ("is_extension", "fedbrew/core/factory.py"),
    ("EXTENSION_TASK_KEYS", "fedbrew/core/factory.py"),
    ("EXTENSION_DATASET_FIELDS", "fedbrew/core/factory.py"),
    ("EXTENSION_CLIENT_KEYS", "fedbrew/core/factory.py"),
    ("_KNOWN_EXTRA_KEYS", "fedbrew/core/config.py"),
    ("CLIENT_METRIC_BASES", "fedbrew/core/config.py"),
    ("client_metric_names", "fedbrew/core/config.py"),
    ("generators", "fedbrew/core/registry.py"),
    ("GeneratorSpec", "fedbrew/core/registry.py"),
    ("_GENERATOR_SECTION_KEYS", "fedbrew/data/generate.py"),
    ("_partition_train_indices", "fedbrew/data/generate.py"),
    ("partition_test_indices_like_train", "fedbrew/data/official_test_partitioning.py"),
    ("reject_unknown_model_keys", "fedbrew/models/config_keys.py"),
    ("_client_distribution_statistics", "fedbrew/core/loop.py"),
    ("aggregate_stream", "fedbrew/servers/base.py"),
)


def _chapter_text() -> str:
    return CHAPTER.read_text(encoding="utf-8")


@pytest.mark.fast
class ChapterShapeTest(unittest.TestCase):
    def test_present_and_agent_facing(self) -> None:
        self.assertTrue(CHAPTER.is_file())
        self.assertIn("\n## For agents\n", _chapter_text())


@pytest.mark.fast
class ExtensionPointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _chapter_text()

    def test_every_named_symbol_exists_where_the_chapter_says(self) -> None:
        for symbol, module in EXTENSION_POINTS:
            with self.subTest(symbol=symbol, module=module):
                path = REPO_ROOT / module
                self.assertTrue(path.is_file(), f"{module} does not exist")
                self.assertIn(
                    symbol,
                    path.read_text(encoding="utf-8"),
                    f"{module} no longer defines {symbol}, which the chapter "
                    "tells a reader to edit",
                )

    def test_the_chapter_names_every_one_of_them(self) -> None:
        for symbol, _ in EXTENSION_POINTS:
            with self.subTest(symbol=symbol):
                self.assertIn(f"`{symbol}`", self.text)

    def test_the_generator_entry_point_shape_is_still_right(self) -> None:
        """The chapter dictates a generator function signature."""

        source = (REPO_ROOT / "fedbrew" / "data" / "generate.py").read_text(encoding="utf-8")
        for argument in ("output_dir", "seed", "client_splits"):
            with self.subTest(argument=argument):
                self.assertIn(argument, source)
        self.assertIn("generate_<name>_from_config", self.text)

    def test_the_writer_modules_exist(self) -> None:
        for module in ("torch_shards", "manifest"):
            with self.subTest(writer=module):
                path = REPO_ROOT / "fedbrew" / "data" / "writers" / f"{module}.py"
                self.assertTrue(path.is_file())
                self.assertIn(f"writers/{module}.py", self.text)


@pytest.mark.fast
class MetricExtensionTest(unittest.TestCase):
    """Section 5's central warning: two places must change together."""

    def test_both_places_read_the_same_tuple(self) -> None:
        config = (REPO_ROOT / "fedbrew" / "core" / "config.py").read_text(encoding="utf-8")
        loop = (REPO_ROOT / "fedbrew" / "core" / "loop.py").read_text(encoding="utf-8")
        self.assertIn("CLIENT_METRIC_BASES", config)
        self.assertIn("CLIENT_METRIC_BASES", loop)
        self.assertIn(
            "client_metric_names` and `_client_distribution_statistics",
            " ".join(_chapter_text().split()),
        )

    def test_the_selection_metric_rule_still_holds(self) -> None:
        from fedbrew.core.checkpointing import SELECTION_METRIC_PREFIXES

        self.assertEqual(SELECTION_METRIC_PREFIXES, ("val_", "personal_val_"))
        text = " ".join(_chapter_text().split())
        self.assertIn("must be a `val_` or `personal_val_`", text)


@pytest.mark.fast
class CitedPathsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _chapter_text()

    def test_cited_modules_exist(self) -> None:
        cited = set(re.findall(r"`(fedbrew/[\w/]+\.py)", self.text))
        self.assertGreaterEqual(len(cited), 10)
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
        self.assertEqual(missing, [])

    def test_cited_directories_exist(self) -> None:
        cited = set(re.findall(r"`(fedbrew/[\w/]+/)`", self.text))
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_dir())
        self.assertEqual(missing, [])

    def test_cited_tests_exist(self) -> None:
        cited = set(re.findall(r"`(tests/test_\w+\.py)`", self.text))
        self.assertGreaterEqual(len(cited), 8)
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
        self.assertEqual(missing, [])


#: A fenced block whose info string carries a second word: ```yaml <path>.
#: The language keeps the block highlighted; the path is the claim this guard
#: checks. A block with no path is prose or a shell transcript and is skipped.
_LABELLED_BLOCK = re.compile(r"^```(\w+) +(\S+)\n(.*?)^```", re.MULTILINE | re.DOTALL)

#: The files §2 must quote. Named rather than counted, because the chapter
#: losing its ``register()`` excerpt and keeping two config ones would leave a
#: count-based check green while the worked case stopped showing the work.
QUOTED_FILES = (
    "examples/drift-quad/problem.py",
    "data/configs/examples/drift-quad.yaml",
    "configs/examples/drift-quad/fedavg.yaml",
)

#: The chapter's `generate` and `--validate-only` commands, as its For agents
#: block writes them, keyed in `WorkedCaseRunsTest.output` by subcommand.
#: `--no-rich` is added so the captured output is flat text.
WORKED_CASE_COMMANDS = (
    ("generate", "--config", "data/configs/examples/drift-quad.yaml"),
    ("run", "--config", "configs/examples/drift-quad/fedavg.yaml", "--validate-only", "--no-rich"),
)


def _first_absent_line(excerpt: str, source: str) -> str:
    """The first line of `excerpt` that `source` does not contain.

    A drifted excerpt usually differs in one line, and that line is the whole
    of what a reader needs. When every line is present individually the
    excerpt is discontiguous -- lines reordered, or a range elided -- and the
    first line is the place to start reading.
    """

    lines = excerpt.splitlines()
    for line in lines:
        if line.strip() and line not in source:
            return line
    return f"(every line appears; the excerpt is not contiguous) {lines[0] if lines else ''}"


@pytest.mark.fast
class VerbatimExcerptTest(unittest.TestCase):
    """Every labelled excerpt is a character-for-character quote of its file.

    The chapter shows a reader what to write. An excerpt that has drifted from
    the file it names is copied into someone's tree before anyone reads the
    file, so this is the one docs check where a near miss is worse than an
    omission.
    """

    def setUp(self) -> None:
        self.blocks = _LABELLED_BLOCK.findall(_chapter_text())

    def test_the_chapter_still_quotes_the_worked_case(self) -> None:
        quoted = {path for _, path, _ in self.blocks}
        missing = sorted(set(QUOTED_FILES) - quoted)
        self.assertEqual(
            missing,
            [],
            f"§2 no longer excerpts {missing}; the worked case is the chapter's "
            "one demonstration that an extension needs nothing but its own file "
            "and two config lines",
        )

    def test_every_labelled_block_names_a_file_that_exists(self) -> None:
        for _, path, _ in self.blocks:
            with self.subTest(path=path):
                self.assertTrue(
                    (REPO_ROOT / path).is_file(),
                    f"an excerpt is labelled {path}, which is not a file",
                )

    def test_every_labelled_block_is_verbatim(self) -> None:
        for language, path, body in self.blocks:
            with self.subTest(path=path):
                # A label naming nothing is the test above's failure, and
                # reporting it twice buries the message that says which.
                if not (REPO_ROOT / path).is_file():
                    continue
                source = (REPO_ROOT / path).read_text(encoding="utf-8")
                if body in source:
                    continue
                # Not `assertIn`: its message prints both operands, which here
                # is the excerpt beside the whole file. A reader of a failure
                # needs the one line that stopped matching, not 40KB of the
                # file they already have open.
                self.fail(
                    f"the ```{language} {path} excerpt is not a quote of that "
                    f"file. First line not found in it:\n"
                    f"    {_first_absent_line(body, source)!r}\n"
                    "Excerpts are contiguous and unedited: re-copy the lines, "
                    "or quote a different range. Eliding with `...` fails here "
                    "on purpose."
                )

    def test_a_labelled_block_is_not_the_whole_file(self) -> None:
        """An excerpt is a quotation, not a copy of the file into the chapter."""

        for _, path, body in self.blocks:
            with self.subTest(path=path):
                source = (REPO_ROOT / path).read_text(encoding="utf-8")
                self.assertLess(
                    len(body),
                    len(source),
                    f"the {path} excerpt is the entire file; link to it instead",
                )


def _run_cli(command: tuple[str, ...]) -> str:
    """Run one `fedbrew` command from a cold interpreter, and return its stdout.

    The console script needs an install a checkout may not have, so the
    equivalent module form is used -- the same entry point, as chapter 02 says.
    """

    completed = subprocess.run(
        [sys.executable, "-m", "fedbrew.cli.dispatch", *command],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"`fedbrew {' '.join(command)}`, which docs/12-extending.md §2 tells "
            f"a reader to run, exited {completed.returncode}\n"
            f"--- stdout ---\n{completed.stdout[-3000:]}\n"
            f"--- stderr ---\n{completed.stderr[-3000:]}"
        )
    return completed.stdout


class WorkedCaseRunsTest(unittest.TestCase):
    """The chapter's claim is that two shipped commands are the whole of it. Run them.

    **One subprocess each, not one for both.** Sharing a process was tried and
    it hides the thing this guard is for: `generate` loads the extension named
    by `dataset.extensions`, and every name it registers is then in the
    registries for the rest of that process -- so the run afterwards resolves
    `drift_quad` and `quad_vector` whether or not its own
    `experiment.extensions` was read at all. Deleting that key from the arm
    config left a shared-process guard passing. Each command therefore starts
    cold, which is also how a reader runs them.

    From the repository root, because the configs use repository-relative
    paths. Nothing here trains: `generate` writes eight rows and
    `--validate-only` returns before the first round, which is why this is in
    the default suite rather than behind the `quickstart` marker. It costs
    about eight seconds, nearly all of it two torch imports.

    `generate` writes into `data/generated/`, which is gitignored and which the
    chapter itself tells a reader to write, and it is idempotent.
    """

    #: Each command's stdout, filled once in `setUpClass`. The two commands are
    #: a pipeline -- the run reads what the generator wrote -- so running them
    #: per test would pay for the generate twice and make the second test
    #: depend on method ordering for its data.
    output: dict[str, str] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cls.output = {command[0]: _run_cli(command) for command in WORKED_CASE_COMMANDS}

    def test_generate_writes_the_data_through_the_out_of_tree_generator(self) -> None:
        self.assertIn("drift_quad", self.output["generate"])
        manifest = REPO_ROOT / "data" / "generated" / "examples" / "drift-quad" / "manifest.json"
        self.assertTrue(manifest.is_file(), "the generator wrote no manifest")

    def test_validate_only_finds_the_task_and_model_the_run_config_names(self) -> None:
        stdout = self.output["run"]
        # The plan header's amber row is written from `experiment.extensions`,
        # so it is the one line that proves the run config's own entry was read
        # rather than inherited from whatever else the interpreter has loaded.
        self.assertIn("Extensions", stdout)
        self.assertIn("drift_quad", stdout)
        self.assertIn("quad_vector", stdout)

    def test_the_chapter_writes_the_commands_this_test_runs(self) -> None:
        """The guard is only worth anything while it runs what the chapter says."""

        text = _chapter_text()
        for command in WORKED_CASE_COMMANDS:
            written = "fedbrew " + " ".join(part for part in command if part not in ("--no-rich",))
            with self.subTest(command=written):
                self.assertIn(written, text)


#: §1.7's two tables: the guards a new algorithm fails on registration, and the
#: two that fire on what it is rather than on its name.
_ALWAYS_ANCHOR = "can answer, asked at the moment they are the only person thinking about it."
_CONDITIONAL_ANCHOR = "Two more fire on what the new name *is* rather than on the name itself:"

#: Test modules that enumerate an algorithm registry and are deliberately not
#: in either table. Each entry is a claim that registering a name does not make
#: the module demand anything, and making that claim is the review moment --
#: the same construction as CROSS_CUTTING_GUARDS in tests/test_docs_testing.py.
REGISTRY_GUARDS_THAT_ASK_FOR_NOTHING = {
    # Asserts the opposite direction: a *built-in* declares no config_keys. A
    # new built-in that declares none passes without being named anywhere.
    "test_extensions_config.py": "checks that built-ins declare no config keys",
    # Walks every registered name looking for dead branches; a new name is
    # swept like the rest and needs no table row.
    "test_no_dead_or_shadowing_paths.py": "sweeps whatever is registered",
    # About where a name came from, not about which names exist.
    "test_registry_origins.py": "checks the origin recorded for a name",
    # About the five examples' own registrations, which are extensions.
    "test_examples_are_extensions.py": "checks examples/, which register out of tree",
}


def _modules_named_in(anchor: str) -> set[str]:
    from docs_sections import table_after

    return set(re.findall(r"`tests/(test_\w+\.py)`", table_after(_chapter_text(), anchor)))


def _modules_that_enumerate_a_registry() -> set[str]:
    """Test modules that name an algorithm registry and enumerate one.

    Two spellings are in use and both have to be caught:
    `client_updates.builtin()` directly, and chapter 01's guard, which holds
    the registry names in a tuple and reaches them with `getattr`. Matching the
    two tokens separately rather than adjacently is what covers the second.
    """

    found = set()
    for path in sorted((REPO_ROOT / "tests").glob("test_*.py")):
        if path.name == Path(__file__).name:
            # This module names both registries and both spellings, in the
            # pattern above. Scanning itself would put it in its own table.
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"\b(?:server_strategies|client_updates)\b", text) and re.search(
            r"\.(?:builtin|list)\(\)", text
        ):
            found.add(path.name)
    return found


@pytest.mark.fast
class RegistryDerivedGuardTableTest(unittest.TestCase):
    """§1.7, in both directions.

    The chapter tells an author which six guards a new algorithm fails and what
    each one wants declared. A table like that is worth exactly as much as its
    agreement with the suite, and it goes stale in the direction nobody looks:
    someone adds a seventh guard, and the chapter still says six.
    """

    def test_every_guard_the_chapter_names_exists(self) -> None:
        named = _modules_named_in(_ALWAYS_ANCHOR) | _modules_named_in(_CONDITIONAL_ANCHOR)
        self.assertTrue(named, "§1.7's tables name no guards; the anchors have moved")
        for module in sorted(named):
            with self.subTest(module=module):
                self.assertTrue((REPO_ROOT / "tests" / module).is_file())

    def test_every_guard_the_chapter_names_enumerates_a_registry(self) -> None:
        named = _modules_named_in(_ALWAYS_ANCHOR) | _modules_named_in(_CONDITIONAL_ANCHOR)
        enumerating = _modules_that_enumerate_a_registry()
        for module in sorted(named):
            with self.subTest(module=module):
                self.assertIn(
                    module,
                    enumerating,
                    f"{module} is in §1.7 but does not enumerate an algorithm "
                    "registry, so a new name cannot be what makes it fail",
                )

    def test_no_registry_derived_guard_is_unclassified(self) -> None:
        """The direction that rots: a new guard lands and the chapter says six."""

        classified = (
            _modules_named_in(_ALWAYS_ANCHOR)
            | _modules_named_in(_CONDITIONAL_ANCHOR)
            | set(REGISTRY_GUARDS_THAT_ASK_FOR_NOTHING)
        )
        unclassified = sorted(_modules_that_enumerate_a_registry() - classified)
        self.assertEqual(
            unclassified,
            [],
            f"these guards enumerate an algorithm registry and are in neither "
            f"§1.7 table nor the exemption list: {unclassified}. Add each to a "
            "table, or to REGISTRY_GUARDS_THAT_ASK_FOR_NOTHING with the reason.",
        )

    def test_the_exemptions_are_real_modules_that_enumerate_a_registry(self) -> None:
        for module in sorted(REGISTRY_GUARDS_THAT_ASK_FOR_NOTHING):
            with self.subTest(module=module):
                self.assertTrue((REPO_ROOT / "tests" / module).is_file())
                self.assertIn(module, _modules_that_enumerate_a_registry())

    def test_the_chapter_counts_the_guards_its_first_table_lists(self) -> None:
        rows = _modules_named_in(_ALWAYS_ANCHOR)
        words = "zero one two three four five six seven eight nine ten".split()
        # The sentence wraps, so compare against the chapter with its line
        # breaks flattened rather than against the raw text.
        flattened = " ".join(_chapter_text().split())
        self.assertIn(
            f"turns the suite red in {words[len(rows)]} places at once",
            flattened,
            f"§1.7's first table lists {len(rows)} guards; the sentence above it "
            "states a different number",
        )

    def test_the_declaration_sites_the_chapter_names_exist(self) -> None:
        from fedbrew.core.config import UNHONOURED_CLIENT_OPTIONS
        from fedbrew.core.validation import AGGREGATION_WEIGHTING_NOTICE

        self.assertTrue(AGGREGATION_WEIGHTING_NOTICE)
        self.assertTrue(UNHONOURED_CLIENT_OPTIONS)
        for module, constant in (
            ("test_ignored_client_options.py", "RULE_CONFIGS"),
            ("test_client_communication_cost.py", "BUILDERS"),
        ):
            with self.subTest(module=module):
                text = (REPO_ROOT / "tests" / module).read_text(encoding="utf-8")
                self.assertTrue(
                    re.search(rf"^{constant}\b", text, flags=re.MULTILINE),
                    f"{constant} is not defined at module level in {module}, so "
                    "§1.7 is pointing an author at something that has moved",
                )


if __name__ == "__main__":
    unittest.main()
