"""docs/05-data-and-partitioning.md must match the generator and a real manifest.

Two things go stale here in different ways. The generator's section allow-lists
change when a generator is added, and the chapter's table of "which sections
does this generator read" is exactly that data restated -- so it is diffed.

The manifest key table is checked against a manifest the test generates, not
against a hand-written list, because a manifest is a description of what the
generator wrote and the only honest source for it is a generator run.

Generation is cheap for the synthetic dataset -- no network, a seeded linear
teacher, 150 examples -- so this stays in the default suite. The heavier
end-to-end path is docs/03's marked guard.
"""

from __future__ import annotations

import fnmatch
import json
import re
import tempfile
import unittest
from pathlib import Path

import pytest

from fedbrew.core.registry import generators, register_builtin_components
from fedbrew.data.generate import _SHARED_SECTIONS


def _generator_sections() -> dict[str, frozenset[str]]:
    """Generator name -> the sections it declares, for the built-in eight."""

    register_builtin_components()
    return {name: generators.get(name).sections for name in generators.builtin()}


REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER = REPO_ROOT / "docs" / "05-data-and-partitioning.md"

#: The strategies _partition_train_indices dispatches on. Kept here rather than
#: imported because the dispatch is an if-chain, not a table -- so this list is
#: a claim, and test_every_strategy_dispatches proves it against the code.
STRATEGIES = ("iid", "dirichlet", "quantity_skew", "label_skew")

#: The fifth. It has no partitioner module and no dispatch entry, because the
#: corpus supplies the clients rather than a cut supplying them -- which is why
#: the chapter called itself "the four partition strategies" while its flagship
#: dataset ran on a strategy it did not list. Proved against femnist.py below.
NATURAL_STRATEGY = "natural"

#: The values client_test_source takes, and the module that writes each.
#: Section 4 asserted the first as universal; femnist.py writes the second, and
#: the chapter's invariant 3 forbade exactly what femnist.py does. The third
#: is written by no generator in the package: it is the value an out-of-tree
#: generator writes when its splits hold the same rows, and the package
#: recognises it -- manifest_validation names it and preflight notes it.
CLIENT_TEST_SOURCES = {
    "partitioned_global_test": ("fedbrew", "data", "generate.py"),
    "within_client_holdout_disjoint_from_eval": ("fedbrew", "data", "femnist.py"),
    "identical_to_train": ("fedbrew", "data", "manifest_validation.py"),
}

#: The values the package writes, as opposed to recognises.
WRITTEN_CLIENT_TEST_SOURCES = frozenset(
    {"partitioned_global_test", "within_client_holdout_disjoint_from_eval"}
)


def _chapter_text() -> str:
    return CHAPTER.read_text(encoding="utf-8")


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


@pytest.mark.fast
class ChapterShapeTest(unittest.TestCase):
    def test_present_and_agent_facing(self) -> None:
        self.assertTrue(CHAPTER.is_file())
        self.assertIn("\n## For agents\n", _chapter_text())


@pytest.mark.fast
class PartitionStrategyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _chapter_text()

    def test_every_strategy_dispatches(self) -> None:
        """The four names must all be reachable in the dispatch chain."""

        source = (REPO_ROOT / "fedbrew" / "data" / "generate.py").read_text(encoding="utf-8")
        dispatched = set(re.findall(r'strategy == "(\w+)"', source))
        dispatched |= set(
            re.findall(r"strategy in \{([^}]+)\}", source)[0]
            .replace('"', "")
            .replace(" ", "")
            .split(",")
            if re.findall(r"strategy in \{([^}]+)\}", source)
            else []
        )
        self.assertEqual(
            dispatched,
            set(STRATEGIES),
            "the partition strategies the generator dispatches on changed",
        )

    def test_each_strategy_has_a_section(self) -> None:
        for index, strategy in enumerate(STRATEGIES, start=1):
            with self.subTest(strategy=strategy):
                self.assertIn(f"### 3.{index} `{strategy}`", self.text)

    def test_each_partitioner_module_exists(self) -> None:
        for strategy in STRATEGIES:
            with self.subTest(strategy=strategy):
                module = REPO_ROOT / "fedbrew" / "data" / "partitioners" / (f"{strategy}.py")
                self.assertTrue(module.is_file())
                self.assertIn(f"partition_{strategy}", self.text)

    def test_the_fifth_strategy_has_a_section_too(self) -> None:
        """It has no partitioner module, which is why it went unlisted."""

        source = (REPO_ROOT / "fedbrew" / "data" / "femnist.py").read_text(encoding="utf-8")
        self.assertIn(f"partition.strategy={NATURAL_STRATEGY}", source)
        self.assertNotIn(NATURAL_STRATEGY, set(STRATEGIES))
        self.assertIn(f"### 3.{len(STRATEGIES) + 1} `{NATURAL_STRATEGY}`", self.text)

    def test_the_heading_counts_every_strategy(self) -> None:
        count = len(STRATEGIES) + 1
        self.assertTrue(
            _states_count(self.text, count, "partition strategies"),
            f"section 3's heading must say {count} partition strategies",
        )

    def test_the_chapter_says_femnist_accepts_only_natural(self) -> None:
        """The refusal is in the generator; a chapter that omits it invites the try."""

        self.assertRegex(
            self.text,
            rf"refuses any other strategy|only `{NATURAL_STRATEGY}`",
        )


@pytest.mark.fast
class GeneratorSectionTest(unittest.TestCase):
    """The 'sections it reads' table must be the allow-lists."""

    def setUp(self) -> None:
        self.text = _chapter_text()

    def _generator_table(self) -> str:
        """Section 2's table only.

        Section 1 also has a row keyed `synthetic_classification`, describing
        the backend rather than the generator, and a whole-file search finds
        that one first.
        """

        start = self.text.index("| Generator | Sections it reads")
        return self.text[start : self.text.index("\n\n", start)]

    def test_every_generator_has_a_row_with_its_own_sections(self) -> None:
        table = self._generator_table()
        for name, sections in _generator_sections().items():
            with self.subTest(generator=name):
                row = re.search(
                    rf"^\| `{re.escape(name)}` \| (.+) \|$",
                    table,
                    flags=re.MULTILINE,
                )
                self.assertIsNotNone(row, f"{name} has no row")
                assert row is not None
                listed = set(re.findall(r"`(\w+)`", row.group(1)))
                self.assertEqual(
                    listed,
                    set(sections),
                    f"the {name} row and the generator registry disagree",
                )

    def test_the_shared_sections_are_documented(self) -> None:
        for section in _SHARED_SECTIONS:
            with self.subTest(section=section):
                self.assertIn(f"`{section}`", self.text)
        self.assertEqual(len(_SHARED_SECTIONS), 2, "the chapter says two")
        self.assertIn("two shared sections", self.text)

    def test_client_splits_is_not_shared(self) -> None:
        """It was the third entry, and the exemption that let the two SFT
        generators take the section and delete it."""

        self.assertNotIn("client_splits", _SHARED_SECTIONS)

    def test_the_generator_count_is_right(self) -> None:
        self.assertTrue(
            _states_count(self.text, len(_generator_sections()), "generators"),
            f"the chapter must say {len(_generator_sections())} generators, which is "
            "how many the registry holds",
        )


@pytest.mark.fast
class ProvenanceTableTest(unittest.TestCase):
    """Section 2's honesty table must stay true as coverage changes.

    The chapter claims cifar10 has no shipped config and no dedicated test, and
    that every other generator has at least one of the two. That is a claim
    about the repository, so it is checked against the repository -- otherwise
    it is exactly the kind of statement that stays on the page after someone
    adds the config it says does not exist.
    """

    def _configs_by_generator(self) -> dict[str, int]:
        import yaml

        counts: dict[str, int] = {}
        for path in sorted((REPO_ROOT / "data" / "configs").glob("*.yaml")):
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                continue
            name = (loaded.get("dataset") or {}).get("name")
            if name:
                counts[name] = counts.get(name, 0) + 1
        return counts

    def test_the_generators_the_chapter_calls_unbacked_really_are(self) -> None:
        counts = self._configs_by_generator()
        for generator in ("tiny_causal_lm", "hf_causal_lm_text", "cifar10"):
            with self.subTest(generator=generator):
                self.assertEqual(
                    counts.get(generator, 0),
                    0,
                    f"{generator} now has a shipped config; section 2's "
                    "provenance table says it has none",
                )

    def test_cifar10_is_still_the_least_supported(self) -> None:
        test_sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (REPO_ROOT / "tests").glob("test_*.py")
            if path.name != "test_docs_data_keys.py"
        )
        self.assertNotIn(
            "cifar10_generator",
            test_sources,
            "cifar10 gained a generator test; update the provenance table",
        )
        self.assertIn("`cifar10` is the one row to treat with suspicion", _chapter_text())

    def test_the_backed_generators_have_what_the_table_claims(self) -> None:
        counts = self._configs_by_generator()
        for generator, expected in (
            ("synthetic_classification", 3),
            ("femnist", 1),
            ("mnist", 4),
            ("oasst1_sft", 3),
            ("generic_sft", 1),
        ):
            with self.subTest(generator=generator):
                self.assertEqual(
                    counts.get(generator, 0),
                    expected,
                    f"{generator} has {counts.get(generator, 0)} shipped "
                    f"configs; the provenance table says {expected}",
                )


class ManifestKeyTest(unittest.TestCase):
    """The manifest table must match a manifest the generator actually writes."""

    @classmethod
    def setUpClass(cls) -> None:
        import yaml

        from fedbrew.data.generate import generate_from_config

        source = REPO_ROOT / "data" / "configs" / "synthetic_label_skew.yaml"
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        cls._scratch = tempfile.TemporaryDirectory()
        # Redirected into a scratch tree so running the suite never touches a
        # dataset a real run may be reading.
        config["dataset"]["output_dir"] = cls._scratch.name
        scratch_config = Path(cls._scratch.name) / "generator.yaml"
        scratch_config.write_text(yaml.safe_dump(config), encoding="utf-8")
        manifest_path = generate_from_config(scratch_config)
        cls.manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        cls.client_record = json.loads(
            (Path(cls._scratch.name) / "clients.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._scratch.cleanup()

    def test_every_manifest_key_is_documented(self) -> None:
        text = _chapter_text()
        missing = sorted(key for key in self.manifest if f"`{key}`" not in text)
        self.assertEqual(
            missing,
            [],
            "the generator writes manifest keys the chapter does not name",
        )

    def test_the_stated_key_count_matches(self) -> None:
        self.assertTrue(
            _states_count(_chapter_text(), len(self.manifest), "keys"),
            f"the chapter must say {len(self.manifest)} keys, which is what the "
            "generator writes into the manifest",
        )

    def test_the_client_record_fields_are_documented(self) -> None:
        text = _chapter_text()
        missing = sorted(field for field in self.client_record if f'"{field}"' not in text)
        self.assertEqual(
            missing,
            [],
            "clients.jsonl carries fields the chapter's example omits",
        )

    def test_the_client_test_source_claim_holds(self) -> None:
        """Section 4's central claim, from a real manifest."""

        self.assertEqual(
            self.manifest["client_test_source"],
            "partitioned_global_test",
            "client test splits no longer come from the official test set",
        )
        self.assertIn("partitioned_global_test", _chapter_text())


@pytest.mark.fast
class BothClientTestSourcesAreDocumentedTest(unittest.TestCase):
    """Section 4 asserted one source as universal; there are two.

    The guard above generated a synthetic manifest, found
    partitioned_global_test, and passed -- while femnist.py wrote
    within_client_holdout_disjoint_from_eval and the chapter's invariant 3 said
    "Never take a client's test examples from its own training pool", which is
    what that value means. One generator was checked and the other was the
    flagship.
    """

    def setUp(self) -> None:
        self.text = _chapter_text()

    def test_every_value_the_tree_writes_is_named_in_the_chapter(self) -> None:
        for value, module in sorted(CLIENT_TEST_SOURCES.items()):
            with self.subTest(value=value):
                source = (REPO_ROOT.joinpath(*module)).read_text(encoding="utf-8")
                if value in WRITTEN_CLIENT_TEST_SOURCES:
                    self.assertIn(
                        f'"client_test_source": "{value}"',
                        source,
                        f"{'/'.join(module)} no longer writes {value}",
                    )
                else:
                    self.assertIn(
                        f'"{value}"',
                        source,
                        f"{'/'.join(module)} no longer recognises {value}",
                    )
                self.assertIn(value, self.text, f"the chapter does not name {value}")

    def test_the_chapter_writes_no_value_the_tree_does_not(self) -> None:
        """The other direction: a promised source nothing produces is a lie."""

        named = set(
            re.findall(r"`(partitioned_global_test|within_client_\w+|identical_to_\w+)`", self.text)
        )
        self.assertEqual(named - set(CLIENT_TEST_SOURCES), set())

    def test_the_invariant_no_longer_forbids_what_femnist_does(self) -> None:
        """The exact sentence that contradicted the generator."""

        self.assertNotIn(
            "Never take a client's test examples from its own\n   training pool",
            self.text,
        )
        self.assertNotIn(
            "The client test split comes from the official test set, not from the",
            self.text,
        )

    def test_the_per_writer_limitation_is_stated_here(self) -> None:
        """This chapter owns the per-writer limitation and states it in full."""

        for phrase in (
            "per-writer",
            "seen writers",
            "unseen writers",
            "no external test set",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.text.lower())


@pytest.mark.fast
class OfficialTestPartitionerTest(unittest.TestCase):
    """The test-set partitioner is production code, and is named like it.

    It was `fedbrew/data/test_partitioning.py`, which matches pytest's
    `test_*.py` discovery pattern; only `testpaths = ["tests"]` kept it from
    being imported as a test suite, and `pytest fedbrew/` or any tool with its
    own discovery would have done so. Renamed rather than left behind one
    config key.
    """

    def test_it_is_imported_by_the_generator(self) -> None:
        source = (REPO_ROOT / "fedbrew" / "data" / "generate.py").read_text(encoding="utf-8")
        self.assertIn(
            "from fedbrew.data.official_test_partitioning import partition_test_indices_like_train",
            source,
        )

    def test_no_production_module_matches_the_discovery_pattern(self) -> None:
        """The general rule the rename established, not just the one file."""

        offenders = sorted(
            str(path.relative_to(REPO_ROOT))
            for path in (REPO_ROOT / "fedbrew").rglob("*.py")
            if fnmatch.fnmatch(path.name, "test_*.py")
        )
        self.assertEqual(
            offenders,
            [],
            f"production modules named like tests: {offenders}. pytest would "
            "import them as test suites but for testpaths; rename them.",
        )

    def test_the_chapter_no_longer_calls_it_a_test_file(self) -> None:
        text = _chapter_text()
        self.assertIn("official_test_partitioning.py", text)
        self.assertNotIn("data/test_partitioning.py", text)


@pytest.mark.fast
class CitedPathsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _chapter_text()

    def test_cited_modules_exist(self) -> None:
        cited = set(re.findall(r"`(fedbrew/[\w/]+\.py)", self.text))
        self.assertGreaterEqual(len(cited), 10)
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
        self.assertEqual(missing, [])

    def test_cited_tests_exist(self) -> None:
        cited = set(re.findall(r"`(tests/test_\w+\.py)`", self.text))
        self.assertTrue(cited)
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
        self.assertEqual(missing, [])

    def test_cited_configs_and_directories_exist(self) -> None:
        for path in set(re.findall(r"(data/configs/[\w/]+\.yaml)", self.text)):
            with self.subTest(path=path):
                self.assertTrue((REPO_ROOT / path).is_file())
        for path in set(re.findall(r"`((?:data/)?configs/[\w/]*)`", self.text)):
            if path.endswith("/"):
                with self.subTest(path=path):
                    self.assertTrue((REPO_ROOT / path).is_dir())


if __name__ == "__main__":
    unittest.main()
