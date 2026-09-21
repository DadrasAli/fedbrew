"""docs/08-metrics.md must name the columns the code actually emits.

The previous documentation set was deleted because it had drifted two CLI
generations stale while still reading as authoritative. A metrics chapter is
the worst place for that to happen: a wrong column name is not noticed when
the docs are read, it is noticed when an analysis script returns an empty
Series three weeks later.

So the chapter's column names are not hand-kept. Every name it lists is
checked here against the function that produces it -- client_metric_names for
the aggregates, the two frozen field lists in artifacts.py for the fixed CSV
schemas, and CLIENT_METRIC_BASES for the two base metrics.

The direction of the check matters. It is bidirectional: a column the code
emits and the chapter omits is a gap, and a column the chapter names and the
code cannot produce is a lie. Both fail here.

Note on `bottom10`: fedbrew/core/logging.py's console description table still
names test_accuracy_bottom10, which the loop stopped emitting when the suffix
became worst{P}. That table is cosmetic -- it never reaches an artifact -- so
it is deliberately NOT the source used here, and the chapter documents the
discrepancy rather than repeating it.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import pytest

from fedbrew.core.artifacts import (
    _CLIENT_EVALUATION_FIELDS,
    _ROUND_TIMING_FIELDS,
)
from fedbrew.core.config import (
    CLIENT_METRIC_BASES,
    ClientStatisticsConfig,
    client_metric_names,
    worst_percent_label,
)
from fedbrew.core.metrics import (
    METRIC_BASE_GLOSSES,
    METRIC_SUFFIX_GLOSSES,
    metric_gloss,
)

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER = REPO_ROOT / "docs" / "08-metrics.md"

#: Splits the loop aggregates over, from loop.py's `for split in (...)`.
SPLITS = ("train", "val", "test")


def _chapter_text() -> str:
    return CHAPTER.read_text(encoding="utf-8")


def _section(text: str, heading: str, until: str) -> str:
    """The chapter between two headings.

    A whole-file scan is the standard way a documentation guard passes a
    mutation it should have caught: the pattern matches an identically shaped
    row somewhere else in the chapter. Section 12 grew a table of
    `_validate_*` function names, which the section-5 suffix regex read as two
    more suffixes. Locate the section first.
    """

    start = text.index(heading)
    return text[start : text.index(until, start)]


def _fenced_block_after(text: str, heading: str) -> str:
    """The first ``` block following `heading`, so a table cannot be mistaken
    for the authoritative list."""

    start = text.index(heading)
    opening = text.index("```", start)
    closing = text.index("```", opening + 3)
    return text[opening + 3 : closing]


class ChapterExistsTest(unittest.TestCase):
    def test_the_chapter_is_present(self) -> None:
        self.assertTrue(
            CHAPTER.is_file(),
            f"{CHAPTER.relative_to(REPO_ROOT)} is missing; the metrics chapter "
            "is the one every other chapter cites for column names",
        )

    def test_it_ends_with_a_for_agents_section(self) -> None:
        self.assertIn(
            "\n## For agents\n",
            _chapter_text(),
            "every chapter must carry a '## For agents' section",
        )


class AggregateColumnNamesTest(unittest.TestCase):
    """The twelve-column block in 5.2 must equal client_metric_names()."""

    def setUp(self) -> None:
        self.text = _chapter_text()

    def test_the_default_column_block_matches_the_emitter(self) -> None:
        block = _fenced_block_after(self.text, "### 5.2 Full column list")
        documented = set(block.split())

        # client_metric_names builds "{split}_loss_avg"; the chapter writes the
        # same names with the literal {split} placeholder kept, so passing the
        # placeholder through as the split name makes the two directly
        # comparable without any string surgery on either side.
        expected = client_metric_names("{split}", ClientStatisticsConfig())

        self.assertEqual(
            documented,
            expected,
            "docs/08-metrics.md 5.2 lists a different column set than "
            "fedbrew.core.config.client_metric_names produces at the "
            "ClientStatisticsConfig defaults",
        )

    def test_every_documented_suffix_is_reachable(self) -> None:
        """No suffix in the 5 table exists that no toggle can produce."""

        every_toggle = ClientStatisticsConfig(
            std=True, variance=True, min=True, max=True, worst_percent=10.0
        )
        reachable = client_metric_names("test", every_toggle)
        # The worst-percent suffix carries the configured percentage in its
        # name, so the chapter documents it as the family `worst{P}` rather
        # than as one arbitrary member of it. Normalise the emitted name to
        # the same spelling so the comparison is about which suffixes exist,
        # not about which percentage this test happened to pick.
        concrete = f"worst{worst_percent_label(every_toggle.worst_percent)}"
        suffixes = {
            "worst{P}" if suffix == concrete else suffix
            for suffix in (
                name[len("test_loss_") :] for name in reachable if name.startswith("test_loss_")
            )
        }

        table = _section(self.text, "**Which clients count:**", "### 5.1")
        documented = set(re.findall(r"^\| `_(\w+(?:\{P\})?)` \|", table, flags=re.MULTILINE))
        self.assertEqual(
            documented,
            suffixes,
            "the suffix table in section 5 and the suffixes "
            "client_metric_names can emit have diverged",
        )

    def test_the_worst_percent_spelling_is_documented_correctly(self) -> None:
        for percent, expected in ((10, "worst10"), (2.5, "worst2p5"), (5, "worst5")):
            self.assertEqual(f"worst{worst_percent_label(percent)}", expected)
            self.assertIn(
                expected,
                self.text,
                f"the chapter must show the {expected} spelling",
            )

    def test_the_client_count_is_emitted_beside_the_aggregates(self) -> None:
        """Not a {metric}_{suffix} name, and deliberately in the set anyway."""

        names = client_metric_names("val", ClientStatisticsConfig())
        self.assertIn("val_num_clients", names)
        self.assertIn("`{split}_num_clients` is emitted beside these", self.text)

    def test_the_personal_prefix_doubles_the_set(self) -> None:
        defaults = ClientStatisticsConfig()
        per_split = len(client_metric_names("test", defaults))
        # Twelve aggregates plus {split}_num_clients, which is not one of
        # them -- it is the |C| every aggregate is over. P07-F06.
        self.assertEqual(per_split, 13, "the chapter states thirteen per split")
        self.assertIn(
            f"Three splits gives {per_split * len(SPLITS)} columns",
            self.text,
        )
        self.assertIn(
            f"doubles that to {per_split * len(SPLITS) * 2}",
            self.text,
        )


class GlossTest(unittest.TestCase):
    """Section 5's Gloss column is quoted from the code, not written here.

    The glosses are what the plan header prints beside every column before a
    run starts. They are the chapter's most quotable content and the content
    most likely to be improved in one place only, so the chapter quotes the
    dictionary rather than paraphrasing it.
    """

    def setUp(self) -> None:
        self.text = _chapter_text()
        self.table = _section(self.text, "| Suffix | Formula | Gloss |", "### 5.1")

    def test_every_suffix_row_quotes_the_gloss_verbatim(self) -> None:
        rows = re.findall(r"^\| `_(\S+)` \| (.*?) \| (.*?) \| ", self.table, flags=re.MULTILINE)
        self.assertEqual(
            len(rows),
            len(METRIC_SUFFIX_GLOSSES),
            "section 5's suffix table and METRIC_SUFFIX_GLOSSES have different numbers of rows",
        )
        documented = {suffix: gloss for suffix, _, gloss in rows}
        self.assertEqual(
            documented,
            dict(METRIC_SUFFIX_GLOSSES),
            "docs/08-metrics.md section 5 and "
            "fedbrew.core.metrics.METRIC_SUFFIX_GLOSSES disagree; the chapter "
            "quotes that dictionary and the plan header prints it",
        )

    def test_the_base_glosses_are_quoted(self) -> None:
        for base, gloss in METRIC_BASE_GLOSSES.items():
            self.assertIn(
                f"{base:<8} -> {gloss}",
                self.text,
                f"the chapter must quote the {base!r} base gloss",
            )

    def test_the_worked_example_is_the_composed_sentence(self) -> None:
        """The chapter shows one composed gloss in full. It has to be the one
        the code actually composes, or the illustration teaches the wrong
        shape."""

        self.assertIn(f"`{metric_gloss('test_accuracy_avg')}`", self.text)

    def test_a_gloss_exists_for_every_column_the_defaults_emit(self) -> None:
        """Bidirectional, like every other check here: a column with no gloss
        is a blank cell in the plan header, which reads as a broken run."""

        for split in SPLITS:
            for name in sorted(client_metric_names(split, ClientStatisticsConfig())):
                gloss = metric_gloss(name)
                self.assertTrue(gloss.endswith("."), f"{name}: {gloss!r} is not a sentence")
                self.assertNotIn(
                    "after local training.",
                    gloss,
                    f"{name} fell through to the generated fallback instead of "
                    "being composed from its base and suffix",
                )


class BaseMetricsTest(unittest.TestCase):
    def test_the_chapter_quotes_the_real_base_tuple(self) -> None:
        text = _chapter_text()
        rendered = "CLIENT_METRIC_BASES = " + repr(tuple(CLIENT_METRIC_BASES)).replace("'", '"')
        self.assertIn(
            rendered,
            text,
            "section 2 must quote fedbrew.core.config.CLIENT_METRIC_BASES "
            "verbatim; it is what every aggregate name is built from",
        )


class FixedCsvSchemaTest(unittest.TestCase):
    """The two hand-fixed column lists in artifacts.py, as quoted in section 9."""

    def setUp(self) -> None:
        self.text = _chapter_text()

    def test_client_metrics_columns_match(self) -> None:
        block = _fenced_block_after(self.text, "### 9.2 `client_metrics.csv`")
        documented = [name.strip() for name in block.replace("\n", " ").split(",")]
        documented = [name for name in documented if name]
        self.assertEqual(
            documented,
            list(_CLIENT_EVALUATION_FIELDS),
            "docs/08-metrics.md 9.2 and artifacts._CLIENT_EVALUATION_FIELDS "
            "disagree; that file has a fixed schema and a resume compares "
            "against its header",
        )

    def test_round_timing_columns_match(self) -> None:
        documented = set(re.findall(r"^\| `([a-z_]+_sec)` \|", self.text, flags=re.MULTILINE))
        self.assertEqual(
            documented,
            set(_ROUND_TIMING_FIELDS),
            "docs/08-metrics.md section 8 and artifacts._ROUND_TIMING_FIELDS disagree",
        )

    def test_round_metrics_column_order_is_documented(self) -> None:
        block = _fenced_block_after(self.text, "### 9.1 `round_metrics.csv`")
        for leading in ("round_id", "num_clients", "num_examples"):
            self.assertIn(leading, block)
        # The timings are appended after the metrics, and the chapter has to
        # say so in the right order or a reader will index the wrong column.
        for column in _ROUND_TIMING_FIELDS:
            self.assertIn(column, block)
        self.assertLess(
            block.index("num_examples"),
            block.index(next(iter(_ROUND_TIMING_FIELDS))),
            "timing columns are appended after the metric columns",
        )


class DocumentedModulePathsTest(unittest.TestCase):
    """Every module the chapter cites must exist."""

    def test_cited_modules_exist(self) -> None:
        text = _chapter_text()
        cited = set(re.findall(r"`(fedbrew/[\w/]+\.py)", text))
        self.assertTrue(cited, "the chapter must cite the modules it derives from")
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
        self.assertEqual(
            missing,
            [],
            f"docs/08-metrics.md cites modules that do not exist: {missing}",
        )

    def test_cited_tests_exist(self) -> None:
        text = _chapter_text()
        cited = set(re.findall(r"`(tests/test_[\w]+\.py)`", text))
        self.assertTrue(cited, "the For agents section must name its guards")
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
        self.assertEqual(
            missing,
            [],
            f"docs/08-metrics.md names tests that do not exist: {missing}",
        )


class PerClientCsvGateTest(unittest.TestCase):
    """One switch gates both per-client CSVs, and the chapter must say so.

    The first draft of section 9 described client_update_metrics.csv as
    written every round from the fit path, with the per_client_csv gate
    mentioned only for its sibling. Running the quickstart disproved it: a
    default run writes round_metrics.csv and run.json and nothing else. The
    claim is cheap to pin, so it is pinned.
    """

    def test_both_per_client_files_are_gated_together(self) -> None:
        from fedbrew.core.runner import _artifact_file_names

        config = _minimal_config()
        config.client_statistics.per_client_csv = False
        self.assertEqual(
            _artifact_file_names(config),
            ["round_metrics.csv", "run.json"],
            "with per_client_csv off, a run writes two metric artifacts",
        )

        config.client_statistics.per_client_csv = True
        self.assertEqual(
            _artifact_file_names(config),
            [
                "round_metrics.csv",
                "client_metrics.csv",
                "client_update_metrics.csv",
                "run.json",
            ],
            "one switch turns on both per-client CSVs",
        )

    def test_the_chapter_says_both_are_gated(self) -> None:
        text = _chapter_text()
        self.assertIn("gates *both* per-client CSVs", text)


def _minimal_config():
    """The smallest FullConfig _artifact_file_names will accept."""

    from fedbrew.core.config import (
        ClientConfig,
        DataConfig,
        ExperimentConfig,
        FullConfig,
        ModelConfig,
        RuntimeConfig,
        ServerConfig,
        TaskConfig,
    )

    return FullConfig(
        experiment=ExperimentConfig(seed=1, output_dir="outputs/docs-test"),
        server=ServerConfig(strategy="fedavg", global_rounds=1, participation_rate=1.0, metrics=[]),
        client=ClientConfig(update_rule="local_sgd", local_iterations=1, batch_size=4, metrics=[]),
        task=TaskConfig(name="classification"),
        data=DataConfig(name="synthetic_classification"),
        model=ModelConfig(name="mlp"),
        runtime=RuntimeConfig(device="cpu", use_amp=False),
    )


if __name__ == "__main__":
    unittest.main()
