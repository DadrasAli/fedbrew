"""Every column a run writes has a sentence, and the right one.

The plan header prints one gloss per column before a run starts. Two things
have to hold for that to be worth printing.

**Every column has one.** The glosses are composed from the base metric and
the suffix -- the same two pieces `client_metric_names` builds the name from --
so a column cannot exist without a gloss. What can go wrong is the two pieces
drifting apart: a third base metric added to `CLIENT_METRIC_BASES` with no
gloss beside it would fall through to a sentence generated from the column
name, which reads like a definition and is not one.

**`_avg` and `_sample_weighted_avg` differ in words.** They are one token apart
in the name and both plausible as numbers, and on a cross-device split they are
substantially different quantities. A reader checking which one a table quotes
has only the gloss to go on, so a gloss that distinguishes them only by
restating the formula would be no help at the moment it is needed.
"""

from __future__ import annotations

import unittest

import pytest

from fedbrew.core.config import (
    CLIENT_METRIC_BASES,
    ClientStatisticsConfig,
    client_metric_names,
    worst_percent_label,
)
from fedbrew.core.metrics import (
    CLIENT_UNFILTERED_FIT_METRICS,
    FIXED_METRIC_GLOSSES,
    METRIC_BASE_GLOSSES,
    METRIC_SUFFIX_GLOSSES,
    SERVER_DIAGNOSTIC_METRICS,
    SPLIT_GLOSSES,
    metric_gloss,
)

pytestmark = pytest.mark.fast

SPLITS = ("train", "val", "test")


class TheGlossKeysTrackTheEmittersTest(unittest.TestCase):
    def test_the_base_glosses_are_exactly_the_base_metrics(self) -> None:
        """metrics.py is a leaf and cannot import CLIENT_METRIC_BASES without
        a cycle -- config.py imports *it*, lazily, for that reason. This is
        the check that stands in for the import."""

        self.assertEqual(
            set(METRIC_BASE_GLOSSES),
            set(CLIENT_METRIC_BASES),
            "fedbrew.core.metrics.METRIC_BASE_GLOSSES and "
            "fedbrew.core.config.CLIENT_METRIC_BASES have diverged; a base "
            "metric with no gloss prints a generated sentence in the plan "
            "header, and a gloss with no base metric is dead text",
        )

    def test_the_suffix_glosses_cover_every_suffix_a_toggle_can_produce(self) -> None:
        every_toggle = ClientStatisticsConfig(
            std=True, variance=True, min=True, max=True, worst_percent=10.0
        )
        emitted = client_metric_names("test", every_toggle)
        suffixes = {name[len("test_loss_") :] for name in emitted if name.startswith("test_loss_")}
        concrete = f"worst{worst_percent_label(every_toggle.worst_percent)}"
        documented = {"worst{P}" if s == concrete else s for s in suffixes}
        self.assertEqual(documented, set(METRIC_SUFFIX_GLOSSES))

    def test_every_split_the_loop_aggregates_has_a_gloss(self) -> None:
        self.assertEqual(set(SPLIT_GLOSSES), set(SPLITS))


class TheComposedSentenceTest(unittest.TestCase):
    def test_every_default_column_composes_rather_than_falling_through(self) -> None:
        for split in SPLITS:
            for name in sorted(client_metric_names(split, ClientStatisticsConfig())):
                with self.subTest(column=name):
                    gloss = metric_gloss(name)
                    self.assertTrue(gloss.endswith("."))
                    self.assertNotIn("after local training.", gloss, "fell through to the fallback")
                    if name == f"{split}_num_clients":
                        # Not a {split}_{base}_{suffix} name, so there is
                        # nothing to compose from: it is a fixed gloss, and
                        # the assertion for it is the one below. P07-F06.
                        continue
                    self.assertIn(SPLIT_GLOSSES[split], gloss)

    def test_the_client_count_names_its_split_and_what_it_excludes(self) -> None:
        for split in SPLITS:
            with self.subTest(split=split):
                gloss = metric_gloss(f"{split}_num_clients")
                self.assertIn(f"{split} aggregates", gloss)
                self.assertIn(f"zero {split} examples", gloss)

    def test_the_two_averages_are_distinguished_in_words(self) -> None:
        pooled = metric_gloss("test_accuracy_sample_weighted_avg")
        uniform = metric_gloss("test_accuracy_avg")

        self.assertNotEqual(pooled, uniform)
        # Not merely different strings: each has to say which way it weights,
        # in words a reader can act on without re-deriving the formula.
        self.assertIn("pooled over examples", pooled)
        self.assertIn("averaged over clients", uniform)
        self.assertNotIn("averaged over clients", pooled)
        self.assertNotIn("pooled over examples", uniform)

    def test_a_personal_column_says_which_model_measured_it(self) -> None:
        """`personal_` prefixes the split but names the *model*: the same
        examples, a different set of weights. A gloss that repeated the global
        one would describe the wrong measurement."""

        personal = metric_gloss("personal_test_accuracy_avg")
        self.assertIn("own model", personal)
        self.assertNotIn("own model", metric_gloss("test_accuracy_avg"))
        self.assertIn(SPLIT_GLOSSES["test"], personal)

    def test_the_worst_percent_gloss_carries_the_configured_percentage(self) -> None:
        for label, percent in (("worst10", "10"), ("worst2p5", "2.5"), ("worst5", "5")):
            with self.subTest(suffix=label):
                self.assertIn(f"worst {percent}% of clients", metric_gloss(f"test_loss_{label}"))

    def test_a_split_override_changes_what_the_gloss_claims_was_measured(self) -> None:
        """`evaluation.train.clients: participating` measures this round's
        trainers, not every client. The column name cannot carry that, so the
        gloss has to."""

        default = metric_gloss("train_loss_avg")
        participating = metric_gloss(
            "train_loss_avg", split_glosses={**SPLIT_GLOSSES, "train": "the selected clients"}
        )
        self.assertIn("client train data", default)
        self.assertIn("the selected clients", participating)


class TheCentralPassFallbackTest(unittest.TestCase):
    """`_evaluate_central_test_set` passes through any finite numeric key a
    task's `evaluate_global` reports, not just loss/accuracy -- so unlike the
    split/base/suffix vocabulary above, central_test_ names cannot be composed
    or enumerated in FIXED_METRIC_GLOSSES ahead of time."""

    def test_a_task_supplied_central_metric_gets_a_real_sentence(self) -> None:
        gloss = metric_gloss("central_test_optimality_gap")
        self.assertTrue(gloss.endswith("."))
        self.assertIn("Optimality Gap", gloss)
        self.assertIn("global test set", gloss)
        self.assertNotIn("after local training.", gloss, "fell through to the generic fallback")

    def test_it_does_not_shadow_the_two_names_with_a_gloss_on_file(self) -> None:
        self.assertEqual(
            metric_gloss("central_test_loss"), FIXED_METRIC_GLOSSES["central_test_loss"]
        )
        self.assertEqual(
            metric_gloss("central_test_accuracy"), FIXED_METRIC_GLOSSES["central_test_accuracy"]
        )


class TheDiagnosticsHaveGlossesTest(unittest.TestCase):
    """The columns a `client.metrics`/`server.metrics` list cannot remove.

    They appear in every run of the strategy that emits them, and had no gloss
    at all before the plan header needed one -- `communicated_bytes` described
    itself as an example-weighted mean of a loss.
    """

    def test_every_unfilterable_column_has_a_written_gloss(self) -> None:
        names = set()
        for metrics in SERVER_DIAGNOSTIC_METRICS.values():
            names |= set(metrics)
        for metrics in CLIENT_UNFILTERED_FIT_METRICS.values():
            names |= set(metrics)
        missing = sorted(name for name in names if name not in FIXED_METRIC_GLOSSES)
        self.assertEqual(
            missing,
            [],
            "these columns reach round_metrics.csv on every run of their "
            f"strategy and have no gloss: {missing}",
        )


if __name__ == "__main__":
    unittest.main()
