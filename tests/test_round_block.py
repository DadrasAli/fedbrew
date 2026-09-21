"""Every column a run can produce classifies, and an unknown one is loud.

The round block's split labels and qualifiers are derived from the column
name. That replaced sixteen hand-written display labels, and it replaced a
failure mode with a different one: a table with a missing entry fell through
to `name.title()` and printed a plausible label for a column nobody had
thought about, while a derivation with a lenient fallback prints a column
under a *heading that is not true of it* -- "algorithm" over a loss, or
"train" over a server quantity.

So the derivation has no lenient fallback. A name neither authority knows
lands in an `unclassified` group drawn in a colour used nowhere else, and this
module fails if any name the real vocabulary can produce lands there.

The vocabulary is enumerated from the code that defines it -- the split
aggregates from `client_metric_names` over every `client_statistics` shape,
the rest from `FIXED_METRIC_GLOSSES` -- rather than listed here, so a column
added tomorrow is checked without editing this file.
"""

from __future__ import annotations

import itertools
import unittest

import pytest

from fedbrew.core.config import (
    CLIENT_METRIC_BASES,
    PERSONAL_SPLIT_PREFIX,
    ClientStatisticsConfig,
    client_metric_names,
)
from fedbrew.core.console import (
    GOLD,
    SPLIT_CENTRAL,
    SPLIT_TEST,
    SPLIT_VAL,
    UNCLASSIFIED,
)
from fedbrew.core.logging import _METRIC_GROUPS, _format_metric, classify_metric
from fedbrew.core.metrics import FIXED_METRIC_GLOSSES

pytestmark = pytest.mark.fast

#: The three splits a client evaluation pass reports under, and the personal
#: twin of each. `_model_scope_splits` builds the same names for `model_scope:
#: personal`, which is what puts a second set of columns in the CSV.
_SPLITS = ("train", "val", "test")


def _every_statistics_shape() -> list[ClientStatisticsConfig]:
    """Every combination of the toggles that add a column.

    `worst_percent` carries its value into the column name, so both a plain
    integer and the fractional form that becomes `worst2p5` are exercised.
    """

    shapes = []
    for std, variance, minimum, maximum in itertools.product((True, False), repeat=4):
        for worst in (None, 10.0, 2.5):
            shapes.append(
                ClientStatisticsConfig(
                    std=std,
                    variance=variance,
                    min=minimum,
                    max=maximum,
                    worst_percent=worst,
                )
            )
    return shapes


#: `central_test_*` is an open class -- `_evaluate_central_test_set` passes
#: through any finite numeric key a task's `evaluate_global` reports, not
#: just the two fixed ones -- so unlike every other column here it cannot be
#: enumerated from code this module can see. These stand in for "any name a
#: task supplies" and exercise the prefix-keyed fallback the same way a real
#: one would.
_TASK_SUPPLIED_CENTRAL_METRICS = (
    "central_test_optimality_gap",
    "central_test_distance_to_optimum",
)


def _producible_columns() -> set[str]:
    """Every column name a run can write, from the code that defines them."""

    names = set(FIXED_METRIC_GLOSSES)
    names.update(_TASK_SUPPLIED_CENTRAL_METRICS)
    for statistics in _every_statistics_shape():
        for split in _SPLITS:
            aggregates = client_metric_names(split, statistics)
            names.update(aggregates)
            names.update(f"{PERSONAL_SPLIT_PREFIX}{name}" for name in aggregates)
    return names


_GROUP_KEYS = {key for key, _, _ in _METRIC_GROUPS}


class EveryProducibleColumnClassifiesTest(unittest.TestCase):
    def test_the_enumeration_found_the_whole_vocabulary(self) -> None:
        """A scan-based guard asserts its scan found something first."""

        columns = _producible_columns()
        # Three splits x two bases x eight statistic suffixes, doubled for
        # the personal twins, plus every fixed diagnostic, plus the
        # representative task-supplied central_test_ names, plus the three
        # personal_{split}_num_clients twins -- the bare three are counted
        # already, being FIXED_METRIC_GLOSSES entries as well as
        # client_metric_names ones.
        self.assertEqual(
            len(columns),
            3 * 2 * 8 * 2
            + len(FIXED_METRIC_GLOSSES)
            + len(_TASK_SUPPLIED_CENTRAL_METRICS)
            + len(_SPLITS),
        )
        for expected in (
            "fit_loss",
            "central_test_accuracy",
            "val_loss_sample_weighted_avg",
            "test_accuracy_worst2p5",
            "personal_val_accuracy_std",
            "optimizer_steps",
            "central_test_optimality_gap",
            "val_num_clients",
            "personal_val_num_clients",
        ):
            self.assertIn(expected, columns)

    def test_none_of_them_is_unclassified(self) -> None:
        offenders = sorted(
            name for name in _producible_columns() if classify_metric(name)[0] == "unclassified"
        )
        self.assertEqual(
            offenders,
            [],
            "these columns print under the 'unclassified' heading: " + ", ".join(offenders),
        )

    def test_every_group_a_column_lands_in_is_one_the_block_prints(self) -> None:
        for name in sorted(_producible_columns()):
            with self.subTest(column=name):
                self.assertIn(classify_metric(name)[0], _GROUP_KEYS)

    def test_no_qualifier_is_empty(self) -> None:
        """An empty cell reads as a rendering fault; an em dash reads as
        'this column has nothing further to say'."""

        for name in sorted(_producible_columns()):
            with self.subTest(column=name):
                self.assertTrue(classify_metric(name)[3].strip())


class TheFallbackIsLoudTest(unittest.TestCase):
    """Guards the guard. A derivation that cannot report a miss is worse than
    the table it replaced, because the miss still prints."""

    def test_a_name_neither_authority_knows_is_unclassified(self) -> None:
        group, label, tone, qualifier = classify_metric("a_column_nobody_declared")
        self.assertEqual(group, "unclassified")
        self.assertEqual(tone, UNCLASSIFIED)
        self.assertEqual(label, "?")
        # The whole name, because nothing about it is redundant with a
        # heading that says only "unclassified".
        self.assertEqual(qualifier, "a_column_nobody_declared")

    def test_a_near_miss_does_not_pass_as_a_real_column(self) -> None:
        """`val_loss_averge` is the shape of a typo that a lenient split on
        underscores would file under "loss" and print as if it were real."""

        for name in ("val_loss_averge", "val_lsos_avg", "vall_loss_avg", "test_accuracy_wors10"):
            with self.subTest(column=name):
                self.assertEqual(classify_metric(name)[0], "unclassified")

    def test_the_unclassified_colour_is_used_for_nothing_else(self) -> None:
        tones = {tone for _, _, tone in _METRIC_GROUPS if tone != UNCLASSIFIED}
        self.assertNotIn(UNCLASSIFIED, tones)
        for name in sorted(_producible_columns()):
            with self.subTest(column=name):
                self.assertNotEqual(classify_metric(name)[2], UNCLASSIFIED)


class TheSplitIsReadFromTheNameTest(unittest.TestCase):
    def test_each_split_gets_its_own_word_and_hue(self) -> None:
        expected = {
            "train_loss_avg": ("train", GOLD),
            "val_loss_avg": ("validation", SPLIT_VAL),
            "test_loss_avg": ("test", SPLIT_TEST),
            "central_test_loss": ("central", SPLIT_CENTRAL),
        }
        for name, (label, tone) in expected.items():
            with self.subTest(column=name):
                _, actual_label, actual_tone, _ = classify_metric(name)
                self.assertEqual((actual_label, actual_tone), (label, tone))

    def test_central_test_is_not_read_as_the_test_split(self) -> None:
        """The longest split token must match first, or every central column
        reads as a client measurement of a split it never touched."""

        self.assertEqual(classify_metric("central_test_loss")[1], "central")
        self.assertNotEqual(classify_metric("central_test_loss")[2], SPLIT_TEST)

    def test_a_task_supplied_central_metric_reads_as_central_too(self) -> None:
        """`_evaluate_central_test_set` passes through any finite numeric key
        a task's `evaluate_global` reports, not just loss/accuracy --
        classify_metric has to place those under the central split with no
        entry in FIXED_METRIC_GLOSSES to read from."""

        group, label, tone, qualifier = classify_metric("central_test_optimality_gap")
        self.assertEqual((label, tone), ("central", SPLIT_CENTRAL))
        self.assertEqual(group, "algorithm")
        self.assertEqual(qualifier, "central_test_optimality_gap")

    def test_fit_shares_the_train_split_and_is_told_apart_by_its_qualifier(self) -> None:
        """Both measure client train data, one phase apart. What separates
        them on screen is that the fit column has no qualifier."""

        fit_group, fit_label, fit_tone, fit_qualifier = classify_metric("fit_loss")
        train_group, train_label, train_tone, train_qualifier = classify_metric(
            "train_loss_sample_weighted_avg"
        )
        self.assertEqual((fit_label, fit_tone), (train_label, train_tone))
        self.assertEqual(fit_group, train_group)
        self.assertEqual(fit_qualifier, "—")
        self.assertEqual(train_qualifier, "sample_weighted_avg")

    def test_a_column_with_no_split_says_so(self) -> None:
        for name in ("optimizer_steps", "server_control_norm", "communicated_bytes"):
            with self.subTest(column=name):
                self.assertEqual(classify_metric(name)[1], "—")

    def test_personal_goes_in_the_qualifier_not_the_split(self) -> None:
        """It names which model was evaluated, not which data -- and
        "personal validation" does not fit an eleven-column label."""

        group, label, tone, qualifier = classify_metric("personal_val_loss_avg")
        self.assertEqual((group, label, tone), ("loss", "validation", SPLIT_VAL))
        self.assertEqual(qualifier, "personal avg")
        pooled = classify_metric("personal_val_loss_sample_weighted_avg")[3]
        self.assertEqual(pooled, "personal sample_weighted_avg")


class TheGroupIsReadFromTheStatisticTest(unittest.TestCase):
    def test_dispersion_suffixes_are_spread_whatever_they_measure(self) -> None:
        for suffix in ("std", "variance", "min", "max", "worst10", "worst2p5"):
            for base in CLIENT_METRIC_BASES:
                with self.subTest(column=f"test_{base}_{suffix}"):
                    group, _, _, qualifier = classify_metric(f"test_{base}_{suffix}")
                    self.assertEqual(group, "spread")
                    # The heading names neither the metric nor the statistic,
                    # so the qualifier keeps both.
                    self.assertEqual(qualifier, f"{base}_{suffix}")

    def test_the_two_averages_are_the_metric_s_own_group(self) -> None:
        for suffix in ("sample_weighted_avg", "avg"):
            self.assertEqual(classify_metric(f"val_loss_{suffix}")[0], "loss")
            self.assertEqual(classify_metric(f"val_accuracy_{suffix}")[0], "accuracy")

    def test_a_fixed_metric_ending_in_a_base_joins_that_base_s_group(self) -> None:
        """`fit_proximal_loss` belongs beside the loss the run minimises, not
        in the diagnostics."""

        self.assertEqual(classify_metric("fit_proximal_loss")[0], "loss")
        self.assertEqual(classify_metric("fit_proximal_loss")[3], "proximal")
        self.assertEqual(classify_metric("fit_total_loss")[0], "loss")

    def test_everything_else_is_an_algorithm_diagnostic_keeping_its_name(self) -> None:
        for name in ("optimizer_steps", "client_learning_rate", "server_control_norm"):
            with self.subTest(column=name):
                group, _, _, qualifier = classify_metric(name)
                self.assertEqual(group, "algorithm")
                self.assertEqual(qualifier, name)


class CountsRenderAsCountsTest(unittest.TestCase):
    """`optimizer_steps 40.0000` invites a reader to wonder what the
    fractional part of a step would be, and the doubt spreads to the losses
    beside it."""

    def test_an_integral_algorithm_value_loses_its_decimal_places(self) -> None:
        for name, value in (
            ("optimizer_steps", 40.0),
            ("communicated_bytes", 1048576.0),
            ("trainable_parameters", 21840.0),
        ):
            with self.subTest(column=name):
                self.assertEqual(_format_metric(name, value, "algorithm"), str(int(value)))

    def test_a_fractional_one_keeps_them(self) -> None:
        """`local_steps` is an example-weighted mean across clients, so it is
        genuinely fractional and rounding it to a count would be a lie."""

        self.assertEqual(_format_metric("local_steps", 12.5, "algorithm"), "12.5000")
        self.assertEqual(_format_metric("client_learning_rate", 0.05, "algorithm"), "0.0500")

    def test_a_loss_that_lands_on_a_whole_number_is_still_a_loss(self) -> None:
        """The rule is confined to the algorithm group in both directions:
        printing 2.0000 as `2` would claim a precision it does not have."""

        self.assertEqual(_format_metric("fit_loss", 2.0, "loss"), "2.0000")
        self.assertEqual(_format_metric("test_loss_min", 1.0, "spread"), "1.0000")
        self.assertEqual(_format_metric("val_accuracy_avg", 0.5, "accuracy"), "50.00%")


class TimingColumnsAreNotMetricsTest(unittest.TestCase):
    """The round CSV carries `duration_sec` and its four siblings, and the
    block does not: they reach the CSV from `record.timings`, not from
    `record.metrics`. Were they ever merged in, they would print under the
    loud heading rather than as a diagnostic of the model -- which is the
    right answer, and this pins that it stays the answer."""

    def test_a_timing_would_be_loud_rather_than_filed_as_a_diagnostic(self) -> None:
        for name in ("duration_sec", "fit_sec", "aggregate_sec", "client_eval_sec"):
            with self.subTest(column=name):
                self.assertEqual(classify_metric(name)[0], "unclassified")


if __name__ == "__main__":
    unittest.main()
