"""Per-split evaluation schedules and the client-statistics toggles.

Evaluation can be the expensive part of a round (chapter 11 §1), so each split
is priced separately. These tests pin down when a
split runs and which statistics come out of it.
"""

from __future__ import annotations

import unittest

import pytest

from fedbrew.core.config import (
    ClientStatisticsConfig,
    evaluates_round,
    parse_evaluation_schedule,
)
from fedbrew.core.loop import _client_distribution_statistics

pytestmark = pytest.mark.fast


class ScheduleParsingTests(unittest.TestCase):
    def test_the_three_forms_parse(self) -> None:
        self.assertEqual(parse_evaluation_schedule(10, "evaluation.test"), 10)
        self.assertEqual(parse_evaluation_schedule("10", "evaluation.test"), 10)
        self.assertEqual(parse_evaluation_schedule("final", "evaluation.test"), 0)
        self.assertIsNone(parse_evaluation_schedule("never", "evaluation.test"))

    def test_anything_else_is_rejected(self) -> None:
        for value in (0, -1, "sometimes", "", True, 2.5, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_evaluation_schedule(value, "evaluation.test")


class ScheduleTests(unittest.TestCase):
    def test_an_interval_pins_both_endpoints(self) -> None:
        # A 25-round run with every: 10 would otherwise start at round 10 and
        # end on an unmeasured round: no baseline to compare against, and a
        # reported number 5 rounds stale.
        rounds = [r for r in range(1, 26) if evaluates_round(10, r, 25)]

        self.assertEqual(rounds, [1, 10, 20, 25])

    def test_a_sparse_schedule_still_measures_the_start(self) -> None:
        # every: 50 over 100 rounds without round 1 would first measure the
        # model halfway through training, which is the case that makes a
        # plotted curve misleading rather than merely coarse.
        self.assertEqual([r for r in range(1, 101) if evaluates_round(50, r, 100)], [1, 50, 100])

    def test_an_interval_longer_than_the_run_still_gives_two_points(self) -> None:
        self.assertEqual([r for r in range(1, 21) if evaluates_round(50, r, 20)], [1, 20])

    def test_every_round_is_unaffected(self) -> None:
        self.assertEqual([r for r in range(1, 6) if evaluates_round(1, r, 5)], [1, 2, 3, 4, 5])

    def test_final_evaluates_only_the_last_round(self) -> None:
        # Exempt from the round-1 pin: "final" means the last round only, and
        # silently adding a second measurement would override the request.
        self.assertEqual([r for r in range(1, 26) if evaluates_round(0, r, 25)], [25])

    def test_never_evaluates_nothing_including_the_final_round(self) -> None:
        self.assertEqual([r for r in range(1, 26) if evaluates_round(None, r, 25)], [])


class ClientStatisticsTests(unittest.TestCase):
    _ACCURACIES = [0.1, 0.9, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 1.0]

    def _stats(self, metric: str = "accuracy", **overrides: object) -> dict[str, float]:
        config = ClientStatisticsConfig(**overrides)  # type: ignore[arg-type]
        return _client_distribution_statistics(
            f"test_{metric}", metric, list(self._ACCURACIES), config
        )

    def test_each_toggle_is_independent(self) -> None:
        only_std = self._stats(std=True, variance=False, min=False, max=False, worst_percent=None)
        self.assertEqual(set(only_std), {"test_accuracy_std"})

        only_worst = self._stats(std=False, variance=False, min=False, max=False, worst_percent=20)
        self.assertEqual(set(only_worst), {"test_accuracy_worst20"})

    def test_all_off_emits_nothing(self) -> None:
        self.assertEqual(
            self._stats(std=False, variance=False, min=False, max=False, worst_percent=None),
            {},
        )

    def test_worst_is_the_lowest_accuracies(self) -> None:
        stats = self._stats(worst_percent=20, std=False, min=False, max=False)

        # 20% of 10 clients = the two lowest accuracies, 0.1 and 0.2.
        self.assertAlmostEqual(stats["test_accuracy_worst20"], 0.15)

    def test_worst_is_the_highest_losses(self) -> None:
        # "Worst" follows the metric, not the number: a high loss is bad. This
        # is why the name is "worst" and not "bottom" -- bottom would have
        # meant opposite things for accuracy and loss.
        stats = self._stats(metric="loss", worst_percent=20, std=False, min=False, max=False)

        self.assertAlmostEqual(stats["test_loss_worst20"], 0.95)

    def test_worst_always_covers_at_least_one_client(self) -> None:
        stats = self._stats(worst_percent=1, std=False, min=False, max=False)

        self.assertAlmostEqual(stats["test_accuracy_worst1"], 0.1)

    def test_variance_is_the_square_of_std(self) -> None:
        stats = self._stats(std=True, variance=True, min=False, max=False)

        self.assertAlmostEqual(stats["test_accuracy_variance"], stats["test_accuracy_std"] ** 2)

    def test_min_and_max_are_the_literal_extremes(self) -> None:
        stats = self._stats(std=False, min=True, max=True, worst_percent=None)

        self.assertAlmostEqual(stats["test_accuracy_min"], 0.1)
        self.assertAlmostEqual(stats["test_accuracy_max"], 1.0)

    def test_a_fractional_percent_keeps_a_readable_metric_name(self) -> None:
        stats = self._stats(worst_percent=2.5, std=False, min=False, max=False)

        self.assertIn("test_accuracy_worst2p5", stats)


if __name__ == "__main__":
    unittest.main()
