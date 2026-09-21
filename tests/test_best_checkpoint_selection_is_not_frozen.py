"""A round that reports NaN is skipped, not made the run's best forever.

`_is_better_checkpoint_metric` took any first value: `best_metric_value is
None` returned True before anything was compared. A NaN first observation was
therefore accepted, `best.pt` was written at that round, and every later
comparison became `x < nan` or `x > nan` -- both False, in `min` mode and
`max` mode alike. The run kept that checkpoint to the end and reported
`best_metric_value: nan` with `best_round_id: 1`.

A NaN arriving *after* a real value was always rejected correctly, by the same
two comparisons. The defect was the first observation alone, which is why a
test that feeds NaN in the middle of a history passes either way and proves
nothing; `TheFirstObservationTest` puts it first.

The first observation is reachable rather than hypothetical. `_overflow_safe`
answers NaN by design for a statistic that cannot fit -- POST-F02 -- and
`val_loss_avg` goes through `statistics.fmean` and is a legal
`checkpointing.best_metric`. A run whose first evaluated round is already
diverging produces exactly this input.

Infinities are refused the same way rather than compared: `+inf` beats every
later accuracy under `max` and `-inf` every later loss under `min`, so
accepting either freezes `best.pt` identically while looking like a comparison
that worked.

Both call sites are covered, because the live one and the resume one share the
predicate: `_update_checkpoints` runs it each round, and
`_initialize_checkpoint_tracker` replays it over the history read back off
disk -- so a NaN in a resumed run's replayed rounds would have frozen the
resumed run too.
"""

from __future__ import annotations

import io
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from typing import Any

import pytest

from fedbrew.core.loop import (
    _checkpoint_metric_value,
    _initialize_checkpoint_tracker,
    _is_better_checkpoint_metric,
    _warn_once_on_a_non_finite_selection_metric,
)
from fedbrew.core.state import MetricRecord

pytestmark = pytest.mark.fast

#: Not finite, and each freezes the comparison in a different direction.
NON_FINITE = (float("nan"), float("inf"), float("-inf"))


def _accepted_rounds(history: list[float], mode: str) -> tuple[list[int], Any]:
    """Which rounds would write best.pt, and the value the run would report."""

    best: Any = None
    accepted: list[int] = []
    for round_id, value in enumerate(history, start=1):
        if _is_better_checkpoint_metric(value, best, mode):
            best = value
            accepted.append(round_id)
    return accepted, best


class TheFirstObservationTest(unittest.TestCase):
    def test_a_non_finite_first_value_is_not_the_run_s_best(self) -> None:
        for value in NON_FINITE:
            for mode in ("max", "min"):
                with self.subTest(value=value, mode=mode):
                    self.assertFalse(_is_better_checkpoint_metric(value, None, mode))

    def test_the_rest_of_the_run_still_selects(self) -> None:
        """The freeze, stated as what the run would have done."""

        history = [math.nan, 0.2, 0.5, 0.9, 0.1]
        self.assertEqual(_accepted_rounds(history, "max"), ([2, 3, 4], 0.9))
        self.assertEqual(_accepted_rounds(history, "min"), ([2, 5], 0.1))

    def test_an_all_non_finite_run_selects_nothing(self) -> None:
        """Which is right: there is no round whose score can be compared."""

        accepted, best = _accepted_rounds([math.nan] * 5, "max")
        self.assertEqual(accepted, [])
        self.assertIsNone(best)


class ItIsNarrowTest(unittest.TestCase):
    """The finite path must not move; every real run is on it."""

    def test_a_finite_first_value_is_still_taken(self) -> None:
        for mode in ("max", "min"):
            with self.subTest(mode=mode):
                self.assertTrue(_is_better_checkpoint_metric(0.5, None, mode))

    def test_ordinary_comparisons_are_unchanged(self) -> None:
        self.assertTrue(_is_better_checkpoint_metric(0.9, 0.5, "max"))
        self.assertFalse(_is_better_checkpoint_metric(0.1, 0.5, "max"))
        self.assertTrue(_is_better_checkpoint_metric(0.1, 0.5, "min"))
        self.assertFalse(_is_better_checkpoint_metric(0.9, 0.5, "min"))

    def test_zero_is_a_value_and_not_a_missing_one(self) -> None:
        """`if not best_metric_value` would be the obvious wrong rewrite here."""

        self.assertFalse(_is_better_checkpoint_metric(0.5, 0.0, "min"))
        self.assertTrue(_is_better_checkpoint_metric(0.5, 0.0, "max"))

    def test_a_non_finite_value_arriving_later_was_always_refused(self) -> None:
        """So the fix adds nothing here; recorded to keep the scope honest."""

        for value in NON_FINITE:
            for mode in ("max", "min"):
                with self.subTest(value=value, mode=mode):
                    self.assertFalse(_is_better_checkpoint_metric(value, 0.5, mode))


class WhereTheNanComesFromTest(unittest.TestCase):
    """The premise. Without it this guard protects nothing a run can reach."""

    def test_the_metric_reader_passes_nan_through(self) -> None:
        """It answers "is this a number", and NaN is one."""

        self.assertTrue(math.isnan(_checkpoint_metric_value({"m": math.nan}, "m") or 0.0))
        self.assertIsNone(_checkpoint_metric_value({"m": "x"}, "m"))
        self.assertIsNone(_checkpoint_metric_value({"m": True}, "m"))

    def test_overflow_safe_answers_nan_by_design(self) -> None:
        """POST-F02's fix is what puts a NaN in a val_ column of a real run."""

        import statistics

        from fedbrew.core.loop import _overflow_safe

        self.assertTrue(math.isnan(_overflow_safe(statistics.fmean, [1e308] * 3)))


class TheResumePathTest(unittest.TestCase):
    """`_initialize_checkpoint_tracker` replays the same predicate over history."""

    def _tracker(self, values: list[float]) -> dict[str, Any]:
        history = [
            MetricRecord(
                round_id=index,
                num_clients=1,
                num_examples=1,
                metrics={"val_loss_avg": value},
            )
            for index, value in enumerate(values, start=1)
        ]
        # A real directory: the function returns early on None, before it
        # replays anything.
        with tempfile.TemporaryDirectory() as directory:
            return _initialize_checkpoint_tracker(
                directory, {"save_best": True, "best_metric": "val_loss_avg"}, history
            )

    def test_a_replayed_nan_does_not_freeze_the_resumed_run(self) -> None:
        tracker = self._tracker([math.nan, 0.8, 0.3, 0.5])
        self.assertEqual(tracker["best_metric_value"], 0.3)
        self.assertEqual(tracker["best_round_id"], 3)

    def test_a_replayed_finite_history_is_unchanged(self) -> None:
        tracker = self._tracker([0.8, 0.3, 0.5])
        self.assertEqual(tracker["best_metric_value"], 0.3)
        self.assertEqual(tracker["best_round_id"], 2)


class ItSaysSoOnceTest(unittest.TestCase):
    """Refusing silently would leave the other half: best.pt stops tracking."""

    def _lines(self, values: list[float | None]) -> list[str]:
        tracker: dict[str, Any] = {}
        stream = io.StringIO()
        with redirect_stdout(stream):
            for round_id, value in enumerate(values, start=1):
                _warn_once_on_a_non_finite_selection_metric(
                    tracker, "val_loss_avg", value, round_id
                )
        return [line for line in stream.getvalue().splitlines() if line]

    def test_the_first_non_finite_round_says_so(self) -> None:
        lines = self._lines([math.nan])
        self.assertEqual(len(lines), 1)
        self.assertIn("val_loss_avg", lines[0])
        self.assertIn("round 1", lines[0])
        self.assertIn("best.pt", lines[0])

    def test_a_diverging_run_says_it_once_and_not_every_round(self) -> None:
        self.assertEqual(len(self._lines([math.nan] * 500)), 1)

    def test_a_healthy_run_says_nothing(self) -> None:
        self.assertEqual(self._lines([0.5, 0.4, 0.3]), [])

    def test_an_absent_metric_is_not_this_warning_s_business(self) -> None:
        """P10-F04 covers a best_metric no column carries, at config load."""

        self.assertEqual(self._lines([None, None]), [])


if __name__ == "__main__":
    unittest.main()
