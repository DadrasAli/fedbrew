"""The divergence monitor: what stops a run early, and what must not.

An arm that stops learning in its first few rounds would otherwise spend the
rest of its budget, so the detectors have to fire fast. They also have to be silent on
healthy runs, because a false positive silently truncates an experiment and the
truncation looks like a result. These tests pin down both directions.
"""

from __future__ import annotations

import ast
import math
import unittest
from pathlib import Path

import pytest

from fedbrew.core.config import (
    ClientStatisticsConfig,
    DivergenceConfig,
    _build_divergence_config,
    _validate_divergence,
    _validate_known_keys,
)
from fedbrew.core.divergence import (
    STATUS_DIVERGED,
    STATUS_STALLED,
    DivergenceMonitor,
)
from fedbrew.core.loop import _client_distribution_statistics

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(config: DivergenceConfig, values: list[float], metric: str = "fit_loss"):
    """Feed a loss trajectory and return the first verdict, or None."""

    monitor = DivergenceMonitor(config)
    for round_id, value in enumerate(values, start=1):
        verdict = monitor.update(round_id, {metric: value})
        if verdict is not None:
            return verdict
    return None


class NonFiniteTests(unittest.TestCase):
    def test_nan_stops_the_run_on_the_round_it_appears(self) -> None:
        verdict = _run(DivergenceConfig(), [2.3, 2.1, float("nan"), 1.9])

        assert verdict is not None
        self.assertEqual(verdict.status, STATUS_DIVERGED)
        self.assertEqual(verdict.detector, "non_finite")
        self.assertEqual(verdict.round_id, 3)

    def test_inf_is_treated_the_same_as_nan(self) -> None:
        verdict = _run(DivergenceConfig(), [2.3, float("inf")])

        assert verdict is not None
        self.assertEqual(verdict.detector, "non_finite")

    def test_a_non_finite_round_never_poisons_the_baseline_when_disabled(self) -> None:
        # With the detector off, a NaN must not become the anchor the relative
        # check measures against; the run continues on the finite values.
        config = DivergenceConfig(non_finite=False, blowup_factor=10.0)
        verdict = _run(config, [2.0, float("nan"), 2.1, 2.2])

        self.assertIsNone(verdict)


class BlowupTests(unittest.TestCase):
    def test_a_run_that_explodes_is_caught_within_a_round_of_crossing(self) -> None:
        verdict = _run(DivergenceConfig(blowup_factor=10.0), [2.0, 5.0, 19.0, 21.0])

        assert verdict is not None
        self.assertEqual(verdict.status, STATUS_DIVERGED)
        self.assertEqual(verdict.detector, "blowup")
        self.assertEqual(verdict.round_id, 4)
        self.assertAlmostEqual(verdict.threshold or 0.0, 20.0)

    def test_a_healthy_decreasing_run_is_never_touched(self) -> None:
        self.assertIsNone(
            _run(DivergenceConfig(blowup_factor=10.0), [2.3, 2.0, 1.7, 1.4, 1.0, 0.6])
        )

    def test_ordinary_round_to_round_noise_does_not_fire(self) -> None:
        # Sampling a different 1-2% of clients each round makes the metric
        # jump around. Anything that fires on this is unusable in a real sweep.
        noisy = [2.3, 2.5, 2.4, 2.6, 2.35, 2.55, 2.3, 2.45, 2.2, 2.4]
        self.assertIsNone(_run(DivergenceConfig(blowup_factor=10.0), noisy))

    def test_the_anchor_is_the_first_positive_value_not_the_first(self) -> None:
        # On a one-label-per-client split every client fits its single label
        # exactly, so fit_loss is genuinely 0.0 in early rounds. Anchoring
        # there would give a ceiling of zero and disable the detector for the
        # rest of the run.
        verdict = _run(DivergenceConfig(blowup_factor=10.0), [0.0, 0.0, 2.0, 25.0])

        assert verdict is not None
        self.assertEqual(verdict.detector, "blowup")
        self.assertAlmostEqual(verdict.threshold or 0.0, 20.0)


class AbsoluteCeilingTests(unittest.TestCase):
    def test_it_catches_a_run_that_was_already_pathological_when_first_seen(
        self,
    ) -> None:
        # The relative check has nothing to anchor on here: the first positive
        # observation is itself the blow-up. This is the real trajectory of an
        # lr=500 MNIST run, which the relative detector alone missed entirely.
        traj = [0.0, 0.0, 10246.99, 99.2, 0.23, 23.5]
        # 10x ln(10), the random-guess cross-entropy for 10 classes.
        verdict = _run(DivergenceConfig(blowup_absolute=23.0), traj)

        assert verdict is not None
        self.assertEqual(verdict.detector, "blowup_absolute")
        self.assertEqual(verdict.round_id, 3)

    def test_it_stays_silent_below_the_ceiling(self) -> None:
        self.assertIsNone(_run(DivergenceConfig(blowup_absolute=23.0), [2.3, 5.0, 22.9]))


class PatienceTests(unittest.TestCase):
    def test_a_plateau_is_stalled_not_diverged(self) -> None:
        # A run that stopped improving is a different claim from one that blew
        # up, and reporting it as divergence would misstate the sweep.
        config = DivergenceConfig(patience=3, blowup_factor=None)
        verdict = _run(config, [2.0, 1.0, 1.1, 1.2, 1.05, 1.3])

        assert verdict is not None
        self.assertEqual(verdict.status, STATUS_STALLED)
        self.assertEqual(verdict.detector, "patience")
        self.assertAlmostEqual(verdict.threshold or 0.0, 1.0)

    def test_any_improvement_resets_the_counter(self) -> None:
        config = DivergenceConfig(patience=3, blowup_factor=None)
        # Improves on round 4, so the three flat rounds after it are not enough.
        self.assertIsNone(_run(config, [2.0, 2.1, 2.05, 1.5, 1.6, 1.55]))

    def test_min_delta_ignores_noise_sized_improvements(self) -> None:
        # A 0.1% gain should not keep a stalled run alive when 5% is required.
        config = DivergenceConfig(patience=2, blowup_factor=None, min_delta=0.05)
        verdict = _run(config, [1.0, 0.999, 0.998, 0.997])

        assert verdict is not None
        self.assertEqual(verdict.status, STATUS_STALLED)

    def test_patience_is_off_by_default(self) -> None:
        # It is the only detector that can be wrong, and the right value is
        # dataset-dependent, so it must be asked for explicitly.
        self.assertIsNone(DivergenceConfig().patience)


class MonitorContractTests(unittest.TestCase):
    def test_every_detector_off_never_fires(self) -> None:
        """Off is expressed by the detectors, not by a separate bool."""

        config = DivergenceConfig(
            non_finite=False,
            blowup_factor=None,
            blowup_absolute=None,
            patience=None,
        )
        self.assertFalse(config.active)
        self.assertIsNone(_run(config, [1.0, float("nan"), 1e9]))

    def test_rounds_without_the_metric_are_skipped_not_counted(self) -> None:
        # A schedule-gated metric is absent on most rounds. Those rounds must
        # not burn patience, or the counter would measure the schedule rather
        # than the run.
        monitor = DivergenceMonitor(
            DivergenceConfig(metric="val_loss_avg", patience=2, blowup_factor=None)
        )
        monitor.update(1, {"val_loss_avg": 1.0})
        for round_id in range(2, 20):
            self.assertIsNone(monitor.update(round_id, {"fit_loss": 0.5}))
        self.assertIsNone(monitor.update(20, {"val_loss_avg": 1.1}))
        self.assertIsNotNone(monitor.update(21, {"val_loss_avg": 1.2}))

    def test_only_the_first_verdict_is_returned(self) -> None:
        monitor = DivergenceMonitor(DivergenceConfig())
        monitor.update(1, {"fit_loss": 2.0})
        first = monitor.update(2, {"fit_loss": float("nan")})

        self.assertIsNotNone(first)
        self.assertIsNone(monitor.update(3, {"fit_loss": float("nan")}))
        self.assertIs(monitor.verdict, first)

    def test_the_termination_block_does_not_repeat_the_status(self) -> None:
        # status is a top-level key in run.json, and run.json's contract is
        # that nothing appears in it twice.
        monitor = DivergenceMonitor(DivergenceConfig())
        monitor.update(1, {"fit_loss": 2.0})
        verdict = monitor.update(2, {"fit_loss": float("nan")})

        assert verdict is not None
        self.assertNotIn("status", verdict.as_dict())
        self.assertEqual(
            set(verdict.as_dict()),
            {"detector", "round_id", "metric", "value", "threshold", "reason"},
        )


class NonFiniteReachesTheMonitorTests(unittest.TestCase):
    """A diverged run is a recorded outcome, so nothing may crash ahead of the monitor.

    Regression test for a crash that beat the monitor to it. The per-client
    split aggregation runs
    86 lines before monitor.update(); statistics.pstdev/pvariance refuse a NaN
    with ValueError, so a blow-up on an evaluation round killed the process and
    the arm left no run.json, no round_metrics.csv and no runs_index.jsonl row --
    exactly the arms the sweep record exists to remember. min/max/sorted are the
    quieter half of the same problem: NaN compares False against everything, so
    they return whichever neighbour it landed next to.
    """

    #: Everything on, as every tracked config sets it.
    _ALL = ClientStatisticsConfig(std=True, variance=True, min=True, max=True, worst_percent=25.0)

    def _stats(self, values: list[float], metric: str = "loss") -> dict[str, float]:
        return _client_distribution_statistics("val_loss", metric, values, self._ALL)

    def test_a_non_finite_client_does_not_raise(self) -> None:
        for position, name in ((0, "first"), (1, "middle"), (2, "last")):
            values = [1.0, 2.0, 3.0]
            values[position] = float("nan")
            with self.subTest(nan=name):
                self._stats(values)  # must not raise

    def test_every_dispersion_statistic_is_nan_not_a_neighbour(self) -> None:
        """min/max/worst must not report a real client's value as the summary."""

        computed = self._stats([1.0, float("nan"), 3.0])
        self.assertTrue(computed)
        for name, value in computed.items():
            with self.subTest(metric=name):
                self.assertTrue(math.isnan(value), f"{name} = {value}")

    def test_the_column_set_is_identical_finite_or_not(self) -> None:
        """A round that dropped columns would change the CSV schema mid-run."""

        self.assertEqual(
            sorted(self._stats([1.0, 2.0, 3.0])),
            sorted(self._stats([1.0, float("nan"), 3.0])),
        )

    def test_the_monitor_then_stops_the_run_cleanly(self) -> None:
        """The NaN the statistics now emit is what the non_finite detector reads."""

        computed = self._stats([1.0, float("nan"), 3.0])
        verdict = DivergenceMonitor(DivergenceConfig(metric="val_loss_std")).update(7, computed)

        assert verdict is not None
        self.assertEqual(verdict.status, STATUS_DIVERGED)
        self.assertEqual(verdict.detector, "non_finite")

    def test_finite_values_are_untouched(self) -> None:
        computed = self._stats([1.0, 2.0, 3.0])
        self.assertAlmostEqual(computed["val_loss_min"], 1.0)
        self.assertAlmostEqual(computed["val_loss_max"], 3.0)
        self.assertAlmostEqual(computed["val_loss_worst25"], 3.0)

    def test_worst_still_follows_the_metric_direction_when_finite(self) -> None:
        accuracy = _client_distribution_statistics(
            "val_accuracy", "accuracy", [0.1, 0.5, 0.9], self._ALL
        )
        self.assertAlmostEqual(accuracy["val_accuracy_worst25"], 0.1)


class PrimingAfterResumeTests(unittest.TestCase):
    """Regression tests for a resumed run rebuilding the monitor empty.

    A resumed run builds a fresh monitor. FEMNIST fit_loss starts near
    ln 62 = 4.1 -- ceiling 41 at blowup_factor 10 -- and is ~0.3 by round 300,
    so a requeue there re-anchored the ceiling at ~3 and could record a noisy
    round on a 1% participation draw as diverged for an arm that would have
    completed had it never been preempted.
    """

    #: rounds 1..5 of a healthy FEMNIST-shaped run
    _HISTORY = [(r, {"fit_loss": v}) for r, v in enumerate([4.1, 2.0, 1.0, 0.5, 0.3], start=1)]

    def test_the_ceiling_stays_anchored_on_round_one(self) -> None:
        monitor = DivergenceMonitor(DivergenceConfig(blowup_absolute=None))
        monitor.prime(self._HISTORY)

        # 10x the resumed round (0.3) is 3; 10x round 1 (4.1) is 41.
        self.assertIsNone(monitor.update(6, {"fit_loss": 5.0}))
        verdict = monitor.update(7, {"fit_loss": 50.0})
        assert verdict is not None
        self.assertEqual(verdict.detector, "blowup")

    def test_without_priming_the_same_round_is_called_diverged(self) -> None:
        """Pins the defect, so the test above cannot pass vacuously."""

        monitor = DivergenceMonitor(DivergenceConfig(blowup_absolute=None))
        monitor.update(6, {"fit_loss": 0.3})
        verdict = monitor.update(7, {"fit_loss": 5.0})

        assert verdict is not None
        self.assertEqual(verdict.detector, "blowup")

    def test_priming_never_returns_a_verdict_for_a_past_round(self) -> None:
        """Those rounds already happened and the run continued past them."""

        monitor = DivergenceMonitor(DivergenceConfig(blowup_absolute=None))
        monitor.prime([(1, {"fit_loss": 0.1}), (2, {"fit_loss": 99.0})])

        self.assertIsNone(monitor.verdict)

    def test_patience_carries_its_counter_across_the_resume(self) -> None:
        """Otherwise a requeue silently resets a stalled run's patience."""

        config = DivergenceConfig(patience=3, blowup_factor=None, blowup_absolute=None)
        monitor = DivergenceMonitor(config)
        # Worsening, so each observation is a real non-improvement. (At
        # min_delta 0 a flat metric counts as improving, by _check_patience's
        # own `value <= required`.)
        monitor.prime([(1, {"fit_loss": 1.0}), (2, {"fit_loss": 1.1}), (3, {"fit_loss": 1.2})])

        verdict = monitor.update(4, {"fit_loss": 1.3})
        assert verdict is not None
        self.assertEqual(verdict.status, STATUS_STALLED)
        self.assertEqual(verdict.detector, "patience")

    def test_priming_skips_rounds_without_the_metric(self) -> None:
        monitor = DivergenceMonitor(DivergenceConfig(blowup_absolute=None))
        monitor.prime([(1, None), (2, {"other": 1.0}), (3, {"fit_loss": 4.0})])

        self.assertIsNone(monitor.update(4, {"fit_loss": 30.0}))

    def test_priming_a_disabled_monitor_does_nothing(self) -> None:
        monitor = DivergenceMonitor(
            DivergenceConfig(
                non_finite=False,
                blowup_factor=None,
                blowup_absolute=None,
                patience=None,
            )
        )
        monitor.prime(self._HISTORY)

        self.assertIsNone(monitor.update(6, {"fit_loss": 1e9}))


class EarlyStoppingMetricGuardTests(unittest.TestCase):
    """Regression tests for early stopping pointed at a test metric.

    divergence with patience set IS early stopping: it ends the run and writes
    status "stalled" with stopped_round. Pointed at a test metric it stops each
    arm at the round its TEST loss stopped improving -- model selection on the
    test set through a different door than the one checkpointing.best_metric
    already refuses with a paragraph of explanation. The bias is invisible
    afterwards, because run.json records the round it stopped at, not what it
    watched.
    """

    def test_a_test_metric_with_patience_is_refused(self) -> None:
        for metric in (
            "central_test_accuracy",
            "central_test_loss",
            "test_loss_sample_weighted_avg",
            "personal_test_accuracy_avg",
        ):
            with self.subTest(metric=metric):
                with self.assertRaises(ValueError) as caught:
                    _validate_divergence(DivergenceConfig(metric=metric, patience=3))
                self.assertIn("early stopping on the test set", str(caught.exception))

    def test_a_validation_metric_with_patience_is_allowed(self) -> None:
        for metric in ("fit_loss", "val_loss_sample_weighted_avg"):
            with self.subTest(metric=metric):
                _validate_divergence(DivergenceConfig(metric=metric, patience=3))

    def test_a_test_metric_without_patience_is_still_allowed(self) -> None:
        """non_finite and blowup are safety stops, not selection."""

        _validate_divergence(DivergenceConfig(metric="central_test_loss", patience=None))


class UnobservedMetricTests(unittest.TestCase):
    """A metric name nothing emits silences every detector, non_finite included."""

    def test_a_monitor_that_never_saw_its_metric_says_so(self) -> None:
        monitor = DivergenceMonitor(DivergenceConfig(metric="fit_los"))
        for round_id in range(1, 6):
            monitor.update(round_id, {"fit_loss": 1.0})

        self.assertFalse(monitor.observed)

    def test_a_monitor_that_saw_its_metric_reports_it(self) -> None:
        monitor = DivergenceMonitor(DivergenceConfig(metric="fit_loss"))
        monitor.update(1, {"fit_loss": 1.0})

        self.assertTrue(monitor.observed)

    def test_a_schedule_gated_metric_still_counts_once_seen(self) -> None:
        monitor = DivergenceMonitor(DivergenceConfig(metric="val_loss"))
        monitor.update(1, {"fit_loss": 1.0})
        self.assertFalse(monitor.observed)
        monitor.update(2, {"fit_loss": 1.0, "val_loss": 0.5})
        self.assertTrue(monitor.observed)

    def test_priming_counts_as_observation(self) -> None:
        monitor = DivergenceMonitor(DivergenceConfig(metric="fit_loss"))
        monitor.prime([(1, {"fit_loss": 1.0})])

        self.assertTrue(monitor.observed)


class ConfigValidationTests(unittest.TestCase):
    def test_defaults_are_valid(self) -> None:
        _validate_divergence(DivergenceConfig())

    def test_every_detector_off_is_valid_and_inactive(self) -> None:
        """This used to be rejected as a contradiction with enabled: true.

        With the bool gone there is nothing to contradict: every detector off
        is the way to say off, and validate_config has no second opinion.
        """

        config = DivergenceConfig(
            non_finite=False,
            blowup_factor=None,
            blowup_absolute=None,
            patience=None,
        )
        _validate_divergence(config)
        self.assertFalse(config.active)

    def test_the_defaults_are_active(self) -> None:
        self.assertTrue(DivergenceConfig().active)

    def test_a_null_divergence_block_resolves_to_every_detector_off(self) -> None:
        config = _build_divergence_config(None)

        self.assertFalse(config.active)
        _validate_divergence(config)

    def test_the_removed_bool_names_what_replaced_it(self) -> None:
        config = DivergenceConfig()
        config.extra["enabled"] = False

        with self.assertRaises(ValueError) as caught:
            _validate_known_keys("divergence", config.extra)

        message = str(caught.exception)
        self.assertIn("divergence.enabled has been removed", message)
        self.assertIn("divergence: null", message)

    def test_a_blowup_factor_of_one_or_less_is_rejected(self) -> None:
        for factor in (1.0, 0.5, -2.0):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                _validate_divergence(DivergenceConfig(blowup_factor=factor))

    def test_out_of_range_values_are_rejected(self) -> None:
        for kwargs in (
            {"patience": 0},
            {"patience": -1},
            {"min_delta": 1.0},
            {"min_delta": -0.1},
            {"blowup_absolute": 0.0},
            {"blowup_absolute": float("inf")},
            {"metric": "   "},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                _validate_divergence(DivergenceConfig(**kwargs))  # type: ignore[arg-type]


class TheChapterCitesTheCounterRatherThanTheFileTest(unittest.TestCase):
    """docs/08 section 12's patience citation has to resolve to the counter.

    It cited `config.py:288-296` for a false-alarm rate. Those lines are the
    tail of min_delta's comment, the extra field and the opening of
    active.__doc__ -- nothing about patience firing on noise.
    `test_cited_modules_exist` passed because config.py exists, which is the
    whole failure: a line-range citation is worse than a bare module name,
    because it looks checked.

    This was the first instance of that trap, and it was fixed by keeping the
    range and checking it contained the counter. docs/14 section 6.8 is the
    same trap found across all of docs/ -- 33 of 177 citations already pointing
    somewhere their own sentence contradicted -- and it rejects exactly that
    fix: once the range has to contain the symbol, the symbol is doing the
    work and the range is a number a human has to maintain forever. So the
    chapter names the symbol alone now, and the range this test reads comes
    from the module's own AST rather than from the prose.

    What is guarded is unchanged: that the chapter cites `_check_patience`,
    that it exists, and that it really is the counter. Only the number nobody
    could keep true is gone.

    This guards the half that has a code authority. The false-alarm arithmetic
    has none, and the chapter now says so rather than implying otherwise.
    """

    CHAPTER = REPO_ROOT / "docs" / "08-metrics.md"
    SOURCE = REPO_ROOT / "fedbrew" / "core" / "divergence.py"

    def setUp(self) -> None:
        self.text = " ".join(self.CHAPTER.read_text(encoding="utf-8").split())

    def test_the_cited_symbol_is_where_the_chapter_says(self) -> None:
        self.assertIn(
            "`_check_patience` (`fedbrew/core/divergence.py`)",
            self.text,
            "section 12 must cite the counter by symbol, beside the file it is "
            "in -- the form tests/test_docs_references_resolve.py checks",
        )
        self.assertIn("def _check_patience(", self.SOURCE.read_text(encoding="utf-8"))

    def test_the_symbol_the_chapter_names_is_the_counter(self) -> None:
        """The check the module-path guard cannot make: the symbol, not the file.

        A citation naming a real symbol in a real file can still name the wrong
        one. This reads that symbol's own source and requires the three things
        the chapter's claim rests on to be in it.
        """

        source = self.SOURCE.read_text(encoding="utf-8")
        found = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name == "_check_patience"
        ]
        self.assertEqual(len(found), 1, "_check_patience is not a single definition")
        body = "\n".join(source.splitlines()[found[0].lineno - 1 : found[0].end_lineno])
        for token in ("patience", "self._since_best", "STATUS_STALLED"):
            with self.subTest(token=token):
                self.assertIn(token, body, "the cited symbol is not the counter")

    def test_the_false_alarm_rate_is_marked_as_unverified(self) -> None:
        """It is a derivation with no code authority; the chapter must say so."""

        self.assertIn("1/(k+1)!", self.text)
        self.assertIn("is a derivation, and nothing here checks", self.text)
        self.assertNotIn("(`config.py:288-296`)", self.text)


if __name__ == "__main__":
    unittest.main()
