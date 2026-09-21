"""What the terminal says while a run is running, and what --quiet leaves.

Three surfaces meet here, and each replaced a silence.

**Between rounds.** The fit and evaluation phases visit a known, bounded
sequence of clients one at a time and said nothing while they did it, so a
round whose fit phase ran for twenty minutes had nothing to report for twenty
minutes. One line now, redrawn in place -- and only on a terminal, because a
carriage-return redraw is the one thing a redirected stream must not receive.

**Which rounds report.** Every round used to print, which for a 500-round run is
500 blocks of which 450 repeat the previous evaluation's numbers unchanged. The
default is now evaluation rounds only, decided by the *schedules* through the
loop's own `evaluates_round` -- not by looking at which columns a round happened
to produce, which would be inferring intent from output.

**--quiet.** It used to print nothing at all, which is indistinguishable from a
job that never started. It now prints one line, and that line has to carry the
outcome: a divergence stop exits zero on purpose, so the exit code cannot.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest
import yaml
from console_env import setUpModule, tearDownModule  # noqa: F401

from fedbrew.core import console, runner
from fedbrew.core import logging as terminal_logging
from fedbrew.core.config import (
    CentralTestConfig,
    ClientConfig,
    ClientStatisticsConfig,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    FullConfig,
    ModelConfig,
    RuntimeConfig,
    ServerConfig,
    SplitEvaluationConfig,
    TaskConfig,
    evaluates_round,
    load_config,
    validate_config,
)
from fedbrew.core.refusal import RunRefused
from fedbrew.core.state import MetricRecord, RoundTimings

pytestmark = pytest.mark.fast


def _config(
    *,
    rounds: int = 12,
    train: object = 10,
    val: object = 5,
    test: object = 10,
    central: object = 10,
    runtime_extra: dict[str, object] | None = None,
) -> FullConfig:
    return FullConfig(
        experiment=ExperimentConfig(seed=3, output_dir="outputs/run-reporting", name="reporting"),
        server=ServerConfig(
            strategy="fedavg", global_rounds=rounds, participation_rate=1.0, metrics=[]
        ),
        client=ClientConfig(
            update_rule="local_sgd",
            local_iterations=1,
            batch_size=8,
            metrics=[],
            learning_rate=0.1,
        ),
        task=TaskConfig(name="classification"),
        data=DataConfig(name="synthetic_classification"),
        model=ModelConfig(name="mlp"),
        runtime=RuntimeConfig(device="cpu", use_amp=False, extra=dict(runtime_extra or {})),
        client_statistics=ClientStatisticsConfig(),
        evaluation=EvaluationConfig(
            train=SplitEvaluationConfig(every=train, clients="all"),
            val=SplitEvaluationConfig(every=val, clients="all"),
            test=SplitEvaluationConfig(every=test, clients="all"),
            central_test=CentralTestConfig(every=central),
        ),
    )


def _record(round_id: int, *, duration: float = 1.0, metrics: dict | None = None) -> MetricRecord:
    return MetricRecord(
        round_id=round_id,
        metrics=dict(metrics or {"fit_loss": 1.0}),
        num_clients=5,
        num_examples=97,
        timings=RoundTimings(total=duration),
    )


def _reported_rounds(config: FullConfig) -> list[int]:
    """Which rounds the reporter actually prints, over a whole run."""

    seen: list[int] = []
    with mock.patch.object(
        runner,
        "print_round_metrics",
        side_effect=lambda round_id, *a, **k: seen.append(round_id),
    ):
        report = runner._round_progress_reporter(config)
        for round_id in range(1, (config.server.global_rounds or 0) + 1):
            report(_record(round_id))
    return seen


class WhichRoundsReportTest(unittest.TestCase):
    def test_only_evaluation_rounds_report_by_default(self) -> None:
        # val every 5 and the rest every 10, over 12 rounds: the schedules pin
        # round 1 and the final round as well as their own multiples.
        self.assertEqual(_reported_rounds(_config(rounds=12)), [1, 5, 10, 12])

    def test_verbose_reports_every_round(self) -> None:
        config = _config(rounds=12, runtime_extra={"verbose": True})
        self.assertEqual(_reported_rounds(config), list(range(1, 13)))

    def test_a_run_that_evaluates_nothing_reports_every_round(self) -> None:
        """There is no evaluation round to wait for, and silence for the whole
        run is worse than a line per round."""

        config = _config(rounds=5, train="never", val="never", test="never", central="never")
        self.assertEqual(_reported_rounds(config), [1, 2, 3, 4, 5])

    def test_a_final_only_schedule_reports_the_final_round(self) -> None:
        config = _config(rounds=8, train="final", val="never", test="never", central="never")
        self.assertEqual(_reported_rounds(config), [8])

    def test_the_gate_is_the_loop_s_own_predicate(self) -> None:
        """Not a reimplementation: the rounds reported must be exactly the
        rounds evaluated, including the two endpoints evaluates_round pins."""

        from fedbrew.core.config import evaluates_round

        intervals = runner._evaluation_intervals(_config(rounds=20))
        for round_id in range(1, 21):
            expected = any(
                evaluates_round(interval, round_id, 20)
                for interval in intervals
                if interval is not None
            )
            self.assertEqual(
                runner._reports_round(intervals, round_id, 20),
                expected,
                f"round {round_id}",
            )

    def test_a_skipped_round_still_counts_towards_the_eta(self) -> None:
        """The ETA is the median of recent rounds. Dropping the unreported ones
        would make it describe only the expensive rounds, which are exactly the
        ones that are not typical."""

        etas: list[float | None] = []

        def capture(round_id, metrics, num_clients, config, **kwargs):
            etas.append(kwargs.get("eta_sec"))

        config = _config(rounds=12)
        with mock.patch.object(runner, "print_round_metrics", side_effect=capture):
            report = runner._round_progress_reporter(config)
            report(_record(1, duration=10.0))
            for round_id in range(2, 5):
                report(_record(round_id, duration=1.0))
            report(_record(5, duration=1.0))

        # Round 5's ETA is the median of five recorded durations
        # (10, 1, 1, 1, 1) = 1.0, times the seven rounds remaining. Had the
        # three unreported rounds been dropped it would have been 5.5 * 7.
        self.assertEqual(etas[-1], 7.0)


#: Three rounds on in-process synthetic data: the smallest config that runs.
SYNTHETIC = Path(__file__).resolve().parent.parent / "configs" / "dev" / "synthetic.yaml"


class PrintEveryTest(unittest.TestCase):
    """--print-every N, runtime.print_every: which rounds print, and nothing else.

    It replaces the evaluation schedule as the reporter's gate rather than
    thinning it, through the loop's own `evaluates_round`, so round 1 and the
    final round always print. Everything the loop records was decided before
    the reporter is called; tests/test_planned_columns_are_written.py runs one
    and checks that it did not move.
    """

    def test_it_replaces_the_evaluation_schedule_rather_than_thinning_it(self) -> None:
        # Evaluation lands on 1, 5, 10 and 12. N = 4 prints 4 and 8, which no
        # split evaluates, and not 5 or 10, which some do.
        self.assertEqual(_reported_rounds(_config(rounds=12)), [1, 5, 10, 12])
        config = _config(rounds=12, runtime_extra={"print_every": 4})
        self.assertEqual(_reported_rounds(config), [1, 4, 8, 12])

    def test_the_gate_is_the_loop_s_own_predicate(self) -> None:
        for every in (1, 2, 3, 7, 25):
            config = _config(rounds=20, runtime_extra={"print_every": every})
            expected = [r for r in range(1, 21) if evaluates_round(every, r, 20)]
            with self.subTest(print_every=every):
                self.assertEqual(_reported_rounds(config), expected)

    def test_it_decides_the_rounds_under_verbose_too(self) -> None:
        """verbose still widens each block; which rounds print is N's."""

        config = _config(rounds=12, runtime_extra={"verbose": True, "print_every": 5})
        self.assertEqual(_reported_rounds(config), [1, 5, 10, 12])

    def test_it_applies_to_a_run_that_evaluates_nothing(self) -> None:
        config = _config(
            rounds=5,
            train="never",
            val="never",
            test="never",
            central="never",
            runtime_extra={"print_every": 2},
        )
        self.assertEqual(_reported_rounds(config), [1, 2, 4, 5])

    def test_a_round_it_skips_still_counts_towards_the_eta(self) -> None:
        seen: list[tuple[int, float | None]] = []

        def capture(round_id, metrics, num_clients, config, **kwargs):
            seen.append((round_id, kwargs.get("eta_sec")))

        config = _config(rounds=12, runtime_extra={"print_every": 4})
        with mock.patch.object(runner, "print_round_metrics", side_effect=capture):
            report = runner._round_progress_reporter(config)
            report(_record(1, duration=10.0))
            for round_id in range(2, 5):
                report(_record(round_id, duration=1.0))

        # Round 4's ETA is the median of all four durations (10, 1, 1, 1) = 1.0
        # times the eight rounds left; without the two skipped rounds it would
        # be 5.5 * 8.
        self.assertEqual(seen, [(1, 110.0), (4, 8.0)])

    def test_quiet_in_a_config_wins(self) -> None:
        """So a driver that always passes --quiet can run a config that sets it."""

        def printed(runtime_extra: dict[str, object]) -> str:
            buffer = io.StringIO()
            config = _config(rounds=4, runtime_extra=runtime_extra)
            with (
                mock.patch.object(
                    terminal_logging,
                    "_surface_for",
                    side_effect=lambda cfg: console.build_surface(
                        quiet=terminal_logging._is_quiet(cfg), file=buffer, force_rich=False
                    ),
                ),
            ):
                report = runner._round_progress_reporter(config)
                for round_id in range(1, 5):
                    report(_record(round_id))
            return buffer.getvalue()

        self.assertIn("round 2/4", printed({"print_every": 2}))
        self.assertEqual(printed({"print_every": 2, "quiet": True}), "")


class PrintEveryIsCheckedTest(unittest.TestCase):
    def _load(self, value: object) -> FullConfig:
        raw = yaml.safe_load(SYNTHETIC.read_text(encoding="utf-8"))
        raw["runtime"]["print_every"] = value
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            config = load_config(path)
        validate_config(config)
        return config

    def test_the_config_key_takes_a_positive_integer(self) -> None:
        for good in (1, 7):
            with self.subTest(value=good):
                self.assertEqual(self._load(good).runtime.extra["print_every"], good)

    def test_the_config_key_refuses_anything_else(self) -> None:
        for bad in (0, -1, True, "5", 2.5):
            with self.subTest(value=bad), self.assertRaises(RunRefused) as refused:
                self._load(bad)
            self.assertIn("runtime.print_every", str(refused.exception))

    def test_the_flag_writes_the_key(self) -> None:
        args = runner.parse_args(["--config", str(SYNTHETIC), "--print-every", "5"])
        config = runner.apply_cli_overrides(load_config(SYNTHETIC), args)
        self.assertEqual(config.runtime.extra["print_every"], 5)

    def test_the_parser_refuses_what_is_not_a_positive_integer(self) -> None:
        for bad in ("0", "-3", "2.5", "x"):
            with (
                self.subTest(value=bad),
                self.assertRaises(SystemExit) as caught,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                runner.parse_args(["--print-every", bad])
            self.assertEqual(caught.exception.code, 2)

    def test_the_parser_refuses_it_beside_quiet(self) -> None:
        """As --quiet --verbose is: one command line asking for both."""

        stderr = io.StringIO()
        with self.assertRaises(SystemExit) as caught, contextlib.redirect_stderr(stderr):
            runner.parse_args(["--quiet", "--print-every", "5"])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("--print-every", stderr.getvalue())


class TheLiveClientLineTest(unittest.TestCase):
    def _reporter(self, config: FullConfig, buffer: io.StringIO):
        return mock.patch.object(
            terminal_logging,
            "_surface_for",
            return_value=console.build_surface(file=buffer, force_rich=True),
        )

    def test_a_redirected_stream_gets_no_reporter_at_all(self) -> None:
        """None rather than a no-op callback, so the loop skips the call
        instead of paying for it once per client per round."""

        self.assertIsNone(terminal_logging.client_progress_reporter(_config()))

    def test_quiet_gets_no_reporter(self) -> None:
        config = _config(runtime_extra={"quiet": True})
        self.assertIsNone(terminal_logging.client_progress_reporter(config))

    def test_it_states_the_round_and_the_selection_against_the_roster(self) -> None:
        """`10/1000 clients`, not a counter climbing to the same total every
        round. Participation is decided when the round is selected; a number
        that counts up to a figure it always reaches says less each read."""

        buffer = io.StringIO()
        progress = terminal_logging.RoundProgress(total_rounds=12, roster=1000)
        with self._reporter(_config(), buffer):
            report = terminal_logging.client_progress_reporter(_config(), progress)
        self.assertIsNotNone(report)

        report(3, 5, 10, "fit")
        rendered = _strip(buffer.getvalue())
        self.assertIn("round 3/12", rendered)
        self.assertIn("10/1000 clients", rendered)
        self.assertNotIn("fit 5/10", rendered, "the count is a --verbose addition")

    def test_the_selection_is_the_one_the_fit_phase_actually_visited(self) -> None:
        """Taken from the phase rather than participation_rate x roster: what
        happened, not what should have."""

        buffer = io.StringIO()
        progress = terminal_logging.RoundProgress(total_rounds=12, roster=1000)
        with self._reporter(_config(), buffer):
            report = terminal_logging.client_progress_reporter(_config(), progress)

        report(3, 1, 7, "fit")
        self.assertEqual(progress.sampled, 7)
        # The evaluation phase visits a different population -- at
        # `client_scope: all` it visits every client -- and must not be read
        # as this round's selection.
        report(3, 1, 1000, "client_eval")
        self.assertEqual(progress.sampled, 7)

    def test_a_round_that_selects_no_client_shows_zero_not_the_last_count(self) -> None:
        """Under participation_probability a round can select nobody, and then no
        client finishes to update the count. The loop's done=0 announcement is
        what the footer has to take, or it shows the previous round's."""

        buffer = io.StringIO()
        progress = terminal_logging.RoundProgress(total_rounds=12, roster=1000)
        with self._reporter(_config(), buffer):
            report = terminal_logging.client_progress_reporter(_config(), progress)

        report(3, 7, 7, "fit")
        report(4, 0, 0, "fit")
        self.assertEqual(progress.sampled, 0)
        last_draw = _strip(buffer.getvalue()).rstrip("\r").split("\r")[-1]
        self.assertIn("round 4/12", last_draw)
        self.assertIn(" 0/1000 clients", last_draw)

    def test_verbose_counts_the_clients_within_the_phase(self) -> None:
        """`client_scope: all` on FEMNIST is 3,500 clients in one pass. That
        is long enough that a reader wants to know where in it they are, and
        short of --verbose the pulse and the clock are the only liveness."""

        buffer = io.StringIO()
        config = _config(runtime_extra={"verbose": True})
        progress = terminal_logging.RoundProgress(total_rounds=12, roster=3500)
        with mock.patch.object(
            terminal_logging,
            "_surface_for",
            return_value=console.build_surface(verbose=True, file=buffer, force_rich=True),
        ):
            report = terminal_logging.client_progress_reporter(config, progress)
        self.assertIsNotNone(report)

        report(3, 340, 3500, "client_eval")
        rendered = _strip(buffer.getvalue())
        # The loop calls the evaluation phase "client_eval"; on screen it sits
        # beside "fit" and reads as a word of the same size.
        self.assertIn("eval 340/3500", rendered)

    def test_it_redraws_in_place_rather_than_scrolling(self) -> None:
        buffer = io.StringIO()
        with self._reporter(_config(), buffer):
            report = terminal_logging.client_progress_reporter(_config())
        for done in range(1, 6):
            report(7, done, 5, "fit")

        rendered = buffer.getvalue()
        self.assertIn("\r", rendered)
        self.assertNotIn("\n", rendered, "a progress line that scrolls is 2,500 lines a run")

    def test_the_last_client_always_draws_despite_the_throttle(self) -> None:
        """The throttle exists for `client_scope: all` on FEMNIST -- 3,500
        callbacks a pass -- but the final count must never be the one it
        swallows, or the line rests on a stale number."""

        buffer = io.StringIO()
        config = _config(runtime_extra={"verbose": True})
        progress = terminal_logging.RoundProgress(total_rounds=12, roster=100)
        with mock.patch.object(
            terminal_logging,
            "_surface_for",
            return_value=console.build_surface(verbose=True, file=buffer, force_rich=True),
        ):
            report = terminal_logging.client_progress_reporter(config, progress)
        for done in range(1, 101):
            report(7, done, 100, "fit")

        rendered = _strip(buffer.getvalue())
        self.assertIn("fit 100/100", rendered)
        self.assertLess(rendered.count("clients"), 100, "the throttle did nothing")

    def test_the_line_fills_the_width_so_the_next_print_leaves_no_tail(self) -> None:
        """Exactly one column short of the terminal, every time.

        The footer is redrawn with a carriage return, which moves the cursor
        without erasing: a shorter line than the one before it leaves that
        one's right-hand end on screen, and the leftover is a stale ETA
        sitting beside a fresh one. Filling the width is what makes each
        redraw overwrite the whole of the last.
        """

        buffer = io.StringIO()
        surface = console.build_surface(file=buffer, force_rich=True)
        progress = terminal_logging.RoundProgress(total_rounds=12, roster=1000)
        with mock.patch.object(terminal_logging, "_surface_for", return_value=surface):
            report = terminal_logging.client_progress_reporter(_config(), progress)

        report(7, 100, 100, "fit")
        drawn = _strip(buffer.getvalue()).split("\r")[0]
        self.assertEqual(len(drawn), surface.width - 1)

    def test_the_settled_header_and_the_live_footer_draw_the_same_bar(self) -> None:
        """A header is the footer with the pulse blanked and the clock
        dropped. Sized from the text in hand instead, the header's bar would
        be longer than the footer's below it -- and longer at `round 1/12`
        than at `round 12/12`, so no two blocks of one run would line up."""

        for verbose in (False, True):
            with self.subTest(verbose=verbose):
                buffer = io.StringIO()
                surface = console.build_surface(verbose=verbose, file=buffer, force_rich=True)
                config = _config(runtime_extra={"verbose": True} if verbose else None)
                progress = terminal_logging.RoundProgress(total_rounds=12, roster=1000)
                progress.record(0.5)

                with mock.patch.object(terminal_logging, "_surface_for", return_value=surface):
                    report = terminal_logging.client_progress_reporter(config, progress)
                    report(1, 10, 10, "fit")
                    footer = _strip(buffer.getvalue()).split("\r")[0]

                    buffer.truncate(0)
                    buffer.seek(0)
                    terminal_logging.print_round_metrics(
                        12, {"fit_loss": 1.0}, 10, config, progress=progress
                    )
                    header = _strip(buffer.getvalue()).splitlines()[0]

                self.assertEqual(_bar_cells(header), _bar_cells(footer))
                # And the text starts in the same column in both.
                self.assertEqual(header.index("round"), footer.index("round"))


class RoundProgressTest(unittest.TestCase):
    """The one object both halves of the run surface read.

    Split out because the rate under the bar and the estimate in the block are
    the same number seen twice: computed in two places they would drift, and
    the drift would only show late in a long run where nobody is watching for
    it.
    """

    def test_the_estimate_is_a_median_not_a_mean(self) -> None:
        """Round one carries setup nothing else pays -- CUDA context, shard
        cache fill, lazily built clients -- and a mean would carry it for the
        rest of the job."""

        progress = terminal_logging.RoundProgress(total_rounds=10)
        for duration in (10.0, 1.0, 1.0, 1.0, 1.0):
            progress.record(duration)

        self.assertEqual(progress.median_seconds, 1.0)
        self.assertEqual(progress.eta_seconds(5), 5.0)

    def test_it_forgets_rounds_old_enough_to_be_a_different_regime(self) -> None:
        """Ten rounds, not all of them: a job whose rounds got slower halfway
        through should say so rather than average the two halves forever."""

        progress = terminal_logging.RoundProgress(total_rounds=100)
        for _ in range(20):
            progress.record(1.0)
        for _ in range(10):
            progress.record(4.0)

        self.assertEqual(progress.median_seconds, 4.0)

    def test_a_round_that_printed_nothing_still_counts(self) -> None:
        progress = terminal_logging.RoundProgress(total_rounds=10)
        progress.record(2.0)
        progress.record(2.0)
        self.assertEqual(len(progress.durations), 2)
        # A round with no timing at all contributes nothing rather than a zero,
        # which would halve every estimate after it.
        progress.record(None)
        self.assertEqual(len(progress.durations), 2)

    def test_there_is_no_estimate_before_a_round_has_finished(self) -> None:
        progress = terminal_logging.RoundProgress(total_rounds=10)
        self.assertIsNone(progress.median_seconds)
        self.assertIsNone(progress.eta_seconds(1))

    def test_the_final_round_has_nothing_left_to_estimate(self) -> None:
        """`0s left` on the last round is worse than no number: it reads as a
        prediction rather than as the end."""

        progress = terminal_logging.RoundProgress(total_rounds=10)
        progress.record(1.0)
        self.assertEqual(progress.eta_seconds(9), 1.0)
        self.assertIsNone(progress.eta_seconds(10))
        self.assertIsNone(progress.eta_seconds(11))

    def test_the_bar_never_overfills_on_a_resumed_run(self) -> None:
        """A checkpoint from a longer run resumed under a lower
        global_rounds reports a round past its own total."""

        progress = terminal_logging.RoundProgress(total_rounds=10)
        self.assertEqual(progress.fraction(5), 0.5)
        buffer = io.StringIO()
        surface = console.build_surface(file=buffer, force_rich=True)
        self.assertEqual(len(surface.bar(progress.fraction(30), 20)[0][0]), 20)
        self.assertEqual(surface.bar(progress.fraction(30), 20)[1][0], "")

    def test_the_clients_line_states_the_selection_against_the_roster(self) -> None:
        self.assertEqual(
            terminal_logging.RoundProgress(10, roster=1000, sampled=10).clients_text(),
            "10/1000 clients",
        )
        # A dataset that would not say how many clients it has still reports
        # the selection, which is the half this run decided.
        self.assertEqual(
            terminal_logging.RoundProgress(10, sampled=10).clients_text(), "10 clients"
        )
        # And before the first fit phase of a run, the roster alone.
        self.assertEqual(
            terminal_logging.RoundProgress(10, roster=1000).clients_text(), "1000 clients"
        )
        self.assertEqual(terminal_logging.RoundProgress(10).clients_text(), "")


class TheQuietFinalLineTest(unittest.TestCase):
    def _final(self, config: FullConfig, history, termination=None) -> str:
        buffer = io.StringIO()
        with mock.patch.object(
            terminal_logging,
            "_surface_for",
            return_value=console.build_surface(file=buffer, quiet=True),
        ):
            terminal_logging.print_experiment_end(history, "outputs/reporting", config, termination)
        return buffer.getvalue()

    def test_quiet_prints_exactly_one_line(self) -> None:
        config = _config(runtime_extra={"quiet": True})
        rendered = self._final(config, [_record(1), _record(2)])

        self.assertEqual(rendered.count("\n"), 1, f"not one line: {rendered!r}")
        self.assertIn("completed 2 rounds", rendered)
        self.assertIn("outputs/reporting", rendered)

    def test_the_line_says_a_diverged_run_diverged(self) -> None:
        """The exit code cannot: a divergence stop exits zero on purpose, so a
        sweep grepping its logs has only this line to go on."""

        config = _config(runtime_extra={"quiet": True})
        rendered = self._final(
            config,
            [_record(1), _record(12)],
            {"status": "diverged", "reason": "fit_loss went to NaN", "detector": "non_finite"},
        )

        self.assertEqual(rendered.count("\n"), 1)
        self.assertIn("diverged at round 12", rendered)
        self.assertIn("fit_loss went to NaN", rendered)

    def test_a_run_with_no_timed_rounds_still_reports(self) -> None:
        config = _config(runtime_extra={"quiet": True})
        rendered = self._final(config, [])
        self.assertIn("completed 0 rounds", rendered)


class TheValidationSplitIsReportedTest(unittest.TestCase):
    """`evaluation.val` defaults to every 5 against test's every 10, so the
    most frequent evaluation round in a default run is a val round. It had no
    curated entries at all, which was survivable while every round printed and
    became a defect the moment the terminal reported evaluation rounds only."""

    def test_a_val_round_shows_what_it_measured(self) -> None:
        buffer = io.StringIO()
        config = _config()
        metrics = {
            "fit_loss": 1.08,
            "val_loss_sample_weighted_avg": 1.18,
            "val_accuracy_sample_weighted_avg": 0.30,
            "val_loss_min": 0.94,
        }
        with mock.patch.object(
            terminal_logging,
            "_surface_for",
            return_value=console.build_surface(file=buffer),
        ):
            terminal_logging.print_round_metrics(5, metrics, 5, config, num_examples=97)

        rendered = _strip(buffer.getvalue())
        self.assertIn("loss", rendered)
        self.assertIn("accuracy", rendered)
        # Two rows under two headings, both naming the validation split.
        self.assertEqual(rendered.count("validation "), 2)
        self.assertIn("sample_weighted_avg", rendered)
        # Still curated: the dispersion columns went to the CSV, not the
        # screen, so no "client spread" group appears at all.
        self.assertNotIn("loss_min", rendered)
        self.assertNotIn("client spread", rendered)
        self.assertIn("1 more column", rendered)

    def test_the_column_selection_reads_is_shown_even_when_uncurated(self) -> None:
        """A config selecting on val_accuracy_worst10 needs to see it: it is
        the reason a round became the new best."""

        config = _config(runtime_extra={"checkpointing": {"best_metric": "val_accuracy_worst10"}})
        names = terminal_logging._client_val_metric_names(config)
        self.assertIn("val_accuracy_worst10", names)
        self.assertIn("val_accuracy_sample_weighted_avg", names)

    def test_a_selection_metric_no_run_emits_is_not_promised(self) -> None:
        config = _config(runtime_extra={"checkpointing": {"best_metric": "val_accuracy_worst10"}})
        config.client_statistics = ClientStatisticsConfig(worst_percent=None)
        self.assertNotIn("val_accuracy_worst10", terminal_logging._client_val_metric_names(config))


def _bar_cells(line: str) -> int:
    return line.count(console.BAR_FILLED) + line.count(console.BAR_EMPTY)


def _strip(rendered: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", rendered)


if __name__ == "__main__":
    unittest.main()
