"""Tests for concise terminal progress definitions."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

import pytest
from console_env import setUpModule, tearDownModule  # noqa: F401

from fedbrew.core import console
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
    client_metric_names,
)

pytestmark = pytest.mark.fast


def _make_config(
    *,
    strategy: str = "fedavg",
    update_rule: str = "local_sgd",
    server_metrics: list[str] | None = None,
    client_metrics: list[str] | None = None,
    runtime_extra: dict[str, object] | None = None,
    evaluate_test: bool = True,
    evaluate_central: bool = True,
    task: str = "classification",
    client_statistics: ClientStatisticsConfig | None = None,
) -> FullConfig:
    return FullConfig(
        experiment=ExperimentConfig(
            name="logging-test",
            seed=7,
            output_dir="outputs/logging-test",
        ),
        server=ServerConfig(
            strategy=strategy,
            global_rounds=3,
            participation_rate=1.0,
            metrics=list(
                ["fit_loss", "fit_accuracy"] if server_metrics is None else server_metrics
            ),
            extra={},
        ),
        client=ClientConfig(
            update_rule=update_rule,
            local_iterations=1,
            batch_size=8,
            learning_rate=0.1,
            metrics=list(
                ["fit_loss", "fit_accuracy"] if client_metrics is None else client_metrics
            ),
        ),
        task=TaskConfig(name=task),
        data=DataConfig(name="test-data"),
        model=ModelConfig(name="test-model"),
        runtime=RuntimeConfig(
            device="cpu",
            use_amp=False,
            extra=dict(runtime_extra or {}),
        ),
        client_statistics=(
            ClientStatisticsConfig() if client_statistics is None else client_statistics
        ),
        evaluation=EvaluationConfig(
            # "all" so the definitions read as the whole-population wording;
            # the participating wording is covered by its own test.
            train=SplitEvaluationConfig(every=1, clients="all"),
            test=SplitEvaluationConfig(every=1 if evaluate_test else "never", clients="all"),
            central_test=CentralTestConfig(every=1 if evaluate_central else "never"),
        ),
    )


def _ansi(tone: str) -> str:
    """The escape sequence one palette tone renders as.

    Derived by rendering through the same console the code under test uses,
    rather than pasted in: a hard-coded "\\x1b[1;33m" is a second copy of the
    palette living in the test suite, and it was one -- these assertions
    outlived the colours they named once already.
    """

    buffer = io.StringIO()
    surface = console.build_surface(file=buffer, force_rich=True)
    surface.line("x", tone=tone)
    return buffer.getvalue().split("x")[0]


def _plan_header(config, **kwargs) -> None:
    """``print_plan_header`` with the two determinism flags supplied.

    They are required arguments -- the header prints what the run resolved
    and keeps no default of its own -- and no config in this module sets
    them, so every call here passes the resolved-off pair.
    """

    terminal_logging.print_plan_header(
        config, deterministic=False, deterministic_warn_only=False, **kwargs
    )


def _render_rich(render, *args) -> str:
    """Run one logging function against a forced-interactive surface."""

    buffer = io.StringIO()
    with mock.patch.object(
        terminal_logging,
        "_surface_for",
        return_value=console.build_surface(file=buffer, force_rich=True),
    ):
        render(*args)
    return buffer.getvalue()


class LegendTracksTheEmittedColumnsTests(unittest.TestCase):
    """The legend must name columns the run will actually write.

    It used to promise "test_accuracy_bottom10" -- a spelling the loop stopped
    emitting when the suffix became worst{P} -- so the legend described a
    column no run produced and omitted the one every run did. It also promised
    _std and _min unconditionally, which is wrong for any config that turns
    those toggles off. Both are now derived from client_metric_names, and
    these are the checks that keep them derived.
    """

    def _legend(self, statistics: ClientStatisticsConfig) -> list[str]:
        return [
            name
            for name, _ in terminal_logging._progress_definitions(
                _make_config(client_statistics=statistics)
            )
        ]

    def test_every_legend_metric_is_a_column_the_run_emits(self) -> None:
        for statistics in (
            ClientStatisticsConfig(),
            ClientStatisticsConfig(std=False, min=False, max=False, worst_percent=None),
            ClientStatisticsConfig(worst_percent=2.5),
            ClientStatisticsConfig(variance=True, worst_percent=5),
        ):
            with self.subTest(statistics=statistics):
                emitted = set()
                for split in ("train", "val", "test"):
                    emitted |= client_metric_names(split, statistics)
                claimed = {
                    name
                    for name in self._legend(statistics)
                    if name.startswith(("train_", "val_", "test_")) and name != "test"
                }
                self.assertTrue(
                    claimed <= emitted,
                    f"legend names columns no run emits: {sorted(claimed - emitted)}",
                )

    def test_the_worst_percent_entry_follows_the_configured_percentage(self) -> None:
        for percent, expected in ((10, "worst10"), (2.5, "worst2p5"), (5, "worst5")):
            with self.subTest(percent=percent):
                statistics = ClientStatisticsConfig(worst_percent=percent)
                legend = self._legend(statistics)
                self.assertIn(f"test_accuracy_{expected}", legend)
                self.assertNotIn("test_accuracy_bottom10", legend)

    def test_the_worst_percent_definition_states_the_configured_percentage(self) -> None:
        definitions = dict(
            terminal_logging._progress_definitions(
                _make_config(client_statistics=ClientStatisticsConfig(worst_percent=2.5))
            )
        )
        self.assertIn(
            "worst 2.5% of clients",
            definitions["test_accuracy_worst2p5"],
        )

    def test_switching_a_toggle_off_removes_its_legend_entry(self) -> None:
        legend = self._legend(
            ClientStatisticsConfig(std=False, min=False, max=False, worst_percent=None)
        )
        for absent in ("test_accuracy_std", "test_accuracy_min"):
            self.assertNotIn(absent, legend)
        self.assertFalse(
            [name for name in legend if "worst" in name],
            "worst_percent None must leave no worst entry in the legend",
        )
        # The two unconditional averages are still there: they are what the
        # split means, and no toggle removes them.
        for present in ("test_loss_sample_weighted_avg", "test_accuracy_avg"):
            self.assertIn(present, legend)


class ProgressDefinitionTests(unittest.TestCase):
    def test_standard_definitions_explain_both_test_metric_groups(self) -> None:
        definitions = terminal_logging._progress_definitions(_make_config())

        self.assertEqual(
            [name for name, _ in definitions],
            [
                "round",
                "clients",
                "examples",
                "time",
                "eta",
                "eval",
                "test",
                "global_test",
                "train_loss_sample_weighted_avg",
                "central_test_loss",
                # The validation split is curated too, and shorter than the
                # test one: val exists to select a checkpoint, so its two
                # averages and whatever selection reads are what a reader
                # needs from it.
                "val_loss_sample_weighted_avg",
                "val_loss_avg",
                "test_loss_sample_weighted_avg",
                "test_loss_avg",
                "train_accuracy_sample_weighted_avg",
                "central_test_accuracy",
                "val_accuracy_sample_weighted_avg",
                "val_accuracy_avg",
                "test_accuracy_sample_weighted_avg",
                "test_accuracy_avg",
                "test_accuracy_std",
                "test_accuracy_min",
                "test_accuracy_worst10",
            ],
        )
        by_name = dict(definitions)
        self.assertIn("client train data", by_name["train_loss_sample_weighted_avg"])
        self.assertIn("complete global test set", by_name["central_test_loss"])
        self.assertIn("total correct predictions", by_name["central_test_accuracy"])
        # The pair the glosses exist to separate. Same column name but for one
        # token; different numbers on any non-uniform split.
        self.assertIn("pooled over examples", by_name["test_accuracy_sample_weighted_avg"])
        self.assertIn("averaged over clients", by_name["test_accuracy_avg"])
        self.assertIn("averaged over clients", by_name["test_loss_avg"])
        self.assertIn("spread across clients", by_name["test_accuracy_std"])
        self.assertIn("single lowest client value", by_name["test_accuracy_min"])
        self.assertIn("worst 10% of clients", by_name["test_accuracy_worst10"])
        self.assertIn("validation", by_name["eval"].lower())
        self.assertIn("local client test data", by_name["test"])
        self.assertIn("centralized/global test data", by_name["global_test"])

    def test_global_test_only_excludes_client_test_metric_definitions(self) -> None:
        definitions = terminal_logging._progress_definitions(_make_config(evaluate_test=False))

        self.assertEqual(
            [name for name, _ in definitions],
            [
                "round",
                "clients",
                "examples",
                "time",
                "eta",
                "eval",
                "test",
                "global_test",
                "train_loss_sample_weighted_avg",
                "central_test_loss",
                "val_loss_sample_weighted_avg",
                "val_loss_avg",
                "train_accuracy_sample_weighted_avg",
                "central_test_accuracy",
                "val_accuracy_sample_weighted_avg",
                "val_accuracy_avg",
            ],
        )
        by_name = dict(definitions)
        self.assertIn("client train data", by_name["train_loss_sample_weighted_avg"])
        self.assertIn("complete global test set", by_name["central_test_loss"])
        self.assertNotIn("test_accuracy_sample_weighted_avg", by_name)

    def test_empty_metric_lists_use_built_in_client_defaults(self) -> None:
        definitions = dict(
            terminal_logging._progress_definitions(
                _make_config(
                    strategy="scaffold",
                    update_rule="scaffold",
                    server_metrics=[],
                    client_metrics=[],
                )
            )
        )

        for name in (
            "train_loss_sample_weighted_avg",
            "central_test_loss",
            "train_accuracy_sample_weighted_avg",
            "central_test_accuracy",
            "control_delta_norm",
            "client_control_norm",
            "local_steps",
        ):
            self.assertIn(name, definitions)

    def test_scaffold_definitions_cover_all_displayed_diagnostics(self) -> None:
        scaffold_metrics = [
            "fit_loss",
            "fit_accuracy",
            "control_delta_norm",
            "client_control_norm",
            "local_steps",
        ]
        definitions = dict(
            terminal_logging._progress_definitions(
                _make_config(
                    strategy="scaffold",
                    update_rule="scaffold",
                    server_metrics=scaffold_metrics,
                    client_metrics=scaffold_metrics,
                )
            )
        )

        for name in (
            "control_delta_norm",
            "client_control_norm",
            "local_steps",
            "mean_client_control_delta_norm",
            "server_control_norm",
        ):
            self.assertIn(name, definitions)
        self.assertIn("Example-weighted", definitions["control_delta_norm"])
        self.assertIn("Unweighted", definitions["mean_client_control_delta_norm"])
        self.assertIn("/ all clients", definitions["server_control_norm"])

    def test_fedprox_definitions_state_the_objective_terms(self) -> None:
        fedprox_metrics = [
            "fit_loss",
            "fit_accuracy",
            "fit_proximal_loss",
            "fit_total_loss",
        ]
        definitions = dict(
            terminal_logging._progress_definitions(
                _make_config(
                    update_rule="fedprox",
                    server_metrics=fedprox_metrics,
                    client_metrics=fedprox_metrics,
                )
            )
        )

        self.assertIn("(mu/2)", definitions["fit_proximal_loss"])
        self.assertIn("w_round_start", definitions["fit_proximal_loss"])
        self.assertIn("cross-entropy + proximal loss", definitions["fit_total_loss"])

    def test_the_definitions_reach_the_plan_header_and_precede_training(self) -> None:
        """The legend moved into the plan header's metrics block, beside the
        columns it defines. What has to stay true is that a reader meets the
        definitions before the first round, not after it."""

        output = io.StringIO()
        config = _make_config(runtime_extra={"no_rich": True})

        with redirect_stdout(output):
            _plan_header(config)
            terminal_logging.print_round_table_header(config)

        rendered = output.getvalue()
        self.assertLess(
            rendered.index("train_loss_sample_weighted_avg"),
            rendered.index("TRAINING"),
        )
        self.assertIn("central_test_accuracy", rendered)
        self.assertNotIn("[", rendered)

    def test_rich_definitions_are_toned_and_precede_progress(self) -> None:
        """The interactive path, forced: pytest's captured stdout is not a
        terminal, so the gate in fedbrew.core.console would otherwise pick the
        plain path for us and this would assert nothing.

        The expected escape sequence is derived from the palette constant
        rather than written out, so renaming a colour fails at the palette and
        not here -- but painting a label with the wrong *role* still fails,
        which is the wiring this test is for.
        """

        rendered = _render_rich(_plan_header, _make_config())

        self.assertIn(f"{_ansi(console.DIM)}train_loss_sample_weighted_avg", rendered)
        self.assertIn(f"{_ansi(console.IVORY)}metrics", rendered)

    def test_a_definition_is_not_painted_as_a_value(self) -> None:
        """Gold means "a measured value". A sentence is not one, and that
        distinction is the whole reason the palette has more than one tone."""

        rendered = _render_rich(_plan_header, _make_config())

        # A fragment short enough to survive the hang-indent wrapping the
        # interactive path applies to long values.
        self.assertIn("Cross-entropy on client train data,", rendered)
        self.assertNotIn(f"{_ansi(console.GOLD)}  Cross-entropy", rendered)

    def test_quiet_mode_prints_nothing(self) -> None:
        output = io.StringIO()
        config = _make_config(runtime_extra={"quiet": True})

        with redirect_stdout(output):
            _plan_header(config)
            terminal_logging.print_round_table_header(config)

        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
