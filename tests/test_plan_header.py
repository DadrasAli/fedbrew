"""The plan header: resolved values, four blocks, and scarce amber.

Printed twice -- by `--validate-only` and at the start of a real run -- and it
is the only place a reader is told what the run is about to do before it does
it. Three things have to be true for that to be worth reading.

**The values are resolved, not configured.** After the defaults, after the CLI
overrides, after `device: auto` became something concrete. A header that echoed
the YAML would answer a question nobody asks -- the file is right there -- and
hide the two facts a reader needs, which are what the defaults filled in and
what the flags changed.

**Amber marks five things and nothing else**: a matmul precision that changes
the numbers, determinism downgraded to warn-only, an output directory that
already holds files, a resumed run, and components loaded from outside the
package. Each is a thing a reader would otherwise assume was not the case. A
sixth amber makes the other five ordinary, so the count itself is guarded
here.

**The metrics block names columns the run will actually write.** It is built
from `client_metric_names` and from the two dictionaries in
fedbrew.core.metrics naming what a metrics list cannot filter out -- the same
sources preflight uses -- so a column listed and not written, or written and
not listed, fails somewhere rather than misleading quietly.
"""

from __future__ import annotations

import io
import re
import unittest
import unittest.mock
from pathlib import Path
from tempfile import TemporaryDirectory

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
from fedbrew.core.runner import _runtime_extra_bool

pytestmark = pytest.mark.fast

BLOCKS = ("data", "federation", "algorithm", "metrics")


def _config(
    *,
    output_dir: str = "outputs/plan-header-test",
    runtime_extra: dict[str, object] | None = None,
    strategy: str = "fedavg",
    update_rule: str = "local_sgd",
    model_scope: str = "global",
    participation_rate: float | None = 0.25,
    participation_probability: float | None = None,
    num_clients: int | None = 20,
    extensions: list[str] | None = None,
) -> FullConfig:
    return FullConfig(
        experiment=ExperimentConfig(
            seed=7,
            output_dir=output_dir,
            name="plan-header-test",
            extensions=list(extensions or []),
        ),
        server=ServerConfig(
            strategy=strategy,
            global_rounds=200,
            participation_rate=participation_rate,
            participation_probability=participation_probability,
            metrics=[],
        ),
        client=ClientConfig(
            update_rule=update_rule,
            local_iterations=2,
            batch_size=32,
            metrics=[],
            learning_rate=0.05,
        ),
        task=TaskConfig(name="classification"),
        data=DataConfig(name="synthetic_classification", num_clients=num_clients),
        model=ModelConfig(name="mlp"),
        runtime=RuntimeConfig(device="cpu", use_amp=False, extra=dict(runtime_extra or {})),
        client_statistics=ClientStatisticsConfig(),
        evaluation=EvaluationConfig(
            train=SplitEvaluationConfig(every=10, clients="participating"),
            val=SplitEvaluationConfig(every=5, clients="all"),
            test=SplitEvaluationConfig(every=10, clients="all"),
            central_test=CentralTestConfig(every=10),
            model_scope=model_scope,
        ),
    )


def _determinism(config: FullConfig) -> dict[str, bool]:
    """The two flags the header no longer reads, resolved the way run() does.

    Through runner's own helper rather than a literal, so a test that sets
    them in ``runtime_extra`` still drives the header, and so the resolver
    and the header cannot drift apart without a test noticing.
    """

    return {
        "deterministic": _runtime_extra_bool(config, "deterministic", False),
        "deterministic_warn_only": _runtime_extra_bool(config, "deterministic_warn_only", False),
    }


def _render(config: FullConfig, **kwargs) -> str:
    """The header as a redirected stream receives it: no colour, one line per
    row, which is also the form a SLURM log holds."""

    buffer = io.StringIO()
    terminal_logging.print_plan_header(
        config,
        **_determinism(config),
        surface=console.build_surface(file=buffer, **kwargs.pop("surface_options", {}))
        if "surface_options" in kwargs
        else console.build_surface(file=buffer),
        **kwargs,
    )
    return buffer.getvalue()


def _amber_lines(config: FullConfig, **kwargs) -> list[str]:
    """Every line of the header carrying the amber tone, decoloured."""

    buffer = io.StringIO()
    surface = console.build_surface(file=buffer, force_rich=True)
    terminal_logging.print_plan_header(config, **_determinism(config), surface=surface, **kwargs)
    amber = _ansi(console.AMBER)
    return [
        re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
        for line in buffer.getvalue().splitlines()
        if amber in line
    ]


def _ansi(tone: str) -> str:
    buffer = io.StringIO()
    console.build_surface(file=buffer, force_rich=True).line("x", tone=tone)
    return buffer.getvalue().split("x")[0]


class TheFourBlocksTest(unittest.TestCase):
    def test_all_four_are_present_and_in_order(self) -> None:
        rendered = _render(_config())
        positions = [rendered.index(f"\n{name}\n") for name in BLOCKS]
        self.assertEqual(positions, sorted(positions), f"blocks out of order: {rendered}")

    def test_one_label_column_runs_through_every_block(self) -> None:
        """The blocks share a row builder and a label column. Measured per
        block instead, the values step in and out four times down one screen."""

        rendered = _render(_config())
        columns = set()
        for line in rendered.splitlines():
            if not line.strip() or line.startswith("-") or line.strip() in BLOCKS:
                continue
            match = re.match(r"^(?:.*?\S)?\s{2,}(\S.*)$", line)
            if match is not None:
                columns.add(len(line) - len(match.group(1)))
        self.assertGreater(len(columns), 0, f"no rows parsed out of: {rendered}")
        self.assertEqual(len(columns), 1, f"more than one value column: {sorted(columns)}")

    def test_no_markers_appear_in_the_header(self) -> None:
        """Markers belong to the rail, where a line's state changes. Every row
        here is settled the moment it is printed."""

        rendered = _render(_config())
        for marker in (console.DONE, console.PENDING, console.WARN, console.FAIL, console.RAIL):
            self.assertNotIn(marker, rendered)


class TheValuesAreResolvedTest(unittest.TestCase):
    def test_clients_per_round_is_computed_not_echoed(self) -> None:
        """participation_rate 0.25 over 20 clients is the fact a reader wants;
        "0.25" is the one the config already gave them."""

        rendered = _render(_config(participation_rate=0.25, num_clients=20))
        self.assertIn("Clients per round", rendered)
        self.assertRegex(rendered, r"Clients per round\s+5\b")

    def test_a_measured_roster_overrides_the_configured_one(self) -> None:
        """A manifest dataset's roster is a counted fact and the config's
        num_clients is a guess that may not even be set."""

        rendered = _render(_config(num_clients=20), client_count=97)
        self.assertRegex(rendered, r"Clients\s+97\b")
        self.assertRegex(rendered, r"Clients per round\s+25\b")

    def test_a_full_participation_rate_takes_every_client(self) -> None:
        rendered = _render(_config(participation_rate=1.0, num_clients=20))
        self.assertRegex(rendered, r"Clients per round\s+20\b")

    def test_a_participation_probability_states_the_mean_and_how_often_none(self) -> None:
        """Under Bernoulli participation the count is a draw, so one number would
        be false; and a round with no client is not aggregated, which a reader
        should know before round one rather than find in the metrics."""

        rendered = _render(
            _config(participation_rate=None, participation_probability=0.2, num_clients=5)
        )
        self.assertRegex(rendered, r"Participation probability\s+0\.2\b")
        self.assertNotIn("Participation rate", rendered)
        # 5 x 0.2 = 1 on average; 0.8 ** 5 = 32.8% of rounds select nobody.
        self.assertRegex(
            rendered,
            r"Clients per round\s+1 on average, varying by round; none in 32\.8% of rounds",
        )

    def test_a_negligible_chance_of_an_empty_round_is_not_stated(self) -> None:
        rendered = _render(
            _config(participation_rate=None, participation_probability=0.25, num_clients=100)
        )
        self.assertRegex(
            rendered, re.compile(r"Clients per round\s+25 on average, varying by round$", re.M)
        )

    def test_a_participation_probability_of_one_takes_every_client(self) -> None:
        rendered = _render(
            _config(participation_rate=None, participation_probability=1.0, num_clients=20)
        )
        self.assertRegex(rendered, re.compile(r"Clients per round\s+20$", re.M))

    def test_the_schedules_are_read_back_as_english(self) -> None:
        rendered = _render(_config())
        self.assertIn("every 10 rounds, participating clients", rendered)
        self.assertIn("every 5 rounds, all clients", rendered)

    def test_a_split_that_never_runs_is_not_listed(self) -> None:
        config = _config()
        config.evaluation.test = SplitEvaluationConfig(every="never", clients="all")
        rendered = _render(config)
        self.assertNotIn("evaluation.test ", rendered)
        self.assertNotIn("test_accuracy_avg", rendered)


class AmberIsScarceTest(unittest.TestCase):
    def test_an_ordinary_run_has_no_amber_at_all(self) -> None:
        with TemporaryDirectory() as empty:
            self.assertEqual(_amber_lines(_config(output_dir=empty)), [])

    def test_loaded_extensions_are_amber(self) -> None:
        """The fifth member of the fixed set: some of what this run is built
        from is not the package, and the component names would otherwise read
        as naming the shipped ones."""

        with TemporaryDirectory() as empty:
            lines = _amber_lines(
                _config(output_dir=empty, extensions=["examples/thing/problem.py"])
            )
        self.assertEqual(len(lines), 1)
        self.assertIn("Extensions", lines[0])
        self.assertIn("examples/thing/problem.py", lines[0])

    def test_a_resumed_run_is_amber(self) -> None:
        lines = _amber_lines(_config(), resume_from="outputs/x/checkpoints/latest.pt")
        self.assertEqual(len(lines), 1)
        self.assertIn("Resumed from", lines[0])

    def test_warn_only_determinism_is_amber_and_plain_determinism_is_not(self) -> None:
        with TemporaryDirectory() as empty:
            strict = _config(
                output_dir=empty,
                runtime_extra={"deterministic": True},
            )
            self.assertEqual(_amber_lines(strict), [])

            warn_only = _config(
                output_dir=empty,
                runtime_extra={"deterministic": True, "deterministic_warn_only": True},
            )
            lines = _amber_lines(warn_only)
            self.assertEqual(len(lines), 1)
            self.assertIn("warn_only", lines[0])

    def test_only_a_numerics_changing_matmul_precision_is_amber(self) -> None:
        with TemporaryDirectory() as empty:
            highest = _config(
                output_dir=empty,
                runtime_extra={"performance": {"matmul_precision": "highest"}},
            )
            self.assertEqual(
                _amber_lines(highest),
                [],
                "'highest' is what torch does anyway; ambering it would spend "
                "the colour on the case where nothing changed",
            )

            high = _config(
                output_dir=empty,
                runtime_extra={"performance": {"matmul_precision": "high"}},
            )
            lines = _amber_lines(high)
            self.assertEqual(len(lines), 1)
            self.assertIn("high", lines[0])

    def test_a_non_empty_output_directory_is_amber(self) -> None:
        with TemporaryDirectory() as directory:
            (Path(directory) / "round_metrics.csv").write_text("round_id\n", encoding="utf-8")
            lines = _amber_lines(_config(output_dir=directory))
            self.assertEqual(len(lines), 1)
            self.assertIn("already holds 1 file", lines[0])

    def test_a_resumed_run_does_not_also_amber_its_own_output_directory(self) -> None:
        """A resume's output directory holds the earlier attempt by definition.
        Two ambers for one fact is how amber stops meaning anything."""

        with TemporaryDirectory() as directory:
            (Path(directory) / "run.json").write_text("{}", encoding="utf-8")
            lines = _amber_lines(
                _config(output_dir=directory, runtime_extra={"resume_from": "checkpoints/last.pt"})
            )
            self.assertEqual(len(lines), 1)
            self.assertIn("Resumed from", lines[0])


class TheMetricsBlockTest(unittest.TestCase):
    def test_each_listed_column_carries_a_gloss(self) -> None:
        rendered = _render(_config())
        for name in ("test_accuracy_avg", "test_accuracy_sample_weighted_avg"):
            self.assertIn(name, rendered)
        self.assertIn("pooled over examples", rendered)
        self.assertIn("averaged over clients", rendered)

    def test_the_train_gloss_follows_the_configured_client_scope(self) -> None:
        rendered = _render(_config())
        self.assertIn("selected clients' train data", rendered)

    def test_verbose_lists_every_column_and_normal_lists_a_subset(self) -> None:
        config = _config()
        normal = _render(config)
        verbose = _render(config, surface_options={"verbose": True})

        planned = terminal_logging._planned_metric_names(config)
        self.assertIn(f"{len(planned)}, all listed", verbose)
        self.assertIn(f"of {len(planned)} listed", normal)
        for name in planned:
            self.assertIn(name, verbose)
        self.assertLess(len(normal.splitlines()), len(verbose.splitlines()))

    def test_the_verbose_column_list_is_what_the_run_will_write(self) -> None:
        """Built from client_metric_names and the two unfilterable-metric
        dictionaries, not from a second hand-kept list."""

        config = _config(strategy="scaffold", update_rule="scaffold")
        planned = set(terminal_logging._planned_metric_names(config))

        for split in ("train", "val", "test"):
            self.assertTrue(
                client_metric_names(split, config.client_statistics) <= planned,
                f"the {split} aggregates are missing from the planned columns",
            )
        self.assertIn("central_test_loss", planned)
        # SCAFFOLD's server diagnostics survive any server.metrics list, so a
        # run of it writes them whether or not the config asked.
        self.assertIn("server_control_norm", planned)
        self.assertIn("mean_client_control_delta_norm", planned)

    def test_model_scope_both_doubles_the_column_set(self) -> None:
        single = terminal_logging._planned_metric_names(_config(model_scope="global"))
        both = terminal_logging._planned_metric_names(_config(model_scope="both"))
        self.assertTrue(any(name.startswith("personal_") for name in both))
        self.assertFalse(any(name.startswith("personal_") for name in single))
        self.assertGreater(len(both), len(single))


class TheHeaderRespectsVerbosityTest(unittest.TestCase):
    def test_quiet_prints_nothing(self) -> None:
        buffer = io.StringIO()
        config = _config(runtime_extra={"quiet": True})
        terminal_logging.print_plan_header(
            config,
            **_determinism(config),
            surface=console.build_surface(file=buffer, quiet=True),
        )
        self.assertEqual(buffer.getvalue(), "")

    def test_the_config_alone_can_ask_for_quiet(self) -> None:
        """The surface is built from the config when none is passed, which is
        how the two call sites in runner.py reach it."""

        buffer = io.StringIO()
        with unittest.mock.patch("sys.stdout", buffer):
            quiet = _config(runtime_extra={"quiet": True})
            terminal_logging.print_plan_header(quiet, **_determinism(quiet))
        self.assertEqual(buffer.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
