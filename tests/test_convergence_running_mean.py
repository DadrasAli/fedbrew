"""``convergence.metrics``: the exact mean over the run's iterates, written every round.

A run asked for it (``fedbrew/core/convergence.py``) evaluates the chosen
metrics at every round's global model and writes ``<column>_running_mean``,
the mean over the iterates ``x_1 .. x_t`` of rounds 1 to ``t``. This module
holds:

- the sum is exact: ``ExactSum`` is ``math.fsum`` after every addition, over
  values of wildly different magnitudes;
- the column on round ``t`` is ``math.fsum`` of the metric's values on rounds
  1 to ``t``, divided by ``t``, bit for bit -- the values taken from a run of
  the same config that evaluates the metric every round itself;
- what the section changes about a run is what it adds: with the config's own
  evaluation schedules left as they were, every other cell and every checkpoint
  tensor is the run's without the section, the evaluations the mean forces
  appear only as the mean, and a metric the schedule asks for is still written;
- the resident round, the per-round batched path and the sequential executor
  write the same means (bit for bit between the two batched paths, to the
  executor's tolerance against the sequential one);
- a resumed run continues the means from its checkpoint and writes the run's
  own, and a checkpoint that holds none refuses the resume;
- a metric nothing evaluates, a repeat, and a task without a gradient are refused;
- the section off: no object, no schedule changed, no column.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import yaml

from fedbrew.core import runner
from fedbrew.core.checkpointing import load_checkpoint
from fedbrew.core.config import ConvergenceConfig, load_config
from fedbrew.core.convergence import (
    ExactSum,
    RunningMeans,
    resolve_convergence_metrics,
    running_mean_column,
    set_up,
)
from fedbrew.core.refusal import RunRefused
from tests.test_batched_executor_tolerance import (
    _rows,
    example_config,
    set_performance,
)
from tests.test_resident_round import ResidentRuns, _clean, _executor, _stopping, per_round

GAP = "central_test_optimality_gap"
GRAD = "grad_norm_sq"


def with_means(config: dict[str, Any], *metrics: str) -> dict[str, Any]:
    config = copy.deepcopy(config)
    config["convergence"] = {"metrics": list(metrics)}
    return config


def column(output: Path, name: str, file: str = "round_metrics.csv") -> list[str]:
    return [row[name] for row in _rows(output / file)]


@pytest.mark.fast
class ExactSumTest(unittest.TestCase):
    def test_it_is_fsum_after_every_addition(self) -> None:
        for seed in range(5):
            rng = random.Random(seed)
            values = [
                rng.choice((-1, 1)) * rng.random() * 10.0 ** rng.randint(-30, 30)
                for _ in range(400)
            ]
            total = ExactSum()
            for count, value in enumerate(values, start=1):
                total.add(value)
                self.assertEqual(total.value(), math.fsum(values[:count]), (seed, count))

    def test_the_naive_sum_is_not_that(self) -> None:
        values = [1e16, 1.0, -1e16] * 7
        naive = 0.0
        for value in values:
            naive += value
        total = ExactSum()
        for value in values:
            total.add(value)
        self.assertEqual(total.value(), 7.0)
        self.assertNotEqual(naive, 7.0)

    def test_a_restored_sum_continues_exactly(self) -> None:
        rng = random.Random(1)
        values = [rng.random() * 10.0 ** rng.randint(-8, 8) for _ in range(100)]
        whole = ExactSum()
        first = ExactSum()
        for value in values[:60]:
            whole.add(value)
            first.add(value)
        second = ExactSum(first.partials)
        for value in values[60:]:
            whole.add(value)
            second.add(value)
        self.assertEqual(second.value(), whole.value())
        self.assertEqual(second.value(), math.fsum(values))


@pytest.mark.fast
class TheMeansObjectTest(unittest.TestCase):
    def _means(self, **schedules: Any) -> RunningMeans:
        return RunningMeans(
            [GAP, GRAD],
            central_schedule=schedules.get("central", 3),
            grad_norm_schedule=schedules.get("grad_norm", None),
            global_rounds=schedules.get("rounds", 6),
        )

    def test_the_mean_of_the_iterates_seen_so_far(self) -> None:
        means = self._means(central=1, grad_norm=1)
        gaps, norms = [0.5, 0.25, 0.1, 0.07], [4.0, 2.0, 1.0, 0.5]
        for round_id, (gap, norm) in enumerate(zip(gaps, norms, strict=True), start=1):
            metrics = {GAP: gap, GRAD: norm}
            means.observe(round_id, metrics)
            self.assertEqual(
                metrics[running_mean_column(GAP)], math.fsum(gaps[:round_id]) / round_id
            )
            self.assertEqual(
                metrics[running_mean_column(GRAD)], math.fsum(norms[:round_id]) / round_id
            )

    def test_an_iterate_may_not_be_skipped_or_repeated(self) -> None:
        means = self._means()
        means.observe(1, {GAP: 1.0, GRAD: 1.0})
        with self.assertRaisesRegex(RuntimeError, "skipped or repeated"):
            means.observe(3, {GAP: 1.0, GRAD: 1.0})
        with self.assertRaisesRegex(RuntimeError, "skipped or repeated"):
            means.observe(1, {GAP: 1.0, GRAD: 1.0})

    def test_a_value_that_is_missing_or_not_finite_makes_the_mean_not_finite(self) -> None:
        for bad in (None, math.inf, math.nan):
            with self.subTest(bad=bad):
                means = self._means(central=1, grad_norm=1)
                first = {GAP: 1.0, GRAD: 1.0}
                means.observe(1, first)
                second = {GRAD: 1.0} if bad is None else {GAP: bad, GRAD: 1.0}
                means.observe(2, second)
                third = {GAP: 1.0, GRAD: 1.0}
                means.observe(3, third)
                self.assertTrue(math.isnan(second[running_mean_column(GAP)]))
                self.assertTrue(math.isnan(third[running_mean_column(GAP)]))
                self.assertEqual(third[running_mean_column(GRAD)], 1.0)

    def test_what_a_pass_forced_is_dropped_where_the_schedule_did_not_ask(self) -> None:
        # central every 3 of 6 rounds: asked on 1, 3, 6; grad_norm never asked.
        means = self._means(central=3, grad_norm=None)
        kept = {}
        for round_id in range(1, 7):
            metrics = {GAP: 1.0, "central_test_loss": 2.0, GRAD: 3.0, "fit_loss": 4.0}
            means.observe(round_id, metrics)
            kept[round_id] = sorted(name for name in metrics if not name.endswith("_running_mean"))
            # The means themselves are central_test_ and grad_norm_sq named, and always stay.
            self.assertIn(running_mean_column(GAP), metrics)
            self.assertIn(running_mean_column(GRAD), metrics)
        central_round = sorted([GAP, "central_test_loss", "fit_loss"])
        self.assertEqual(kept[1], central_round)
        self.assertEqual(kept[2], ["fit_loss"])
        self.assertEqual(kept[3], central_round)
        self.assertEqual(kept[4], ["fit_loss"])
        self.assertEqual(kept[6], central_round)

    def test_the_schedules_are_every_round_where_a_pass_is_needed(self) -> None:
        only_grad = RunningMeans(
            [GRAD], central_schedule=10, grad_norm_schedule=None, global_rounds=20
        )
        self.assertEqual(only_grad.schedules(), (10, 1))
        only_central = RunningMeans(
            [GAP], central_schedule=None, grad_norm_schedule=5, global_rounds=20
        )
        self.assertEqual(only_central.schedules(), (1, 5))

    def test_the_section_off_builds_nothing_and_changes_no_schedule(self) -> None:
        self.assertEqual(set_up(ConvergenceConfig(), 5, 7, 10), (None, 5, 7))
        self.assertEqual(set_up(None, None, None, 10), (None, None, None))

    def test_a_state_is_restored_and_continues(self) -> None:
        whole = self._means(central=1, grad_norm=1)
        part = self._means(central=1, grad_norm=1)
        rng = random.Random(3)
        history = [{GAP: rng.random(), GRAD: rng.random() * 1e-9} for _ in range(8)]
        for round_id, values in enumerate(history[:5], start=1):
            whole.observe(round_id, dict(values))
            part.observe(round_id, dict(values))
        resumed = self._means(central=1, grad_norm=1)
        resumed.restore({"convergence": part.state()}, 5, "latest.pt")
        for round_id, values in enumerate(history[5:], start=6):
            a, b = dict(values), dict(values)
            whole.observe(round_id, a)
            resumed.observe(round_id, b)
            self.assertEqual(a, b)

    def test_a_checkpoint_without_the_state_refuses_the_resume(self) -> None:
        means = self._means()
        with self.assertRaisesRegex(RunRefused, "no convergence state"):
            means.restore({"round_id": 3}, 3, "latest.pt")
        with self.assertRaisesRegex(RunRefused, "state of round 2, not 3"):
            means.restore({"convergence": {"rounds": 2, "metrics": {}}}, 3, "latest.pt")
        with self.assertRaisesRegex(RunRefused, "running means of"):
            means.restore({"convergence": {"rounds": 3, "metrics": {GAP: {}}}}, 3, "latest.pt")


@pytest.mark.fast
class TheNamesResolveTest(unittest.TestCase):
    CENTRAL = ("loss", "optimality_gap")

    def _resolve(self, names: list[str], grad_norm: bool = True) -> list[str]:
        return resolve_convergence_metrics(
            names, central=self.CENTRAL, grad_norm=grad_norm, task="example"
        )

    def test_a_bare_central_metric_is_its_central_column(self) -> None:
        self.assertEqual(
            self._resolve(["optimality_gap", GRAD, "central_test_loss"]),
            [GAP, GRAD, "central_test_loss"],
        )

    def test_what_nothing_evaluates_is_refused(self) -> None:
        with self.assertRaisesRegex(RunRefused, "does not evaluate"):
            self._resolve(["accuracy"])
        with self.assertRaisesRegex(RunRefused, "does not evaluate"):
            self._resolve(["fit_loss"])
        with self.assertRaisesRegex(RunRefused, "declares no gradient"):
            self._resolve([GRAD], grad_norm=False)

    def test_a_repeat_is_refused(self) -> None:
        with self.assertRaisesRegex(RunRefused, "twice"):
            self._resolve(["optimality_gap", GAP])


class ConvergenceRuns(ResidentRuns):
    """fed-lasso-l2's FedAvg arm, 4 rounds, with the section on."""

    def config(self, **evaluation: Any) -> dict[str, Any]:
        config = _clean(example_config("fed-lasso-l2", "fedavg"))
        config.setdefault("evaluation", {}).update(evaluation)
        return config

    def assertColumnsMatch(self, a: Path, b: Path, skip: tuple[str, ...] = ()) -> None:
        """Every non-timing cell of round_metrics.csv except ``skip`` is the same text."""

        from tests.test_reproducibility import TIMING

        rows_a, rows_b = _rows(a / "round_metrics.csv"), _rows(b / "round_metrics.csv")
        self.assertEqual(len(rows_a), len(rows_b))
        for row_a, row_b in zip(rows_a, rows_b, strict=True):
            for name in row_b:
                if name in TIMING or name in skip:
                    continue
                self.assertEqual(row_a[name], row_b[name], name)


class TheColumnIsTheMeanOfEveryRoundsMetricTest(ConvergenceRuns):
    def test_exactly_the_mean_of_the_values_of_rounds_one_to_t(self) -> None:
        # The reference evaluates both metrics itself at every round, with no section.
        reference = self.run_config(
            self.config(central_test={"every": 1}, grad_norm={"every": 1}), "batched"
        )
        gaps = [float(value) for value in column(reference, GAP)]
        norms = [float(value) for value in column(reference, GRAD)]
        # The run with the section keeps the config's own schedules: central every 3, no grad_norm.
        held = self.run_config(
            with_means(self.config(central_test={"every": 3}), "optimality_gap", GRAD), "batched"
        )
        for round_id, (mean_gap, mean_norm) in enumerate(
            zip(
                column(held, running_mean_column(GAP)),
                column(held, running_mean_column(GRAD)),
                strict=True,
            ),
            start=1,
        ):
            self.assertEqual(float(mean_gap), math.fsum(gaps[:round_id]) / round_id, round_id)
            self.assertEqual(float(mean_norm), math.fsum(norms[:round_id]) / round_id, round_id)

    def test_the_run_writes_the_values_the_mean_is_of(self) -> None:
        # Evaluated every round by the config too: the raw column and the mean are one run's.
        config = with_means(
            self.config(central_test={"every": 1}, grad_norm={"every": 1}),
            "optimality_gap",
            GRAD,
            "loss",
        )
        held = self.run_config(config, "batched")
        for metric in (GAP, GRAD, "central_test_loss"):
            values = [float(value) for value in column(held, metric)]
            means = [float(value) for value in column(held, running_mean_column(metric))]
            self.assertEqual(
                means, [math.fsum(values[:t]) / t for t in range(1, len(values) + 1)], metric
            )

    def test_nothing_else_about_the_run_changes(self) -> None:
        asked = self.config(central_test={"every": 3})
        without = self.run_config(asked, "batched")
        held = self.run_config(with_means(asked, "optimality_gap", GRAD), "batched")
        means = (running_mean_column(GAP), running_mean_column(GRAD))
        names_without = list(_rows(without / "round_metrics.csv")[0])
        names_held = list(_rows(held / "round_metrics.csv")[0])
        self.assertEqual([n for n in names_held if n not in means], names_without)
        self.assertEqual(sorted(n for n in names_held if n in means), sorted(means))
        self.assertColumnsMatch(held, without, skip=means)
        # Round 2 is not a central round: its central cells are blank, as without the section,
        # while the mean is written on it.
        self.assertEqual(column(held, GAP)[1], "")
        self.assertNotEqual(column(held, running_mean_column(GAP))[1], "")
        self.assertNotIn(GRAD, names_held)
        for name in ("client_update_metrics.csv", "client_metrics.csv"):
            self.assertEqual(
                (held / name).read_text().splitlines()[1:],
                (without / name).read_text().splitlines()[1:],
            )
        asked_checkpoints = sorted((without / "checkpoints").glob("round_*.pt"))
        self.assertEqual(len(asked_checkpoints), 4)
        for path in asked_checkpoints:
            a, b = load_checkpoint(path), load_checkpoint(held / "checkpoints" / path.name)
            for key in a["model_state"]:
                self.assertTrue((a["model_state"][key] == b["model_state"][key]).all(), key)

    def test_a_metric_the_schedule_asks_for_is_still_written(self) -> None:
        held = self.run_config(with_means(self.config(grad_norm={"every": 2}), GRAD), "batched")
        self.assertEqual([bool(v) for v in column(held, GRAD)], [True, True, False, True])
        self.assertTrue(all(column(held, running_mean_column(GRAD))))

    def test_the_checkpoint_holds_the_state_of_its_round(self) -> None:
        held = self.run_config(with_means(self.config(), GRAD), "batched")
        for round_id, path in enumerate(sorted((held / "checkpoints").glob("round_*.pt")), start=1):
            state = load_checkpoint(path)["convergence"]
            self.assertEqual(state["rounds"], round_id)
            self.assertEqual(list(state["metrics"]), [GRAD])
        self.assertEqual(
            load_checkpoint(held / "checkpoints" / "latest.pt")["convergence"]["rounds"], 4
        )

    def test_a_run_without_the_section_has_no_column_and_no_state(self) -> None:
        plain = self.run_config(self.config(), "batched")
        self.assertFalse([n for n in _rows(plain / "round_metrics.csv")[0] if "running_mean" in n])
        self.assertNotIn("convergence", load_checkpoint(plain / "checkpoints" / "latest.pt"))


class TheExecutorsWriteTheSameMeansTest(ConvergenceRuns):
    def test_resident_per_round_and_sequential(self) -> None:
        config = with_means(self.config(central_test={"every": 3}), "optimality_gap", GRAD, "loss")
        held, reference = self.pair(config)
        self.assertSameRun(held, reference)
        sequential = self.run_config(config, "sequential")
        self.assertAgree(held, sequential)

    def test_graph_replay_is_not_asked_for_and_changes_nothing(self) -> None:
        config = _clean(with_means(self.config(), "optimality_gap"))
        eager = self.run_config(config, "batched")
        with per_round():
            reference = self.run_config(config, "batched")
        self.assertEqual(_executor(eager)["rounds"], {"used": "resident"})
        self.assertSameRun(eager, reference)


class AResumeContinuesTheMeansTest(ConvergenceRuns):
    def _stopped(self, config: dict[str, Any], name: str, stop: int) -> tuple[Path, Path]:
        path = self.root / f"{name}.yaml"
        config = copy.deepcopy(config)
        config["experiment"]["output_dir"] = str(self.root / name)
        set_performance(config, executor="batched")
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with mock.patch.object(runner, "_round_progress_reporter", _stopping(stop)):
            with self.assertRaises(KeyboardInterrupt):
                runner.run(path, args=None)
        return path, Path(config["experiment"]["output_dir"])

    def test_stopped_between_two_flushes_and_resumed(self) -> None:
        config = with_means(self.config(central_test={"every": 3}), "optimality_gap", GRAD)
        config["schedule"]["rounds"] = 6
        config["runtime"]["flush_every"] = 2
        whole = self.run_config(config, "batched")
        path, output = self._stopped(config, "stopped", 5)
        self.assertEqual(load_checkpoint(output / "checkpoints" / "latest.pt")["round_id"], 4)
        runner.run(path, args=argparse.Namespace(resume_latest=True))
        self.assertSameRun(output, whole)
        for name in (running_mean_column(GAP), running_mean_column(GRAD)):
            self.assertEqual(column(output, name), column(whole, name))

    def test_a_checkpoint_from_a_run_without_the_section_refuses_the_resume(self) -> None:
        config = self.config()
        config["schedule"]["rounds"] = 6
        config["runtime"]["flush_every"] = 2
        path, output = self._stopped(config, "without", 5)
        asked = with_means(config, GRAD)
        asked["experiment"]["output_dir"] = str(output)
        set_performance(asked, executor="batched")
        path.write_text(yaml.safe_dump(asked), encoding="utf-8")
        before = sorted(p.name for p in output.rglob("*") if p.is_file())
        with self.assertRaisesRegex(RunRefused, "carries no convergence state"):
            runner.run(path, args=argparse.Namespace(resume_latest=True))
        self.assertEqual(sorted(p.name for p in output.rglob("*") if p.is_file()), before)


class TheConfigTest(ConvergenceRuns):
    def _load(self, **section: Any) -> Any:
        config = copy.deepcopy(self.config())
        config["convergence"] = section
        path = self.root / "load.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return load_config(path)

    def test_names_are_resolved_to_columns(self) -> None:
        self.assertEqual(
            self._load(metrics=["optimality_gap", GRAD]).convergence.metrics, [GAP, GRAD]
        )

    def test_off_is_the_default(self) -> None:
        self.assertEqual(self._load().convergence.metrics, [])
        config = self.config()
        path = self.root / "default.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        self.assertEqual(load_config(path).convergence.metrics, [])

    def test_refusals(self) -> None:
        with self.assertRaisesRegex(RunRefused, "does not evaluate"):
            self._load(metrics=["accuracy"])
        with self.assertRaisesRegex(RunRefused, "twice"):
            self._load(metrics=["optimality_gap", GAP])
        with self.assertRaisesRegex(RunRefused, "list of metric names"):
            self._load(metrics=[1, 2])
        with self.assertRaisesRegex(RunRefused, "convergence"):
            self._load(metrics=[GRAD], mean_of="all")


def _cuda_usable() -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
    except RuntimeError:
        return False
    return True


@pytest.mark.cuda
@unittest.skipUnless(_cuda_usable(), "needs a usable CUDA device")
class OnCudaTest(ResidentRuns):
    """The means of a network on the device: resident, every round, the mean of what is written."""

    def test_the_column_is_the_mean_of_the_rounds_values_and_resident(self) -> None:
        from tests.test_batched_executor_tolerance import classification_config
        from tests.test_resident_round import FEDAVG

        config = _clean(classification_config(**FEDAVG, update_mode="single_batch"))
        config["runtime"]["device"] = "cuda"
        config["evaluation"] = {"central_test": {"every": 1}, "grad_norm": {"every": 1}}
        config["convergence"] = {"metrics": ["loss", GRAD]}
        held = self.run_config(config, "batched")
        self.assertEqual(_executor(held)["rounds"], {"used": "resident"})
        for metric in ("central_test_loss", GRAD):
            values = [float(value) for value in column(held, metric)]
            means = [float(value) for value in column(held, running_mean_column(metric))]
            self.assertEqual(
                means, [math.fsum(values[:t]) / t for t in range(1, len(values) + 1)], metric
            )
        # What the section forces is dropped where the config's own schedule does not ask.
        config["evaluation"] = {"central_test": {"every": 3}, "grad_norm": {"every": "never"}}
        sparse = self.run_config(config, "batched")
        self.assertEqual(column(sparse, "central_test_loss")[1], "")
        self.assertNotIn(GRAD, _rows(sparse / "round_metrics.csv")[0])
        self.assertTrue(all(column(sparse, running_mean_column(GRAD))))


if __name__ == "__main__":
    unittest.main()
