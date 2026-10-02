"""``convergence.iterates: before_update``: row ``t`` is the model before round ``t``'s update.

The default counts the iterates ``x_1 .. x_t``, the model after each round.
``before_update`` counts them as a method's analysis often does: row ``t`` is
``x_{t-1}``, the initial model included, and the last round applies no
update, so ``T`` rounds apply ``T - 1`` updates (``fedbrew/core/convergence.py``).
This module holds:

- the means object writes the held columns of the round before and holds the
  round's own, starting from the initial model's, and refuses a row it has no
  measurement for;
- on a run, row 1 is the initial model's measurement and row ``t > 1`` is the
  default run's row ``t - 1``, every global-model column bit for bit; the
  running mean is ``math.fsum`` of the shifted column; the last round samples
  no client, and the run ends on the model of ``T - 1`` updates;
- the resident round, the per-round batched path and the sequential executor
  write the same rows (bit for bit between the batched two);
- a resumed run continues the held column from its checkpoint, and a checkpoint
  of the other convention refuses the resume;
- the default is unchanged, and a run that asks for the shift without the means,
  or with a client pass the shift would not move, is refused.
"""

from __future__ import annotations

import argparse
import copy
import math
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import yaml

from fedbrew.core import runner
from fedbrew.core.checkpointing import load_checkpoint
from fedbrew.core.config import load_config
from fedbrew.core.convergence import RunningMeans, running_mean_column
from fedbrew.core.refusal import RunRefused
from tests.test_batched_executor_tolerance import _rows, set_performance
from tests.test_convergence_running_mean import GAP, GRAD, ConvergenceRuns, column
from tests.test_resident_round import _stopping

LOSS = "central_test_loss"


def before(config: dict[str, Any], *metrics: str) -> dict[str, Any]:
    config = copy.deepcopy(config)
    config["convergence"] = {"metrics": list(metrics), "iterates": "before_update"}
    return config


class BeforeUpdateRuns(ConvergenceRuns):
    """fed-lasso-l2's FedAvg arm, 4 rounds, with no client split pass (the shift refuses one)."""

    def config(self, **evaluation: Any) -> dict[str, Any]:
        never = {split: {"every": "never"} for split in ("train", "val", "test")}
        return super().config(**{**never, **evaluation})


def _model(output: Path, name: str = "latest.pt") -> dict[str, Any]:
    return load_checkpoint(output / "checkpoints" / name)["model_state"]


@pytest.mark.fast
class TheMeansObjectTest(unittest.TestCase):
    def _means(self, rounds: int = 4) -> RunningMeans:
        return RunningMeans(
            [GAP, GRAD],
            central_schedule=1,
            grad_norm_schedule=1,
            global_rounds=rounds,
            before_update=True,
        )

    def test_each_row_is_given_the_round_befores_columns(self) -> None:
        means = self._means()
        means.start({GAP: 9.0, LOSS: 8.0, GRAD: 7.0})
        after = [(0.5, 0.4, 4.0), (0.25, 0.2, 2.0), (0.1, 0.09, 1.0), (0.05, 0.04, 0.5)]
        written = [(9.0, 8.0, 7.0), *after[:-1]]
        for round_id, ((gap, loss, norm), shown) in enumerate(
            zip(after, written, strict=True), start=1
        ):
            metrics = {GAP: gap, LOSS: loss, GRAD: norm, "fit_loss": 3.0}
            means.observe(round_id, metrics)
            self.assertEqual((metrics[GAP], metrics[LOSS], metrics[GRAD]), shown, round_id)
            self.assertEqual(metrics["fit_loss"], 3.0)
            gaps = [row[0] for row in written[:round_id]]
            self.assertEqual(metrics[running_mean_column(GAP)], math.fsum(gaps) / round_id)

    def test_only_the_last_round_skips_its_update(self) -> None:
        means = self._means(rounds=5)
        self.assertEqual([means.skips_update(t) for t in range(1, 6)], [False] * 4 + [True])
        default = RunningMeans([GAP], central_schedule=1, grad_norm_schedule=None, global_rounds=5)
        self.assertFalse(default.skips_update(5))

    def test_a_row_with_no_measurement_before_it_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "measure_start"):
            self._means().observe(1, {GAP: 1.0, GRAD: 1.0})

    def test_the_held_columns_are_in_the_state_and_restored(self) -> None:
        means = self._means()
        means.start({GAP: 9.0, GRAD: 7.0})
        means.observe(1, {GAP: 0.5, GRAD: 4.0})
        state = means.state()
        self.assertEqual(state["held"], {GAP: 0.5, GRAD: 4.0})
        restored = self._means()
        restored.restore({"convergence": state}, 1, "here")
        metrics = {GAP: 0.25, GRAD: 2.0}
        restored.observe(2, metrics)
        self.assertEqual((metrics[GAP], metrics[GRAD]), (0.5, 4.0))
        self.assertEqual(metrics[running_mean_column(GAP)], math.fsum([9.0, 0.5]) / 2)

    def test_a_checkpoint_of_the_other_convention_refuses_the_resume(self) -> None:
        default = RunningMeans([GAP], central_schedule=1, grad_norm_schedule=None, global_rounds=4)
        default.observe(1, {GAP: 0.5})
        shifted = RunningMeans(
            [GAP], central_schedule=1, grad_norm_schedule=None, global_rounds=4, before_update=True
        )
        with self.assertRaisesRegex(RunRefused, "after each update"):
            shifted.restore({"convergence": default.state()}, 1, "here")
        shifted.start({GAP: 9.0})
        shifted.observe(1, {GAP: 0.5})
        fresh = RunningMeans([GAP], central_schedule=1, grad_norm_schedule=None, global_rounds=4)
        with self.assertRaisesRegex(RunRefused, "before each update"):
            fresh.restore({"convergence": shifted.state()}, 1, "here")


class TheRowsAreTheModelBeforeTheUpdateTest(BeforeUpdateRuns):
    def _every_round(self) -> dict[str, Any]:
        return self.config(central_test={"every": 1}, grad_norm={"every": 1})

    def test_row_t_is_the_default_runs_row_t_minus_one(self) -> None:
        default = self.run_config(self._every_round(), "batched")
        shifted = self.run_config(before(self._every_round(), "optimality_gap", GRAD), "batched")
        start = self.run_config(
            _rounds(before(self._every_round(), "optimality_gap", GRAD), 1), "batched"
        )
        names = [n for n in _rows(default / "round_metrics.csv")[0] if n.startswith("central_")]
        self.assertIn(GAP, names)
        for name in [*names, GRAD]:
            shown, measured = column(shifted, name), column(default, name)
            self.assertEqual(shown[1:], measured[:-1], name)
            self.assertEqual(shown[0], column(start, name)[0], name)
            self.assertNotEqual(shown[0], measured[0], name)
        for metric in (GAP, GRAD):
            values = [float(value) for value in column(shifted, metric)]
            means = [float(value) for value in column(shifted, running_mean_column(metric))]
            self.assertEqual(means, [math.fsum(values[:t]) / t for t in range(1, 5)], metric)

    def test_t_rounds_apply_t_minus_one_updates(self) -> None:
        config = before(self._every_round(), "optimality_gap")
        shifted = self.run_config(config, "batched")
        three = self.run_config(_rounds(self._every_round(), 3), "batched")
        for key, value in _model(three).items():
            self.assertTrue((_model(shifted)[key] == value).all(), key)
        for key, value in _model(shifted, "round_003.pt").items():
            self.assertTrue((_model(shifted)[key] == value).all(), key)
        rounds = {row["round_id"] for row in _rows(shifted / "client_update_metrics.csv")}
        self.assertEqual(rounds, {"1", "2", "3"})
        self.assertEqual(column(shifted, "num_clients")[-1], "0")

    def test_resident_per_round_and_sequential(self) -> None:
        config = before(self.config(central_test={"every": 3}), "optimality_gap", GRAD, "loss")
        held, reference = self.pair(config)
        self.assertSameRun(held, reference)
        self.assertAgree(held, self.run_config(config, "sequential"))

    def test_a_schedule_is_kept_on_the_shifted_rows(self) -> None:
        shifted = self.run_config(before(self.config(grad_norm={"every": 2}), GRAD), "batched")
        self.assertEqual([bool(v) for v in column(shifted, GRAD)], [True, True, False, True])


class AResumeContinuesTheHeldColumnTest(BeforeUpdateRuns):
    def _stopped(self, config: dict[str, Any], stop: int) -> tuple[Path, Path]:
        path = self.root / "stopped.yaml"
        config = copy.deepcopy(config)
        config["experiment"]["output_dir"] = str(self.root / "stopped")
        set_performance(config, executor="batched")
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with mock.patch.object(runner, "_round_progress_reporter", _stopping(stop)):
            with self.assertRaises(KeyboardInterrupt):
                runner.run(path, args=None)
        return path, Path(config["experiment"]["output_dir"])

    def test_stopped_and_resumed(self) -> None:
        config = _rounds(before(self.config(central_test={"every": 1}), "optimality_gap", GRAD), 6)
        config["runtime"]["flush_every"] = 2
        whole = self.run_config(config, "batched")
        path, output = self._stopped(config, 5)
        self.assertEqual(load_checkpoint(output / "checkpoints" / "latest.pt")["round_id"], 4)
        runner.run(path, args=argparse.Namespace(resume_latest=True))
        self.assertSameRun(output, whole)


class TheConfigTest(BeforeUpdateRuns):
    def _load(self, **section: Any) -> Any:
        config = copy.deepcopy(self.config())
        config["convergence"] = section
        path = self.root / "load.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return load_config(path)

    def test_after_the_update_is_the_default(self) -> None:
        self.assertEqual(self._load(metrics=[GRAD]).convergence.iterates, "after_update")
        loaded = self._load(metrics=[GRAD], iterates="before_update")
        self.assertEqual(loaded.convergence.iterates, "before_update")

    def test_refusals(self) -> None:
        with self.assertRaisesRegex(RunRefused, "one of after_update, before_update"):
            self._load(metrics=[GRAD], iterates="pre")
        with self.assertRaisesRegex(RunRefused, "convergence.metrics is empty"):
            self._load(iterates="before_update")
        config = before(self.config(test={"every": 1}), GRAD)
        path = self.root / "split.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with self.assertRaisesRegex(RunRefused, "evaluation.test"):
            load_config(path)


def _rounds(config: dict[str, Any], rounds: int) -> dict[str, Any]:
    config = copy.deepcopy(config)
    config["schedule"]["rounds"] = rounds
    return config
