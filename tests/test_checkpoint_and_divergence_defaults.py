"""Checkpoint defaults that do not depend on the block, and task-aware selection and divergence.

Three defaults assumed every run was a classification run with its
checkpointing block written out:

- an absent ``runtime.checkpointing`` block took a policy of its own -- every
  round checkpointed, no latest.pt or best.pt, no pruning -- the opposite of
  the per-key defaults a written block gets. Now an absent block is an empty
  one (``checkpoint_config_with_defaults``);
- ``best_metric`` defaulted to ``val_accuracy_sample_weighted_avg``, a column
  a task without accuracy never writes. It defaults from the task's declared
  metrics now, and ``save_best`` to whether the run evaluates validation at
  all (``config.resolved_checkpointing``);
- the divergence monitor took every watched metric to be better lower, so
  patience on ``fit_accuracy`` called learning a stall and a blow-up ceiling
  called it a divergence. The direction comes from the task's declared
  metrics (``config.divergence_direction``), and a ceiling on a metric that
  is better higher is refused at load.

Every shipped config states its checkpointing block, names ``best_metric``
wherever ``save_best`` is on, and watches ``fit_loss``, so none of them runs
differently; tests/test_shipped_configs_resolve_as_recorded.py holds that.
"""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import yaml

from fedbrew.core import registry
from fedbrew.core.checkpointing import checkpoint_config_with_defaults
from fedbrew.core.config import (
    default_selection_metric,
    divergence_direction,
    load_config,
    resolved_checkpointing,
)
from fedbrew.core.divergence import DivergenceMonitor, divergence_monitor
from fedbrew.core.metrics import declared_direction
from fedbrew.core.refusal import RunRefused

pytestmark = pytest.mark.fast

BASE = textwrap.dedent("""
    experiment:
      seed: 1
      output_dir: out
    server:
      strategy: fedavg
      participation_rate: 1
    client:
      update_rule: fedavg
      update_mode: sequential_epoch
      batch_size: 4
      learning_rate: 0.05
      learning_rate_schedule: constant
      momentum: 0.0
      weight_decay: 0.0
    data:
      num_clients: 2
      samples_per_client: 8
      input_dim: 4
      num_classes: 2
    model:
      name: mlp
      input_dim: 4
      hidden_dim: 8
      num_classes: 2
    runtime:
      device: cpu
    schedule:
      rounds: 1
      local_iterations: 1
    """)

#: What a quadratic example task declares: no accuracy.
NO_ACCURACY = {"loss": "min", "optimality_gap": "min"}


class _Directory(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def _load(self, edit: Any = None) -> Any:
        raw = yaml.safe_load(BASE)
        if edit is not None:
            edit(raw)
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return load_config(path)


class TheBlockDoesNotChangeTheDefaultsTest(_Directory):
    def test_an_absent_block_is_an_empty_one(self) -> None:
        self.assertEqual(checkpoint_config_with_defaults(None), checkpoint_config_with_defaults({}))

    def test_the_defaults_are_the_per_key_ones(self) -> None:
        policy = checkpoint_config_with_defaults(None)
        self.assertEqual(
            {key: policy[key] for key in ("save_last", "save_every_round", "keep_last")},
            {"save_last": True, "save_every_round": False, "keep_last": 3},
        )

    def test_a_run_without_the_block_resolves_as_one_with_an_empty_block(self) -> None:
        def empty(raw: dict[str, Any]) -> None:
            raw["runtime"]["checkpointing"] = {}

        self.assertEqual(
            resolved_checkpointing(self._load()), resolved_checkpointing(self._load(empty))
        )

    def test_run_json_records_what_the_file_stated(self) -> None:
        self.assertNotIn("checkpointing", self._load().runtime.extra)


class TheSelectionMetricDefaultsFromTheTaskTest(_Directory):
    def test_a_task_that_reports_accuracy_selects_on_it(self) -> None:
        config = self._load()
        self.assertEqual(default_selection_metric(config), "val_accuracy_sample_weighted_avg")
        self.assertEqual(
            resolved_checkpointing(config)["best_metric"], default_selection_metric(config)
        )

    def test_a_task_without_accuracy_selects_on_the_loss(self) -> None:
        config = self._load()
        with mock.patch.object(registry.tasks, "metrics", return_value=NO_ACCURACY):
            self.assertEqual(default_selection_metric(config), "val_loss_sample_weighted_avg")

    def test_the_personal_pass_carries_its_prefix(self) -> None:
        config = self._load()
        config.evaluation.model_scope = "personal"
        self.assertEqual(
            default_selection_metric(config), "personal_val_accuracy_sample_weighted_avg"
        )

    def test_a_stated_metric_is_kept(self) -> None:
        def state(raw: dict[str, Any]) -> None:
            raw["runtime"]["checkpointing"] = {"best_metric": "val_loss_avg"}

        self.assertEqual(resolved_checkpointing(self._load(state))["best_metric"], "val_loss_avg")

    def test_the_plan_header_says_the_default_is_the_task_s(self) -> None:
        from fedbrew.core.logging import _metrics_rows

        rows = {row.label: row.value for row in _metrics_rows(self._load(), verbose=False)}
        self.assertEqual(
            rows["checkpoint selects on"], "val_accuracy_sample_weighted_avg (the task's default)"
        )


class SaveBestFollowsTheValidationScheduleTest(_Directory):
    def _val(self, every: object, **checkpointing: object) -> Any:
        def edit(raw: dict[str, Any]) -> None:
            raw["evaluation"] = {"val": {"every": every}}
            if checkpointing:
                raw["runtime"]["checkpointing"] = dict(checkpointing)

        return self._load(edit)

    def test_on_when_validation_is_evaluated(self) -> None:
        self.assertTrue(resolved_checkpointing(self._val(1))["save_best"])

    def test_off_when_it_is_not_so_a_minimal_config_loads(self) -> None:
        self.assertFalse(resolved_checkpointing(self._val("never"))["save_best"])

    def test_a_stated_save_best_without_validation_is_still_refused(self) -> None:
        with self.assertRaisesRegex(RunRefused, "save_best is on but evaluation.val.every"):
            self._val("never", save_best=True)


class TheDirectionIsTheTasksTest(unittest.TestCase):
    DIRECTIONS = {"loss": "min", "accuracy": "max"}

    def test_names_that_carry_a_declared_metric(self) -> None:
        for name, expected in (
            ("fit_loss", "min"),
            ("fit_accuracy", "max"),
            ("val_accuracy_sample_weighted_avg", "max"),
            ("personal_val_accuracy_worst2p5", "max"),
            ("test_loss_max", "min"),
            ("central_test_accuracy", "max"),
        ):
            with self.subTest(name=name):
                self.assertEqual(declared_direction(name, self.DIRECTIONS), expected)

    def test_names_that_do_not(self) -> None:
        for name in (
            "fit_proximal_loss",
            "val_accuracy_std",
            "val_num_clients",
            "momentum_norm",
            "communicated_bytes",
        ):
            with self.subTest(name=name):
                self.assertIsNone(declared_direction(name, self.DIRECTIONS))


class ACeilingOnAMaxMetricIsRefusedTest(_Directory):
    def _watch(self, **divergence: object) -> Any:
        def edit(raw: dict[str, Any]) -> None:
            raw["divergence"] = {"metric": "fit_accuracy", "non_finite": True, **divergence}

        return self._load(edit)

    def test_the_default_blowup_factor_is_refused(self) -> None:
        with self.assertRaisesRegex(RunRefused, r"divergence\.blowup_factor .*better higher"):
            self._watch()

    def test_an_absolute_ceiling_is_refused_too(self) -> None:
        with self.assertRaisesRegex(RunRefused, r"divergence\.blowup_absolute"):
            self._watch(blowup_factor=None, blowup_absolute=0.9)

    def test_patience_and_non_finite_load(self) -> None:
        config = self._watch(blowup_factor=None, patience=3)
        self.assertEqual(divergence_direction(config), "max")

    def test_a_metric_the_task_does_not_declare_is_watched_as_before(self) -> None:
        def edit(raw: dict[str, Any]) -> None:
            raw["divergence"] = {"metric": "communicated_bytes", "non_finite": True}

        self.assertEqual(divergence_direction(self._load(edit)), "min")


class TheMonitorWatchesTheDeclaredSideTest(unittest.TestCase):
    def _monitor(self, direction: str, patience: int = 2) -> DivergenceMonitor:
        from fedbrew.core.config import DivergenceConfig

        config = DivergenceConfig(
            metric="fit_accuracy",
            non_finite=True,
            blowup_factor=None,
            blowup_absolute=None,
            patience=patience,
            min_delta=0.0,
        )
        return divergence_monitor(config, direction)

    def _feed(self, monitor: DivergenceMonitor, values: list[float]) -> Any:
        verdict = None
        for round_id, value in enumerate(values, start=1):
            verdict = monitor.update(round_id, {"fit_accuracy": value}) or verdict
        return verdict

    def test_a_rising_accuracy_is_not_a_stall(self) -> None:
        self.assertIsNone(self._feed(self._monitor("max"), [0.1, 0.2, 0.3, 0.4, 0.5]))

    def test_a_falling_one_is(self) -> None:
        verdict = self._feed(self._monitor("max"), [0.5, 0.4, 0.3])
        self.assertEqual((verdict.status, verdict.threshold, verdict.round_id), ("stalled", 0.5, 3))

    def test_the_lower_is_better_side_is_the_monitor_every_shipped_config_uses(self) -> None:
        self.assertIs(type(self._monitor("min")), DivergenceMonitor)
        self.assertIsNotNone(self._feed(self._monitor("min"), [0.1, 0.2, 0.3]))

    def test_non_finite_fires_either_way(self) -> None:
        verdict = self._feed(self._monitor("max"), [0.1, float("nan")])
        self.assertEqual(verdict.detector, "non_finite")


if __name__ == "__main__":
    unittest.main()
