"""A group of settings run in one process is each setting run alone.

``fedbrew/core/settings_group.py`` runs several configs that differ only in
numeric hyperparameters as one process, their clients trained together
(docs/11 §10). What it must keep:

- a group of one is its config run alone, bit for bit: every non-timing CSV
  cell, every round's checkpoint -- model, client states and the process-wide
  generator state it records;
- every setting of a group matches its own run alone within the batched
  executor's tolerance (chapter 11 §9), under the stacked results path, the
  per-client one (FedProx), the MLP, and a chunk budget that splits rounds;
- a setting that diverges, or is refused, stops alone: the others are
  bit-identical to a group without it;
- run.json records the group.
"""

from __future__ import annotations

import copy
import json
import random
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import torch
import yaml

from fedbrew.core.checkpointing import load_checkpoint
from fedbrew.core.runner import run
from fedbrew.core.settings_group import _Round, run_group
from tests.test_batched_executor_tolerance import (
    ExecutorRuns,
    classification_rule_config,
    example_config,
)
from tests.test_resident_round import per_round


def _batched(config: dict[str, Any], **performance: Any) -> dict[str, Any]:
    config = copy.deepcopy(config)
    config["runtime"].setdefault("performance", {}).update(executor="batched", **performance)
    return config


def _edited(config: dict[str, Any], **sections: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(config)
    for section, values in sections.items():
        config[section].update(values)
    return config


class GroupRuns(ExecutorRuns):
    """Writes configs, runs them alone or as a group, and compares what they wrote."""

    def write(self, config: dict[str, Any], tag: str) -> Path:
        self._count += 1
        config = copy.deepcopy(config)
        config["experiment"]["output_dir"] = str(self.root / f"{tag}{self._count}")
        path = self.root / f"{tag}{self._count}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    def alone(self, config: dict[str, Any]) -> Path:
        path = self.write(config, "alone")
        run(path, args=None)
        return self.output(path)

    def group(self, configs: list[dict[str, Any]]) -> tuple[list[Path], list[Any]]:
        paths = [self.write(config, "setting") for config in configs]
        outcomes = run_group(paths, ["client.learning_rate"])
        return [self.output(path) for path in paths], outcomes

    @staticmethod
    def output(path: Path) -> Path:
        return Path(yaml.safe_load(path.read_text())["experiment"]["output_dir"])

    def assertSameGenerators(self, first: Path, second: Path) -> None:
        for checkpoint in sorted((first / "checkpoints").glob("round_*.pt")):
            left = load_checkpoint(checkpoint)["rng_state"]
            right = load_checkpoint(second / "checkpoints" / checkpoint.name)["rng_state"]
            self.assertTrue(torch.equal(left["torch"], right["torch"]), checkpoint.name)
            self.assertEqual(left["python"], right["python"], checkpoint.name)


def _lasso() -> dict[str, Any]:
    return _batched(example_config("fed-lasso"))


class AGroupOfOneIsTheRunAloneTest(GroupRuns):
    def test_bit_for_bit(self) -> None:
        for label, config in (
            ("fedavg", _lasso()),
            ("fedprox", _batched(example_config("fed-lasso", "fedprox"))),
            ("small chunks", _batched(example_config("fed-lasso"), executor_chunk_bytes=3000)),
        ):
            with self.subTest(config=label):
                alone = self.alone(config)
                (grouped,), outcomes = self.group([config])
                self.assertEqual(outcomes[0].status, "completed")
                self.assertAgree(grouped, alone, exact=True)
                self.assertSameGenerators(grouped, alone)


class EverySettingIsItsRunAloneTest(GroupRuns):
    """Within the executor's tolerance, each setting against its own run alone."""

    def _each(self, configs: list[dict[str, Any]]) -> None:
        grouped, outcomes = self.group(configs)
        self.assertEqual([outcome.status for outcome in outcomes], ["completed"] * len(configs))
        for config, output in zip(configs, grouped, strict=True):
            alone = self.alone(config)
            self.assertAgree(output, alone)
            self.assertSameGenerators(output, alone)

    def test_fedavg_learning_rates_and_clipping(self) -> None:
        base = _lasso()
        self._each(
            [
                _edited(base, client={"learning_rate": 0.05}),
                _edited(base, client={"learning_rate": 0.1}),
                _edited(base, client={"learning_rate": 0.2, "max_grad_norm": 0.05}),
            ]
        )

    def test_fedprox_one_result_at_a_time(self) -> None:
        base = _batched(example_config("fed-lasso", "fedprox"))
        self._each(
            [
                _edited(base, client={"learning_rate": 0.004, "proximal_mu": 0.01}),
                _edited(base, client={"learning_rate": 0.002, "proximal_mu": 0.1}),
            ]
        )

    def test_server_learning_rate_and_momentum(self) -> None:
        base = _batched(example_config("fed-lasso", "fedavgm"))
        self._each(
            [
                _edited(base, server={"server_learning_rate": 0.03}),
                _edited(base, server={"server_learning_rate": 0.01, "beta1": 0.5}),
            ]
        )

    def test_a_budget_that_splits_every_round(self) -> None:
        base = _batched(example_config("fed-lasso"), executor_chunk_bytes=3000)
        self._each([_edited(base, client={"learning_rate": lr}) for lr in (0.05, 0.1, 0.2)])

    def test_the_mlp_with_momentum_and_weight_decay(self) -> None:
        base = _batched(classification_rule_config({"update_rule": "local_sgd"}))
        self._each(
            [
                _edited(base, client={"learning_rate": 0.05}),
                _edited(base, client={"learning_rate": 0.1, "momentum": 0.5}),
                _edited(base, client={"learning_rate": 0.2, "weight_decay": 0.001}),
            ]
        )


class EachSettingDrawsItsOwnNumbersTest(GroupRuns):
    """A setting that draws from the process-wide generators draws what it draws alone.

    No shipped batchable run draws from them after seeding, so the round's
    planning is made to draw here, from torch's and Python's, every round.
    """

    def test_generators_are_handed_over(self) -> None:
        from fedbrew.core import batched_executor, settings_group

        real = batched_executor._plans

        def drawing(*args: Any, **kwargs: Any) -> Any:
            torch.rand(3)
            random.random()
            return real(*args, **kwargs)

        base = _lasso()
        configs = [_edited(base, client={"learning_rate": lr}) for lr in (0.05, 0.1, 0.2)]
        with (
            mock.patch.object(batched_executor, "_plans", drawing),
            mock.patch.object(settings_group, "_plans", drawing),
        ):
            grouped, _ = self.group(configs)
            # Alone on the per-round path, which calls the drawing _plans; a
            # resident run plans no client of its own.
            with per_round():
                for config, output in zip(configs, grouped, strict=True):
                    self.assertSameGenerators(output, self.alone(config))


class ASettingThatStopsStopsAloneTest(GroupRuns):
    """The others are bit-identical to a group without it."""

    def _others(self, stopping: dict[str, Any], expected: str, position: int) -> None:
        base = _lasso()
        others = [_edited(base, client={"learning_rate": lr}) for lr in (0.05, 0.1)]
        with_it = list(others)
        with_it.insert(position, stopping)
        outputs, outcomes = self.group(with_it)
        self.assertEqual(outcomes[position].status, expected)
        kept = [output for index, output in enumerate(outputs) if index != position]
        without, outcomes_without = self.group(others)
        self.assertEqual([outcome.status for outcome in outcomes_without], ["completed"] * 2)
        for output, reference in zip(kept, without, strict=True):
            self.assertAgree(output, reference, exact=True)
            self.assertSameGenerators(output, reference)

    def test_a_diverging_setting(self) -> None:
        diverging = _edited(_lasso(), client={"learning_rate": 1000.0})
        self._others(diverging, "diverged", 1)

    def test_a_refused_setting(self) -> None:
        # A finished run of another config in its directory: refused before round 1.
        refused = _edited(_lasso(), client={"learning_rate": 0.3})
        path = self.write(_edited(_lasso(), client={"learning_rate": 0.4}), "finished")
        run(path, args=None)
        refused["experiment"]["output_dir"] = str(self.output(path))
        base = _lasso()
        others = [_edited(base, client={"learning_rate": lr}) for lr in (0.05, 0.1)]
        paths = [self.write(config, "setting") for config in others]
        refused_path = self.root / "refused.yaml"
        refused_path.write_text(yaml.safe_dump(refused), encoding="utf-8")
        outcomes = run_group([paths[0], refused_path, paths[1]], [])
        self.assertEqual(
            [outcome.status for outcome in outcomes], ["completed", "refused", "completed"]
        )
        without, _ = self.group(others)
        for path, reference in zip(paths, without, strict=True):
            self.assertAgree(self.output(path), reference, exact=True)


class RunJsonRecordsTheGroupTest(GroupRuns):
    def test_the_record(self) -> None:
        base = _lasso()
        outputs, _ = self.group([_edited(base, client={"learning_rate": lr}) for lr in (0.05, 0.1)])
        records = [json.loads((output / "run.json").read_text()) for output in outputs]
        groups = [record["reproducibility"]["group"] for record in records]
        self.assertEqual({group["id"] for group in groups}, {groups[0]["id"]})
        self.assertEqual([group["position"] for group in groups], [1, 2])
        self.assertEqual({group["size"] for group in groups}, {2})
        self.assertEqual(groups[0]["varies"], ["client.learning_rate"])
        self.assertEqual(groups[0]["largest_chunk_rows"], 16)
        executor = records[0]["reproducibility"]["executor"]
        self.assertEqual(executor, {"used": "batched", "largest_chunk_clients": 8})

    def test_an_auto_budget_is_the_first_settings(self) -> None:
        from fedbrew.core import batched_executor

        base = _batched(_lasso(), executor_chunk_bytes="auto")
        frees = iter((8 << 30, 1 << 20))
        with mock.patch.object(batched_executor, "free_memory", side_effect=lambda _: next(frees)):
            outputs, _ = self.group(
                [_edited(base, client={"learning_rate": lr}) for lr in (0.05, 0.1)]
            )
        budgets = [
            json.loads((output / "run.json").read_text())["reproducibility"]["executor"][
                "chunk_bytes"
            ]
            for output in outputs
        ]
        self.assertEqual(budgets[0], budgets[1])
        self.assertEqual(budgets[0]["used"], 4 << 30)

    def test_a_run_alone_records_none(self) -> None:
        output = self.alone(_lasso())
        record = json.loads((output / "run.json").read_text())
        self.assertNotIn("group", record["reproducibility"])


class UnitsArePackedFirstUnitsFirstTest(unittest.TestCase):
    def test_packing(self) -> None:
        class Prep:
            def __init__(self, costs: list[int]) -> None:
                self.unit_costs = costs
                self.units = [(index, index + 1) for index in range(len(costs))]

        packed = _Round({0: Prep([4, 4]), 1: Prep([4, 4]), 2: Prep([4])}, budget=8)
        self.assertEqual(packed.chunks, [[(0, 0), (1, 0)], [(2, 0), (0, 1)], [(1, 1)]])


if __name__ == "__main__":
    unittest.main()
