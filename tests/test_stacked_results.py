"""A batched round's results handed over stacked are the per-client results, exactly.

The batched executor hands a chunk's results to the aggregator as one
``StackedFitResults`` (``fedbrew/core/stacked_results.py``) when the rule
builds them stacked and the aggregator takes them; otherwise, one
``FitResult`` at a time (chapter 01 §2). What is pinned here:

- what the server is handed: iterated, a stacked result is each client's
  FitResult in request order, with its own row, count, metrics and payload;
- ``FedAvgServer`` folds a stack whole: the round's metrics are the
  per-client fold's to the bit, the model is the same to ``1e-12`` (the rows
  of one stack are summed together rather than interleaved with another's),
  and a server that weighs or checks a result its own way is handed the
  results one by one and folds them exactly as before;
- every refusal is the per-client path's, in its words, naming the same
  client: a non-finite state, weight or metadata;
- a run: FedAvg, local_sgd and local_adamw take the stacked path, FedProx and
  SCAFFOLD the per-client one; with every client in one bucket the stacked
  run is bit-identical to the per-client batched run, the classification
  task's folded post-fit metrics included; on ragged clients the first
  round's per-client records are, and the rest agree to ``1e-12``; an
  observer that records one result at a time is handed each.
"""

from __future__ import annotations

import unittest
from contextlib import nullcontext
from typing import Any
from unittest import mock

import torch

from fedbrew.core import loop
from fedbrew.core.batched_executor import BatchedExecutor
from fedbrew.core.protocol import FitResult, RoundInfo
from fedbrew.core.stacked_results import MetricColumns, StackedFitResults, StackedResults
from fedbrew.core.torch_utils import NonFiniteStateError, StateStack
from fedbrew.servers.fedavg import FedAvgServer, WeightedMetricAccumulator
from tests.test_batched_executor_tolerance import (
    TOLERANCE,
    ExecutorRuns,
    _rows,
    classification_rule_config,
    example_config,
    fedavg_modes,
    float64_classification,
    ragged_clients,
    rule_arms,
    rule_config,
    with_client,
)
from tests.test_reproducibility import TIMING

METADATA = {"model_state_scope": "full"}


def _stacked(generator: torch.Generator, *, poison: dict[int, str] | None = None) -> Any:
    """Five clients in two interleaved stacks, one metric reported by three of them."""

    positions = ([0, 2, 4], [1, 3])
    stacks = []
    for rows in positions:
        tensors = {
            "w": torch.randn(len(rows), 3, 2, generator=generator, dtype=torch.float64),
            "b": torch.randn(len(rows), 2, generator=generator, dtype=torch.float64),
        }
        for row, position in enumerate(rows):
            key = (poison or {}).get(position)
            if key is not None:
                tensors[key][row].view(-1)[0] = float("nan")
        stacks.append((StateStack(tensors), list(rows)))
    columns = MetricColumns(5)
    columns.put("fit_loss", range(5), torch.rand(5, generator=generator).tolist())
    columns.put("fit_accuracy", [0, 3, 4], torch.rand(3, generator=generator).tolist())
    columns.put("optimizer_steps", range(5), [3.0, 3.0, 2.0, 1.0, 3.0])
    metrics, reported = columns.tensors()
    return StackedFitResults(
        round_id=1,
        client_ids=[f"client_{index}" for index in range(5)],
        num_examples=torch.tensor([7, 3, 11, 5, 2]),
        states=stacks,
        metrics=metrics,
        reported=reported,
        payload={"model_state_scope": "full", "model_state_metadata": dict(METADATA)},
    )


def _server(cls: type[FedAvgServer] = FedAvgServer, weighting: str = "examples") -> FedAvgServer:
    server = cls(participation_rate=1.0, seed=0, aggregation_weighting=weighting)
    server._model_state = {
        "w": torch.zeros(3, 2, dtype=torch.float64),
        "b": torch.zeros(2, dtype=torch.float64),
    }
    server._model_state_scope = "full"
    server._model_state_metadata = dict(METADATA)
    return server


class _OwnWeight(FedAvgServer):
    def _result_weight(self, result: FitResult) -> float:
        return float(result.num_examples) + 1.0


def _fold(server: FedAvgServer, results: Any) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    round_info = RoundInfo(round_id=1, total_rounds=1)
    payload = server.aggregate_stream(round_info, results)
    return payload["model_state"], dict(round_info.metrics)


class WhatTheServerIsHandedTest(unittest.TestCase):
    def test_each_client_in_request_order(self) -> None:
        stacked = _stacked(torch.Generator().manual_seed(0))
        results = list(stacked.results())
        self.assertEqual([r.client_id for r in results], stacked.client_ids)
        self.assertEqual([r.num_examples for r in results], [7, 3, 11, 5, 2])
        self.assertEqual(list(results[0].metrics), ["fit_loss", "fit_accuracy", "optimizer_steps"])
        self.assertEqual(list(results[1].metrics), ["fit_loss", "optimizer_steps"])
        for stack, positions in stacked.states:
            for row, position in enumerate(positions):
                self.assertTrue(
                    torch.equal(
                        results[position].payload["model_state"]["w"], stack.tensors["w"][row]
                    )
                )
        self.assertEqual(results[2].payload["model_state_metadata"], METADATA)
        self.assertIsNot(
            results[2].payload["model_state_metadata"], results[3].payload["model_state_metadata"]
        )

    def test_consumed_once(self) -> None:
        handed = StackedResults([_stacked(torch.Generator().manual_seed(0))])
        list(handed)
        with self.assertRaisesRegex(RuntimeError, "consumed once"):
            list(handed)


class TheStackedFoldTest(unittest.TestCase):
    def test_metrics_to_the_bit_and_the_model_to_tolerance(self) -> None:
        for weighting in ("examples", "uniform"):
            with self.subTest(weighting=weighting):
                make = lambda: _stacked(torch.Generator().manual_seed(1))  # noqa: E731
                model, metrics = _fold(
                    _server(weighting=weighting), StackedResults([make(), make()])
                )
                model_one, metrics_one = _fold(
                    _server(weighting=weighting), [*make().results(), *make().results()]
                )
                self.assertEqual(list(metrics), list(metrics_one))
                self.assertEqual(metrics, metrics_one)
                for key, tensor in model_one.items():
                    scale = float(tensor.abs().max())
                    self.assertLessEqual(
                        float((model[key] - tensor).abs().max()) / scale, TOLERANCE
                    )

    def test_one_stack_is_folded_as_its_rows_are(self) -> None:
        generator = torch.Generator().manual_seed(2)
        tensors = {
            "w": torch.randn(4, 3, 2, generator=generator, dtype=torch.float64),
            "b": torch.randn(4, 2, generator=generator, dtype=torch.float64),
        }

        def make() -> StackedFitResults:
            columns = MetricColumns(4)
            columns.put("fit_loss", range(4), [0.5, 0.25, 0.125, 1.0])
            metrics, _ = columns.tensors()
            return StackedFitResults(
                round_id=1,
                client_ids=["a", "b", "c", "d"],
                num_examples=torch.tensor([1, 2, 3, 4]),
                states=[(StateStack(tensors), [0, 1, 2, 3])],
                metrics=metrics,
                payload={"model_state_scope": "full", "model_state_metadata": dict(METADATA)},
            )

        model, metrics = _fold(_server(), StackedResults([make()]))
        model_one, metrics_one = _fold(_server(), list(make().results()))
        self.assertEqual(metrics, metrics_one)
        for key in model:
            self.assertTrue(torch.equal(model[key], model_one[key]), key)

    def test_a_server_with_its_own_weight_is_handed_each_result(self) -> None:
        make = lambda: _stacked(torch.Generator().manual_seed(3))  # noqa: E731
        with mock.patch.object(FedAvgServer, "_accumulate_stacks") as whole:
            model, metrics = _fold(_server(_OwnWeight), StackedResults([make()]))
        whole.assert_not_called()
        model_one, metrics_one = _fold(_server(_OwnWeight), list(make().results()))
        self.assertEqual(metrics, metrics_one)
        for key in model:
            self.assertTrue(torch.equal(model[key], model_one[key]), key)

    def test_metric_columns_add_as_add_adds(self) -> None:
        generator = torch.Generator().manual_seed(4)
        # Across 17 orders of magnitude, where the order of a sum shows.
        values = [
            (u - 0.5) * 10.0 ** (index % 17)
            for index, u in enumerate(
                torch.rand(50, generator=generator, dtype=torch.float64).tolist()
            )
        ]
        counts = torch.randint(0, 9, (50,), generator=generator).tolist()
        mask = [index % 3 != 0 for index in range(50)]
        one, columns = WeightedMetricAccumulator(), WeightedMetricAccumulator()
        for value, count, kept in zip(values, counts, mask, strict=True):
            one.add({"a": value, **({"b": -value} if kept else {})}, count)
        columns.add_columns({"a": values, "b": [-v for v in values]}, counts, {"b": mask})
        self.assertEqual(list(one.result().items()), list(columns.result().items()))


class TheRefusalsAreThePerClientOnesTest(unittest.TestCase):
    def _both(self, **stacked: Any) -> tuple[str, str]:
        messages = []
        for handed in (
            lambda s: StackedResults([s]),
            lambda s: list(s.results()),
        ):
            with self.assertRaises((NonFiniteStateError, ValueError)) as caught:
                _fold(_server(), handed(_stacked(torch.Generator().manual_seed(5), **stacked)))
            messages.append(str(caught.exception))
        return messages[0], messages[1]

    def test_the_first_non_finite_client_in_request_order(self) -> None:
        # Client 4 comes first in its stack, client 1 first in the round.
        stacked, one = self._both(poison={4: "w", 1: "b"})
        self.assertEqual(stacked, one)
        self.assertIn("'client_1'", stacked)

    def test_a_non_finite_weight(self) -> None:
        messages = []
        for handed in (lambda s: StackedResults([s]), lambda s: list(s.results())):
            with (
                mock.patch.object(
                    StackedFitResults, "counts", return_value=[7, 3, float("nan"), 5, 2]
                ),
                self.assertRaises(NonFiniteStateError) as caught,
            ):
                _fold(_server(), handed(_stacked(torch.Generator().manual_seed(6))))
            messages.append(str(caught.exception))
        self.assertEqual(messages[0], messages[1])

    def test_foreign_metadata_names_the_first_client(self) -> None:
        messages = []
        for handed in (lambda s: StackedResults([s]), lambda s: list(s.results())):
            stacked = _stacked(torch.Generator().manual_seed(7))
            stacked.payload["model_state_scope"] = "adapter"
            with self.assertRaises(ValueError) as caught:
                _fold(_server(), handed(stacked))
            messages.append(str(caught.exception))
        self.assertEqual(messages[0], messages[1])
        self.assertIn("'client_0'", messages[0])


def _per_client() -> Any:
    """The batched executor with its stacked path off: every round one result at a time."""

    return mock.patch.object(BatchedExecutor, "fit_stacked", lambda self, *args: None)


class TheStackedPathIsTakenTest(ExecutorRuns):
    def test_by_the_rules_that_build_their_results_stacked(self) -> None:
        taken: dict[str, set[bool]] = {}
        real = BatchedExecutor.fit_stacked
        arms = [("fedavg", example_config("fed-lasso"))]
        arms += [(label, rule_config(client)) for label, client in rule_arms()]
        for label, config in arms:

            def spy(self: BatchedExecutor, *args: Any, label: str = label) -> Any:
                stacks = real(self, *args)
                taken.setdefault(label, set()).add(stacks is not None)
                return stacks

            with mock.patch.object(BatchedExecutor, "fit_stacked", spy):
                self.run_config(config, "batched")
        self.assertEqual(
            taken,
            {label: {not label.startswith(("fedprox", "scaffold"))} for label, _ in arms},
        )


class TheStackedPathIsThePerClientPathTest(ExecutorRuns):
    def _pair(self, config: dict[str, Any], data: Any = None, **performance: Any) -> Any:
        with data() if data else nullcontext():
            stacked = self.run_config(config, "batched", **performance)
            with _per_client():
                per_client = self.run_config(config, "batched", **performance)
        return stacked, per_client

    def test_one_bucket_is_bit_identical(self) -> None:
        arms = [
            (label, with_client(example_config("fed-lasso"), **client))
            for label, client in fedavg_modes()
        ]
        arms += [
            (label, rule_config(client))
            for label, client in rule_arms()
            if not label.startswith(("fedprox", "scaffold"))
        ]
        for label, config in arms:
            with self.subTest(arm=label):
                self.assertAgree(*self._pair(config), exact=True)

    def test_the_classification_task_folded_is_bit_identical(self) -> None:
        for label in ("sequential_epoch", "full_gradient"):
            with self.subTest(mode=label):
                config = classification_rule_config({"update_mode": label})
                self.assertAgree(*self._pair(config, float64_classification), exact=True)

    def test_ragged_clients(self) -> None:
        for label, client in fedavg_modes():
            with self.subTest(mode=label):
                config = with_client(example_config("fed-lasso"), **client)
                stacked, per_client = self._pair(config, ragged_clients)
                first = [
                    [
                        row
                        for row in _rows(path / "client_update_metrics.csv")
                        if row["round_id"] == "1"
                    ]
                    for path in (stacked, per_client)
                ]
                self.assertTrue(first[0])
                timing = TIMING
                self.assertEqual(
                    [{k: v for k, v in row.items() if k not in timing} for row in first[0]],
                    [{k: v for k, v in row.items() if k not in timing} for row in first[1]],
                )
                self.assertAgree(stacked, per_client)

    def test_an_observer_of_one_result_at_a_time_is_handed_each(self) -> None:
        config = example_config("fed-lasso")
        stacked = self.run_config(config, "batched")
        with mock.patch.object(loop._RoundFitObserver, "fitted_stack", None):
            each = self.run_config(config, "batched")
        self.assertAgree(stacked, each, exact=True)


if __name__ == "__main__":
    unittest.main()
