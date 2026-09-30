"""What the batched executor keeps of fedbrew's guarantees, beyond its numbers.

``tests/test_batched_executor_tolerance.py`` holds its results to the
sequential executor's. What is pinned here (chapter 11 §9):

- selection: ``runtime.performance.executor`` takes ``sequential`` or
  ``batched`` and ``executor_chunk_bytes`` a positive integer or ``auto``,
  half the device's free memory at the start, recorded; a batchable
  run records ``used: batched`` and its largest chunk in run.json, and one
  that is not runs sequentially, says why in the plan header, and records the
  reason;
- client isolation: permuting the sampled clients permutes the results, and a
  NaN in one client's data leaves every other client's result bit-identical;
- refusals: a client with no training batches is refused with the sequential
  executor's message before any client runs, and a non-finite client state is
  refused naming the client and tensor the sequential run names;
- randomness: neither executor draws from the process-wide generator on a
  seeded, batchable run;
- memory: a chunk holds as many clients as its budget allows, and no earlier
  chunk's stack is alive when the next is built.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
import weakref
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch
import yaml

from fedbrew.clients.batched_update import plan_round
from fedbrew.core import batched_executor
from fedbrew.core.batched_executor import BatchedExecutor
from fedbrew.core.config import load_config, validate_config
from fedbrew.core.factory import build_components
from fedbrew.core.logging import _executor_rows
from fedbrew.core.loop import SequentialExecutor
from fedbrew.core.protocol import FitRequest, FitResult
from fedbrew.core.refusal import RunRefused
from fedbrew.core.torch_utils import StateStack
from tests.test_batched_executor_tolerance import ExecutorRuns, example_config

EXECUTOR_KEYS = ("executor", "executor_chunk_bytes")


class _Observer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def fitted(self, result: FitResult, seconds: float, done: int, total: int) -> None:
        self.calls.append((result.client_id, done, total))


def _components(config: dict[str, Any], root: Path) -> Any:
    config = copy.deepcopy(config)
    config["experiment"]["output_dir"] = str(root / "unused")
    path = root / "components.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return build_components(load_config(path))


def _requests(components: Any, clients: list[str] | None = None) -> list[FitRequest]:
    payload = components.server.initialize()
    ids = clients if clients is not None else list(components.clients)
    return [FitRequest(round_id=1, client_id=cid, payload=payload, total_rounds=4) for cid in ids]


def _fit(executor: Any, components: Any, requests: list[FitRequest]) -> dict[str, FitResult]:
    results = executor.fit(components.clients, requests, _Observer())
    return {result.client_id: result for result in results}


@pytest.mark.fast
class TheKeysAreCheckedTest(unittest.TestCase):
    def _validate(self, **performance: Any) -> None:
        config = load_config(Path(__file__).resolve().parent.parent / "configs/dev/smoke.yaml")
        config.runtime.extra.setdefault("performance", {}).update(performance)
        validate_config(config)

    def test_the_two_executors_and_a_positive_budget_are_accepted(self) -> None:
        self._validate(executor="sequential")
        self._validate(executor="batched", executor_chunk_bytes=1 << 20)
        self._validate(executor="batched", executor_chunk_bytes="auto")
        self._validate(executor="batched", gradient_form="summed")
        self._validate(executor="batched", gradient_form="vmap_grad")
        self._validate(executor="batched", gradient_form="closed_form")

    def test_anything_else_is_refused(self) -> None:
        for performance in (
            {"executor": "vmap"},
            {"executor": True},
            {"executor_chunk_bytes": 0},
            {"executor_chunk_bytes": -1},
            {"executor_chunk_bytes": True},
            {"executor_chunk_bytes": 1.5},
            {"executor_chunk_bytes": "Auto"},
            {"executor_chunk_bytes": "half"},
            {"executor": "batched", "gradient_form": "backward"},
            {"executor": "sequential", "gradient_form": "summed"},
        ):
            with self.subTest(**performance), self.assertRaises(RunRefused):
                self._validate(**performance)


class SelectionIsRecordedTest(ExecutorRuns):
    def _record(self, output: Path) -> dict[str, Any]:
        return json.loads((output / "run.json").read_text())["reproducibility"]["executor"]

    def test_a_batchable_run_records_its_largest_chunk(self) -> None:
        output = self.run_config(example_config("fed-lasso"), "batched")
        self.assertEqual(
            self._record(output),
            {
                "used": "batched",
                "largest_chunk_clients": 8,
                # Planned from the roster in this process: a CPU run starts no
                # planner workers (round_planner).
                "planner": {"used": "on", "workers": 0, "waited_sec": 0.0},
                # Its rounds held on the device (fedbrew/core/resident.py).
                "rounds": {"used": "resident"},
            },
        )

    def test_a_gradient_form_a_config_sets_is_recorded_and_taken(self) -> None:
        """fed-lasso declares ``summed``; asked for ``vmap_grad``, its stacks take that."""

        from fedbrew.core import batched_executor

        taken: list[bool] = []
        real = batched_executor._Bucket._run_summed

        def summed(self: Any, *args: Any) -> Any:
            taken.append(True)
            return real(self, *args)

        config = example_config("fed-lasso")
        with mock.patch.object(batched_executor._Bucket, "_run_summed", summed):
            output = self.run_config(config, "batched", gradient_form="vmap_grad")
            self.assertEqual(self._record(output)["gradient_form"], "vmap_grad")
            self.assertEqual(taken, [])
            self.assertNotIn("gradient_form", self._record(self.run_config(config, "batched")))
            self.assertTrue(taken)

    def test_closed_form_is_refused_for_a_task_without_one(self) -> None:
        from tests.test_batched_executor_tolerance import classification_config

        with self.assertRaisesRegex(RunRefused, "gives no closed-form gradient"):
            self.run_config(classification_config(), "batched", gradient_form="closed_form")

    def test_the_sequential_default_is_recorded(self) -> None:
        output = self.run_config(example_config("fed-lasso"), "sequential")
        self.assertEqual(self._record(output), {"used": "sequential"})

    def test_an_unbatchable_run_falls_back_and_says_why(self) -> None:
        config = example_config("fed-lasso", "fedlalr")
        batched, sequential = self.both(config)
        record = self._record(batched)
        self.assertEqual(record["used"], "sequential")
        self.assertIn("TorchFedLALRClient declares no batched update", record["fallback"])
        # The fallback is the sequential executor, so it computes its numbers.
        self.assertAgree(batched, sequential, exact=True)
        rows = _executor_rows(record)
        self.assertEqual(len(rows), 1)
        self.assertIn("batched falls back", rows[0].value)
        self.assertIsNotNone(rows[0].tone)

    def test_dropout_falls_back(self) -> None:
        from tests.test_batched_executor_tolerance import classification_config

        config = classification_config()
        config["model"]["dropout"] = 0.3
        record = self._record(self.run_config(config, "batched"))
        self.assertEqual(record["used"], "sequential")
        self.assertIn("dropout at p = 0.3", record["fallback"])

    def test_the_plan_header_names_a_batched_run(self) -> None:
        self.assertEqual(_executor_rows({"used": "batched"})[0].value, "batched")
        self.assertEqual(_executor_rows({"used": "sequential"}), [])


class ClientsAreIsolatedTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)
        self.config = example_config("fed-lasso")

    def test_permuting_the_clients_permutes_the_results(self) -> None:
        components = _components(self.config, self.root)
        forward = _fit(BatchedExecutor(), components, _requests(components))
        order = list(reversed(list(components.clients)))
        backward = _fit(BatchedExecutor(), components, _requests(components, order))
        self.assertEqual(set(forward), set(backward))
        for client, result in forward.items():
            for key, value in result.payload["model_state"].items():
                self.assertTrue(torch.equal(value, backward[client].payload["model_state"][key]))
            self.assertEqual(result.metrics, backward[client].metrics)
            self.assertEqual(result.num_examples, backward[client].num_examples)

    def test_a_nan_in_one_client_leaves_the_others_bit_identical(self) -> None:
        components = _components(self.config, self.root)
        clean = _fit(BatchedExecutor(), components, _requests(components))
        poisoned = _components(self.config, self.root)
        client = poisoned.clients["client_3"]
        train = dict(client.client_data["train"])
        train["y"] = train["y"].clone()
        train["y"][0] = float("nan")
        client.client_data = {**client.client_data, "train": train}
        dirty = _fit(BatchedExecutor(), poisoned, _requests(poisoned))
        self.assertTrue(torch.isnan(dirty["client_3"].payload["model_state"]["x"]).any())
        for name, result in clean.items():
            if name == "client_3":
                continue
            self.assertTrue(
                torch.equal(
                    result.payload["model_state"]["x"], dirty[name].payload["model_state"]["x"]
                ),
                name,
            )
            self.assertEqual(result.metrics, dirty[name].metrics, name)


class RefusalsAreTheSequentialOnesTest(ExecutorRuns):
    def test_no_training_batches_is_refused_before_any_client_runs(self) -> None:
        messages = []
        for executor in (SequentialExecutor(), BatchedExecutor()):
            components = _components(example_config("fed-lasso"), self.root)
            client = components.clients["client_5"]
            train = {key: value[:0] for key, value in client.client_data["train"].items()}
            client.client_data = {**client.client_data, "train": train}
            observer = _Observer()
            with self.assertRaises(ValueError) as caught:
                list(executor.fit(components.clients, _requests(components), observer))
            messages.append(str(caught.exception))
            if isinstance(executor, BatchedExecutor):
                self.assertEqual(observer.calls, [])
        self.assertEqual(messages[0], messages[1])
        self.assertIn("client_5", messages[0])

    def test_a_non_finite_client_is_named_as_the_sequential_run_names_it(self) -> None:
        from fedbrew.data.manifest_dataset import ManifestFederatedDataset

        real = ManifestFederatedDataset.get_client_data

        def poisoned(self: Any, client_id: str) -> dict[str, Any]:
            shard = real(self, client_id)
            if client_id != "client_6":
                return shard
            train = {**shard["train"], "y": shard["train"]["y"] * float("inf")}
            return {**shard, "train": train}

        batched, sequential = self.both(
            example_config("fed-lasso"),
            data=lambda: mock.patch.object(ManifestFederatedDataset, "get_client_data", poisoned),
        )
        terminations = [
            json.loads((output / "run.json").read_text())["termination"]
            for output in (batched, sequential)
        ]
        self.assertEqual(terminations[0], terminations[1])
        self.assertEqual(terminations[0]["detector"], "non_finite_client_state")
        self.assertIn("client_6", terminations[0]["reason"])


class NothingDrawsFromTheGlobalGeneratorTest(unittest.TestCase):
    def test_either_executor_leaves_the_generator_where_it_was(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            components = _components(example_config("fed-lasso"), Path(directory))
            for executor in (SequentialExecutor(), BatchedExecutor()):
                before = torch.get_rng_state()
                _fit(executor, components, _requests(components))
                self.assertTrue(torch.equal(before, torch.get_rng_state()), type(executor).__name__)


class OneChunkAtATimeTest(ExecutorRuns):
    def _stacks(self, chunk_clients: int) -> list[int]:
        """Stack sizes built during one round at a budget of ``chunk_clients`` clients."""

        components = _components(example_config("fed-lasso"), self.root)
        requests = _requests(components)
        sizes: list[int] = []
        alive: list[weakref.ref[StateStack]] = []
        real_init = StateStack.__init__

        def recording(stack: StateStack, tensors: Any) -> None:
            real_init(stack, tensors)
            # At most the previous chunk's: its last result is still the
            # consumer's loop variable, as the previous client's is under the
            # sequential executor.
            self.assertLessEqual(len([ref for ref in alive if ref() is not None]), 1)
            alive.append(weakref.ref(stack))
            sizes.append(stack.size)

        template = components.task.build_model(components.clients["client_0"].model_config)
        plan = components.clients["client_0"].batched_plan(requests[0], template)
        train, _ = plan_round([plan], 1)
        per_client = _client_cost(components, template, plan, train)
        with mock.patch.object(StateStack, "__init__", recording):
            results = BatchedExecutor(chunk_bytes=per_client * chunk_clients).fit(
                components.clients, requests, _Observer()
            )
            for _ in results:
                pass  # each result dropped at the next, as the aggregator drops it
        return sizes

    def test_the_budget_decides_the_chunk(self) -> None:
        for chunk_clients in (1, 4, 8):
            with self.subTest(chunk_clients=chunk_clients):
                sizes = self._stacks(chunk_clients)
                self.assertEqual(sum(sizes), 8)
                self.assertEqual(max(sizes), chunk_clients)

    def test_a_chunked_run_agrees_with_the_sequential_one(self) -> None:
        components = _components(example_config("fed-lasso"), self.root)
        template = components.task.build_model(components.clients["client_0"].model_config)
        plan = components.clients["client_0"].batched_plan(_requests(components)[0], template)
        train, _ = plan_round([plan], 1)
        budget = _client_cost(components, template, plan, train) * 3
        self.assertAgree(*self.both(example_config("fed-lasso"), executor_chunk_bytes=budget))


class AnAutoBudgetIsHalfTheFreeMemoryTest(ExecutorRuns):
    def _record(self, output: Path) -> dict[str, Any]:
        return json.loads((output / "run.json").read_text())["reproducibility"]["executor"]

    def _cost(self) -> int:
        components = _components(example_config("fed-lasso"), self.root)
        template = components.task.build_model(components.clients["client_0"].model_config)
        plan = components.clients["client_0"].batched_plan(_requests(components)[0], template)
        train, _ = plan_round([plan], 1)
        return _client_cost(components, template, plan, train)

    def test_the_budget_is_measured_and_recorded(self) -> None:
        free = self._cost() * 3 * 2 + 1
        with mock.patch.object(batched_executor, "free_memory", return_value=free):
            auto = self.run_config(
                example_config("fed-lasso"), "batched", executor_chunk_bytes="auto"
            )
        record = self._record(auto)
        self.assertEqual(
            record["chunk_bytes"],
            {"asked": "auto", "used": free // 2, "free": free, "fraction": 0.5},
        )
        self.assertEqual(record["largest_chunk_clients"], 3)
        self.assertIn("auto: 0.5 of", _executor_rows(record)[1].value)

    def test_a_budget_as_large_as_the_default_runs_as_the_default(self) -> None:
        auto = self.run_config(example_config("fed-lasso"), "batched", executor_chunk_bytes="auto")
        self.assertAgree(auto, self.run_config(example_config("fed-lasso"), "batched"), exact=True)
        self.assertGreater(self._record(auto)["chunk_bytes"]["free"], 0)

    def test_unreadable_free_memory_takes_the_default_and_says_so(self) -> None:
        with mock.patch.object(batched_executor, "free_memory", return_value=None):
            auto = self.run_config(
                example_config("fed-lasso"), "batched", executor_chunk_bytes="auto"
            )
        record = self._record(auto)["chunk_bytes"]
        self.assertEqual(record["used"], batched_executor.DEFAULT_EXECUTOR_CHUNK_BYTES)
        self.assertIn("could not be read", record["fallback"])

    def test_a_cgroup_limit_is_headroom_and_no_limit_is_none(self) -> None:
        for limit, usage, headroom in (
            ("1000", "400", 600),
            ("max", "400", None),
            (str(1 << 63), "400", None),
            ("300", "400", 0),
        ):
            with self.subTest(limit=limit):
                (self.root / "limit").write_text(limit + "\n")
                (self.root / "usage").write_text(usage + "\n")
                self.assertEqual(
                    batched_executor._headroom(str(self.root / "limit"), str(self.root / "usage")),
                    headroom,
                )


def _client_cost(components: Any, template: Any, plan: Any, train: Any) -> int:
    """One client's estimate, as ``BatchedExecutor._chunks`` makes it."""

    parameter_bytes = sum(p.numel() * p.element_size() for p in template.parameters())
    row_bytes = sum(
        t[:1].numel() * t.element_size() for t in components.task.split_rows(plan.train_data)
    )
    longest = int(train.lengths[plan.slot].max())
    slots = batched_executor._WORKING_SLOTS + plan.program.optimizer.state_slots
    return parameter_bytes * slots + row_bytes * (plan.eval_rows + 2 * longest)


if __name__ == "__main__":
    unittest.main()


class StackedMetricsTest(unittest.TestCase):
    """The classification task's stacked post-fit metrics are compute_metrics, split by split."""

    def test_each_split_gets_its_own_compute_metrics(self) -> None:
        from fedbrew.tasks.classification.torch_classification import TorchClassificationTask

        task = TorchClassificationTask({"name": "mlp"})
        generator = torch.Generator().manual_seed(3)
        counts = [4, 1, 3, 0, 4, 2]
        positions = max(counts)
        outputs = []
        for position in range(positions):
            total = torch.randint(0, 5, (len(counts),), generator=generator).to(torch.float64)
            past = torch.tensor([position >= count for count in counts])
            if position == 0:
                total[1] = 0.0  # a split whose only batch holds no example
            outputs.append(
                {
                    # A split's positions past its count are padding: an
                    # empty batch's mean loss, NaN.
                    "loss": torch.where(
                        past,
                        float("nan"),
                        torch.rand(len(counts), generator=generator, dtype=torch.float32),
                    ),
                    "correct": torch.minimum(
                        torch.randint(0, 5, (len(counts),), generator=generator),
                        total.long(),
                    ),
                    "total": total,
                }
            )
        folded, examples = task.stacked_metrics(outputs, counts)
        for split, count in enumerate(counts):
            records = [
                {key: float(outputs[p][key][split]) for key in ("loss", "correct", "total")}
                for p in range(count)
            ]
            expected = task.compute_metrics(records)
            for name, value in expected.items():
                self.assertAlmostEqual(
                    float(folded[name][split]), value, places=12, msg=(split, name)
                )
            self.assertEqual(int(examples[split]), int(sum(r["total"] for r in records)))
