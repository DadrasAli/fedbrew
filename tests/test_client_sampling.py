"""``client.sampling: with_replacement``: every training pass one batch drawn with replacement.

``fedbrew/clients/sampling.py`` makes any task's training loader the iid
oracle -- one batch of ``batch_size`` rows drawn uniformly with replacement per
pass, from the seed the task's loader would have had -- and declares the same
draws to the batched planner. Held here:

- the draws: uniform over the rows, with replacement, independent across
  passes and seeds, and a loader rebuilt from a seed draws the same batches;
  a batch is the task's rows at the drawn indices;
- the planner's batches are the loader's draws, so the three paths train on
  the same rows: a run sequential, batched per round and resident agree --
  batched against sequential within the executor tolerance, resident against
  per round bit for bit -- under ``single_batch``, ``sequential_epoch`` and
  SCAFFOLD's own loop;
- off is the default: a config stating ``without_replacement`` writes what one
  without the key writes, bit for bit;
- refused beside ``full_gradient`` and ``drop_last: true``, and for a task that
  gives no rows.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import pytest
import torch

from fedbrew.clients.batch_orders import LocalLoop, plan_orders
from fedbrew.clients.sampling import (
    ReplacementBatches,
    ReplacementLoader,
    replacement_order,
)
from fedbrew.clients.torch_sgd_client import TorchSGDClient
from fedbrew.core.refusal import RunRefused
from tests.test_batch_orders import planned
from tests.test_batched_executor_tolerance import (
    classification_rule_config,
    float64_classification,
)
from tests.test_resident_round import ResidentRuns

FEDAVG = {"update_rule": "fedavg", "frozen_gradient_weighting": "examples"}
WITH = {"sampling": "with_replacement"}


@pytest.mark.fast
class TheDrawsTest(unittest.TestCase):
    def test_uniform_with_replacement(self) -> None:
        rows, size = 5, 20_000
        (batch,) = list(ReplacementBatches(rows, size, seed=3))
        self.assertEqual(batch.shape, (size,))
        counts = torch.bincount(batch, minlength=rows)
        self.assertEqual(len(counts), rows)
        # Each count is Binomial(20000, 1/5): 4000 +- 57; five sigma apart.
        self.assertLess(float((counts - size / rows).abs().max()), 5 * (size * 0.2 * 0.8) ** 0.5)
        # A batch larger than the split, and one of a single row, both draw.
        self.assertEqual(len(list(ReplacementBatches(1, 7, seed=0))[0]), 7)

    def test_passes_and_seeds_are_independent_and_a_seed_repeats(self) -> None:
        loader = ReplacementBatches(32, 16, seed=11)
        first, second = (list(loader)[0] for _ in range(2))
        self.assertFalse(torch.equal(first, second))
        again = ReplacementBatches(32, 16, seed=11)
        self.assertEqual(
            [list(again)[0].tolist() for _ in range(2)], [first.tolist(), second.tolist()]
        )
        self.assertFalse(torch.equal(list(ReplacementBatches(32, 16, seed=12))[0], first))
        self.assertEqual(len(loader), 1)

    def test_a_batch_is_the_rows_at_the_draws(self) -> None:
        features = torch.arange(24.0).reshape(12, 2)
        targets = torch.arange(12)
        (index,) = list(ReplacementBatches(12, 9, seed=4))
        (batch,) = list(ReplacementLoader((features, targets), 9, seed=4))
        self.assertTrue(torch.equal(batch[0], features[index]))
        self.assertTrue(torch.equal(batch[1], targets[index]))

    def test_the_planner_draws_the_loaders_batches(self) -> None:
        cases = (
            (20, 3, 5, LocalLoop(epochs=6, single_batch=True)),
            (7, 16, 2**40, LocalLoop(epochs=3)),
        )
        orders = [replacement_order(rows, size, seed) for rows, size, seed, _ in cases]
        result = plan_orders(orders, [loop for *_, loop in cases])
        for client, (rows, size, seed, loop) in enumerate(cases):
            loader = ReplacementBatches(rows, size, seed)
            expected = [list(loader)[0].tolist() for _ in range(loop.epochs)]
            with self.subTest(client=client):
                self.assertEqual([update[0] for update in planned(result, client)], expected)


class _NoRows:
    """A task that gives no rows: its loader is its own business."""


@pytest.mark.fast
class RefusedTest(unittest.TestCase):
    def test_beside_full_gradient_and_drop_last(self) -> None:
        import tempfile

        import yaml

        from fedbrew.core.config import load_config
        from tests.test_batched_executor_tolerance import set_performance

        for extra, words in (
            ({"update_mode": "full_gradient"}, "full_gradient"),
            ({"update_mode": "single_batch", "drop_last": True}, "drop_last"),
            ({"update_mode": "single_batch", "sampling": "sometimes"}, "must be one of"),
        ):
            config = classification_rule_config({**FEDAVG, **WITH, **extra})
            set_performance(config, executor="sequential")
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.yaml"
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                with self.subTest(extra=extra), self.assertRaisesRegex(RunRefused, words):
                    load_config(path)

    def test_a_task_without_rows(self) -> None:
        with self.assertRaisesRegex(RunRefused, "gives none"):
            TorchSGDClient(
                client_id="c",
                task=_NoRows(),  # type: ignore[arg-type]
                model_config={},
                client_data={"train": {}},
                local_iterations=1,
                batch_size=2,
                learning_rate=0.1,
                train_sampling="with_replacement",
            )


class TheThreePathsTrainOnTheSameRowsTest(ResidentRuns):
    """In float64, so the batched path is held to the executor's tolerance."""

    def _check(self, client: dict[str, Any]) -> None:
        config = classification_rule_config(client)
        config["runtime"]["checkpointing"].update(save_every_round=True)
        with float64_classification():
            sequential = self.run_config(config, "sequential")
        held, per_round = self.pair(config, float64_classification)
        self.assertSameRun(held, per_round)
        self.assertAgree(per_round, sequential)

    def test_fedavg_single_batch_and_sequential_epoch(self) -> None:
        for mode in ("single_batch", "sequential_epoch"):
            with self.subTest(update_mode=mode):
                self._check({**FEDAVG, **WITH, "update_mode": mode})

    def test_scaffold(self) -> None:
        config = classification_rule_config({"update_rule": "scaffold", **WITH})
        config["runtime"]["checkpointing"].update(save_every_round=True)
        self.assertAgree(*self.both(config, data=float64_classification))


class OffIsTheDefaultTest(ResidentRuns):
    def test_stating_without_replacement_changes_nothing(self) -> None:
        client = {**FEDAVG, "update_mode": "sequential_epoch"}
        for executor in ("sequential", "batched"):
            with self.subTest(executor=executor):
                unset = self.run_config(classification_rule_config(client), executor)
                stated = self.run_config(
                    classification_rule_config({**client, "sampling": "without_replacement"}),
                    executor,
                )
                self.assertAgree(unset, stated, exact=True)


if __name__ == "__main__":
    unittest.main()
