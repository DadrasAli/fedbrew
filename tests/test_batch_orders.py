"""Every client's batch order, computed together, is the order its loader yields.

``fedbrew/clients/batch_orders.py`` plans a round's batch orders for every
client together from each task's ``LoaderOrder`` declaration, drawing each
permutation with the calls the loader makes rather than through the loader. So
each order here is compared with the loader's own: the task's ``build_dataloader`` built
on a split whose rows are their own numbers and iterated as each update rule's
loop iterates it (``sgd_mode_updates``, ``own_loop_updates``), for every update
mode, shuffled and not, with and without ``drop_last`` and
``max_local_steps``, over splits of different lengths planned in one call.
"""

from __future__ import annotations

import itertools
import unittest
from pathlib import Path
from typing import Any

import torch

from fedbrew.clients.batch_orders import LocalLoop, plan_orders
from fedbrew.clients.batched_update import own_loop_updates, sgd_mode_updates
from fedbrew.core import extensions
from fedbrew.core.seeding import dataloader_seed, dataloader_seeds
from fedbrew.tasks.base import LoaderOrder
from fedbrew.tasks.classification.torch_classification import TorchClassificationTask

REPO_ROOT = Path(__file__).resolve().parent.parent

#: (label, update mode, local iterations, max_local_steps, which loop).
LOOPS = [
    *(("single_batch", k, None, "sgd") for k in (1, 3, 8)),
    *(("sequential_epoch", k, None, "sgd") for k in (1, 3)),
    *(("frozen_batch_gradients", k, None, "sgd") for k in (1, 2)),
    *(("full_gradient", k, None, "sgd") for k in (1, 3)),
    *(("sequential_epoch", k, cap, "own") for k in (1, 3) for cap in (None, 1, 4, 7)),
    *(("full_gradient", k, cap, "own") for k in (1, 3) for cap in (None, 2)),
]


def local_loop(mode: str, iterations: int, cap: int | None, loop: str) -> LocalLoop:
    """What a rule's loop takes, as ``batched_plan`` declares it."""

    per_update = "epoch" if mode in {"frozen_batch_gradients", "full_gradient"} else "batch"
    return LocalLoop(
        epochs=iterations,
        per_update=per_update,
        single_batch=mode == "single_batch",
        max_updates=cap if loop == "own" else None,
    )


def reference(loader: Any, mode: str, iterations: int, cap: int | None, loop: str) -> Any:
    """The rule's own loop over the loader: the batches of each update."""

    if loop == "sgd":
        return sgd_mode_updates(
            loader, local_iterations=iterations, update_mode=mode, client_id="c"
        )
    return own_loop_updates(
        loader,
        local_iterations=iterations,
        update_mode=mode,
        max_local_steps=cap,
        client_id="c",
    )


def planned(orders: Any, client: int) -> list[list[list[int]]]:
    """Client ``client``'s updates, each a list of batches of row numbers."""

    updates, step = [], 0
    for count in orders.structure[client]:
        batches = []
        for _ in range(count):
            length = int(orders.lengths[client, step])
            batches.append(orders.indices[client, step, :length].tolist())
            step += 1
        updates.append(batches)
    assert step == int(orders.steps[client])
    return updates


class _Case:
    """A task, a way to build a numbered split of n rows, and a way to read a batch's rows."""

    def __init__(self, label: str, task: Any, split: Any, numbers: Any) -> None:
        self.label, self.task, self.split, self.numbers = label, task, split, numbers


def classification_cases() -> list[_Case]:
    def split(rows: int) -> dict[str, Any]:
        return {"X": torch.arange(rows, dtype=torch.float32).unsqueeze(1), "y": torch.zeros(rows)}

    def numbers(batch: Any) -> list[int]:
        return [int(value) for value in batch[0][:, 0].tolist()]

    return [
        _Case(
            f"classification/fast_batching={fast}",
            TorchClassificationTask({"name": "mlp"}, batch_size=4, fast_batching=fast),
            split,
            numbers,
        )
        for fast in (True, False)
    ]


def example_cases() -> list[_Case]:
    cases = []
    for name, dataset in (
        ("fed-lasso", "fed_lasso"),
        ("simplex-lsq", "simplex_lsq"),
        ("drift-quad", "drift_quad"),
        ("nonconvex-simplex", "nonconvex_simplex"),
        ("pl-1d", "pl_1d"),
    ):
        module = extensions._import_file(REPO_ROOT / "examples" / name / "problem.py")
        task = _example_task(module, dataset)
        cases.append(_Case(name, task, _example_split(name), _example_numbers(name)))
    return cases


def _example_task(module: Any, dataset: str) -> Any:
    for value in vars(module).values():
        if (
            isinstance(value, type)
            and value.__module__ == module.__name__
            and hasattr(value, "loader_order")
        ):
            task = value.__new__(value)
            task.device = torch.device("cpu")
            return task
    raise AssertionError(f"no task class in {dataset}")


def _example_split(name: str) -> Any:
    def split(rows: int) -> dict[str, Any]:
        numbers = torch.arange(rows, dtype=torch.float64)
        if name in {"fed-lasso", "simplex-lsq"}:
            return {"x": numbers.unsqueeze(1), "y": numbers}
        if name == "pl-1d":
            return {"x": numbers}
        return {"x": numbers.unsqueeze(1)}

    return split


def _example_numbers(name: str) -> Any:
    def numbers(batch: Any) -> list[int]:
        first = batch[0]
        return [int(value) for value in first.reshape(len(first), -1)[:, 0].tolist()]

    return numbers


class PlannedOrdersAreTheLoadersTest(unittest.TestCase):
    """For each task, each loop, each loader setting: the planned batches are the loader's."""

    def check(self, case: _Case, splits: list[tuple[int, dict[str, Any]]], loop_spec: Any) -> None:
        orders, loops, expected = [], [], []
        for rows, config in splits:
            data = case.split(rows)
            order = case.task.loader_order(data, config)
            self.assertIsInstance(order, LoaderOrder)
            orders.append(order)
            loops.append(local_loop(*loop_spec))
            loader = case.task.build_dataloader(data, config)
            try:
                updates = reference(loader, *loop_spec)
            except ValueError:
                updates = []
            expected.append([[case.numbers(batch) for batch in update] for update in updates])
        result = plan_orders(orders, loops)
        for client, updates in enumerate(expected):
            self.assertEqual(planned(result, client), updates, f"client {client}")

    def configurations(self) -> list[dict[str, Any]]:
        return [
            {"batch_size": size, "shuffle": shuffle, "drop_last": drop, "seed": seed}
            for size, shuffle, drop, seed in itertools.product(
                (3, 4, 16), (False, True), (False, True), (7, 2**31 + 5)
            )
        ]

    def test_every_task_loop_and_loader_setting(self) -> None:
        for case in classification_cases() + example_cases():
            for loop_spec in LOOPS:
                with self.subTest(task=case.label, loop=loop_spec):
                    # One call over splits of 1 to 13 rows under every setting:
                    # each its own length, seed and batching, planned together.
                    splits = [
                        (rows, config)
                        for rows, config in zip(
                            itertools.cycle((1, 2, 3, 5, 8, 9, 13)),
                            self.configurations(),
                            strict=False,
                        )
                    ]
                    self.check(case, splits, loop_spec)

    def test_many_clients_planned_together(self) -> None:
        case = classification_cases()[0]
        for count in (5, 520):
            with self.subTest(clients=count):
                splits = [
                    (3 + client % 11, {"batch_size": 4, "shuffle": True, "seed": 1000 + client})
                    for client in range(count)
                ]
                self.check(case, splits, ("sequential_epoch", 3, None, "sgd"))

    def test_a_split_with_no_batch_gets_no_steps(self) -> None:
        case = classification_cases()[0]
        data = case.split(3)
        order = case.task.loader_order(data, {"batch_size": 4, "drop_last": True})
        result = plan_orders([order], [LocalLoop(epochs=2)])
        self.assertEqual(int(result.steps[0]), 0)
        self.assertEqual(result.structure[0], ())


class DataloaderSeedsTest(unittest.TestCase):
    def test_the_seeds_are_dataloader_seeds(self) -> None:
        clients = ["client_0", "client_17", "a b", "ü"]
        for base, round_id, phase in ((42, 1, "fit"), (0, 900, "eval"), (2**40, 3, "fit")):
            self.assertEqual(
                dataloader_seeds(base, round_id, clients, phase),
                [dataloader_seed(base, round_id, client, phase) for client in clients],
            )


if __name__ == "__main__":
    unittest.main()
