"""What the batched executor stopped doing a step changes no bit of what it computes.

Two per-step costs are gone from a stack's steps, and each is held here against
the executor that still pays it -- the same run with the change turned off --
bit for bit, in every CSV and every round's checkpoint:

- the combination of a pass's gradients and the update run on the stacked
  tensors where they are elementwise (``_Bucket._stepwise``: unclipped, not
  compiled), not under ``torch.func.vmap``: over FedAvg's four modes and each
  frozen weighting, local SGD with momentum, Nesterov and weight decay under a
  cosine schedule, AdamW, FedProx and SCAFFOLD, on the float64 linear example
  and on the float32 classification task, under both gradient forms;
- a step whose rows are an earlier step's for every client -- a loader that
  permutes once and is iterated again, a single-batch loop taking its first
  batch each time -- reuses that step's gathered batch where steps are
  gathered one at a time (``_Steps.period``), and nowhere else.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from typing import Any
from unittest import mock

import torch

from fedbrew.core import batched_executor
from fedbrew.tasks.base import LoaderOrder
from tests.test_batched_executor_tolerance import (
    ExecutorRuns,
    classification_rule_config,
    example_config,
    fedavg_modes,
    ragged_clients,
    rule_arms,
    rule_config,
    with_client,
)


def _unclipped(client: dict[str, Any]) -> bool:
    return client.get("max_grad_norm") is None


def cases() -> Iterator[tuple[str, dict[str, Any], Any]]:
    for mode, client in fedavg_modes():
        if _unclipped(client):
            yield f"fedavg/{mode}", with_client(example_config("fed-lasso-l2"), **client), None
    for label, client in rule_arms():
        yield label, rule_config(client), ragged_clients
    for mode in ("single_batch", "sequential_epoch", "frozen_batch_gradients", "full_gradient"):
        client = {
            "update_rule": "fedavg",
            "update_mode": mode,
            "frozen_gradient_weighting": "examples",
        }
        yield f"mlp/fedavg/{mode}", classification_rule_config(client), None
    yield (
        "mlp/local_sgd",
        classification_rule_config({"momentum": 0.9, "nesterov": True, "weight_decay": 0.01}),
        None,
    )
    yield "mlp/scaffold", classification_rule_config({"update_rule": "scaffold"}), None


@contextmanager
def paying(stepwise: bool, repeats: bool) -> Iterator[None]:
    """The executor with the removed costs put back, as asked."""

    with ExitStack() as stack:
        if not stepwise:
            stack.enter_context(
                mock.patch.object(batched_executor._Bucket, "_stepwise", lambda self: False)
            )
        if not repeats:
            stack.enter_context(
                mock.patch.object(batched_executor._Steps, "_period", lambda self: None)
            )
        yield


class NothingMovesTest(ExecutorRuns):
    def _pair(self, config: dict[str, Any], data: Any, **off: bool) -> tuple[Any, Any]:
        with data() if data is not None else nullcontext():
            now = self.run_config(config, "batched")
            with paying(**off):
                before = self.run_config(config, "batched")
        return now, before

    def test_the_stacked_arithmetic_under_either_gradient_form(self) -> None:
        for form in ("vmap_grad", "summed"):
            for label, config, data in cases():
                with (
                    self.subTest(form=form, case=label),
                    mock.patch.object(
                        batched_executor, "gradient_form", lambda task, form=form: form
                    ),
                ):
                    now, before = self._pair(config, data, stepwise=False, repeats=True)
                    self.assertAgree(now, before, exact=True)

    def test_a_repeated_step_reuses_its_gather(self) -> None:
        """Gathered a step at a time, as a stack too large to gather at once is."""

        cases = [
            (f"fed-lasso-l2/{mode}", with_client(example_config("fed-lasso-l2"), update_mode=mode))
            for mode in ("single_batch", "full_gradient", "sequential_epoch")
        ]
        # One graph per client, stepped three times a round: the steps repeat
        # with period 1. Its reuse of a view into a gather of every step moved
        # the last bits (vectorised sums at another address); it is kept to
        # the per-step gathers, where a reused step is a gather of its own.
        cases.append(("nonconvex-simplex", example_config("nonconvex-simplex")))
        for label, config in cases:
            for one_at_a_time in (True, False):
                budget = 1 if one_at_a_time else batched_executor._GATHER_AT_ONCE
                with (
                    self.subTest(case=label, one_at_a_time=one_at_a_time),
                    mock.patch.object(batched_executor, "_GATHER_AT_ONCE", budget),
                ):
                    now, before = self._pair(config, ragged_clients, stepwise=True, repeats=False)
                    self.assertAgree(now, before, exact=True)

    def test_the_period_is_found_only_where_the_steps_repeat(self) -> None:
        from fedbrew.clients.batch_orders import LocalLoop, plan_orders

        orders = [
            LoaderOrder(12, 4, True, False, seed, per_epoch=False, keep_single_batch=True)
            for seed in (1, 2, 3)
        ]
        planned = plan_orders(orders, [LocalLoop(epochs=5)] * 3)
        stack = _FakeRows(3, 12)
        # Small enough to gather every step at once: each step a view, none kept.
        self.assertIsNone(batched_executor._Steps(stack, planned, [0, 1, 2], torch.float64).period)
        # Found at the first step, against the rows then bound.
        with mock.patch.object(batched_executor, "_GATHER_AT_ONCE", 1):
            steps = batched_executor._Steps(stack, planned, [0, 1, 2], torch.float64)
            self.assertEqual(steps.period, 3)
        for step in range(15):
            fresh = steps._batch(step)[0][0]
            self.assertTrue(torch.equal(steps.batch(step)[0][0], fresh))
        iid = [
            LoaderOrder(12, 4, True, False, seed, per_epoch=False, replacement=True)
            for seed in (1, 2, 3)
        ]
        drawn = plan_orders(iid, [LocalLoop(epochs=5, single_batch=True)] * 3)
        with mock.patch.object(batched_executor, "_GATHER_AT_ONCE", 1):
            iid_steps = batched_executor._Steps(stack, drawn, [0, 1, 2], torch.float64)
            self.assertIsNone(iid_steps.period)


class _FakeRows:
    """A stack of ``size`` splits of ``rows`` numbered rows, as ``_Rows`` holds them."""

    def __init__(self, size: int, rows: int) -> None:
        self.stacked = True
        self.device = torch.device("cpu")
        self.longest = rows
        numbers = torch.arange(size * rows, dtype=torch.float64).view(size, rows, 1)
        self.tensors = (numbers, numbers.squeeze(2))
