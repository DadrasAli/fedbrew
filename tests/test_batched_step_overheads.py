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
  gathered one at a time (``_Steps.period``), and nowhere else; a step that
  slices the rows an earlier one sliced takes that step's views
  (``_Steps._sliced``);
- a pass's combination on the stack is taken in place, into a total of the
  pass's own, by step weights and update denominators uploaded and shaped once
  a round (``_Bucket._pass_add``, ``_pass_combined``), against ``accumulate``
  and ``divide`` a step: over the same cases, under every gradient form;
- a stack's unclipped step makes each operation into the tensor it would have
  replaced, where the stack owns it (``_owned``), against the step that makes
  a new tensor each time: over the same cases, under every gradient form.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from typing import Any
from unittest import mock

import torch

from fedbrew.clients.batched_update import accumulate, divide
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


def _fresh(self: Any, first: int, width: int, stacked: bool) -> Any:
    """A sliced step's views made anew, as every step made them."""

    return tuple(
        tensor[:, first : first + width] if stacked else tensor[first : first + width]
        for tensor in self.rows.tensors
    )


def _accumulated(self: Any, total: Any, grads: Any, step: int) -> Any:
    return accumulate(total, grads, self._weights(step)[0])


def _divided(self: Any, total: Any, number: int) -> Any:
    if self.program.combine != "full":
        return total
    first = sum(self.structure[: number - 1])
    return divide(total, self._denominators(first, self.structure[number - 1])[0])


@contextmanager
def paying(stepwise: bool, repeats: bool, in_place: bool = True) -> Iterator[None]:
    """The executor with the removed costs put back, as asked."""

    with ExitStack() as stack:
        if not in_place:
            stack.enter_context(
                mock.patch.object(batched_executor._Bucket, "_pass_add", _accumulated)
            )
            stack.enter_context(
                mock.patch.object(batched_executor._Bucket, "_pass_combined", _divided)
            )
        if not stepwise:
            stack.enter_context(
                mock.patch.object(batched_executor._Bucket, "_stepwise", lambda self: False)
            )
        if not repeats:
            stack.enter_context(
                mock.patch.object(batched_executor._Steps, "_period", lambda self: None)
            )
            stack.enter_context(mock.patch.object(batched_executor._Steps, "_sliced", _fresh))
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
                        batched_executor, "gradient_form", lambda task, asked=None, form=form: form
                    ),
                ):
                    now, before = self._pair(config, data, stepwise=False, repeats=True)
                    self.assertAgree(now, before, exact=True)

    def test_the_pass_combined_in_place(self) -> None:
        for form in ("vmap_grad", "summed", "closed_form"):
            for label, config, data in cases():
                if form == "closed_form" and label.startswith("mlp/"):
                    continue
                with self.subTest(form=form, case=label):
                    with data() if data is not None else nullcontext():
                        now = self.run_config(config, "batched", gradient_form=form)
                        with paying(stepwise=True, repeats=True, in_place=False):
                            before = self.run_config(config, "batched", gradient_form=form)
                    self.assertAgree(now, before, exact=True)

    def test_the_stacked_step_in_place(self) -> None:
        """Each operation into the tensor it replaced (``_owned``), against a new tensor each."""

        for form in ("vmap_grad", "summed", "closed_form"):
            for label, config, data in cases():
                if form == "closed_form" and label.startswith("mlp/"):
                    continue
                with self.subTest(form=form, case=label):
                    with data() if data is not None else nullcontext():
                        now = self.run_config(config, "batched", gradient_form=form)
                        with mock.patch.object(batched_executor, "_STACKED_IN_PLACE", False):
                            before = self.run_config(config, "batched", gradient_form=form)
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

    def test_the_host_facts_are_the_tensor_reductions(self) -> None:
        """Each step's width, fullness and alignment, read on the host, are the tensors' own."""

        generator = torch.Generator().manual_seed(9)

        class Orders:
            def __init__(self, lengths: torch.Tensor, starts: torch.Tensor) -> None:
                self.lengths, self.starts = lengths, starts
                self.indices = torch.zeros((*lengths.shape, 4), dtype=torch.long)
                self.contiguous = torch.zeros(len(lengths), dtype=torch.bool)

        for clients, steps, spread in ((1, 1, 1), (3, 5, 1), (4, 6, 3), (32, 10, 2)):
            lengths = torch.randint(1, 1 + spread, (clients, steps), generator=generator)
            starts = torch.randint(0, spread, (clients, steps), generator=generator)
            orders = Orders(lengths, starts)
            slots = list(range(clients))
            stack = _FakeRows(clients, 12)
            on_host = batched_executor._Steps(stack, orders, slots, torch.float64)
            with mock.patch.object(batched_executor, "_HOST_FACTS", 0):
                reduced = batched_executor._Steps(stack, orders, slots, torch.float64)
            with self.subTest(clients=clients, steps=steps, spread=spread):
                for name in ("widths", "full", "aligned"):
                    self.assertEqual(getattr(on_host, name), getattr(reduced, name), name)

    def test_the_period_is_the_fingerprint_search_s(self) -> None:
        """``_period`` finds what a search over every step's fingerprint found.

        Over steps drawn from few values, so that steps coincide by chance --
        in one split, in its fingerprint, or everywhere -- tiled with a period
        and then disturbed in one place, under a bound on what is kept that
        some candidates pass and some do not.
        """

        def fingerprint_search(indices: torch.Tensor, lengths: torch.Tensor, per_row: int) -> Any:
            size, steps, widest = indices.shape
            weights = torch.arange(1, size * widest + 1, dtype=torch.long).view(size, 1, -1)
            prints = (indices * weights).sum(dim=(0, 2)) + lengths.sum(dim=0)
            for candidate in torch.nonzero(prints == prints[0]).view(-1).tolist()[1:]:
                if size * candidate * widest * per_row > batched_executor._KEEP_AT_ONCE:
                    return None
                if torch.equal(indices[:, candidate:], indices[:, : steps - candidate]) and (
                    torch.equal(lengths[:, candidate:], lengths[:, : steps - candidate])
                ):
                    return int(candidate)
            return None

        generator = torch.Generator().manual_seed(0)
        found = set()
        for case in range(3000):
            size = int(torch.randint(2, 5, (), generator=generator))
            steps = int(torch.randint(2, 13, (), generator=generator))
            widest = int(torch.randint(1, 4, (), generator=generator))
            values = int(torch.randint(1, 4, (), generator=generator))
            period = int(torch.randint(1, steps + 1, (), generator=generator))
            base = torch.randint(0, values, (size, period, widest), generator=generator)
            indices = base.repeat(1, steps // period + 1, 1)[:, :steps].contiguous()
            lengths = torch.randint(widest, widest + 1, (size, steps), generator=generator)
            if case % 3 == 0:
                split, step = (
                    int(torch.randint(0, n, (), generator=generator)) for n in (size, steps)
                )
                indices[split, step, 0] += 1
            if case % 5 == 0:
                lengths[0, int(torch.randint(0, steps, (), generator=generator))] -= 1
            stack = _FakeRows(size, values + 1)
            held = batched_executor._Steps.__new__(batched_executor._Steps)
            held.rows, held.size, held.sliced = stack, size, False
            held._indices, held._lengths = indices, lengths
            per_row = sum(tensor[0, :1].numel() for tensor in stack.tensors)
            bound = int(torch.randint(1, 60, (), generator=generator))
            with (
                mock.patch.object(batched_executor, "_GATHER_AT_ONCE", 1),
                mock.patch.object(batched_executor, "_KEEP_AT_ONCE", bound),
            ):
                expected = fingerprint_search(indices, lengths, per_row)
                self.assertEqual(held._period(), expected, (case, indices, lengths, bound))
            found.add(expected)
        # Periods found and not, both.
        self.assertIn(None, found)
        self.assertGreater(len(found), 5)


class _FakeRows:
    """A stack of ``size`` splits of ``rows`` numbered rows, as ``_Rows`` holds them."""

    def __init__(self, size: int, rows: int) -> None:
        self.stacked = True
        self.device = torch.device("cpu")
        self.longest = rows
        numbers = torch.arange(size * rows, dtype=torch.float64).view(size, rows, 1)
        self.tensors = (numbers, numbers.squeeze(2))
