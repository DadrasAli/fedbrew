"""The resident round's evaluation: measured on the device at the round, read back at the flush.

``BatchedEvaluator`` (``fedbrew/core/batched_evaluator.py``) measures a
round's due clients together at the broadcast model, and its central pass
the server's; both read their outputs back as they finish. A resident round
(``fedbrew/core/resident.py``) measures the same splits the same way when it
trains -- every split's clients in the evaluator's order and chunks, their
rows gathered from stacks held for the run, ``measure_splits`` at the
round's mean where the fold left it -- and stages what it measured with the
round's other values. The flush reads them back in the round's one copy, and
builds each client's ``EvalResult`` with its rule's own
``batched_evaluation_result`` and the central metrics with the task's own
``compute_metrics``, as the evaluator does after its copy.

The central pass is measured on the device where the task's
``evaluate_model`` is the classification task's -- ``eval_step`` over the
global test rows in order, then ``compute_metrics`` -- which
``functional_eval`` computes batch for batch (``BatchableTask``). Any other
task's central pass, and a rule the batched evaluator does not measure, run
at the flush through the evaluator itself, as the per-round path runs them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from fedbrew.clients.batch_orders import LocalLoop
from fedbrew.clients.batched_update import ClientEvalPlan, loader_seeds, round_orders
from fedbrew.core.batched_executor import (
    StagedLayout,
    _Steps,
    finished_values,
    folded_part,
    measure_splits,
    staged_values,
    unfolded,
)
from fedbrew.core.protocol import EvalRequest


@dataclass(slots=True)
class _Planned:
    """One split's clients this round: entries (work index, split position), orders, chunks."""

    split: str
    entries: list[tuple[int, int]]
    orders: Any
    chunks: list[list[tuple[int, int]]]


@dataclass(slots=True)
class EvalStage:
    """A round's client evaluation, measured and staged: what the flush builds results from."""

    work: list[tuple[int, list[str]]]
    plans: list[ClientEvalPlan]
    missing: list[str | None]
    chunks: list[tuple[list[tuple[int, int]], bool]]
    layout: StagedLayout
    staged: Tensor | None


@dataclass(slots=True)
class CentralStage:
    """A round's central pass, measured and staged."""

    layout: StagedLayout
    staged: Tensor | None


class ResidentEvaluation:
    """Measures a resident round's due splits and central pass at the round's mean."""

    def __init__(self, rounds: Any) -> None:
        self.rounds = rounds
        self.task = rounds.task
        self.template = rounds.template
        self.buffers = rounds.buffers
        representative = rounds.representative
        supported = getattr(representative, "batched_evaluation_supported", None)
        self.clients = bool(callable(supported) and supported())
        self.eval_config = representative._eval_loader_config(1, seeded=False)
        self._data: dict[int, Any] = {}
        self._rows: dict[str, Any] = {}
        self._planned: dict[tuple[Any, ...], _Planned] = {}
        self.central = _central_rows(rounds)

    # -- the client splits ---------------------------------------------------

    def enqueue(
        self, round_id: int, work: list[tuple[int, list[str]]], params: dict[str, Tensor]
    ) -> EvalStage | None:
        """Measure ``work`` -- (roster place, splits) in the evaluator's order -- at ``params``."""

        if not work or not self.clients:
            return None
        plans, missing = zip(*(self._plan(place, splits) for place, splits in work), strict=True)
        done: list[tuple[list[tuple[int, int]], Any, bool]] = []
        dtype = next(iter(params.values())).dtype
        for planned in self._splits(list(plans), work, round_id):
            rows = self._split_rows(planned.split)
            slot_of = {entry: slot for slot, entry in enumerate(planned.entries)}
            step_counts = planned.orders.steps.tolist()
            for chunk in planned.chunks:
                places = [work[index][0] for index, _ in chunk]
                slots = [slot_of[entry] for entry in chunk]
                counts = [step_counts[slot] for slot in slots]
                steps = _Steps(rows.bucket(places), planned.orders, slots, dtype)
                outputs = measure_splits(
                    self.task, self.template, self.buffers, params, None, steps, counts
                )
                done.append((chunk, *folded_part(self.task, outputs, counts)))
        for rows in self._rows.values():
            rows.round_done()
        staged, layout = staged_values([part for _, part, _ in done], [])
        return EvalStage(
            work,
            list(plans),
            list(missing),
            [(chunk, folded) for chunk, _, folded in done],
            layout,
            staged,
        )

    def _plan(self, place: int, splits: list[str]) -> tuple[ClientEvalPlan, str | None]:
        """``batched_evaluation_plan``'s plan for a client, and the split it refuses, if any."""

        client_id = self.rounds.roster.client_ids[place]
        plan = ClientEvalPlan(
            splits=list(splits), client_id=client_id, seed=self.rounds.representative.base_seed
        )
        data = self._client_data(place)
        for split in splits:
            split_data = _evaluation_split(data, split)
            if split_data is None:
                if split != "val":
                    return plan, split
                plan.data.append(None)
                plan.orders.append(None)
                plan.batches.append(None)
                continue
            plan.data.append(split_data)
            plan.orders.append(self.task.loader_order(split_data, self.eval_config))
            plan.batches.append(None)
        return plan, None

    def _client_data(self, place: int) -> Any:
        data = self._data.get(place)
        if data is None:
            client_id = self.rounds.roster.client_ids[place]
            data = self._data[place] = self.rounds.context.dataset.get_client_data(client_id)
        return data

    def _splits(
        self, plans: list[ClientEvalPlan], work: list[tuple[int, list[str]]], round_id: int
    ) -> list[_Planned]:
        """``BatchedEvaluator._plan``: per split, in order of first appearance, its orders."""

        segments: dict[str, list[tuple[int, int]]] = {}
        for index, plan in enumerate(plans):
            for position, data in enumerate(plan.data):
                if data is not None:
                    segments.setdefault(plan.splits[position], []).append((index, position))
        planned = []
        for split, entries in segments.items():
            places = tuple(work[index][0] for index, _ in entries)
            declared = [plans[index].orders[position] for index, position in entries]
            shuffled = any(order is not None and order.shuffle for order in declared)
            key = (split, places, tuple(position for _, position in entries))
            held = self._planned.get(key)
            if held is None or shuffled:
                held = self._planned[key] = _Planned(
                    split,
                    entries,
                    self._orders(plans, entries, declared, round_id),
                    self._chunks(plans, entries),
                )
            planned.append(held)
        return planned

    def _orders(
        self,
        plans: list[ClientEvalPlan],
        entries: list[tuple[int, int]],
        declared: list[Any],
        round_id: int,
    ) -> Any:
        return round_orders(
            declared,
            loader_seeds(
                declared,
                [(plans[index].client_id, plans[index].seed) for index, _ in entries],
                round_id,
                "eval",
            ),
            [LocalLoop(epochs=1)] * len(entries),
            lambda slot: [],  # every order is declared: a resident task has loader_order
        )

    def _chunks(
        self, plans: list[ClientEvalPlan], entries: list[tuple[int, int]]
    ) -> list[list[tuple[int, int]]]:
        """``BatchedEvaluator._chunks``: consecutive splits whose rows fit the chunk budget."""

        budget = self.rounds.executor.chunk_bytes
        chunks: list[list[tuple[int, int]]] = []
        current: list[tuple[int, int]] = []
        used = 0
        row_bytes: int | None = None
        for index, position in entries:
            data = plans[index].data[position]
            if row_bytes is None:
                row_bytes = sum(
                    tensor[:1].numel() * tensor.element_size()
                    for tensor in self.task.split_rows(data)
                )
            cost = row_bytes * 2 * plans[index].row_count(position)
            if current and used + cost > budget:
                chunks.append(current)
                current, used = [], 0
            current.append((index, position))
            used += cost
        if current:
            chunks.append(current)
        return chunks

    def _split_rows(self, split: str) -> Any:
        rows = self._rows.get(split)
        if rows is None:
            from fedbrew.core.resident import ResidentRows

            splits = [
                _evaluation_split(self._client_data(place), split)
                for place in range(len(self.rounds.roster))
            ]
            rows = self._rows[split] = ResidentRows(self.task, splits)
        return rows

    def results(
        self,
        stage: EvalStage,
        values: Sequence[float],
        round_id: int,
        server_payload: dict[str, Any],
        model_scope: str,
        on_progress: Any,
    ) -> list[tuple[Any, list[str]]]:
        """``evaluate_clients``'s results from the staged values, in the work's order."""

        from fedbrew.core.loop import _release_client, _validate_client_evaluation

        groups, _ = finished_values(values, stage.layout)
        measured: list[list[Any]] = [[None] * len(plan.data) for plan in stage.plans]
        for (chunk, folded), group in zip(stage.chunks, groups, strict=True):
            for (index, position), split_values in zip(chunk, group, strict=True):
                measured[index][position] = unfolded(split_values, folded)
        base_payload = dict(server_payload)
        base_payload.update({"metrics": ["loss", "accuracy"], "model_scope": model_scope})
        members = [self.rounds.member(place) for place, _ in stage.work]
        results: list[tuple[Any, list[str]]] = []
        try:
            for done, (member, (_, splits), plan, missing, outputs) in enumerate(
                zip(members, stage.work, stage.plans, stage.missing, measured, strict=True), start=1
            ):
                if missing is not None:
                    plan.refusal = member._missing_split_refusal(missing)
                request = EvalRequest(
                    round_id=round_id,
                    client_id=member.client_id,
                    payload={**base_payload, "splits": list(splits)},
                )
                result = member.batched_evaluation_result(
                    request, plan, outputs, dict(self.rounds.metadata)
                )
                _validate_client_evaluation(result, splits, model_scope)
                results.append((result, splits))
                if on_progress is not None:
                    on_progress(round_id, done, len(stage.work), "client_eval")
        finally:
            for member in members:
                _release_client(self.rounds.context.client, member.client_id)
        return results

    # -- the central pass ------------------------------------------------------

    def enqueue_central(self, params: dict[str, Tensor]) -> CentralStage | None:
        """``evaluate_model``'s eval steps over the global rows at ``params``; None if not here."""

        if self.central is None:
            return None
        features, targets = self.central
        size = self.task.eval_batch_size
        outputs = []
        self.template.eval()
        with torch.no_grad():
            for first in range(0, len(targets), size):
                batch = (features[first : first + size], targets[first : first + size])
                outputs.append(
                    self.task.functional_eval(self.template, params, self.buffers, batch, None)
                )
        staged, layout = staged_values([(outputs, [len(outputs)])], [])
        return CentralStage(layout, staged)

    def central_metrics(self, stage: CentralStage, values: Sequence[float]) -> dict[str, float]:
        """The ``central_test_*`` metrics the evaluator's central pass reports, from its steps."""

        from fedbrew.core.loop import _central_test_metrics

        groups, _ = finished_values(values, stage.layout)
        outputs = groups[0][0] if groups and groups[0] else []
        metrics = self.task.compute_metrics(outputs)
        return _central_test_metrics({f"global_{name}": value for name, value in metrics.items()})


def _evaluation_split(data: Any, split: str) -> Any:
    from fedbrew.clients.torch_sgd_client import _get_evaluation_split

    return _get_evaluation_split(data, split)


def _central_rows(rounds: Any) -> tuple[Tensor, Tensor] | None:
    """The global test rows on the device, where the central pass can be measured there; else None.

    Where the task's ``evaluate_model`` is the classification task's and the
    server evaluates with FedAvg's ``evaluate_global``: then the pass is
    ``eval_step`` over the global rows in order, in batches of the task's
    ``eval_batch_size``, and ``compute_metrics`` of the steps.
    """

    from fedbrew.core.batched_evaluator import BatchedEvaluator
    from fedbrew.servers.fedavg import FedAvgServer
    from fedbrew.tasks.classification.torch_classification import TorchClassificationTask

    task, server = rounds.task, rounds.context.server
    if type(task).evaluate_model is not TorchClassificationTask.evaluate_model:
        return None
    if type(server).evaluate_global is not FedAvgServer.evaluate_global:
        return None
    if type(rounds.context.evaluator).evaluate_central is not BatchedEvaluator.evaluate_central:
        return None
    try:
        data = rounds.context.dataset.get_global_data()
    except (FileNotFoundError, KeyError):
        return None
    if data is None:
        return None
    features, targets = task.split_rows(data)
    return features, targets
