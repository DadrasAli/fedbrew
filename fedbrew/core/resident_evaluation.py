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

``grad_norm_sq`` (``evaluation.grad_norm``) is measured on the device from the
rows the round trains on: every client's, gathered in chunks of consecutive
rows from the stacks held for the run, one ``functional_loss`` and one
backward per chunk at the round's mean (``fedbrew/core/grad_norm.py``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
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
    free_memory,
    measure_splits,
    staged_values,
    unfolded,
)
from fedbrew.core.protocol import EvalRequest
from fedbrew.core.torch_utils import set_training


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


@dataclass(slots=True)
class _Central:
    """The global rows a central pass measured on the device reads, and how it cuts them.

    ``in_parts``: the task's pass is ``CentralPassInParts``' -- its terms are
    measured beside the steps and its metrics made by ``central_metrics``; else
    the classification task's, ``compute_metrics`` of the steps. ``prepared``:
    the rows in one batch as the task's closed form reads them, which its
    ``closed_form_central`` measures the steps and terms from, where the
    round's training stack holds them (``ResidentRounds._share_central``).
    """

    features: Tensor
    targets: Tensor
    batch_size: int
    in_parts: bool
    prepared: tuple[Tensor, ...] | None = None


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
        self._plans: dict[tuple[int, tuple[str, ...]], tuple[ClientEvalPlan, str | None]] = {}
        self._work: dict[tuple[Any, ...], Any] = {}
        self.central = _central_rows(rounds)
        #: The gradient pass's chunks, as positions into the flattened train
        #: stacks: made on its first scheduled round and kept for the run.
        self._grad_index: list[Tensor] | None = None
        #: The gradient pass's chunks of rows, gathered on its first scheduled
        #: round and kept for the run where a second copy of the rows fits
        #: (decided then: ``_grad_kept``).
        self._grad_rows: list[tuple[Tensor, ...]] | None = None
        self._grad_kept: bool | None = None

    # -- the client splits ---------------------------------------------------

    def enqueue(
        self, round_id: int, work: list[tuple[int, list[str]]], params: dict[str, Tensor]
    ) -> EvalStage | None:
        """Measure ``work`` -- (roster place, splits) in the evaluator's order -- at ``params``."""

        if not work or not self.clients:
            return None
        plans, missing = self._work_plans(work)
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

    def _work_plans(
        self, work: list[tuple[int, list[str]]]
    ) -> tuple[tuple[ClientEvalPlan, ...], tuple[str | None, ...]]:
        """Every client's plan, and the split it refuses: for the same work, the same lists."""

        key = tuple((place, tuple(splits)) for place, splits in work)
        held = self._work.get(key)
        if held is None:
            held = tuple(zip(*(self._plan(place, splits) for place, splits in work), strict=True))
            if len(self._work) < 8:
                self._work[key] = held
        return held  # type: ignore[return-value]

    def _plan(self, place: int, splits: list[str]) -> tuple[ClientEvalPlan, str | None]:
        """``batched_evaluation_plan``'s plan for a client, and the split it refuses, if any.

        The same every round for the same client and splits -- its data and its
        unseeded loader orders -- so it is made once.
        """

        key = (place, tuple(splits))
        held = self._plans.get(key)
        if held is None:
            held = self._plans[key] = self._new_plan(place, splits)
        return held

    def _new_plan(self, place: int, splits: list[str]) -> tuple[ClientEvalPlan, str | None]:
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
            # The entries themselves, not only whose and which split they
            # are: they index this round's work, and a round that evaluates
            # the same clients' split beside other work holds them at other
            # indices, where the kept chunks would read another client's plan.
            key = (split, places, tuple(entries))
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

        central = self.central
        if central is None:
            return None
        features, targets, size = central.features, central.targets, central.batch_size
        outputs = []
        set_training(self.template, False)
        if central.prepared is not None:
            with torch.no_grad():
                outputs, terms = self.task.closed_form_central(
                    self.template, params, self.buffers, central.prepared
                )
            staged, layout = staged_values([(outputs, [len(outputs)]), ([terms], [1])], [])
            return CentralStage(layout, staged)
        with torch.no_grad():
            for first in range(0, len(targets), size):
                batch = (features[first : first + size], targets[first : first + size])
                outputs.append(
                    self.task.functional_eval(self.template, params, self.buffers, batch, None)
                )
            parts = [(outputs, [len(outputs)])]
            if central.in_parts:
                parts.append(([self.task.central_terms(self.template, params)], [1]))
        staged, layout = staged_values(parts, [])
        return CentralStage(layout, staged)

    def enqueue_fused(self, fused: Any, params: dict[str, Tensor]) -> tuple[CentralStage, Tensor]:
        """The central pass and ``grad_norm_sq`` at ``params`` in one pass (``FusedPass``).

        Staged as ``enqueue_central`` and ``enqueue_grad_norm`` stage theirs, so
        the flush reads them as it reads those.
        """

        # In eval mode afterwards, as the central pass leaves the template.
        set_training(self.template, False)
        outputs, terms, value = fused.measure(self.template, params, self.buffers)
        parts = [(outputs, [len(outputs)])]
        if fused.in_parts:
            parts.append(([terms], [1]))
        staged, layout = staged_values(parts, [])
        return CentralStage(layout, staged), value

    # -- the gradient of the global objective ----------------------------------

    def enqueue_grad_norm(self, params: dict[str, Tensor]) -> Tensor:
        """``grad_norm_sq`` at ``params`` over every client's train rows, as a 0-d device tensor."""

        from fedbrew.core.grad_norm import flat_chunk_gradient

        return flat_chunk_gradient(
            self.task, self.template, params, self.buffers, self._grad_chunks()
        )

    def _grad_chunks(self) -> Iterable[tuple[Tensor, ...]]:
        """Each chunk's rows, gathered from the train stacks.

        The stacks and the chunks are the run's, so their rows are the same
        every round: gathered on the first round and kept, where a copy of
        every train row fits in a quarter of the device's free memory, as the
        rows themselves must; otherwise gathered again each round, a chunk at
        a time.
        """

        if self._grad_rows is not None:
            return self._grad_rows
        rows = self.rounds.rows
        stacks = [tensor.reshape(-1, *tensor.shape[2:]) for tensor in rows.tensors]
        index = self._grad_norm_index()
        chunks = (
            tuple(stack.index_select(0, positions) for stack in stacks) for positions in index
        )
        if self._grad_kept is None:
            from fedbrew.core.resident import ROWS_FRACTION

            gathered = sum(int(positions.numel()) for positions in index) * rows.row_bytes
            free = free_memory(rows.tensors[0].device)
            self._grad_kept = free is None or gathered <= ROWS_FRACTION * free
        if not self._grad_kept:
            return chunks
        self._grad_rows = list(chunks)
        return self._grad_rows

    def _grad_norm_index(self) -> list[Tensor]:
        """Each chunk's rows, as positions into the train stacks flattened to one row axis."""

        if self._grad_index is None:
            from fedbrew.core.grad_norm import chunk_pieces, chunk_rows
            from fedbrew.core.torch_utils import uploaded

            rows = self.rounds.rows
            cap = chunk_rows(self.task, rows.row_bytes, self.rounds.executor.chunk_bytes)
            self._grad_index = [
                uploaded(
                    torch.cat(
                        [
                            torch.arange(first, stop, dtype=torch.long) + split * rows.longest
                            for split, first, stop in chunk
                        ]
                    ),
                    rows.tensors[0].device,
                )
                for chunk in chunk_pieces(rows.lengths, cap)
            ]
        return self._grad_index

    def central_metrics(self, stage: CentralStage, values: Sequence[float]) -> dict[str, float]:
        """The ``central_test_*`` metrics the evaluator's central pass reports, from its steps."""

        from fedbrew.core.loop import _central_test_metrics

        groups, _ = finished_values(values, stage.layout)
        outputs = groups[0][0] if groups and groups[0] else []
        if self.central is not None and self.central.in_parts:
            metrics = self.task.central_metrics(outputs, groups[1][0][0])
        else:
            metrics = self.task.compute_metrics(outputs)
        return _central_test_metrics({f"global_{name}": value for name, value in metrics.items()})


def _evaluation_split(data: Any, split: str) -> Any:
    from fedbrew.clients.torch_sgd_client import _get_evaluation_split

    return _get_evaluation_split(data, split)


def _central_rows(rounds: Any) -> _Central | None:
    """The global test rows on the device, where the central pass can be measured there; else None.

    Where the server's ``evaluate_global`` is the task's ``evaluate_model``
    (``central_pass_is_the_tasks``: FedAvg's and SCAFFOLD's), and that is
    either the classification task's -- ``eval_step`` over the global rows in
    order, in batches of the task's
    ``eval_batch_size``, and ``compute_metrics`` of the steps -- or a
    ``CentralPassInParts`` task's whose loader neither shuffles nor drops a
    batch: ``eval_step`` over the rows in order, in the batches its
    ``loader_order`` declares, and its terms.
    """

    from fedbrew.core.batched_evaluator import BatchedEvaluator
    from fedbrew.servers.fedavg import central_pass_is_the_tasks
    from fedbrew.tasks.base import CentralPassInParts
    from fedbrew.tasks.classification.torch_classification import TorchClassificationTask

    task, server = rounds.task, rounds.context.server
    classification = type(task).evaluate_model is TorchClassificationTask.evaluate_model
    in_parts = isinstance(task, CentralPassInParts) and callable(
        getattr(task, "loader_order", None)
    )
    if not (classification or in_parts):
        return None
    if not central_pass_is_the_tasks(server):
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
    if classification:
        return _Central(features, targets, int(task.eval_batch_size), False)
    order = task.loader_order(data, task.central_loader_config())
    if order.shuffle or order.replacement or order.drop_last:
        return None
    return _Central(features, targets, int(order.batch_size), True)
