"""The batched Evaluator: a round's due clients measured together.

``SequentialEvaluator`` (``fedbrew/core/loop.py``) builds each due client,
loads the broadcast into a model and runs ``eval_step`` over each of its
requested splits, one batch at a time; the central pass rebuilds the model,
loads the server's state into it and reads the global test shard from disk.
This evaluator is what the batched executor runs with
(``runtime.performance.executor: batched``), and computes the same numbers,
to summation order:

- client evaluation: every due client's requested splits are held as the
  task's rows, stacked, and batch ``k`` of every split is measured in one
  ``torch.func.vmap`` of the task's ``functional_eval`` at the broadcast
  model, which all of them share. Each client's batches are its own
  evaluation loader's, and each client's result is built by its rule from
  its share, with the code its own ``evaluate`` ends with. A split measured
  alone is not vmapped, and is ``eval_step``'s arithmetic bit for bit;
- the central pass: one model, built once and kept, into which the server's
  state is copied in place each time, and the global test shard read once
  and served again for as long as it is not edited.

Splits are taken into a chunk until its rows reach ``chunk_bytes``
(``runtime.performance.executor_chunk_bytes``), as the executor chunks
clients. Refusals are the sequential ones, raised for the same client: every
plan is made first, and a client's refusal is raised when its result would
have been built.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from torch import nn

from fedbrew.clients.batch_orders import LocalLoop
from fedbrew.clients.batched_update import loader_seeds, round_orders
from fedbrew.core.batched_executor import (
    DEFAULT_EXECUTOR_CHUNK_BYTES,
    _placed,
    _Rows,
    _Steps,
    folded_part,
    host_floats,
    measure_splits,
    unfolded,
)
from fedbrew.core.execution import ClientPool, ProgressCallback
from fedbrew.core.protocol import ClientInfo, EvalRequest, EvalResult
from fedbrew.data.cached_payload import CachedPayload
from fedbrew.data.dataset import FederatedDataset
from fedbrew.servers.base import ServerStrategy


class BatchedEvaluator:
    """An Evaluator that measures a round's due clients together.

    For clients whose rule declares ``batched_evaluation_plan`` (every rule
    the batched executor runs) and the global model scope; anything else is
    measured by the reference, ``SequentialEvaluator``.
    """

    def __init__(
        self,
        chunk_bytes: int = DEFAULT_EXECUTOR_CHUNK_BYTES,
        executor: Any = None,
    ) -> None:
        """Args:
        chunk_bytes: The rows one chunk of splits may take.
        executor: The BatchedExecutor this runs beside, whose kept train
            rows are the same tensors a train split's evaluation reads.
        """

        from fedbrew.core.loop import SequentialEvaluator

        if isinstance(chunk_bytes, bool) or int(chunk_bytes) <= 0:
            raise ValueError("chunk_bytes must be a positive integer")
        self.chunk_bytes = int(chunk_bytes)
        self._reference = SequentialEvaluator()
        #: The stacked rows of the last evaluation's buckets, kept while it
        #: was one chunk: the same clients' unchanged splits, every round
        #: their evaluation is due.
        self._rows: dict[tuple[Any, ...], _Rows] = {}
        self._executor = executor
        #: The central pass's model, built once, and the global test shard.
        self._central_model: nn.Module | None = None
        self._global: tuple[FederatedDataset, CachedPayload | None] | None = None

    # -- the clients ---------------------------------------------------------

    def evaluate_clients(
        self,
        clients: ClientPool,
        round_id: int,
        work: list[tuple[ClientInfo, list[str]]],
        server_payload: Mapping[str, Any],
        model_scope: str,
        on_progress: ProgressCallback | None,
    ) -> list[tuple[EvalResult, list[str]]]:
        members = [
            clients[info.client_id] if isinstance(clients, Mapping) else clients for info, _ in work
        ]
        if not work or model_scope != "global" or not _all_supported(members):
            return self._reference.evaluate_clients(
                clients, round_id, work, server_payload, model_scope, on_progress
            )
        from fedbrew.core.loop import _release_client, _validate_client_evaluation

        base_payload = dict(server_payload)
        base_payload.update({"metrics": ["loss", "accuracy"], "model_scope": model_scope})
        requests = [
            EvalRequest(
                round_id=round_id,
                client_id=info.client_id,
                payload={**base_payload, "splits": list(splits)},
            )
            for info, splits in work
        ]
        try:
            outputs, metadata = self._measure(members, requests)
            total = len(work)
            results: list[tuple[EvalResult, list[str]]] = []
            for done, (member, request, (_, splits), (plan, measured)) in enumerate(
                zip(members, requests, work, outputs, strict=True), start=1
            ):
                result = member.batched_evaluation_result(request, plan, measured, dict(metadata))
                _validate_client_evaluation(result, splits, model_scope)
                results.append((result, splits))
                if on_progress is not None:
                    on_progress(round_id, done, total, "client_eval")
        finally:
            for request in requests:
                _release_client(clients, request.client_id)
        return results

    def _measure(
        self, members: list[Any], requests: list[EvalRequest]
    ) -> tuple[list[tuple[Any, list[list[dict[str, float]] | None]]], dict[str, Any]]:
        """Every client's plan and, per requested split, its eval-step outputs as floats."""

        first = members[0]
        task = first.task
        template = task.build_model(first.model_config)
        # One broadcast, checked as each client's evaluate checks it: every
        # client builds the same model from the same config.
        model_state, metadata = first._checked_federated_payload(
            template, requests[0].payload, context="evaluation request"
        )
        parameters = dict(template.named_parameters())
        params = {name: _placed(model_state[name], parameters[name]) for name in parameters}
        buffers = dict(template.named_buffers())

        plans = [
            member.batched_evaluation_plan(request)
            for member, request in zip(members, requests, strict=True)
        ]
        planned, keep = self._plan(task, plans, requests[0].round_id)
        kept_rows = self._rows
        self._rows = {}
        executor_rows = getattr(self._executor, "_rows", None) or {}
        dtype = next(iter(params.values())).dtype
        done: list[tuple[list[tuple[int, int]], Any, bool]] = []
        for entries, orders, split_chunks in planned:
            slot_of = {entry: slot for slot, entry in enumerate(entries)}
            step_counts = orders.steps.tolist()
            for chunk in split_chunks:
                sources = [plans[index].data[position] for index, position in chunk]
                key = tuple(id(source) for source in sources)
                rows = kept_rows.get(key) or executor_rows.get(key)
                if rows is None or not rows.holds(sources):
                    rows = _Rows(task, sources)
                if keep:
                    self._rows[key] = rows
                slots = [slot_of[entry] for entry in chunk]
                counts = [step_counts[slot] for slot in slots]
                steps = _Steps(rows, orders, slots, dtype)
                outputs = measure_splits(task, template, buffers, params, None, steps, counts)
                done.append((chunk, *folded_part(task, outputs, counts)))
        # Every chunk's outputs cross to the host together, once for the round.
        measured: list[list[Any]] = [[None] * len(plan.data) for plan in plans]
        floats = host_floats([part for _, part, _ in done])
        for (chunk, _, folded), values in zip(done, floats, strict=True):
            for (index, position), split_values in zip(chunk, values, strict=True):
                measured[index][position] = unfolded(split_values, folded)
        return list(zip(plans, measured, strict=True)), metadata

    def _plan(
        self, task: Any, plans: list[Any], round_id: int
    ) -> tuple[list[tuple[list[tuple[int, int]], Any, list[list[tuple[int, int]]]]], bool]:
        """Per split name: its entries, their orders planned together, and its chunks.

        An entry is (client position, split position). Also whether the
        whole evaluation fits one chunk's budget, which is when its rows are
        kept for the next evaluation round.
        """

        segments: dict[str, list[tuple[int, int]]] = {}
        for index, plan in enumerate(plans):
            for position, data in enumerate(plan.data):
                if data is not None:
                    segments.setdefault(plan.splits[position], []).append((index, position))
        planned = []
        cost = 0
        for entries in segments.values():
            declared = [plans[index].orders[position] for index, position in entries]
            orders = round_orders(
                declared,
                loader_seeds(
                    declared,
                    [(plans[index].client_id, plans[index].seed) for index, _ in entries],
                    round_id,
                    "eval",
                ),
                [LocalLoop(epochs=1)] * len(entries),
                lambda slot, entries=entries: [
                    [batch] for batch in _replayed(plans, entries[slot])
                ],
            )
            split_chunks, split_cost = self._chunks(task, plans, entries)
            planned.append((entries, orders, split_chunks))
            cost += split_cost
        return planned, cost <= self.chunk_bytes

    def _chunks(
        self, task: Any, plans: list[Any], entries: list[tuple[int, int]]
    ) -> tuple[list[list[tuple[int, int]]], int]:
        """Consecutive runs of splits whose rows fit ``chunk_bytes``, and their total cost."""

        chunks: list[list[tuple[int, int]]] = []
        current: list[tuple[int, int]] = []
        used = total = 0
        row_bytes: int | None = None
        for index, position in entries:
            data = plans[index].data[position]
            if row_bytes is None:
                row_bytes = sum(
                    tensor[:1].numel() * tensor.element_size() for tensor in task.split_rows(data)
                )
            cost = row_bytes * 2 * plans[index].row_count(position)
            if current and used + cost > self.chunk_bytes:
                chunks.append(current)
                current, used = [], 0
            current.append((index, position))
            used += cost
            total += cost
        if current:
            chunks.append(current)
        return chunks, total

    # -- the central pass ----------------------------------------------------

    def evaluate_central(
        self,
        server: ServerStrategy,
        dataset: FederatedDataset,
    ) -> dict[str, float]:
        from fedbrew.core.loop import _central_test_metrics, _evaluate_central_test_set

        task = getattr(server, "task", None)
        if task is None or not callable(getattr(server, "evaluate_global", None)):
            return self._reference.evaluate_central(server, dataset)
        global_data = self._global_data(dataset)
        if global_data is None:
            # The reference's refusal, in the reference's words.
            return _evaluate_central_test_set(server, dataset)
        if self._central_model is None:
            # A private copy: a task that caches its models hands every
            # caller the same instance, and this one's weights are ours.
            self._central_model = copy.deepcopy(task.build_model(server.model_config))
        return _central_test_metrics(server.evaluate_global(global_data, model=self._central_model))

    def _global_data(self, dataset: FederatedDataset) -> Any:
        """The global test shard: read once, and refused if edited since."""

        if self._global is None or self._global[0] is not dataset:
            try:
                data = dataset.get_global_data()
            except (FileNotFoundError, KeyError):
                data = None
            self._global = (dataset, None if data is None else CachedPayload(data))
        cached = self._global[1]
        if cached is None:
            return None
        return cached.serve("the global test shard")


def _replayed(plans: list[Any], entry: tuple[int, int]) -> list[Any]:
    """The evaluation loader's batches of one split whose task declares no order."""

    index, position = entry
    return list(plans[index].batches[position])


def _all_supported(members: list[Any]) -> bool:
    """Whether every client's rule lets the batched evaluator measure it, asked once per class."""

    answers: dict[type, bool] = {}
    for member in members:
        cls = type(member)
        if cls not in answers:
            supported = getattr(member, "batched_evaluation_supported", None)
            answers[cls] = bool(callable(supported) and supported())
        if not answers[cls]:
            return False
    return True


def evaluator_for(executor: Any) -> BatchedEvaluator | None:
    """The evaluator a run with ``executor`` measures through: batched beside a batched one.

    None -- the reference, ``SequentialEvaluator`` -- beside the sequential
    executor, including when ``batched`` was asked for and fell back.
    """

    if executor is None:
        return None
    return BatchedEvaluator(executor.chunk_bytes, executor=executor)
