"""The batched ClientExecutor: a round's sampled clients trained together.

``SequentialExecutor`` (``fedbrew/core/loop.py``) runs each sampled client's
local update alone, one eager autograd step at a time. This executor runs them
together: every client's parameters stacked on a leading client dimension, and
each step one ``torch.func.vmap(torch.func.grad(...))`` of the task's
``functional_loss`` over the stack, followed by the rule's step arithmetic
(``fedbrew/clients/batched_update.py``) over the same dimension. It computes
what the sequential executor computes, per client:

- the same batches, in the same order: each client's rule declares the loop
  its update takes (``batched_plan``), and every client's batches are
  computed together from what the task declares its loaders yield
  (``plan_round``, ``fedbrew/clients/batch_orders.py``);
- the same per-client records, weights and state: each client's rule builds
  its FitResult from its share of the stack (``batched_result``), with the
  code its own ``fit`` ends with;
- the same results, to summation order: the one-client stack is not vmapped at
  all, and is the sequential arithmetic bit for bit
  (``tests/test_batched_executor_tolerance.py``).

The data. Each client's train split is held as its rows
(``task.split_rows``); the clients of a bucket are stacked, padded to the
longest, and a step gathers every client's batch from that in one
``index_select`` (``_Steps``), or slices it where the order is unshuffled. A
client whose batch at some step is shorter than the others' is padded with its
own rows and masked, and the task's functions take the mean over its real
rows.

Buckets and chunks. Clients whose updates have the same shape -- the same
number of updates, each over the same number of batches -- are stepped
together as one bucket. Consecutive clients are taken into a chunk until the
chunk's estimated memory reaches ``chunk_bytes``
(``runtime.performance.executor_chunk_bytes``); a chunk is trained, its
results are yielded in request order, and only then is the next one stacked,
so peak memory is one chunk as it was one client. Results stream as rows of
the chunk's stack (``StackedRow``), which ``WeightedStateAccumulator`` folds
in one weighted reduction per tensor.

Refusals. Every client's update is planned before any is run, so a client
with no training batches is refused with its rule's own message before any
work is done.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from torch import Tensor, nn

from fedbrew.clients.batch_orders import RoundOrders
from fedbrew.clients.batched_update import (
    ClientBatchFit,
    ClientBatchPlan,
    ProgramValues,
    accumulate,
    apply_update,
    data_versions,
    divide,
    initial_optimizer_state,
    plan_round,
    update_weights,
)
from fedbrew.clients.torch_sgd_client import trainable_parameter_count
from fedbrew.core.execution import ClientPool, FitObserver
from fedbrew.core.protocol import FitRequest, FitResult
from fedbrew.core.stacked_results import StackedFitResults
from fedbrew.core.torch_utils import StateStack
from fedbrew.tasks.base import BatchableTask

#: ``runtime.performance.executor_chunk_bytes`` when unset: 1 GiB.
DEFAULT_EXECUTOR_CHUNK_BYTES = 1 << 30

#: Model-sized tensors a client holds while it is stepped, beside its
#: optimizer's: its parameters, one gradient, and one sum or update.
_WORKING_SLOTS = 3


class BatchedExecutor:
    """A ClientExecutor that trains a round's sampled clients together.

    Only for clients whose rule, task and model are batchable
    (``batched_unsupported``, ``fedbrew/core/runner.py``'s selection); each
    client is asked for its plan, and one that has none raises.
    """

    def __init__(
        self,
        chunk_bytes: int = DEFAULT_EXECUTOR_CHUNK_BYTES,
        record: dict[str, Any] | None = None,
        context: StepContext | None = None,
    ) -> None:
        """Args:
        chunk_bytes: The memory one chunk of clients may take, estimated
            from parameters, gradients, optimizer state and data rows.
            At least one client is taken whatever it costs.
        record: Kept current with ``largest_chunk_clients``, the most
            clients one chunk has held -- run.json's record of the executor.
        context: How the training step runs (``runtime.performance.precision``);
            None is the reference, at the model's own precision.
        """

        if isinstance(chunk_bytes, bool) or int(chunk_bytes) <= 0:
            raise ValueError("chunk_bytes must be a positive integer")
        self.chunk_bytes = int(chunk_bytes)
        self.record = record if record is not None else {}
        self.record.setdefault("largest_chunk_clients", 0)
        self.context = context
        #: The stacked rows of the last round's buckets, kept while the round is
        #: one chunk: at full participation the same clients' unchanged data
        #: would otherwise be stacked again every round.
        self._rows: dict[tuple[int, ...], _Rows] = {}

    def fit(
        self,
        clients: ClientPool,
        requests: Sequence[FitRequest],
        observer: FitObserver,
    ) -> Iterator[FitResult]:
        return self._fit(clients, list(requests), observer)

    def _fit(
        self,
        clients: ClientPool,
        requests: list[FitRequest],
        observer: FitObserver,
    ) -> Iterator[FitResult]:
        members = [
            clients[request.client_id] if isinstance(clients, Mapping) else clients
            for request in requests
        ]
        yield from client_results(
            members, requests, observer, self._trained(members, requests, _train_chunk)
        )

    def fit_stacked(
        self,
        clients: ClientPool,
        requests: Sequence[FitRequest],
        observer: FitObserver,
    ) -> Iterator[StackedFitResults] | None:
        """The round's results as one stacked result per chunk, or None if a rule cannot build them.

        What ``fit`` yields, handed over a chunk at a time: each chunk's
        clients' results as one :class:`StackedFitResults`, built by their
        rule (``batched_stacked_results``) from the chunk's stacked tensors,
        in request order. Decided before anything runs: None when some
        client's rule builds its results only one by one, and the round then
        goes through ``fit``.
        """

        requests = list(requests)
        members = [
            clients[request.client_id] if isinstance(clients, Mapping) else clients
            for request in requests
        ]
        for member in {type(member): member for member in members}.values():
            supported = getattr(member, "batched_stacked_supported", None)
            if not callable(supported) or not supported():
                return None
        return self._fit_stacked(members, requests, observer)

    def _fit_stacked(
        self, members: list[Any], requests: list[FitRequest], observer: FitObserver
    ) -> Iterator[StackedFitResults]:
        yield from stacked_client_results(
            members, requests, observer, self._trained(members, requests, _train_chunk_stacked)
        )

    def _trained(
        self,
        members: list[Any],
        requests: list[FitRequest],
        train_chunk: Callable[..., Any],
    ) -> Iterator[tuple[int, int, list[ClientBatchPlan], Any, float]]:
        """Plan the round, then train it a chunk at a time with ``train_chunk``.

        Yields each chunk's bounds, the round's plans, what ``train_chunk``
        returned, and the seconds it took.
        """

        if not requests:
            return
        task = members[0].task
        template = task.build_model(members[0].model_config)
        plans = _plans(members, requests, template)
        orders = plan_round(plans, requests[0].round_id)
        chunks = list(self._chunks(task, template, plans, orders[0]))
        # Kept only while a round is one chunk, so what is held between rounds
        # is what one chunk holds anyway.
        kept = self._rows if len(chunks) == 1 else None
        self._rows = {}
        for start, stop in chunks:
            self.record["largest_chunk_clients"] = max(
                self.record["largest_chunk_clients"], stop - start
            )
            chunk_started = time.perf_counter()
            trained = train_chunk(
                task, template, plans[start:stop], orders, kept, self._rows, self.context
            )
            if kept is None:
                self._rows = {}
            yield start, stop, plans, trained, time.perf_counter() - chunk_started

    def _chunks(
        self, task: Any, template: nn.Module, plans: list[ClientBatchPlan], train: RoundOrders
    ) -> Iterator[tuple[int, int]]:
        """Consecutive runs of clients whose estimated memory fits ``chunk_bytes``."""

        return cut_chunks(client_costs(task, template, plans, train), self.chunk_bytes)


#: What a trained chunk is handed over as: its bounds in the round's requests,
#: the round's plans, the chunk's share for the rules (``chunk_fits`` or
#: ``chunk_fit_stacked``), and the seconds it took.
TrainedChunk = tuple[int, int, list[ClientBatchPlan], Any, float]


def client_results(
    members: list[Any],
    requests: list[FitRequest],
    observer: FitObserver,
    trained: Iterable[TrainedChunk],
) -> Iterator[FitResult]:
    """Each client's result from its chunk's share (``chunk_fits``), in request order."""

    done, total = 0, len(requests)
    for start, stop, plans, fits, seconds in trained:
        share = seconds / (stop - start)
        for index in range(start, stop):
            result_started = time.perf_counter()
            result = members[index].batched_result(
                requests[index], plans[index], fits[index - start]
            )
            fits[index - start] = None
            done += 1
            observer.fitted(result, share + time.perf_counter() - result_started, done, total)
            yield result
            # Held no longer than the consumer holds it: a result is a row
            # of its chunk's stack and keeps the whole stack alive.
            result = None  # type: ignore[assignment]


def stacked_client_results(
    members: list[Any],
    requests: list[FitRequest],
    observer: FitObserver,
    trained: Iterable[TrainedChunk],
) -> Iterator[StackedFitResults]:
    """Each chunk's results as one stacked result, built by its rule (``chunk_fit_stacked``)."""

    report = getattr(observer, "fitted_stack", None)
    done, total = 0, len(requests)
    for start, stop, plans, chunk, seconds in trained:
        built = time.perf_counter()
        stacked = members[start].batched_stacked_results(
            requests[start:stop], plans[start:stop], chunk, members[start:stop]
        )
        chunk = None
        seconds += time.perf_counter() - built
        if callable(report):
            done += len(stacked)
            report(stacked, seconds, done, total)
        else:
            # An observer that records one result at a time is handed
            # each, with its share of the chunk's time.
            for result in stacked.results():
                done += 1
                observer.fitted(result, seconds / len(stacked), done, total)
        yield stacked
        # Held no longer than the consumer holds it: it keeps the stacks alive.
        stacked = None  # type: ignore[assignment]


def client_costs(
    task: Any, template: nn.Module, plans: Sequence[ClientBatchPlan], train: RoundOrders
) -> list[int]:
    """Each client's estimated memory while it is stepped, in bytes.

    Its parameters times the model-sized tensors it holds -- parameters,
    gradient, sum or update, its optimizer's slots and any persistent state --
    plus its split's rows and two of its longest batch.
    """

    parameter_bytes = sum(p.numel() * p.element_size() for p in template.parameters())
    row_bytes = sum(
        tensor[:1].numel() * tensor.element_size()
        for tensor in task.split_rows(plans[0].train_data)
    )
    longest = (
        train.lengths.amax(dim=1) if train.lengths.shape[1] else torch.zeros(len(plans))
    ).tolist()
    costs = []
    for index, plan in enumerate(plans):
        slots = _WORKING_SLOTS + plan.program.optimizer.state_slots + int(plan.program.scaffold)
        costs.append(
            parameter_bytes * slots + row_bytes * (plan.eval_rows + 2 * int(longest[index]))
        )
    return costs


def cut_chunks(costs: Sequence[int], budget: int) -> Iterator[tuple[int, int]]:
    """Consecutive runs of ``costs`` whose sum fits ``budget``; at least one each."""

    start, used = 0, 0
    for index, cost in enumerate(costs):
        if index > start and used + cost > budget:
            yield start, index
            start, used = index, 0
        used += cost
    yield start, len(costs)


def _plans(
    members: list[Any], requests: list[FitRequest], template: nn.Module
) -> list[ClientBatchPlan]:
    """Every client's plan, the broadcast checked once per rule and payload.

    Every client of a rule checks the same payload against the same model the
    same way (``batched_start``), so the first client's check is each one's.
    """

    starts: dict[tuple[Any, int], tuple[Any, Mapping[str, Any]]] = {}
    plans = []
    for member, request in zip(members, requests, strict=True):
        key = (type(member).batched_start, id(request.payload))
        held = starts.get(key)
        if held is None or held[0] is not request.payload:
            held = starts[key] = (request.payload, member.batched_start(request, template))
        plans.append(member.batched_plan(request, template, start=held[1]))
    return plans


def trained_state_keys(task: Any, template: nn.Module) -> list[str]:
    """The federated state's keys, which must be exactly the model's parameters."""

    state_keys = list(task.get_federated_model_state(template))
    names = [name for name, _ in template.named_parameters()]
    if set(state_keys) != set(names):
        raise ValueError(
            "the batched executor trains parameters, and this model's federated state "
            f"is not exactly its parameters: {sorted(set(state_keys) ^ set(names))}"
        )
    return state_keys


def _train_buckets(
    task: Any,
    template: nn.Module,
    plans: Sequence[ClientBatchPlan],
    orders: tuple[RoundOrders, RoundOrders],
    kept: Mapping[tuple[int, ...], _Rows] | None,
    keep: dict[tuple[int, ...], _Rows] | None,
    context: StepContext | None = None,
) -> tuple[list[str], list[tuple[list[int], dict[str, Tensor], Any, Any]]]:
    """Train one chunk's clients, bucket by bucket.

    Returns the federated state's keys, and per bucket its clients (indices
    into ``plans``), its trained stack, and both passes' outputs, still on
    the device.
    """

    state_keys = trained_state_keys(task, template)
    buffers = dict(template.named_buffers())
    buckets: dict[tuple[Any, ...], list[int]] = {}
    for index, plan in enumerate(plans):
        buckets.setdefault(plan.bucket, []).append(index)
    trained = [
        (
            members,
            *_run_bucket(
                task,
                template,
                buffers,
                [plans[i] for i in members],
                orders,
                kept,
                keep,
                context=context,
            ),
        )
        for members in buckets.values()
    ]
    return state_keys, trained


def _train_chunk(
    task: Any,
    template: nn.Module,
    plans: Sequence[ClientBatchPlan],
    orders: tuple[RoundOrders, RoundOrders],
    kept: Mapping[tuple[int, ...], _Rows] | None = None,
    keep: dict[tuple[int, ...], _Rows] | None = None,
    context: StepContext | None = None,
) -> list[ClientBatchFit]:
    """Train one chunk's clients, bucket by bucket, and return each one's share.

    ``orders`` are the round's training and post-fit batches
    (``plan_round``); ``kept`` holds stacked rows from the last round, reused
    for a bucket of the same clients whose data is the same objects, unedited;
    ``keep`` receives this chunk's.
    """

    return chunk_fits(
        task, template, plans, *_train_buckets(task, template, plans, orders, kept, keep, context)
    )


def chunk_fits(
    task: Any,
    template: nn.Module,
    plans: Sequence[ClientBatchPlan],
    state_keys: list[str],
    trained: list[tuple[list[int], dict[str, Tensor], Any, Any]],
) -> list[ClientBatchFit]:
    """Each client's share of a trained chunk (``_train_buckets``), for its rule's result."""

    metadata = task.federated_model_state_metadata(template)
    trainable = trainable_parameter_count(template)
    # Every bucket's outputs cross to the host together, once for the chunk,
    # a stacked bucket's post-fit outputs folded into metrics first.
    parts: list[Any] = []
    kinds: list[bool | None] = []
    for _, _, training, evaluated in trained:
        parts.append(training)
        if evaluated is None:
            kinds.append(None)
            continue
        part, folded = folded_part(task, *evaluated)
        parts.append(part)
        kinds.append(folded)
    floats = iter(host_floats(parts))
    fits: list[ClientBatchFit | None] = [None] * len(plans)
    for (members, stack, _, _), folded in zip(trained, kinds, strict=True):
        training_outputs = next(floats)
        evaluated_values = next(floats) if folded is not None else None
        states = StateStack({key: stack[key] for key in state_keys})
        for position, index in enumerate(members):
            evaluation = (
                None
                if evaluated_values is None
                else unfolded(evaluated_values[position], bool(folded))
            )
            fits[index] = ClientBatchFit(
                model_state=states.row(position),
                training_outputs=training_outputs[position],
                eval_outputs=evaluation if isinstance(evaluation, list) else None,
                optimizer_steps=len(plans[index].structure),
                model_state_metadata=dict(metadata),
                trainable_parameters=trainable,
                start=plans[index].start,
                eval_metrics=evaluation if isinstance(evaluation, tuple) else None,
            )
    return [fit for fit in fits if fit is not None]


def _train_chunk_stacked(
    task: Any,
    template: nn.Module,
    plans: Sequence[ClientBatchPlan],
    orders: tuple[RoundOrders, RoundOrders],
    kept: Mapping[tuple[int, ...], _Rows] | None = None,
    keep: dict[tuple[int, ...], _Rows] | None = None,
    context: StepContext | None = None,
) -> ChunkFit:
    """``_train_chunk``, for a rule that builds its results stacked: each bucket as columns.

    What a rule's result reads of the passes' outputs is kept per bucket over
    its clients, not per client: each training step's ``total``, and the
    post-fit pass's metrics and example counts as the task folded them
    (``stacked_metrics``). A bucket the task does not fold -- one client, or
    a task without the hook -- keeps each client's post-fit outputs. All of
    it crosses to the host in one copy for the chunk.
    """

    return chunk_fit_stacked(
        task, template, *_train_buckets(task, template, plans, orders, kept, keep, context)
    )


def chunk_fit_stacked(
    task: Any,
    template: nn.Module,
    state_keys: list[str],
    trained: list[tuple[list[int], dict[str, Tensor], Any, Any]],
) -> ChunkFit:
    """A trained chunk (``_train_buckets``) as its buckets' columns, for stacked results."""

    parts: list[Any] = []
    columns: list[Tensor] = []
    layout: list[tuple[int | None, int | None, list[str], bool]] = []
    for _, _, (outputs, counts), evaluated in trained:
        totals = None
        if outputs and "total" in outputs[0]:
            totals = len(columns)
            columns.append(torch.stack([output["total"] for output in outputs]))
        eval_part, names, folded = None, [], False
        if evaluated is not None:
            stacked_metrics = getattr(task, "stacked_metrics", None)
            if callable(stacked_metrics) and len(evaluated[1]) > 1 and evaluated[0]:
                metrics, examples = stacked_metrics(*evaluated)
                names, folded = list(metrics), True
                eval_part = len(columns)
                columns.extend([*metrics.values(), examples])
            else:
                eval_part = len(parts)
                parts.append(evaluated)
        layout.append((totals, eval_part, names, folded))
        del counts
    groups, values = host_values(parts, columns)
    buckets = []
    for (members, stack, _, _), (totals, eval_part, names, folded) in zip(
        trained, layout, strict=True
    ):
        bucket = BucketFit(
            positions=members,
            states=StateStack({key: stack[key] for key in state_keys}),
            step_totals=None if totals is None else values[totals],
        )
        if folded:
            assert eval_part is not None
            bucket.eval_metrics = {
                name: values[eval_part + offset] for offset, name in enumerate(names)
            }
            bucket.eval_examples = [int(count) for count in values[eval_part + len(names)]]
        elif eval_part is not None:
            bucket.eval_outputs = groups[eval_part]
        buckets.append(bucket)
    return ChunkFit(
        buckets=buckets,
        model_state_metadata=task.federated_model_state_metadata(template),
        trainable_parameters=trainable_parameter_count(template),
    )


@dataclass(slots=True)
class BucketFit:
    """One bucket's share of a trained chunk, over its clients (``_train_chunk_stacked``)."""

    #: The bucket's clients, as positions in the chunk, in the order of its rows.
    positions: list[int]
    #: Their trained states: row ``r`` is client ``positions[r]``'s.
    states: StateStack
    #: Each training step's ``total``, step-major -- client ``r``'s steps are
    #: ``step_totals[r::len(positions)]`` -- or None when the steps report none.
    step_totals: list[float] | None = None
    #: The post-fit pass's ``compute_metrics`` per name over the rows, and its
    #: example counts, as the task folded them; None when it did not.
    eval_metrics: dict[str, list[float]] | None = None
    eval_examples: list[int] | None = None
    #: Each row's post-fit outputs, when the pass ran and was not folded.
    eval_outputs: list[list[dict[str, float]]] | None = None


@dataclass(slots=True)
class ChunkFit:
    """A trained chunk, bucket by bucket, for a rule that builds its results stacked."""

    buckets: list[BucketFit]
    #: ``task.federated_model_state_metadata`` of the model.
    model_state_metadata: dict[str, Any]
    trainable_parameters: int


def _run_bucket(
    task: Any,
    template: nn.Module,
    buffers: Mapping[str, Tensor],
    plans: list[ClientBatchPlan],
    orders: tuple[RoundOrders, RoundOrders],
    kept: Mapping[tuple[int, ...], _Rows] | None,
    keep: dict[tuple[int, ...], _Rows] | None,
    parts: Sequence[tuple[int, int]] | None = None,
    context: StepContext | None = None,
) -> tuple[dict[str, Tensor], Any, Any]:
    """One bucket's clients trained on their stacked rows, kept rows reused.

    ``parts`` are ranges of the bucket's rows each measured on its own in the
    post-fit pass (``_Bucket``).
    """

    sources = [plan.train_data for plan in plans]
    key = tuple(id(source) for source in sources)
    rows = (kept or {}).get(key)
    if rows is None or not rows.holds(sources):
        rows = _Rows(task, sources)
    if keep is not None:
        keep[key] = rows
    return _Bucket(task, template, buffers, plans, rows, orders, parts, context).run()


class _Rows:
    """Splits as the task's rows, each tensor padded to the longest and stacked.

    A split's rows are ``tensors[k][position, :lengths[position]]``; the
    padding is zeros and is never read unmasked. One split is held as its own
    rows, unstacked.
    """

    def __init__(self, task: Any, sources: list[Any]) -> None:
        rows = [task.split_rows(source) for source in sources]
        self.sources = sources
        self.versions = [data_versions(source) for source in sources]
        self.lengths = [int(len(split_rows[0])) for split_rows in rows]
        self.longest = max(self.lengths)
        self.stacked = len(rows) > 1
        if not self.stacked:
            self.tensors = rows[0]
        else:
            self.tensors = tuple(
                torch.nn.utils.rnn.pad_sequence(list(parts), batch_first=True)
                for parts in zip(*rows, strict=True)
            )
        first = self.tensors[0]
        self.device = first.device
        self.bytes = sum(tensor.numel() * tensor.element_size() for tensor in self.tensors)
        self._cast: dict[torch.dtype, _Rows] = {}

    def as_dtype(self, dtype: torch.dtype) -> _Rows:
        """These rows with every floating tensor in ``dtype``: themselves, when it is theirs.

        The copy is kept with the rows, so rows kept for the next round are
        cast once.
        """

        if all(not tensor.is_floating_point() or tensor.dtype == dtype for tensor in self.tensors):
            return self
        if dtype not in self._cast:
            rows = copy.copy(self)
            rows.tensors = tuple(
                tensor.to(dtype) if tensor.is_floating_point() else tensor
                for tensor in self.tensors
            )
            rows._cast = {}
            self._cast[dtype] = rows
        return self._cast[dtype]

    def serving(self, sources: list[Any]) -> _Rows:
        """These rows, held as the rows of other splits over the same tensors, unedited.

        A group's settings are served their clients' splits as mappings of
        their own over the dataset's shared tensors
        (``fedbrew/core/settings_group.py``); each setting's evaluator asks
        for its rows by its own splits (``holds``).
        """

        rows = copy.copy(self)
        rows.sources = sources
        rows.versions = [data_versions(source) for source in sources]
        return rows

    def holds(self, sources: list[Any]) -> bool:
        """Whether these are the rows of ``sources``: the same objects, not edited since."""

        return all(
            held is source and version == data_versions(source)
            for held, source, version in zip(self.sources, sources, self.versions, strict=True)
        )


#: At most this many elements of a stack's rows are gathered for every step
#: at once (``_Steps._gathered``); more are gathered a step at a time.
_GATHER_AT_ONCE = 1 << 22


class _Steps:
    """A group's batches, one step at a time, gathered from its rows.

    ``orders`` are the round's (``RoundOrders``) and ``slots`` the group's rows
    of them, in the order of ``rows``' splits. A batch shorter than the step's
    widest is padded -- from the split's own rows, or its zero padding -- and
    masked. Where every order is unshuffled and the step starts at the same
    row for all, the batch is a slice of the stacked rows, not a copy.
    """

    def __init__(
        self, rows: _Rows, orders: RoundOrders, slots: Sequence[int], mask_dtype: torch.dtype
    ) -> None:
        where = torch.tensor(list(slots), dtype=torch.long)
        self.rows = rows
        self.dtype = mask_dtype
        self.size = len(slots)
        lengths = orders.lengths[where]
        starts = orders.starts[where]
        self._lengths = lengths
        self._indices = orders.indices[where]
        # What each step needs to know on the host, read off once.
        if lengths.shape[1]:
            widths = lengths.amax(dim=0)
            self.widths = widths.tolist()
            self.full = (lengths == widths.unsqueeze(0)).all(dim=0).tolist()
            self.aligned = (starts == starts[:1]).all(dim=0).tolist()
        else:
            self.widths, self.full, self.aligned = [], [], []
        self.first_starts = starts[0].tolist() if len(starts) else []
        self.first_lengths = lengths[0].tolist() if len(lengths) else []
        self.sliced = bool(orders.contiguous[where].all())
        self._on_device: tuple[Tensor, Tensor] | None = None
        self._every: tuple[Tensor, ...] | None = None

    def _small(self) -> bool:
        """Whether every step's batches together are few enough elements to gather at once."""

        steps, widest = self._indices.shape[1], self._indices.shape[2]
        per_row = sum(tensor[0, :1].numel() for tensor in self.rows.tensors)
        return self.size * steps * widest * per_row <= _GATHER_AT_ONCE

    def _gathered(self) -> tuple[Tensor, ...]:
        """Every step's batches of every split, gathered in one index_select per tensor.

        What a step reads is then a view: for a small stack stepped many
        times -- fed-lasso's eight clients over twelve steps -- one gather a
        round rather than one a step.
        """

        if self._every is None:
            indices, _ = self._device()
            flat = indices.reshape(-1)
            shape = indices.shape
            self._every = tuple(
                tensor.reshape(-1, *tensor.shape[2:])
                .index_select(0, flat)
                .reshape(*shape, *tensor.shape[2:])
                for tensor in self.rows.tensors
            )
        return self._every

    def _device(self) -> tuple[Tensor, Tensor]:
        """The indices, as rows of the flattened stack, and the lengths, on the rows' device."""

        if self._on_device is None:
            device = self.rows.device
            # Split k's row r is row k * longest + r of the stack seen as one
            # tensor, so a step's batches are one index_select, which is
            # several times faster than indexing by (split, row) pairs.
            offsets = torch.arange(self.size, dtype=torch.long) * self.rows.longest
            flat = self._indices + offsets.view(-1, 1, 1)
            self._on_device = (flat.to(device), self._lengths.to(device))
        return self._on_device

    def batch(self, step: int) -> tuple[tuple[Tensor, ...], Tensor | None]:
        """Step ``step``'s batch of every split, and the mask of its real rows."""

        rows = self.rows
        width = self.widths[step]
        if not rows.stacked:
            length = self.first_lengths[step]
            if self.sliced:
                first = self.first_starts[step]
                return tuple(tensor[first : first + length] for tensor in rows.tensors), None
            index = self._indices[0, step, :length].to(rows.device)
            return tuple(tensor.index_select(0, index) for tensor in rows.tensors), None
        if self.sliced and self.aligned[step]:
            first = self.first_starts[step]
            batch = tuple(tensor[:, first : first + width] for tensor in rows.tensors)
        elif self._every is not None or self._small():
            batch = tuple(gathered[:, step, :width] for gathered in self._gathered())
        else:
            indices, _ = self._device()
            index = indices[:, step, :width].reshape(-1)
            batch = tuple(
                tensor.reshape(-1, *tensor.shape[2:])
                .index_select(0, index)
                .reshape(self.size, width, *tensor.shape[2:])
                for tensor in rows.tensors
            )
        if self.full[step]:
            return batch, None
        _, lengths = self._device()
        positions = torch.arange(width, device=rows.device).unsqueeze(0)
        return batch, (positions < lengths[:, step].unsqueeze(1)).to(self.dtype)

    @property
    def lengths(self) -> Tensor:
        """Each split's batch lengths, ``(splits, steps)``, on the CPU."""

        return self._lengths


def measure_splits(
    task: Any,
    model: nn.Module,
    buffers: Mapping[str, Tensor],
    params: Mapping[str, Tensor],
    params_dim: int | None,
    steps: _Steps,
    counts: Sequence[int],
) -> list[dict[str, Tensor]]:
    """``functional_eval`` over each split's batches, all splits together.

    ``steps`` gathers each split's batches from its rows; ``counts[k]`` is how
    many split ``k`` has. ``params`` are every split's own (stacked,
    ``params_dim`` 0) or one model all share (``params_dim`` None). Batch
    ``position`` of every split is measured in one vmapped call, a split with
    fewer batches padded with an empty one whose outputs are never read; with
    one split nothing is vmapped, and the call is ``eval_step``'s arithmetic.
    Returns the outputs per position, each a tensor per key over the splits.
    """

    def measure(params: Any, batch: Any, mask: Any) -> Any:
        return task.functional_eval(model, params, buffers, batch, mask)

    outputs: list[dict[str, Tensor]] = []
    model.eval()
    with torch.no_grad():
        for position in range(max(counts, default=0)):
            batch, mask = steps.batch(position)
            if not steps.rows.stacked:
                outputs.append(measure(params, batch, mask))
                continue
            dims = (params_dim, 0, None if mask is None else 0)
            outputs.append(torch.func.vmap(measure, in_dims=dims)(params, batch, mask))
    return outputs


def per_split_floats(
    outputs: list[dict[str, Tensor]], counts: Sequence[int]
) -> list[list[dict[str, float]]]:
    """Per-position outputs as each split's list of float dicts, in one host copy."""

    return host_floats([(outputs, counts)])[0]


def folded_part(
    task: Any, outputs: list[dict[str, Tensor]], counts: Sequence[int]
) -> tuple[tuple[list[dict[str, Tensor]], Sequence[int]], bool]:
    """A stack's eval outputs, ready for the host copy: folded by the task where it can.

    With ``stacked_metrics`` and more than one split, each split's
    ``compute_metrics`` and example count are computed on the device and the
    part is one dict per split; otherwise it is the outputs themselves.
    Returns the part and whether it was folded (``unfolded`` reads it back).
    """

    stacked_metrics = getattr(task, "stacked_metrics", None)
    if callable(stacked_metrics) and len(counts) > 1 and outputs:
        metrics, examples = stacked_metrics(outputs, counts)
        return ([{**metrics, "examples": examples}], [1] * len(counts)), True
    return (outputs, counts), False


def unfolded(values: list[dict[str, float]], folded: bool) -> Any:
    """One split's share of a host copy: its outputs, or its (metrics, examples)."""

    if not folded:
        return values
    metrics = dict(values[0])
    return metrics, int(metrics.pop("examples"))


def host_floats(
    parts: Sequence[tuple[list[dict[str, Tensor]], Sequence[int]]],
) -> list[list[list[dict[str, float]]]]:
    """Several groups' per-position outputs as float dicts, in one host copy for all.

    ``parts[g]`` is a group's outputs -- per position, a tensor per key over
    its splits -- and each split's count of positions; each group becomes, per
    split, its list of float dicts. Every value is widened to float64 on its
    device and all of them cross to the host together: a float32 value and a
    count below 2**53 are exact in float64, so each float is the one its own
    ``float()`` would give.
    """

    return host_values(parts, [])[0]


def host_values(
    parts: Sequence[tuple[list[dict[str, Tensor]], Sequence[int]]],
    columns: Sequence[Tensor],
) -> tuple[list[list[list[dict[str, float]]]], list[list[float]]]:
    """``host_floats`` of ``parts``, and each of ``columns`` as a list of floats, in one copy.

    A column is read in its own order, flattened: a ``(steps, clients)``
    tensor becomes its values step by step.
    """

    pieces: list[Tensor] = []
    layout: list[tuple[list[str], int, int]] = []
    for outputs, counts in parts:
        keys = list(outputs[0]) if outputs else []
        for key in keys:
            pieces.append(
                torch.stack([output[key] for output in outputs]).reshape(-1).to(torch.float64)
            )
        layout.append((keys, len(outputs), len(counts)))
    for column in columns:
        pieces.append(column.reshape(-1).to(torch.float64))
    values: list[float] = []
    if pieces:
        # Joined where they are, so a GPU round is one device-to-host copy.
        device = pieces[0].device
        values = torch.cat([piece.to(device) for piece in pieces]).cpu().tolist()
    groups: list[list[list[dict[str, float]]]] = []
    offset = 0
    for (keys, positions, size), (_, counts) in zip(layout, parts, strict=True):
        read = {}
        for key in keys:
            read[key] = values[offset : offset + positions * size]
            offset += positions * size
        groups.append(
            [
                [{key: read[key][step * size + split] for key in keys} for step in range(count)]
                for split, count in enumerate(counts)
            ]
        )
    lists: list[list[float]] = []
    for column in columns:
        lists.append(values[offset : offset + column.numel()])
        offset += column.numel()
    return groups, lists


class _Bucket:
    """Clients whose updates share a shape, stepped together.

    With one client nothing is stacked and nothing is vmapped: every function
    runs on that client's own tensors, which is the sequential arithmetic.

    ``parts``, ranges of the rows, are the settings of a group whose clients
    share the bucket (``fedbrew/core/settings_group.py``). Their post-fit pass
    is measured part by part, each on its own rows as its bucket alone would
    be: a task's evaluation may multiply the stacked parameters by a tensor
    of its own -- fed-lasso's objective does, by its design matrix -- which
    vmap makes one matrix product over every row, and a product's rounding
    depends on how many rows it has.
    """

    def __init__(
        self,
        task: Any,
        model: nn.Module,
        buffers: Mapping[str, Tensor],
        plans: list[ClientBatchPlan],
        rows: _Rows,
        orders: tuple[RoundOrders, RoundOrders],
        parts: Sequence[tuple[int, int]] | None = None,
        context: StepContext | None = None,
    ) -> None:
        self.task = task
        self.plans = plans
        self.program = plans[0].program
        self.structure = plans[0].structure
        self.size = len(plans)
        self.stacked = self.size > 1
        first = next(model.parameters())
        self.device, self.model_dtype = first.device, first.dtype
        self.parameters = dict(model.named_parameters())
        #: The reference unless the run asks for a precision: the step then
        #: runs at ``dtype`` (float32 under f32_f64), under autocast or TF32,
        #: and the post-fit pass as the reference runs it.
        self.context = context or StepContext()
        self.dtype = self.context.train_dtype(self.model_dtype)
        self.model = model
        self.buffers = buffers
        # The pass measures the model as trained, at its own precision.
        self.eval_model, self.eval_buffers = model, buffers
        self.buffers = {
            name: value.to(self.dtype) if value.is_floating_point() else value
            for name, value in self.buffers.items()
        }
        self.rows = rows
        slots = [plan.slot for plan in plans]
        train, evaluation = orders
        self.steps = _Steps(rows.as_dtype(self.dtype), train, slots, self.dtype)
        self.evaluation = evaluation
        self.parts = list(parts) if parts is not None and len(parts) > 1 else None
        self.eval_steps = _Steps(rows, evaluation, slots, self.model_dtype)
        self.eval_counts = evaluation.steps[torch.tensor(slots, dtype=torch.long)].tolist()
        self.weights = update_weights(self.steps.lengths, self.structure, self.program)
        self._weights_on_device: Tensor | None = None
        #: Each client's own learning rate, momentum, ...: the bucket shares
        #: its program's shape, not its values.
        self.values = ProgramValues(
            [plan.program for plan in plans],
            len(self.structure),
            self.dtype,
            self.device,
        )
        #: Whether a step's gradients are taken as one backward through the
        #: stacked losses' sum rather than vmap(grad): the form the task
        #: declares, by measurement (``batched_gradient``); one client is
        #: never vmapped, and takes the sequential gradient either way.
        self.summed = self.stacked and gradient_form(task) == "summed"

    # -- the tensors every client starts from --------------------------------

    def _state(self, states: list[Mapping[str, Any] | None]) -> tuple[Any, int | None]:
        """Per-client model-shaped states, stacked; one shared object is not copied."""

        if states[0] is None:
            return None, None
        placed = [
            {name: _placed(state[name], self.parameters[name]) for name in self.parameters}
            for state in states  # type: ignore[union-attr]
        ]
        placed = [self._trained_dtype(state) for state in placed]
        if not self.stacked:
            return placed[0], None
        if all(state is states[0] for state in states):
            return placed[0], None
        return {name: torch.stack([s[name] for s in placed]) for name in self.parameters}, 0

    def _trained_dtype(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        """A state in the dtype the step runs at: itself, but under f32_f64."""

        if self.dtype == self.model_dtype:
            return state
        return {name: value.to(self.dtype) for name, value in state.items()}

    def _start(self) -> dict[str, Tensor]:
        """Every client's starting parameters.

        Nothing is written into them -- every step is out of place -- so the
        broadcast the clients share is one tensor seen C times, not C copies.
        """

        if all(plan.start is self.plans[0].start for plan in self.plans):
            shared = self._trained_dtype(
                {
                    name: _placed(self.plans[0].start[name], parameter)
                    for name, parameter in self.parameters.items()
                }
            )
            if not self.stacked:
                return shared
            return {
                name: value.unsqueeze(0).expand(self.size, *value.shape)
                for name, value in shared.items()
            }
        start = [
            self._trained_dtype(
                {name: _placed(plan.start[name], self.parameters[name]) for name in self.parameters}
            )
            for plan in self.plans
        ]
        if not self.stacked:
            return start[0]
        return {name: torch.stack([s[name] for s in start]) for name in self.parameters}

    def _gather(self, step: int) -> tuple[tuple[Tensor, ...], Tensor | None]:
        """Each client's batch at ``step``, and the mask of its real rows."""

        return self.steps.batch(step)

    def _call(self, function: Callable[..., Any], arguments: list[tuple[Any, int | None]]) -> Any:
        """``function`` on each client's arguments: vmapped over the stack, or called as is."""

        values = [value for value, _ in arguments]
        if not self.stacked:
            return function(*values)
        dims = tuple(dim if value is not None else None for value, dim in arguments)
        return torch.func.vmap(function, in_dims=dims)(*values)

    def _weights(self, step: int) -> tuple[Any, int | None]:
        """Each client's weight of its ``step``-th batch in its update's combination."""

        if not self.stacked:
            return float(self.weights[0, step]), None
        if self._weights_on_device is None:
            self._weights_on_device = self.weights.to(self.device)
        return self._weights_on_device[:, step], 0

    def _values(self, step: int) -> tuple[dict[str, Tensor], int | None]:
        """Each client's values for the ``step``-th applied update (from 1)."""

        values = self.values.at(step)
        if not self.stacked:
            return {name: value[0] for name, value in values.items()}, None
        return values, 0

    def _denominators(self, first: int, count: int) -> tuple[Any, int | None]:
        """Each client's sum of the weights of the batches ``first`` on of one update."""

        totals = self.weights[:, first : first + count].sum(dim=1)
        if not self.stacked:
            return float(totals[0]), None
        return totals.to(self.device), 0

    # -- the round ------------------------------------------------------------

    def run(self) -> tuple[dict[str, Tensor], Any, Any]:
        """Every step, then the post-fit pass: the trained stack, and both passes' outputs.

        The outputs stay on the device, per position a tensor per key over
        the clients, with each client's count, for the chunk's one host copy.
        """

        with self.context.training(self.device):
            params, outputs, step = self._train()
        return self._finish(params, outputs, step)

    def _train(self) -> tuple[dict[str, Tensor], list[dict[str, Tensor]], int]:
        """Every step: the trained stack, each step's outputs, and how many batches were taken."""

        program = self.program
        batch_update, gradient_sum, combined_update = step_functions(
            self.task, self.model, self.buffers, program, self.context.autocast(self.device)
        )
        client_dim = 0 if self.stacked else None
        params = self._start()
        state: Any = initial_optimizer_state(program.optimizer, params)
        reference = (
            self._state([plan.start for plan in self.plans])
            if program.proximal_mu
            else (None, None)
        )
        client_control = self._state([plan.client_control for plan in self.plans])
        server_control = self._state([plan.server_control for plan in self.plans])
        corrections = [reference, client_control, server_control]
        outputs: list[dict[str, Tensor]] = []

        self.model.train()
        if self.summed:
            return self._run_summed(params, state, corrections, outputs)
        step = 0
        for number, count in enumerate(self.structure, start=1):
            if program.combine == "batch":
                batch, mask = self._gather(step)
                step += 1
                params, state, step_outputs = self._call(
                    batch_update,
                    [
                        (params, client_dim),
                        (state, client_dim),
                        (batch, client_dim),
                        (mask, client_dim),
                        *corrections,
                        self._values(number),
                        (number, None),
                    ],
                )
                outputs.append(step_outputs)
                continue
            total: Any = None
            first = step
            for _ in range(count):
                batch, mask = self._gather(step)
                total, step_outputs = self._call(
                    gradient_sum,
                    [
                        (total, client_dim),
                        (params, client_dim),
                        (batch, client_dim),
                        (mask, client_dim),
                        self._weights(step),
                    ],
                )
                outputs.append(step_outputs)
                step += 1
            params, state = self._call(
                combined_update,
                [
                    (params, client_dim),
                    (state, client_dim),
                    (total, client_dim),
                    self._denominators(first, count),
                    *corrections,
                    self._values(number),
                    (number, None),
                ],
            )

        return params, outputs, step

    def _finish(
        self, params: dict[str, Tensor], outputs: list[dict[str, Tensor]], step: int
    ) -> tuple[dict[str, Tensor], Any, Any]:
        """The trained stack, the step outputs, and the post-fit pass's."""

        if self.dtype != self.model_dtype:
            params = {name: value.to(self.model_dtype) for name, value in params.items()}
        training = (outputs, [step] * self.size)
        evaluated: Any = None
        if self.plans[0].evaluate:
            if self.parts is None:
                evaluated = self._evaluate(params)
            else:
                evaluated = [self._evaluate_part(params, first, stop) for first, stop in self.parts]
        if not self.stacked:
            params = {name: value.unsqueeze(0) for name, value in params.items()}
        return params, training, evaluated

    # -- the summed form of a step's gradient ---------------------------------

    def _run_summed(
        self,
        params: dict[str, Tensor],
        state: Any,
        corrections: list[tuple[Any, int | None]],
        outputs: list[dict[str, Tensor]],
    ) -> tuple[dict[str, Tensor], list[dict[str, Tensor]], int]:
        """``run``'s steps with each gradient taken as ``_summed_gradients`` takes it.

        The rule's step -- correction, clipping, optimizer -- and the
        combination of a pass's gradients are ``run``'s, vmapped over the
        clients as there.
        """

        program = self.program

        def update(  # type: ignore[no-untyped-def]
            params, grads, state, reference, client_control, server_control, values, *, step
        ):
            return apply_update(
                program,
                params,
                grads,
                state,
                step,
                reference,
                client_control,
                server_control,
                values=values,
            )

        def combine(total, grads, weight):  # type: ignore[no-untyped-def]
            return accumulate(total, grads, weight)

        def full_update(  # type: ignore[no-untyped-def]
            params,
            total,
            state,
            denominator,
            reference,
            client_control,
            server_control,
            values,
            *,
            step,
        ):
            if program.combine == "full":
                total = divide(total, denominator)
            return apply_update(
                program,
                params,
                total,
                state,
                step,
                reference,
                client_control,
                server_control,
                values=values,
            )

        step = 0
        for number, count in enumerate(self.structure, start=1):
            if program.combine == "batch":
                grads, step_outputs = self._summed_gradients(params, *self._gather(step))
                step += 1
                if program.max_grad_norm is None:
                    # Unclipped, a step is elementwise -- every operand a
                    # client's tensor, one shared by all, a Python number, or
                    # a value per client, which _per_client shapes to its
                    # rows -- so on the stacked tensors it is the vmapped
                    # arithmetic.
                    params, state = apply_update(
                        program,
                        params,
                        grads,
                        state,
                        number,
                        *(value for value, _ in corrections),
                        values=self.values.at(number),
                    )
                else:
                    params, state = self._call(
                        partial(update, step=number),
                        [
                            (params, 0),
                            (grads, 0),
                            (state, 0),
                            *corrections,
                            self._values(number),
                        ],
                    )
                outputs.append(step_outputs)
                continue
            total: Any = None
            first = step
            for _ in range(count):
                grads, step_outputs = self._summed_gradients(params, *self._gather(step))
                total = self._call(combine, [(total, 0), (grads, 0), self._weights(step)])
                outputs.append(step_outputs)
                step += 1
            params, state = self._call(
                partial(full_update, step=number),
                [
                    (params, 0),
                    (total, 0),
                    (state, 0),
                    self._denominators(first, count),
                    *corrections,
                    self._values(number),
                ],
            )
        return params, outputs, step

    def _summed_gradients(
        self, params: dict[str, Tensor], batch: tuple[Tensor, ...], mask: Tensor | None
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Every client's gradient: one vmapped forward, one backward through the losses' sum.

        Client ``c``'s loss is a function of its own parameters alone, so the
        derivative of the sum with respect to client ``c``'s parameters is the
        derivative of its loss: every other term's is exactly zero. What
        differs from ``vmap(grad)`` is how many times the stack is walked, not
        what each client's gradient is; its arithmetic is the batched kernels'
        either way, to summation order.
        """

        task, model, buffers = self.task, self.model, self.buffers
        autocast = self.context.autocast(self.device)

        def loss(params: Any, batch: Any, mask: Any) -> Any:
            with _autocast(autocast):
                return task.functional_loss(model, params, buffers, batch, mask)

        leaves = {name: value.detach().requires_grad_() for name, value in params.items()}
        with torch.enable_grad():
            losses, outputs = torch.func.vmap(loss, in_dims=(0, 0, None if mask is None else 0))(
                leaves, batch, mask
            )
            grads = torch.autograd.grad(losses.sum(), tuple(leaves.values()))
        return dict(zip(leaves, grads, strict=True)), outputs

    def _evaluate_part(
        self, params: dict[str, Tensor], first: int, stop: int
    ) -> tuple[list[dict[str, Tensor]], list[int]]:
        """The post-fit pass of rows ``first`` to ``stop``, as a bucket of those clients alone."""

        plans = self.plans[first:stop]
        stacked = len(plans) > 1
        own = {
            name: (value[first:stop] if stacked else value[first]).clone()
            for name, value in params.items()
        }
        rows = _Rows(self.task, [plan.train_data for plan in plans])
        steps = _Steps(rows, self.evaluation, [plan.slot for plan in plans], self.dtype)
        counts = self.eval_counts[first:stop]
        outputs = measure_splits(
            self.task,
            self.eval_model,
            self.eval_buffers,
            own,
            0 if stacked else None,
            steps,
            counts,
        )
        return outputs, counts

    def _evaluate(self, params: dict[str, Tensor]) -> tuple[list[dict[str, Tensor]], list[int]]:
        """The post-fit pass: ``functional_eval`` over each client's eval batches."""

        outputs = measure_splits(
            self.task,
            self.eval_model,
            self.eval_buffers,
            params,
            0 if self.stacked else None,
            self.eval_steps,
            self.eval_counts,
        )
        return outputs, self.eval_counts


#: The two forms a stacked step's gradient can be taken in (``batched_gradient``).
GRADIENT_FORMS = ("vmap_grad", "summed")


def gradient_form(task: Any) -> str:
    """The form a task declares its stacked gradients are fastest in; ``vmap_grad`` by default."""

    form = getattr(task, "batched_gradient", "vmap_grad")
    if form not in GRADIENT_FORMS:
        raise ValueError(f"batched_gradient must be one of {GRADIENT_FORMS}, got {form!r}")
    return form


def step_functions(
    task: Any,
    model: nn.Module,
    buffers: Mapping[str, Tensor],
    program: Any,
    autocast: str | None,
) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    """The three per-client functions a bucket's steps are made of.

    ``batch_update``: one batch's gradient and the update on it;
    ``gradient_sum``: one batch's gradient added into a pass's sum;
    ``combined_update``: the update on a pass's combined gradient. The loss
    runs under bfloat16 autocast on ``autocast``'s device type, when given.
    """

    def loss(params: Any, batch: Any, mask: Any) -> Any:
        with _autocast(autocast):
            return task.functional_loss(model, params, buffers, batch, mask)

    gradient = torch.func.grad(loss, has_aux=True)

    def batch_update(  # type: ignore[no-untyped-def]
        params, state, batch, mask, reference, client_control, server_control, values, step
    ):
        grads, outputs = gradient(params, batch, mask)
        params, state = apply_update(
            program,
            params,
            grads,
            state,
            step,
            reference,
            client_control,
            server_control,
            values=values,
        )
        return params, state, outputs

    def gradient_sum(total, params, batch, mask, weight):  # type: ignore[no-untyped-def]
        grads, outputs = gradient(params, batch, mask)
        return accumulate(total, grads, weight), outputs

    def combined_update(  # type: ignore[no-untyped-def]
        params, state, total, denominator, reference, client_control, server_control, values, step
    ):
        if program.combine == "full":
            total = divide(total, denominator)
        return apply_update(
            program,
            params,
            total,
            state,
            step,
            reference,
            client_control,
            server_control,
            values=values,
        )

    return batch_update, gradient_sum, combined_update


def _autocast(device_type: str | None) -> Any:
    """bfloat16 autocast on ``device_type``, or nothing."""

    if device_type is None:
        return nullcontext()
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16)


class StepContext:
    """How a run's batched training step runs: the reference, or the precision it asks for.

    ``f32_f64`` steps a float64 model in float32; ``tf32`` lets float32
    matmuls and convolutions on CUDA use TensorFloat32; ``bf16`` runs the loss
    under bfloat16 autocast. The update arithmetic and every evaluation stay
    at the model's precision.
    """

    def __init__(self, precision: str = "reference") -> None:
        self.precision = precision

    def train_dtype(self, dtype: torch.dtype) -> torch.dtype:
        """The dtype a model of ``dtype`` is stepped in."""

        if self.precision == "f32_f64" and dtype == torch.float64:
            return torch.float32
        return dtype

    def autocast(self, device: torch.device) -> str | None:
        """The device type the loss is autocast on, or None."""

        return device.type if self.precision == "bf16" else None

    @contextmanager
    def training(self, device: torch.device) -> Iterator[None]:
        """Where a bucket's steps run: under TF32 when the run asks for it on CUDA."""

        if self.precision != "tf32" or device.type != "cuda":
            yield
            return
        held = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = held


def _placed(value: Tensor, like: Tensor) -> Tensor:
    """A received tensor on the model's device and in its dtype, as load_state_dict copies it."""

    return value.detach().to(device=like.device, dtype=like.dtype)


#: Layers whose training forward draws from the process-wide generator, which
#: no batched run can replay (chapter 11 §9).
_STOCHASTIC_LAYERS = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)


def batched_unsupported(components: Any) -> str | None:
    """Why this run cannot be batched, or None if it can.

    Checks what can be known before a round runs, in the order a reader would
    fix it: the strategy, the runtime, the task, the model, then the rule
    (``batched_unsupported`` on a built client, which also checks the seed).
    """

    return _batched_check(components)[0]


def _batched_check(components: Any) -> tuple[str | None, nn.Module | None]:
    """``batched_unsupported``'s reason, and the model it built to check, if it got that far."""

    config = components.config
    if config.server.strategy == "centralized":
        return "the centralized strategy trains one client, so there is nothing to batch", None
    if config.runtime.use_amp:
        return "runtime.use_amp is on, and GradScaler's loss scale is sequential state", None
    task = components.task
    if not isinstance(task, BatchableTask):
        return (
            f"task {type(task).__name__} provides no functional_loss and functional_eval "
            "(fedbrew.tasks.base.BatchableTask)"
        ), None
    client_ids = list(components.clients)
    if not client_ids:
        return "the run has no clients", None
    client = components.clients[client_ids[0]]
    unsupported = getattr(client, "batched_unsupported", None)
    if not callable(unsupported):
        return f"update rule {type(client).__name__} declares no batched update", None
    model = task.build_model(getattr(client, "model_config", None))
    reason = _model_unsupported(task, model)
    if reason is not None:
        return reason, model
    return unsupported(), model


def _model_unsupported(task: Any, model: nn.Module) -> str | None:
    if next(model.parameters(), None) is None:
        return "the model has no parameters"
    device = next(model.parameters()).device.type
    if device not in {"cpu", "cuda"}:
        return f"the model is on {device}, and only cpu and cuda are batched"
    names = {name for name, _ in model.named_parameters()}
    federated = set(task.get_federated_model_state(model))
    if federated != names:
        return "the model's federated state is not exactly its parameters"
    if not all(parameter.requires_grad for parameter in model.parameters()):
        return "the model has frozen parameters"
    for name, module in model.named_modules():
        if isinstance(module, _STOCHASTIC_LAYERS) and float(getattr(module, "p", 0.0)) > 0.0:
            return (
                f"the model's {name or type(module).__name__} is dropout at p = {module.p}, "
                "which draws from the process-wide generator"
            )
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            return f"the model's {name} is batch normalisation, whose statistics are state"
    return None


def select_executor(components: Any) -> tuple[BatchedExecutor | None, dict[str, Any]]:
    """The executor ``runtime.performance.executor`` asks for, or the reference if it cannot be.

    Returns it -- None for the sequential reference -- and the record run.json
    keeps under ``reproducibility.executor``: which one ran, and, when
    ``batched`` was asked for and the run could not be batched, why. A batched
    executor keeps its record current as it runs (``largest_chunk_clients``).
    """

    performance = components.config.runtime.extra.get("performance") or {}
    if performance.get("executor", "sequential") != "batched":
        return None, {"used": "sequential"}
    precision_asked = str(performance.get("precision", "reference"))
    reason, model = _batched_check(components)
    if reason is not None:
        record = {"used": "sequential", "fallback": reason}
        _record_modes(record, precision_asked, "reference", "the run is sequential")
        return None, record
    record = {"used": "batched", "largest_chunk_clients": 0}
    assert model is not None
    precision, why = _precision_for(precision_asked, model)
    _record_modes(record, precision_asked, precision, why)
    chunk_bytes = performance.get("executor_chunk_bytes", DEFAULT_EXECUTOR_CHUNK_BYTES)
    context = StepContext(precision)
    return BatchedExecutor(chunk_bytes, record=record, context=context), record


def _precision_for(asked: str, model: nn.Module) -> tuple[str, str | None]:
    """The precision a run's step runs at, and why it is not the one asked for, if it is not."""

    first = next(model.parameters())
    dtype, device = first.dtype, first.device.type
    if asked == "f32_f64" and dtype != torch.float64:
        return "reference", f"f32_f64 steps a float64 model in float32, and this model is {dtype}"
    if asked == "tf32" and device != "cuda":
        return "reference", f"tf32 is a mode of CUDA's float32 matmuls, and this run is on {device}"
    if asked in {"tf32", "bf16"} and dtype != torch.float32:
        return "reference", f"{asked} applies to a float32 model, and this model is {dtype}"
    return asked, None


def _record_modes(record: dict[str, Any], asked: str, precision: str, why: str | None) -> None:
    """What run.json records of the precision a run asked for: what ran, and why, if not that."""

    if asked != "reference":
        record["precision"] = (
            {"used": precision} if precision == asked else {"used": precision, "fallback": why}
        )
