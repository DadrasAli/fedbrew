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

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import partial
from typing import Any

import torch
from torch import Tensor, nn

from fedbrew.clients.batch_orders import RoundOrders
from fedbrew.clients.batched_update import (
    ClientBatchFit,
    ClientBatchPlan,
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
    ) -> None:
        """Args:
        chunk_bytes: The memory one chunk of clients may take, estimated
            from parameters, gradients, optimizer state and data rows.
            At least one client is taken whatever it costs.
        record: Kept current with ``largest_chunk_clients``, the most
            clients one chunk has held -- run.json's record of the executor.
        """

        if isinstance(chunk_bytes, bool) or int(chunk_bytes) <= 0:
            raise ValueError("chunk_bytes must be a positive integer")
        self.chunk_bytes = int(chunk_bytes)
        self.record = record if record is not None else {}
        self.record.setdefault("largest_chunk_clients", 0)
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
        if not requests:
            return
        members = [
            clients[request.client_id] if isinstance(clients, Mapping) else clients
            for request in requests
        ]
        task = members[0].task
        template = task.build_model(members[0].model_config)
        plans = _plans(members, requests, template)
        train, evaluation = plan_round(plans, requests[0].round_id)
        done, total = 0, len(requests)
        chunks = list(self._chunks(task, template, plans, train))
        # Kept only while a round is one chunk, so what is held between rounds
        # is what one chunk holds anyway.
        kept = self._rows if len(chunks) == 1 else None
        self._rows = {}
        for start, stop in chunks:
            self.record["largest_chunk_clients"] = max(
                self.record["largest_chunk_clients"], stop - start
            )
            chunk_started = time.perf_counter()
            fits = _train_chunk(
                task, template, plans[start:stop], (train, evaluation), kept, self._rows
            )
            if kept is None:
                self._rows = {}
            share = (time.perf_counter() - chunk_started) / (stop - start)
            for index in range(start, stop):
                result_started = time.perf_counter()
                result = members[index].batched_result(
                    requests[index], plans[index], fits[index - start]
                )
                fits[index - start] = None  # type: ignore[call-overload]
                done += 1
                observer.fitted(result, share + time.perf_counter() - result_started, done, total)
                yield result
                # Held no longer than the consumer holds it: a result is a row
                # of its chunk's stack and keeps the whole stack alive.
                result = None  # type: ignore[assignment]

    def _chunks(
        self, task: Any, template: nn.Module, plans: list[ClientBatchPlan], train: RoundOrders
    ) -> Iterator[tuple[int, int]]:
        """Consecutive runs of clients whose estimated memory fits ``chunk_bytes``."""

        parameter_bytes = sum(p.numel() * p.element_size() for p in template.parameters())
        row_bytes = sum(
            tensor[:1].numel() * tensor.element_size()
            for tensor in task.split_rows(plans[0].train_data)
        )
        longest = (
            train.lengths.amax(dim=1) if train.lengths.shape[1] else torch.zeros(len(plans))
        ).tolist()
        start, used = 0, 0
        for index, plan in enumerate(plans):
            slots = _WORKING_SLOTS + plan.program.optimizer.state_slots + int(plan.program.scaffold)
            cost = parameter_bytes * slots + row_bytes * (plan.eval_rows + 2 * int(longest[index]))
            if index > start and used + cost > self.chunk_bytes:
                yield start, index
                start, used = index, 0
            used += cost
        yield start, len(plans)


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


def _train_chunk(
    task: Any,
    template: nn.Module,
    plans: Sequence[ClientBatchPlan],
    orders: tuple[RoundOrders, RoundOrders],
    kept: Mapping[tuple[int, ...], _Rows] | None = None,
    keep: dict[tuple[int, ...], _Rows] | None = None,
) -> list[ClientBatchFit]:
    """Train one chunk's clients, bucket by bucket, and return each one's share.

    ``orders`` are the round's training and post-fit batches
    (``plan_round``); ``kept`` holds stacked rows from the last round, reused
    for a bucket of the same clients whose data is the same objects, unedited;
    ``keep`` receives this chunk's.
    """

    state_keys = list(task.get_federated_model_state(template))
    names = [name for name, _ in template.named_parameters()]
    if set(state_keys) != set(names):
        raise ValueError(
            "the batched executor trains parameters, and this model's federated state "
            f"is not exactly its parameters: {sorted(set(state_keys) ^ set(names))}"
        )
    metadata = task.federated_model_state_metadata(template)
    trainable = trainable_parameter_count(template)
    buffers = dict(template.named_buffers())

    buckets: dict[tuple[Any, ...], list[int]] = {}
    for index, plan in enumerate(plans):
        buckets.setdefault(plan.bucket, []).append(index)

    fits: list[ClientBatchFit | None] = [None] * len(plans)
    for members in buckets.values():
        sources = [plans[i].train_data for i in members]
        key = tuple(id(source) for source in sources)
        rows = (kept or {}).get(key)
        if rows is None or not rows.holds(sources):
            rows = _Rows(task, sources)
        if keep is not None:
            keep[key] = rows
        bucket = _Bucket(task, template, buffers, [plans[i] for i in members], rows, orders)
        stack, training_outputs, eval_outputs = bucket.run()
        states = StateStack({key: stack[key] for key in state_keys})
        for position, index in enumerate(members):
            fits[index] = ClientBatchFit(
                model_state=states.row(position),
                training_outputs=training_outputs[position],
                eval_outputs=None if eval_outputs is None else eval_outputs[position],
                optimizer_steps=len(plans[index].structure),
                model_state_metadata=dict(metadata),
                trainable_parameters=trainable,
                start=plans[index].start,
            )
    return [fit for fit in fits if fit is not None]


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

    def holds(self, sources: list[Any]) -> bool:
        """Whether these are the rows of ``sources``: the same objects, not edited since."""

        return all(
            held is source and version == data_versions(source)
            for held, source, version in zip(self.sources, sources, self.versions, strict=True)
        )


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
    """Per-position outputs as each split's list of float dicts, one host copy per key."""

    if not outputs:
        return [[] for _ in counts]
    keys = list(outputs[0])
    size = len(counts)
    values = {
        key: torch.stack([output[key] for output in outputs]).reshape(len(outputs), size).tolist()
        for key in keys
    }
    return [
        [{key: float(values[key][step][position]) for key in keys} for step in range(count)]
        for position, count in enumerate(counts)
    ]


class _Bucket:
    """Clients whose updates share a shape, stepped together.

    With one client nothing is stacked and nothing is vmapped: every function
    runs on that client's own tensors, which is the sequential arithmetic.
    """

    def __init__(
        self,
        task: Any,
        model: nn.Module,
        buffers: Mapping[str, Tensor],
        plans: list[ClientBatchPlan],
        rows: _Rows,
        orders: tuple[RoundOrders, RoundOrders],
    ) -> None:
        self.task = task
        self.model = model
        self.buffers = buffers
        self.plans = plans
        self.program = plans[0].program
        self.structure = plans[0].structure
        self.size = len(plans)
        self.stacked = self.size > 1
        first = next(model.parameters())
        self.device, self.dtype = first.device, first.dtype
        self.parameters = dict(model.named_parameters())
        self.rows = rows
        slots = [plan.slot for plan in plans]
        train, evaluation = orders
        self.steps = _Steps(rows, train, slots, self.dtype)
        self.eval_steps = _Steps(rows, evaluation, slots, self.dtype)
        self.eval_counts = evaluation.steps[torch.tensor(slots, dtype=torch.long)].tolist()
        self.weights = update_weights(self.steps.lengths, self.structure, self.program)
        self._weights_on_device: Tensor | None = None

    # -- the tensors every client starts from --------------------------------

    def _state(self, states: list[Mapping[str, Any] | None]) -> tuple[Any, int | None]:
        """Per-client model-shaped states, stacked; one shared object is not copied."""

        if states[0] is None:
            return None, None
        placed = [
            {name: _placed(state[name], self.parameters[name]) for name in self.parameters}
            for state in states  # type: ignore[union-attr]
        ]
        if not self.stacked:
            return placed[0], None
        if all(state is states[0] for state in states):
            return placed[0], None
        return {name: torch.stack([s[name] for s in placed]) for name in self.parameters}, 0

    def _start(self) -> dict[str, Tensor]:
        """Every client's starting parameters.

        Nothing is written into them -- every step is out of place -- so the
        broadcast the clients share is one tensor seen C times, not C copies.
        """

        if all(plan.start is self.plans[0].start for plan in self.plans):
            shared = {
                name: _placed(self.plans[0].start[name], parameter)
                for name, parameter in self.parameters.items()
            }
            if not self.stacked:
                return shared
            return {
                name: value.unsqueeze(0).expand(self.size, *value.shape)
                for name, value in shared.items()
            }
        start = [
            {name: _placed(plan.start[name], self.parameters[name]) for name in self.parameters}
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

    def _denominators(self, first: int, count: int) -> tuple[Any, int | None]:
        """Each client's sum of the weights of the batches ``first`` on of one update."""

        totals = self.weights[:, first : first + count].sum(dim=1)
        if not self.stacked:
            return float(totals[0]), None
        return totals.to(self.device), 0

    # -- the round ------------------------------------------------------------

    def run(self) -> tuple[dict[str, Tensor], list[list[dict[str, float]]], Any]:
        program = self.program
        task, model, buffers = self.task, self.model, self.buffers

        def loss(params: Any, batch: Any, mask: Any) -> Any:
            return task.functional_loss(model, params, buffers, batch, mask)

        gradient = torch.func.grad(loss, has_aux=True)

        def batch_update(  # type: ignore[no-untyped-def]
            params, state, batch, mask, reference, client_control, server_control, *, step
        ):
            grads, outputs = gradient(params, batch, mask)
            params, state = apply_update(
                program, params, grads, state, step, reference, client_control, server_control
            )
            return params, state, outputs

        def gradient_sum(total, params, batch, mask, weight):  # type: ignore[no-untyped-def]
            grads, outputs = gradient(params, batch, mask)
            return accumulate(total, grads, weight), outputs

        def combined_update(  # type: ignore[no-untyped-def]
            params, state, total, denominator, reference, client_control, server_control, *, step
        ):
            if program.combine == "full":
                total = divide(total, denominator)
            return apply_update(
                program, params, total, state, step, reference, client_control, server_control
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

        model.train()
        step = 0
        for number, count in enumerate(self.structure, start=1):
            if program.combine == "batch":
                batch, mask = self._gather(step)
                step += 1
                params, state, step_outputs = self._call(
                    partial(batch_update, step=number),
                    [
                        (params, client_dim),
                        (state, client_dim),
                        (batch, client_dim),
                        (mask, client_dim),
                        *corrections,
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
                partial(combined_update, step=number),
                [
                    (params, client_dim),
                    (state, client_dim),
                    (total, client_dim),
                    self._denominators(first, count),
                    *corrections,
                ],
            )

        training_outputs = self._per_client(outputs, [step] * self.size)
        eval_outputs = self._evaluate(params) if self.plans[0].evaluate else None
        if not self.stacked:
            params = {name: value.unsqueeze(0) for name, value in params.items()}
        return params, training_outputs, eval_outputs

    def _evaluate(self, params: dict[str, Tensor]) -> list[list[dict[str, float]]]:
        """The post-fit pass: ``functional_eval`` over each client's eval batches."""

        outputs = measure_splits(
            self.task,
            self.model,
            self.buffers,
            params,
            0 if self.stacked else None,
            self.eval_steps,
            self.eval_counts,
        )
        return per_split_floats(outputs, self.eval_counts)

    def _per_client(
        self, outputs: list[dict[str, Tensor]], counts: Sequence[int]
    ) -> list[list[dict[str, float]]]:
        """Step outputs as each client's list of float dicts, one host copy per key."""

        if not outputs:
            return [[] for _ in self.plans]
        return per_split_floats(outputs, counts)


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

    config = components.config
    if config.server.strategy == "centralized":
        return "the centralized strategy trains one client, so there is nothing to batch"
    if config.runtime.use_amp:
        return "runtime.use_amp is on, and GradScaler's loss scale is sequential state"
    task = components.task
    if not isinstance(task, BatchableTask):
        return (
            f"task {type(task).__name__} provides no functional_loss and functional_eval "
            "(fedbrew.tasks.base.BatchableTask)"
        )
    client_ids = list(components.clients)
    if not client_ids:
        return "the run has no clients"
    client = components.clients[client_ids[0]]
    unsupported = getattr(client, "batched_unsupported", None)
    if not callable(unsupported):
        return f"update rule {type(client).__name__} declares no batched update"
    model = task.build_model(getattr(client, "model_config", None))
    reason = _model_unsupported(task, model)
    if reason is not None:
        return reason
    return unsupported()


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
    reason = batched_unsupported(components)
    if reason is not None:
        return None, {"used": "sequential", "fallback": reason}
    record: dict[str, Any] = {"used": "batched", "largest_chunk_clients": 0}
    chunk_bytes = performance.get("executor_chunk_bytes", DEFAULT_EXECUTOR_CHUNK_BYTES)
    return BatchedExecutor(chunk_bytes, record=record), record
