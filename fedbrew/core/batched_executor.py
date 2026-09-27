"""The batched ClientExecutor: a round's sampled clients trained together.

``SequentialExecutor`` (``fedbrew/core/loop.py``) runs each sampled client's
local update alone, one eager autograd step at a time. This executor runs them
together: every client's parameters stacked on a leading client dimension, and
each step one ``torch.func.vmap(torch.func.grad(...))`` of the task's
``functional_loss`` over the stack, followed by the rule's step arithmetic
(``fedbrew/clients/batched_update.py``) over the same dimension. It computes
what the sequential executor computes, per client:

- the same batches, in the same order: each client's rule replays its own
  loader iteration to name them (``batched_plan``);
- the same per-client records, weights and state: each client's rule builds
  its FitResult from its share of the stack (``batched_result``), with the
  code its own ``fit`` ends with;
- the same results, to summation order: the one-client stack is not vmapped at
  all, and is the sequential arithmetic bit for bit
  (``tests/test_batched_executor_tolerance.py``).

The data. Each client's train split is held as its rows
(``task.split_rows``); the clients of a bucket are concatenated, and a step
gathers each client's batch from that by index. A client whose batch at some
step is shorter than the others' is padded with one of its own rows and
masked, and the task's functions take the mean over its real rows.

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

from fedbrew.clients.batched_update import (
    ClientBatchFit,
    ClientBatchPlan,
    accumulate,
    apply_update,
    divide,
    initial_optimizer_state,
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
        plans: list[ClientBatchPlan] = [
            member.batched_plan(request, template)
            for member, request in zip(members, requests, strict=True)
        ]
        done, total = 0, len(requests)
        for start, stop in self._chunks(task, template, plans):
            self.record["largest_chunk_clients"] = max(
                self.record["largest_chunk_clients"], stop - start
            )
            chunk_started = time.perf_counter()
            fits = _train_chunk(task, template, plans[start:stop])
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
        self, task: Any, template: nn.Module, plans: list[ClientBatchPlan]
    ) -> Iterator[tuple[int, int]]:
        """Consecutive runs of clients whose estimated memory fits ``chunk_bytes``."""

        parameter_bytes = sum(p.numel() * p.element_size() for p in template.parameters())
        row_bytes = sum(
            tensor[:1].numel() * tensor.element_size()
            for tensor in task.split_rows(plans[0].train_data)
        )
        start, used = 0, 0
        for index, plan in enumerate(plans):
            longest = max((len(b) for update in plan.updates for b in update), default=0)
            slots = _WORKING_SLOTS + plan.program.optimizer.state_slots + int(plan.program.scaffold)
            cost = parameter_bytes * slots + row_bytes * (plan.eval_count + 2 * longest)
            if index > start and used + cost > self.chunk_bytes:
                yield start, index
                start, used = index, 0
            used += cost
        yield start, len(plans)


def _train_chunk(
    task: Any, template: nn.Module, plans: Sequence[ClientBatchPlan]
) -> list[ClientBatchFit]:
    """Train one chunk's clients, bucket by bucket, and return each one's share."""

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
    rows = [task.split_rows(plan.train_data) for plan in plans]

    buckets: dict[tuple[Any, ...], list[int]] = {}
    for index, plan in enumerate(plans):
        buckets.setdefault((plan.structure, plan.evaluate), []).append(index)

    fits: list[ClientBatchFit | None] = [None] * len(plans)
    for members in buckets.values():
        bucket = _Bucket(
            task, template, buffers, [plans[i] for i in members], [rows[i] for i in members]
        )
        stack, training_outputs, eval_outputs = bucket.run()
        states = StateStack({key: stack[key] for key in state_keys})
        for position, index in enumerate(members):
            fits[index] = ClientBatchFit(
                model_state=states.row(position),
                training_outputs=training_outputs[position],
                eval_outputs=None if eval_outputs is None else eval_outputs[position],
                optimizer_steps=len(plans[index].updates),
                model_state_metadata=dict(metadata),
                trainable_parameters=trainable,
                start=plans[index].start,
            )
    return [fit for fit in fits if fit is not None]


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
        rows: list[tuple[Tensor, ...]],
    ) -> None:
        self.task = task
        self.model = model
        self.buffers = buffers
        self.plans = plans
        self.program = plans[0].program
        self.size = len(plans)
        self.stacked = self.size > 1
        first = next(model.parameters())
        self.device, self.dtype = first.device, first.dtype
        self.parameters = dict(model.named_parameters())
        if self.stacked:
            self.rows = tuple(torch.cat(parts) for parts in zip(*rows, strict=True))
            counts = [len(client_rows[0]) for client_rows in rows]
            self.offsets = [sum(counts[:position]) for position in range(self.size)]
        else:
            self.rows = rows[0]
            self.offsets = [0]

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
        start = [
            {name: _placed(plan.start[name], self.parameters[name]) for name in self.parameters}
            for plan in self.plans
        ]
        if not self.stacked:
            return {name: value.clone() for name, value in start[0].items()}
        return {name: torch.stack([s[name] for s in start]) for name in self.parameters}

    # -- one batch per client, gathered from the rows ------------------------

    def _gather(self, indices: Sequence[Tensor]) -> tuple[tuple[Tensor, ...], Tensor | None]:
        """Each client's batch, padded to the longest with its own first row, and the mask."""

        if not self.stacked:
            index = indices[0].to(self.device)
            return tuple(tensor.index_select(0, index) for tensor in self.rows), None
        lengths = [int(len(index)) for index in indices]
        longest = max(lengths)
        padded = torch.empty((self.size, longest), dtype=torch.long)
        for position, (index, offset) in enumerate(zip(indices, self.offsets, strict=True)):
            padded[position, : lengths[position]] = index + offset
            padded[position, lengths[position] :] = offset
        flat = padded.reshape(-1).to(self.device)
        batch = tuple(
            tensor.index_select(0, flat).reshape(self.size, longest, *tensor.shape[1:])
            for tensor in self.rows
        )
        if min(lengths) == longest:
            return batch, None
        mask = (torch.arange(longest).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)).to(
            device=self.device, dtype=self.dtype
        )
        return batch, mask

    def _call(self, function: Callable[..., Any], arguments: list[tuple[Any, int | None]]) -> Any:
        """``function`` on each client's arguments: vmapped over the stack, or called as is."""

        values = [value for value, _ in arguments]
        if not self.stacked:
            return function(*values)
        dims = tuple(dim if value is not None else None for value, dim in arguments)
        return torch.func.vmap(function, in_dims=dims)(*values)

    def _weights(self, update: int, batch: int) -> tuple[Any, int | None]:
        values = [plan.weights[update][batch] for plan in self.plans]
        if not self.stacked:
            return values[0], None
        return torch.tensor(values, dtype=torch.float64, device=self.device), 0

    def _denominators(self, update: int) -> tuple[Any, int | None]:
        values = []
        for plan in self.plans:
            denominator = 0.0
            for weight in plan.weights[update]:
                denominator += weight
            values.append(denominator)
        if not self.stacked:
            return values[0], None
        return torch.tensor(values, dtype=torch.float64, device=self.device), 0

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
        updates = list(zip(*(plan.updates for plan in self.plans), strict=True))
        for number, batches in enumerate(updates, start=1):
            if program.combine == "batch":
                batch, mask = self._gather([client_batches[0] for client_batches in batches])
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
            for position in range(len(batches[0])):
                batch, mask = self._gather([client_batches[position] for client_batches in batches])
                total, step_outputs = self._call(
                    gradient_sum,
                    [
                        (total, client_dim),
                        (params, client_dim),
                        (batch, client_dim),
                        (mask, client_dim),
                        self._weights(number - 1, position),
                    ],
                )
                outputs.append(step_outputs)
            params, state = self._call(
                partial(combined_update, step=number),
                [
                    (params, client_dim),
                    (state, client_dim),
                    (total, client_dim),
                    self._denominators(number - 1),
                    *corrections,
                ],
            )

        training_outputs = self._per_client(
            outputs, [sum(len(update) for update in plan.updates) for plan in self.plans]
        )
        eval_outputs = self._evaluate(params) if self.plans[0].evaluate else None
        if not self.stacked:
            params = {name: value.unsqueeze(0) for name, value in params.items()}
        return params, training_outputs, eval_outputs

    def _evaluate(self, params: dict[str, Tensor]) -> list[list[dict[str, float]]]:
        """The post-fit pass: ``functional_eval`` over each client's eval batches."""

        task, model, buffers = self.task, self.model, self.buffers

        def measure(params: Any, batch: Any, mask: Any) -> Any:
            return task.functional_eval(model, params, buffers, batch, mask)

        client_dim = 0 if self.stacked else None
        model.eval()
        counts = [len(plan.eval_batches) for plan in self.plans]
        outputs: list[dict[str, Tensor]] = []
        for position in range(max(counts)):
            indices = [
                plan.eval_batches[position] if position < count else plan.eval_batches[0][:0]
                for plan, count in zip(self.plans, counts, strict=True)
            ]
            batch, mask = self._gather(indices)
            outputs.append(
                self._call(measure, [(params, client_dim), (batch, client_dim), (mask, client_dim)])
            )
        return self._per_client(outputs, counts)

    def _per_client(
        self, outputs: list[dict[str, Tensor]], counts: Sequence[int]
    ) -> list[list[dict[str, float]]]:
        """Step outputs as each client's list of float dicts, one host copy per key."""

        if not outputs:
            return [[] for _ in self.plans]
        keys = list(outputs[0])
        values = {
            key: torch.stack([output[key] for output in outputs])
            .reshape(len(outputs), self.size)
            .tolist()
            for key in keys
        }
        return [
            [{key: float(values[key][step][position]) for key in keys} for step in range(count)]
            for position, count in enumerate(counts)
        ]


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
