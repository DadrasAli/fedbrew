"""A batched run's rounds held on its device: the resident round.

``runtime.performance.executor: batched`` trains a round's clients together
(``fedbrew/core/batched_executor.py``) but builds the round around them on the
host: a plan object per client, the round's rows stacked again, the model
moved to the device and back, and every record made as the round runs. For a
run whose rounds allow it, this module keeps all of that on the device for
the whole run:

- **rows**: every client's training rows are stacked once, at the start. A
  bucket's rows are its clients' rows gathered from that stack -- the tensors
  the round's own stacking gives, padded to the bucket's longest split;
- **model**: the server's state is the round's mean, on the device, where the
  fold left it; the next round's clients start from it without a copy;
- **records**: each round's per-client outputs wait on the device until the
  flush, which reads them back in one copy and builds every record from them
  with the code the round itself would run, in the same order: each rule's
  ``batched_stacked_results``, the loop's observer, the server's metric sums;
- **plans**: a round's sampled clients and orders come from the run's
  planner (``fedbrew/core/round_planner.py``) and its program from the
  first client's rule, which every client shares; no per-client plan exists
  until the flush builds a client's record;
- **SCAFFOLD's controls**: every client's ``c_i`` is a row of one table on
  the device and ``c`` a state beside the model. A bucket's clients read
  their rows, and their new ones are written back as Option II computes
  them; the control deltas are summed in the clients' order, as the server
  sums them, into the new ``c``. Each round's trained states wait for the
  flush, where every client's result and ``c``'s update are built by the
  rule's and the server's own code (``batched_result``,
  ``ScaffoldServer.aggregate_folded``) -- the host's ``c_i`` and ``c`` are
  what a checkpoint of that round holds.

A round computes what the batched executor computes, bit for bit on the same
device: the same chunks and buckets from the same costs and structures, the
same kernels on the same tensors (``_Bucket``), the fold's weighted sums added
and divided on the device -- where float addition and a division by a device
tensor round as the CPU's do (measured 2026-09-28) -- or, in a round with a
bucket of one client, on the CPU exactly as the executor adds that client.
``tests/test_resident_round.py`` holds every CSV, checkpoint and model to the
per-round path's.

``resident_rounds_for`` decides, once, whether a run takes it, and records
why not (run.json ``executor.rounds``).
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from fedbrew.clients.batched_update import ProgramValues
from fedbrew.core.batched_executor import (
    _WORKING_SLOTS,
    BatchedExecutor,
    ChunkLayout,
    StagedLayout,
    _Bucket,
    _Rows,
    _Steps,
    cut_chunks,
    finished_chunk,
    finished_values,
    free_memory,
    staged_chunk,
    staged_values,
    trained_state_keys,
    uploaded,
)
from fedbrew.core.federated_state import model_state_size
from fedbrew.core.metrics import filter_metrics
from fedbrew.core.protocol import FitRequest, RoundInfo
from fedbrew.core.resident_flush import FlushWriter, HostCopy, RoundClock, WriterStaged
from fedbrew.core.resident_graphs import RoundGraphs
from fedbrew.core.round_planner import PlannedRound
from fedbrew.core.stacked_results import StackedFitResults
from fedbrew.core.torch_utils import StateStack
from fedbrew.servers.fedavg import WeightedMetricAccumulator

#: The rules whose stacked results the resident round builds: those whose
#: update has no per-client state or correction (no FedProx term, no SCAFFOLD
#: control).
RESIDENT_RULES = ("fedavg", "local_sgd", "local_adamw")

#: The rule whose per-client controls the resident round holds on the device.
SCAFFOLD_RULE = "scaffold"

#: The share of the device's free memory every client's rows may take.
ROWS_FRACTION = 0.25

#: How many chunks' record plans are kept; a run whose clients change every
#: round builds them again.
MAX_KEPT_CHUNKS = 8


# ---------------------------------------------------------------------------
# Whether a run takes it
# ---------------------------------------------------------------------------


def resident_rounds_for(context: Any) -> ResidentRounds | None:
    """The run's resident rounds, or None; the executor's record says which, and why not."""

    executor = context.executor
    if not isinstance(executor, BatchedExecutor):
        return None
    reason = resident_unsupported(context)
    rounds = None
    if reason is None:
        rounds = ResidentRounds(context)
        reason = rounds.unsupported
    executor.record["rounds"] = (
        {"used": "resident"} if reason is None else {"used": "per_round", "reason": reason}
    )
    if reason is not None and executor.cuda_graphs:
        executor.record["cuda_graphs"] = {
            "used": "off",
            "fallback": f"CUDA graphs replay resident rounds; this run's are per round: {reason}",
        }
    return rounds if reason is None else None


def resident_unsupported(context: Any) -> str | None:
    """Why this run's rounds cannot be held on the device, or None if they can."""

    executor = context.executor
    if executor.planner is None:
        return "the run's orders are not planned from its roster"
    for check in (_server_unsupported, _pipeline_unsupported, _rule_unsupported):
        reason = check(context)
        if reason is not None:
            return reason
    return None


def _server_unsupported(context: Any) -> str | None:
    from fedbrew.servers.fedavg import FedAvgServer
    from fedbrew.servers.scaffold import ScaffoldServer

    server = context.server
    cls = type(server)
    own = (
        "aggregate_stream",
        "_accumulate_fit_results",
        "_accumulate_stacks",
        "_federated_payload",
    )
    base = ScaffoldServer if _scaffold(context) else FedAvgServer
    if base is ScaffoldServer:
        own = (*own, "_aggregate", "aggregate_folded", "save_state", "load_state")
    if not isinstance(server, base) or any(
        getattr(cls, name) is not getattr(base, name) for name in own
    ):
        return f"server {cls.__name__} folds its results its own way"
    if not server._folds_stacks():
        return f"server {cls.__name__} checks or weighs a result its own way"
    return None


def _pipeline_unsupported(context: Any) -> str | None:
    from fedbrew.core.batched_evaluator import BatchedEvaluator
    from fedbrew.core.execution import StreamingAggregator

    if type(context.aggregator) is not StreamingAggregator:
        return f"aggregator {type(context.aggregator).__name__} is not the streaming one"
    if not isinstance(context.evaluator, BatchedEvaluator):
        return f"evaluator {type(context.evaluator).__name__} is not the batched one"
    if context.evaluation.model_scope != "global":
        return f"evaluation.model_scope is {context.evaluation.model_scope}"
    return None


def _scaffold(context: Any) -> bool:
    """Whether the run's clients are SCAFFOLD's, whose controls the round holds."""

    roster = context.executor.planner.roster
    return type(context.client[roster.client_ids[0]]).__dict__.get("_batched_rule") == SCAFFOLD_RULE


def _rule_unsupported(context: Any) -> str | None:
    roster = context.executor.planner.roster
    representative = context.client[roster.client_ids[0]]
    rule = type(representative).__dict__.get("_batched_rule")
    if rule == SCAFFOLD_RULE:
        return _scaffold_unsupported(representative)
    if rule not in RESIDENT_RULES:
        return f"update rule {type(representative).__name__} keeps per-client state"
    supported = getattr(representative, "batched_stacked_supported", None)
    if not callable(supported) or not supported():
        return f"update rule {type(representative).__name__} builds its results one by one"
    return None


def _scaffold_unsupported(representative: Any) -> str | None:
    """Why a SCAFFOLD client's results cannot be built from the resident round's, or None.

    They are built by the rule's own ``batched_result``, whose control update
    the round computes on the device with the same arithmetic.
    """

    from fedbrew.clients.torch_scaffold_client import TorchScaffoldClient

    cls = type(representative)
    own = ("batched_program", "batched_result", "_update_client_control", "_scaffold_result")
    if not isinstance(representative, TorchScaffoldClient) or any(
        getattr(cls, name) is not getattr(TorchScaffoldClient, name) for name in own
    ):
        return f"update rule {cls.__name__} updates its controls its own way"
    return None


# ---------------------------------------------------------------------------
# The rows
# ---------------------------------------------------------------------------


class ResidentRows:
    """Clients' rows of one split, stacked once for the run and padded to the longest.

    ``splits[place]`` is roster client ``place``'s split, or None where it has
    none. ``tensors[k][row, :lengths[row]]`` is a client's rows of the task's
    tensor ``k``, as ``split_rows`` gives them, at the row ``index[place]``;
    the padding is zeros, as ``pad_sequence``'s is.
    """

    def __init__(self, task: Any, splits: Sequence[Any]) -> None:
        self.index = {
            place: row
            for row, place in enumerate(
                place for place, split in enumerate(splits) if split is not None
            )
        }
        rows = [task.split_rows(split) for split in splits if split is not None]
        self.lengths = [int(len(split_rows[0])) for split_rows in rows]
        self.longest = max(self.lengths)
        self.tensors = tuple(
            torch.nn.utils.rnn.pad_sequence(list(parts), batch_first=True)
            for parts in zip(*rows, strict=True)
        )
        self.bytes = sum(tensor.numel() * tensor.element_size() for tensor in self.tensors)
        self.row_bytes = sum(
            tensor[0, :1].numel() * tensor.element_size() for tensor in self.tensors
        )
        self._everyone = list(range(len(self.lengths)))
        self._kept: dict[tuple[int, ...], _Rows] = {}
        self._fresh: dict[tuple[int, ...], _Rows] = {}

    def bucket(self, places: list[int]) -> _Rows:
        """The rows of these roster clients, as ``_Rows`` of their splits holds them.

        One client's are its own rows, unpadded; several clients' are padded to
        the longest of them and stacked. Kept for the next round when the same
        clients come back.
        """

        key = tuple(self.index[place] for place in places)
        rows = self._kept.get(key)
        if rows is None:
            rows = self._rows(list(key))
        self._fresh[key] = rows
        return rows

    def round_done(self) -> None:
        """Keep what this round used, and nothing else, for the next."""

        self._kept, self._fresh = self._fresh, {}

    def single(self, row: int) -> _Rows:
        """One client's own rows, unpadded, as ``_Rows`` of one split holds them."""

        return self._rows([row])

    def everyone(self) -> _Rows:
        """Every client's rows, stacked, as ``_Rows`` of every split holds them."""

        rows = self._kept.get(("everyone",))
        if rows is None:
            rows = self._kept[("everyone",)] = self._rows(self._everyone)
        return rows

    def gathered(self, index: Tensor, longest: int) -> _Rows:
        """The rows at the stacked rows ``index`` (on the device), padded to ``longest``."""

        rows: _Rows = object.__new__(_Rows)
        rows.sources, rows.versions, rows.lengths = [], [], []
        rows.longest, rows.stacked = longest, True
        rows.tensors = tuple(
            tensor.index_select(0, index)[:, :longest].contiguous() for tensor in self.tensors
        )
        rows.device = rows.tensors[0].device
        rows.bytes = sum(tensor.numel() * tensor.element_size() for tensor in rows.tensors)
        rows._cast = {}
        return rows

    def _rows(self, places: list[int]) -> _Rows:
        """``_Rows`` of these stacked rows: one's own, or several padded to their longest."""

        rows: _Rows = object.__new__(_Rows)
        rows.sources = []
        rows.versions = []
        rows.lengths = [self.lengths[place] for place in places]
        rows.longest = max(rows.lengths)
        rows.stacked = len(places) > 1
        if not rows.stacked:
            place, length = places[0], rows.lengths[0]
            rows.tensors = tuple(tensor[place, :length] for tensor in self.tensors)
        elif places == self._everyone and rows.longest == self.longest:
            rows.tensors = self.tensors
        else:
            index = uploaded(torch.tensor(places, dtype=torch.long), self.tensors[0].device)
            rows.tensors = tuple(
                tensor.index_select(0, index)[:, : rows.longest].contiguous()
                for tensor in self.tensors
            )
        rows.device = rows.tensors[0].device
        rows.bytes = sum(tensor.numel() * tensor.element_size() for tensor in rows.tensors)
        rows._cast = {}
        return rows


# ---------------------------------------------------------------------------
# One round, on the device
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _MemberPlan:
    """The fields of a client's ``ClientBatchPlan`` a bucket and its rule's records read."""

    program: Any
    structure: tuple[int, ...]
    slot: int
    start: Mapping[str, Tensor]
    evaluate: bool
    eval_rows: int
    train_data: Any
    client_control: Mapping[str, Tensor] | None = None
    server_control: Mapping[str, Tensor] | None = None


@dataclass(slots=True)
class ChunkRun:
    """One chunk of a round: its clients (positions in the round), and its staged records."""

    start: int
    stop: int
    layout: ChunkLayout
    parts: list[Any]
    columns: list[Tensor]
    seconds: float = 0.0


@dataclass(slots=True)
class _RowsShape:
    """What a bucket's ``_Steps`` read of its rows while the round is planned: their shape."""

    longest: int
    stacked: bool
    device: torch.device
    tensors: tuple[Tensor, ...] = ()


@dataclass(slots=True)
class _BucketPlan:
    """One bucket's host half: its clients, its rows, its steps, and where its inputs are."""

    chunk: int
    members: list[int]
    slots: list[int]
    rows_index: list[int]
    structure: tuple[int, ...]
    kind: str
    longest: int
    train: Any
    evaluation: Any
    inputs: dict[Any, int] = field(default_factory=dict)
    #: The clients' roster places, in the order of the bucket's rows.
    places: list[int] = field(default_factory=list)


@dataclass(slots=True)
class RoundPlan:
    """A round's host half: its chunks and buckets, and the tensors its device half reads."""

    planned: PlannedRound
    chunks: list[tuple[int, int]]
    host_fold: bool
    weights: list[float]
    total_weight: float
    buckets: list[_BucketPlan] = field(default_factory=list)
    uploads: list[Tensor] = field(default_factory=list)
    inputs: dict[Any, int] = field(default_factory=dict)
    key: Any = None
    #: SCAFFOLD's fold, whose server is handed its clients' results one by
    #: one: per run of consecutive clients in one bucket, the bucket's index
    #: and the rows, in order (``_Fold.add_runs``); None for a fold by bucket.
    runs: list[tuple[int, list[int]]] | None = None

    def add(self, tensor: Tensor) -> int:
        """Another tensor for the device half to read; its place among the inputs."""

        self.uploads.append(tensor)
        return len(self.uploads) - 1


def _steps_key(steps: Any) -> tuple[Any, ...]:
    """What of a bucket's steps its kernels depend on: their shape, and how each batch is cut."""

    return (
        steps.size,
        tuple(steps._indices.shape),
        tuple(steps.widths),
        tuple(steps.full),
        tuple(steps.aligned),
        steps.sliced,
        tuple(steps.first_starts) if steps.sliced else (),
    )


@dataclass(slots=True)
class DeviceRound:
    """One round's work, enqueued: what the flush reads back and records from."""

    round_id: int
    positions: list[int]
    program: Any
    evaluate: bool
    structures: list[tuple[int, ...]]
    eval_rows: list[int]
    chunks: list[ChunkRun] = field(default_factory=list)
    #: The model after the round, where the fold left it; None for a round
    #: with no clients, whose model is the one before it.
    mean: dict[str, Tensor] | None = None
    finite: Tensor | None = None
    staged: Tensor | None = None
    staged_layouts: list[StagedLayout] = field(default_factory=list)
    #: The clients due an evaluation this round, as (roster place, splits) in
    #: the evaluator's order, and what was measured of them on the device.
    work: list[tuple[int, list[str]]] = field(default_factory=list)
    eval_stage: Any = None
    central_due: bool = False
    central_stage: Any = None
    #: Whether ``grad_norm_sq`` is due, and its value on the device, staged
    #: after the central pass's (``evaluation.grad_norm``).
    grad_norm_due: bool = False
    grad_norm_stage: Tensor | None = None
    #: The staged values' sizes: the chunks', the evaluation's, the central
    #: pass's, the gradient norm's, and 1 for the aggregate's finiteness flag,
    #: when there is one.
    sizes: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
    #: The phase boundaries on the device's timeline, and the event the
    #: flush's copy waits for.
    clock: Any = None
    end: Any = None
    #: SCAFFOLD's trained states, per bucket its round positions and its
    #: stack, for the flush to build each client's result from.
    trained: list[tuple[list[int], dict[str, Tensor]]] = field(default_factory=list)


class ResidentRounds:
    """A run's rounds, trained on its device from its planner's orders.

    Built once, before the first round: the roster's rows, the first client's
    rule (every client's shares its configuration), and the model the server
    holds. ``unsupported`` says why the run cannot take it after all -- rows
    too large for the device, or a client pool that could evict a client.
    """

    def __init__(self, context: Any) -> None:
        self.context = context
        executor: BatchedExecutor = context.executor
        self.executor = executor
        self.planner = executor.planner
        self.roster = self.planner.roster
        self.representative = context.client[self.roster.client_ids[0]]
        self.task = self.representative.task
        self.template = self.task.build_model(self.representative.model_config)
        self.state_keys = trained_state_keys(self.task, self.template)
        self.parameters = dict(self.template.named_parameters())
        self.buffers = dict(self.template.named_buffers())
        self.device = next(iter(self.parameters.values())).device
        self.metadata = self.task.federated_model_state_metadata(self.template)
        self.trainable = sum(
            int(parameter.numel())
            for parameter in self.template.parameters()
            if parameter.requires_grad
        )
        self._row_size = model_state_size(self.task.get_federated_model_state(self.template))
        self.splits = [
            _train_split(context.dataset.get_client_data(client))
            for client in self.roster.client_ids
        ]
        #: SCAFFOLD's: every client's ``c_i`` as a row of one table, filled
        #: from its rule the round it first trains, and ``c``.
        self.scaffold = _scaffold(context)
        self.controls: dict[str, Tensor] = {}
        self._controls_known = [False] * len(self.roster)
        self.server_control: dict[str, Tensor] | None = None
        self.unsupported = self._memory_unsupported() or _pool_unsupported(
            context.client, context.dataset
        )
        self.rows: ResidentRows | None = None
        self.members: list[Any] = [None] * len(self.roster)
        self._values: dict[tuple[Any, ...], ProgramValues] = {}
        if self.unsupported is None:
            self.rows = ResidentRows(self.task, self.splits)
        self.model = self._placed(context.server._model_state)
        if self.scaffold and self.unsupported is None and context.server._server_control is None:
            self.unsupported = "the SCAFFOLD server's control variate is not initialized"
        if self.scaffold and self.unsupported is None:
            self.server_control = self._placed(context.server._server_control)
            self.controls = {
                name: torch.zeros(
                    (len(self.roster), *parameter.shape),
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
                for name, parameter in self.parameters.items()
            }
        if self.unsupported is None:
            self._check_broadcast(context.server)
        self.copier = HostCopy(self.device)
        self._client_states: tuple[Any, dict[str, Any] | None] | None = None
        #: The server's model before the round the flush records, on the host.
        self._host_start: dict[str, Tensor] = {}
        #: Per chunk of clients, their rules and record plans (``_record_plans``).
        self._records: dict[tuple[int, ...], tuple[list[Any], list[_MemberPlan]]] = {}
        self.graphs = RoundGraphs(self.device, executor.cuda_graphs, executor.record)
        #: The dtypes the fold sums the model's tensors in, in order of first use.
        self.accumulation_dtypes = list(
            dict.fromkeys(
                _accumulation_dtype(self.parameters[key].dtype) for key in self.state_keys
            )
        )
        from fedbrew.core.resident_evaluation import ResidentEvaluation

        self.evaluation = ResidentEvaluation(self) if self.unsupported is None else None

    # -- set-up --------------------------------------------------------------

    def _memory_unsupported(self) -> str | None:
        """Why every client's rows cannot stay on the device, or None."""

        rows = sum(_split_bytes(split) for split in self.splits)
        free = free_memory(self.device)
        if free is not None and rows > ROWS_FRACTION * free:
            return (
                f"every client's rows ({rows} bytes) exceed a quarter of the device's "
                f"free memory ({free} bytes)"
            )
        if self.scaffold and free is not None:
            # The control table, and a flush window's trained states at most.
            model = sum(p.numel() * p.element_size() for p in self.parameters.values())
            window = min(self.context.flush_every, self.context.global_rounds)
            controls = model * len(self.roster) * (1 + window)
            if rows + controls > ROWS_FRACTION * free:
                return (
                    f"every client's rows and SCAFFOLD controls, with a flush window's "
                    f"trained states ({rows + controls} bytes), exceed a quarter of the "
                    f"device's free memory ({free} bytes)"
                )
        return None

    def _placed(self, state: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """A server state on the model's device and in its dtypes, as a received state is placed."""

        return {
            name: state[name].detach().to(device=parameter.device, dtype=parameter.dtype)
            for name, parameter in self.parameters.items()
        }

    def _check_broadcast(self, server: Any) -> None:
        """The rule's own check of what the server broadcasts (``batched_start``), once.

        The per-round path makes it for each round's first client, so a rule
        that cannot train this model -- an adapter-scoped one under a
        full-state rule -- refuses before the first update; the resident
        round's broadcast is its server's model throughout, and SCAFFOLD's
        control, which is what ``configure_round`` hands each client.
        """

        payload = server._federated_payload()
        if self.scaffold:
            payload["server_control"] = server._server_control
        self.representative.batched_start(
            FitRequest(round_id=1, client_id=self.representative.client_id, payload=payload),
            self.template,
        )

    # -- a round -------------------------------------------------------------

    def run_round(self, round_id: int, evaluate: bool) -> DeviceRound:
        """Train round ``round_id`` from the model the last round left, and fold it."""

        clock = RoundClock(self.device)
        clock.mark("start")
        planned = self.planner.plan(round_id)
        self._refuse_empty(planned)
        program = self.representative.batched_program(
            FitRequest(round_id=round_id, client_id=self.representative.client_id, payload={})
        )
        eval_rows = planned.evaluation.lengths.sum(dim=1).tolist()
        device_round = DeviceRound(
            round_id,
            planned.positions,
            program,
            evaluate,
            list(planned.train.structure),
            eval_rows,
        )
        device_round.clock = clock
        if planned.positions:
            self._train(planned, device_round)
        clock.mark("trained")
        self._evaluate(device_round)
        self._stage(device_round)
        device_round.end = self.copier.mark()
        return device_round

    def _evaluate(self, device_round: DeviceRound) -> None:
        """The round's due evaluations, measured on the device at the model it leaves."""

        from fedbrew.core.config import evaluates_round
        from fedbrew.core.loop import _round_evaluation_plan

        context, round_id = self.context, device_round.round_id
        selected = [self.roster.client_ids[place] for place in device_round.positions]
        splits_by_client, _ = _round_evaluation_plan(
            context.evaluation_clients, context.schedules, round_id, context.global_rounds, selected
        )
        device_round.work = [
            (self.roster.where[client], splits) for client, splits in splits_by_client.items()
        ]
        device_round.eval_stage = self.evaluation.enqueue(round_id, device_round.work, self.model)
        device_round.clock.mark("clients")
        device_round.central_due = evaluates_round(
            context.central_schedule, round_id, context.global_rounds
        )
        if device_round.central_due:
            device_round.central_stage = self.evaluation.enqueue_central(self.model)
        if context.grad_norm_schedule is not None and evaluates_round(
            context.grad_norm_schedule, round_id, context.global_rounds
        ):
            device_round.grad_norm_due = True
            device_round.grad_norm_stage = self.evaluation.enqueue_grad_norm(self.model)
        device_round.clock.mark("evaluated")

    def _refuse_empty(self, planned: PlannedRound) -> None:
        """Refuse the first client, in request order, whose loader yields no batch."""

        steps = planned.train.steps.tolist()
        for slot, place in enumerate(planned.positions):
            if not steps[slot]:
                member = self.member(place)
                raise member.batched_plan(
                    FitRequest(round_id=planned.round_id, client_id=member.client_id, payload={}),
                    self.template,
                    start=self.model,
                ).refuse()

    def _train(self, planned: PlannedRound, device_round: DeviceRound) -> None:
        """Every chunk's buckets trained and folded, in order; the round's staged records.

        The round's host half is computed first (``_prepare``); its device half
        (``_execute``) reads nothing but the tensors that computes, so it runs
        eagerly or as the graph of its shape (``RoundGraphs``) alike.
        """

        plan = self._prepare(planned, device_round)
        if self.scaffold:
            self._fill_controls(planned.positions)
        for start, stop in plan.chunks:
            self.executor.record["largest_chunk_clients"] = max(
                self.executor.record["largest_chunk_clients"], stop - start
            )
        outputs, graphed = self.graphs.run(
            plan.key,
            plan.uploads,
            self.model,
            lambda inputs, model: self._execute(plan, device_round, inputs, model),
        )
        results, mean, finite = outputs
        for index, (start, stop) in enumerate(plan.chunks):
            parts, columns, layout = staged_chunk(
                self.task,
                [
                    (members, training, evaluated)
                    for chunk, members, training, evaluated in results
                    if chunk == index
                ],
            )
            device_round.chunks.append(ChunkRun(start, stop, layout, parts, columns))
        if graphed:
            # The graph's own outputs are what its next replay writes over.
            mean = {name: value.clone() for name, value in mean.items()}
            finite = finite.clone()
        device_round.mean, device_round.finite = mean, finite
        self.model = mean

    def _prepare(self, planned: PlannedRound, device_round: DeviceRound) -> RoundPlan:
        """The round's host half: its chunks and buckets, and every tensor its device half reads."""

        program = device_round.program
        chunks = list(
            cut_chunks(self._costs(planned, device_round, program), self.executor.chunk_bytes)
        )
        server = self.context.server
        if server.aggregation_weighting == "uniform":
            weights = [1.0] * len(planned.positions)
        else:
            weights = [float(rows) for rows in device_round.eval_rows]
        total_weight = 0.0
        for weight in weights:
            total_weight += float(weight)
        runs = _runs(planned, chunks) if self.scaffold else None
        plan = RoundPlan(
            planned=planned,
            chunks=chunks,
            host_fold=(
                any(len(rows) == 1 for _, rows in runs)
                if runs is not None
                else self._has_single_bucket(planned, chunks)
            ),
            weights=weights,
            total_weight=total_weight,
        )
        for index, (start, stop) in enumerate(chunks):
            groups: dict[tuple[int, ...], list[int]] = {}
            for slot in range(start, stop):
                groups.setdefault(device_round.structures[slot], []).append(slot - start)
            for structure, members in groups.items():
                plan.buckets.append(self._bucket_plan(plan, index, start, structure, members))
        if runs is not None:
            plan.runs = _bucket_runs(plan.buckets, runs)
        if not plan.host_fold:
            for dtype in self.accumulation_dtypes:
                plan.inputs[("divisor", dtype)] = plan.add(torch.tensor(total_weight, dtype=dtype))
        plan.key = self._key(plan, device_round)
        return plan

    def _bucket_plan(
        self,
        plan: RoundPlan,
        chunk: int,
        start: int,
        structure: tuple[int, ...],
        members: list[int],
    ) -> _BucketPlan:
        """One bucket's host half: its clients, its rows' shape, its steps and their indices."""

        rows = self.rows
        assert rows is not None
        slots = [start + member for member in members]
        places = [plan.planned.positions[slot] for slot in slots]
        index = [rows.index[place] for place in places]
        longest = max(rows.lengths[row] for row in index)
        size = len(slots)
        if size == 1:
            kind = "single"
        elif index == rows._everyone and longest == rows.longest:
            kind = "everyone"
        else:
            kind = "gathered"
        shape = _RowsShape(longest, size > 1, self.device)
        context = self.executor.context
        dtype = context.train_dtype(self.template_dtype) if context else self.template_dtype
        bucket = _BucketPlan(
            chunk=chunk,
            members=members,
            slots=slots,
            rows_index=index,
            structure=structure,
            kind=kind,
            longest=longest,
            train=_Steps(shape, plan.planned.train, slots, dtype),  # type: ignore[arg-type]
            evaluation=_Steps(shape, plan.planned.evaluation, slots, self.template_dtype),  # type: ignore[arg-type]
            places=places,
        )
        if self.scaffold:
            bucket.inputs["places"] = plan.add(torch.tensor(places, dtype=torch.long))
        if kind == "gathered":
            bucket.inputs["index"] = plan.add(torch.tensor(index, dtype=torch.long))
        if size > 1:
            offsets = torch.arange(size, dtype=torch.long).view(-1, 1, 1) * longest
            for name, steps in (("train", bucket.train), ("eval", bucket.evaluation)):
                # What ``_Steps._device`` computes and uploads, computed here.
                bucket.inputs[f"{name}_flat"] = plan.add(steps._indices + offsets)
                bucket.inputs[f"{name}_lengths"] = plan.add(steps._lengths)
            if not plan.host_fold:
                for dtype_ in self.accumulation_dtypes:
                    scale = torch.tensor([plan.weights[slot] for slot in slots], dtype=dtype_)
                    bucket.inputs[("scale", dtype_)] = plan.add(scale)
        return bucket

    def _key(self, plan: RoundPlan, device_round: DeviceRound) -> Any:
        """What makes two rounds the same graph: None for a round only eager runs.

        A round folded on the CPU, one with a bucket of one client, one
        whose update combines a pass's gradients (its weights uploaded as it
        steps), SCAFFOLD's, whose controls live outside the round's inputs,
        and a compiled one (``runtime.performance.compile``), which is its
        own graph, run eagerly.
        """

        program = device_round.program
        if plan.host_fold or program.combine != "batch" or self.scaffold:
            return None
        context = self.executor.context
        if context is not None and context.compiling:
            return None
        if any(bucket.kind == "single" for bucket in plan.buckets):
            return None
        return (
            repr(program),
            device_round.evaluate,
            tuple(
                (
                    bucket.chunk,
                    tuple(bucket.members),
                    bucket.structure,
                    bucket.kind,
                    bucket.longest,
                    _steps_key(bucket.train),
                    _steps_key(bucket.evaluation),
                )
                for bucket in plan.buckets
            ),
        )

    def _execute(
        self,
        plan: RoundPlan,
        device_round: DeviceRound,
        inputs: Sequence[Tensor],
        model: dict[str, Tensor],
    ) -> tuple[list[tuple[int, list[int], Any, Any]], dict[str, Tensor], Tensor]:
        """The round's device half: every bucket trained and folded, from ``inputs`` and ``model``.

        Reads no host tensor but the round's inputs, and waits for nothing, so
        a graph can record it (``RoundGraphs``).
        """

        fold = _Fold(self, plan, inputs)
        results = []
        trained = []
        for bucket in plan.buckets:
            rows = self._rows_for(bucket, inputs)
            context = self.executor.context
            dtype = context.train_dtype(self.template_dtype) if context else self.template_dtype
            bucket.train.rows = rows.as_dtype(dtype)
            bucket.evaluation.rows = rows
            # What a step gathers lazily is this call's: a capture that failed
            # left tensors it recorded but never computed.
            bucket.train._every = bucket.evaluation._every = None
            bucket.train._on_device = bucket.evaluation._on_device = None
            bucket.train._repeated.clear()
            bucket.evaluation._repeated.clear()
            if bucket.kind != "single":
                bucket.train._on_device = (
                    inputs[bucket.inputs["train_flat"]],
                    inputs[bucket.inputs["train_lengths"]],
                )
                bucket.evaluation._on_device = (
                    inputs[bucket.inputs["eval_flat"]],
                    inputs[bucket.inputs["eval_lengths"]],
                )
            members = [
                _MemberPlan(
                    program=device_round.program,
                    structure=bucket.structure,
                    slot=slot,
                    start=model,
                    evaluate=device_round.evaluate,
                    eval_rows=device_round.eval_rows[slot],
                    train_data=None,
                )
                for slot in bucket.slots
            ]
            if self.scaffold:
                places = inputs[bucket.inputs["places"]]
                old = {name: table.index_select(0, places) for name, table in self.controls.items()}
                for row, member in enumerate(members):
                    member.client_control = {name: value[row] for name, value in old.items()}
                    member.server_control = self.server_control
            stack, training, evaluated = _Bucket(
                self.task,
                self.template,
                self.buffers,
                members,  # type: ignore[arg-type]
                rows,
                (plan.planned.train, plan.planned.evaluation),
                context=self.executor.context,
                values=self._program_values(
                    device_round.program, len(bucket.slots), len(bucket.structure)
                ),
                steps=(bucket.train, bucket.evaluation),
            ).run()
            if self.scaffold:
                deltas = self._new_controls(bucket, old, stack, model, device_round.program)
                trained.append((bucket, stack, deltas))
            else:
                fold.add(bucket, stack)
            results.append((bucket.chunk, bucket.members, training, evaluated))
        if self.scaffold:
            fold.add_runs([stack for _, stack, _ in trained])
            device_round.trained = [(bucket.slots, stack) for bucket, stack, _ in trained]
        mean, finite = fold.result()
        if self.scaffold:
            finite = torch.stack([finite, self._fold_controls(trained)]).all()
        return results, mean, finite

    # -- SCAFFOLD's controls ----------------------------------------------------

    def _fill_controls(self, positions: Sequence[int]) -> None:
        """The table's rows of the clients training for the first time: their rules' ``c_i``.

        A client's ``c_i`` changes only where the table holds it from then on;
        its rule's copy is brought up to date as each round is recorded.
        """

        for place in positions:
            if self._controls_known[place]:
                continue
            self._controls_known[place] = True
            control = self.member(place)._client_control
            if control is None:
                continue
            for name, table in self.controls.items():
                table[place].copy_(control[name].detach())

    def _new_controls(
        self,
        bucket: _BucketPlan,
        old: Mapping[str, Tensor],
        stack: Mapping[str, Tensor],
        start: Mapping[str, Tensor],
        program: Any,
    ) -> dict[str, Tensor]:
        """Option II's new ``c_i`` of a bucket's clients, written to the table; their deltas.

        ``_update_client_control``'s arithmetic, a client per row:
        ``c_i - c + (x - y_i) / (K * learning_rate)``, and the new minus the old.
        """

        scale = 1.0 / (len(bucket.structure) * program.optimizer.lr)
        server = self.server_control
        assert server is not None
        index = uploaded(torch.tensor(bucket.places, dtype=torch.long), self.device)
        deltas = {}
        for name, table in self.controls.items():
            new = (old[name] - server[name]) + (start[name] - stack[name]) * scale
            deltas[name] = new - old[name]
            table.index_copy_(0, index, new)
        return deltas

    def _fold_controls(self, trained: list[tuple[_BucketPlan, Any, dict[str, Tensor]]]) -> Tensor:
        """``c``'s update from the round's control deltas; whether all of it is finite.

        The deltas are summed a client at a time, in the round's order, from
        zeros, and ``c + sum / N`` taken, as ``ScaffoldServer`` computes them;
        what it refuses -- a delta, their sum or the new ``c`` not finite --
        is what the flag says.
        """

        server = self.server_control
        assert server is not None
        rows: dict[int, tuple[dict[str, Tensor], int]] = {}
        for bucket, _, deltas in trained:
            for row, slot in enumerate(bucket.slots):
                rows[slot] = (deltas, row)
        summed = {name: torch.zeros_like(value) for name, value in server.items()}
        for slot in range(len(rows)):
            deltas, row = rows[slot]
            summed = {name: summed[name] + deltas[name][row] for name in summed}
        scale = 1.0 / len(self.context.client_infos)
        updated = {name: server[name] + summed[name] * scale for name in server}
        checks = [torch.isfinite(deltas[name]).all() for _, _, deltas in trained for name in deltas]
        checks += [torch.isfinite(value).all() for value in (*summed.values(), *updated.values())]
        self.server_control = updated
        return torch.stack(checks).all()

    def _rows_for(self, bucket: _BucketPlan, inputs: Sequence[Tensor]) -> _Rows:
        rows = self.rows
        assert rows is not None
        if bucket.kind == "single":
            return rows.single(bucket.rows_index[0])
        if bucket.kind == "everyone":
            return rows.everyone()
        return rows.gathered(inputs[bucket.inputs["index"]], bucket.longest)

    def _costs(self, planned: PlannedRound, device_round: DeviceRound, program: Any) -> list[int]:
        """``client_costs`` of the round's clients, from the resident rows' sizes."""

        parameter_bytes = sum(p.numel() * p.element_size() for p in self.template.parameters())
        lengths = planned.train.lengths
        longest = (
            lengths.amax(dim=1) if lengths.shape[1] else torch.zeros(len(planned.positions))
        ).tolist()
        slots = _WORKING_SLOTS + program.optimizer.state_slots + int(program.scaffold)
        row_bytes = self.rows.row_bytes  # type: ignore[union-attr]
        return [
            parameter_bytes * slots + row_bytes * (rows + 2 * int(widest))
            for rows, widest in zip(device_round.eval_rows, longest, strict=True)
        ]

    def _has_single_bucket(self, planned: PlannedRound, chunks: list[tuple[int, int]]) -> bool:
        """Whether some chunk has a bucket of one client, whose fold is the CPU's."""

        structures = planned.train.structure
        for start, stop in chunks:
            counts: dict[tuple[int, ...], int] = {}
            for slot in range(start, stop):
                counts[structures[slot]] = counts.get(structures[slot], 0) + 1
            if 1 in counts.values():
                return True
        return False

    def _program_values(self, program: Any, size: int, steps: int) -> ProgramValues:
        """``ProgramValues`` of ``size`` clients sharing ``program``, made once per shape."""

        context = self.executor.context
        dtype = context.train_dtype(self.template_dtype) if context else self.template_dtype
        key = (repr(program), size, steps, dtype)
        values = self._values.get(key)
        if values is None:
            # Compiled, a step's own values are read per step, as the bucket reads them.
            per_step = bool(context and context.compiling) and size > 1
            values = self._values[key] = ProgramValues(
                [program] * size, steps, dtype, self.device, per_step=per_step
            )
        return values

    @property
    def template_dtype(self) -> torch.dtype:
        return next(iter(self.parameters.values())).dtype

    def _stage(self, device_round: DeviceRound) -> None:
        """Join the round's staged values in one float64 tensor, for the flush's copy.

        The chunks' first, then the evaluation's, the central pass's and the
        gradient norm's.
        """

        pieces, layouts = [], []
        for chunk in device_round.chunks:
            staged, layout = staged_values(chunk.parts, chunk.columns)
            layouts.append(layout)
            if staged is not None:
                pieces.append(staged)
            chunk.parts, chunk.columns = [], []
        fit_size = sum(int(piece.numel()) for piece in pieces)
        extra = []
        for stage in (device_round.eval_stage, device_round.central_stage):
            staged = None if stage is None else stage.staged
            extra.append(0 if staged is None else int(staged.numel()))
            if staged is not None:
                pieces.append(staged)
                stage.staged = None
        grad_norm = device_round.grad_norm_stage
        extra.append(0 if grad_norm is None else 1)
        if grad_norm is not None:
            pieces.append(grad_norm.reshape(1))
            device_round.grad_norm_stage = None
        flag = 0
        if device_round.finite is not None:
            pieces.append(device_round.finite.to(torch.float64).reshape(1))
            flag = 1
        device_round.staged_layouts = layouts
        device_round.sizes = (fit_size, extra[0], extra[1], extra[2], flag)
        device_round.staged = torch.cat(pieces) if pieces else None

    # -- the flush's side ----------------------------------------------------

    def member(self, place: int) -> Any:
        """Roster client ``place``'s rule, taken from the pool the first time the round would."""

        member = self.members[place]
        if member is None:
            member = self.members[place] = self.context.client[self.roster.client_ids[place]]
        return member

    def read_back(
        self, rounds: Sequence[DeviceRound]
    ) -> list[tuple[list[float], dict[str, Tensor] | None]]:
        """Every round's staged values and mean on the host, in one copy with one wait.

        The copy waits for the window's last round alone, not for a round
        queued after it (``HostCopy``).
        """

        tensors: list[Tensor] = []
        for device_round in rounds:
            if device_round.staged is not None:
                tensors.append(device_round.staged)
            if device_round.mean is not None:
                tensors.extend(device_round.mean.values())
            for _, stack in device_round.trained:
                tensors.extend(stack.values())
        copies = iter(self.copier.copy(tensors, rounds[-1].end))
        read: list[tuple[list[float], dict[str, Tensor] | None]] = []
        for device_round in rounds:
            values = next(copies).tolist() if device_round.staged is not None else []
            mean = None
            if device_round.mean is not None:
                mean = {name: next(copies) for name in device_round.mean}
            # SCAFFOLD's trained states, now the host's.
            device_round.trained = [
                (slots, {name: next(copies) for name in stack})
                for slots, stack in device_round.trained
            ]
            read.append((values, mean))
        return read

    def stacked_results(
        self, device_round: DeviceRound, values: list[float]
    ) -> list[tuple[StackedFitResults, float]]:
        """The round's stacked results a chunk at a time, built by the rule, with their seconds."""

        results = []
        offset = 0
        for chunk, layout in zip(device_round.chunks, device_round.staged_layouts, strict=True):
            size = _staged_size(layout)
            groups, lists = finished_values(values[offset : offset + size], layout)
            offset += size
            fit = finished_chunk(
                groups,
                lists,
                chunk.layout,
                [self._stand_in(len(members)) for members, *_ in chunk.layout],
                dict(self.metadata),
                self.trainable,
            )
            members, plans = self._record_plans(device_round, chunk.start, chunk.stop)
            built = time.perf_counter()
            stacked = members[0].batched_stacked_results(
                _Requests(device_round.round_id, chunk.stop - chunk.start), plans, fit, members
            )
            results.append((stacked, chunk.seconds + time.perf_counter() - built))
        return results

    def _record_plans(
        self, device_round: DeviceRound, start: int, stop: int
    ) -> tuple[list[Any], list[_MemberPlan]]:
        """A chunk's clients' rules and the plan fields their records read, kept per chunk.

        The same clients in the same places have the same rules, structures,
        rows and splits every round; only the program and whether the post-fit
        pass ran are the round's, and those are set on the kept plans.
        """

        places = tuple(device_round.positions[start:stop])
        held = self._records.get(places)
        if held is None:
            members = [self.member(place) for place in places]
            plans = [
                _MemberPlan(
                    program=device_round.program,
                    structure=device_round.structures[slot],
                    slot=slot,
                    start={},
                    evaluate=device_round.evaluate,
                    eval_rows=device_round.eval_rows[slot],
                    train_data=self.splits[device_round.positions[slot]],
                )
                for slot in range(start, stop)
            ]
            held = (members, plans)
            if len(self._records) < MAX_KEPT_CHUNKS:
                self._records[places] = held
        members, plans = held
        if plans[0].program != device_round.program or plans[0].evaluate != device_round.evaluate:
            for plan in plans:
                plan.program, plan.evaluate = device_round.program, device_round.evaluate
        for plan, slot in zip(plans, range(start, stop), strict=True):
            # A client's structure and rows are its own every round; checked, not assumed.
            if (
                plan.structure != device_round.structures[slot]
                or plan.eval_rows != (device_round.eval_rows[slot])
            ):
                raise RuntimeError(f"round {device_round.round_id}: client {slot}'s plan changed")
        return members, plans

    def _stand_in(self, size: int) -> StateStack:
        """A bucket's state stack as its records read it: its row size, and a row per client.

        The records never read a trained state: the model is the round's mean,
        already folded. A row here is the template's parameters, seen ``size``
        times, for an observer that is handed each client's result.
        """

        stack = StateStack(
            {
                key: self.parameters[key].detach()[None].expand(size, *self.parameters[key].shape)
                for key in self.state_keys
            }
        )
        stack.row_size = self._row_size
        return stack

    def aggregate(
        self,
        device_round: DeviceRound,
        values: list[float],
        round_info: RoundInfo,
        observer: Any,
        host_mean: dict[str, Tensor],
        start: dict[str, Tensor] | None = None,
    ) -> dict[str, Any]:
        """What ``aggregate_stacked`` does with the round, from its staged values and its mean.

        ``start`` is the model before the round, on the host: SCAFFOLD's
        control update reads it.

        The observer is handed each chunk's stacked results as the executor
        hands them; the server's metrics are summed as ``_accumulate_stacks``
        sums them; its state becomes the round's mean.
        """

        if self.scaffold:
            assert start is not None
            self._host_start = start
            return self._scaffold_aggregate(device_round, values, round_info, observer, host_mean)
        server = self.context.server
        metric_accumulator = WeightedMetricAccumulator()
        done, total = 0, len(device_round.positions)
        report = getattr(observer, "fitted_stack", None)
        for stacked, seconds in self.stacked_results(device_round, values):
            if callable(report):
                done += len(stacked)
                report(stacked, seconds, done, total)
            else:
                for result in stacked.results():
                    done += 1
                    observer.fitted(result, seconds / len(stacked), done, total)
            if not len(stacked):
                continue
            columns, reported = stacked.metric_columns()
            metric_accumulator.add_columns(columns, stacked.counts(), reported)
        server._model_state = host_mean
        metrics = filter_metrics(metric_accumulator.result(), server.metrics)
        round_info.metrics.update(metrics)
        return server._federated_payload(metrics=metrics)

    def _scaffold_aggregate(
        self,
        device_round: DeviceRound,
        values: list[float],
        round_info: RoundInfo,
        observer: Any,
        host_mean: dict[str, Tensor],
    ) -> dict[str, Any]:
        """SCAFFOLD's round as the per-round path aggregates it, from the round's trained states.

        Each client's result is its rule's ``batched_result`` -- the control
        update, kept in the rule, and the metrics -- from its trained state,
        the model and ``c`` before the round, and its post-fit outputs; the
        observer is handed each in order, and the server folds the controls
        (``aggregate_folded``) with the model the device folded.
        """

        from fedbrew.clients.batched_update import ClientBatchFit
        from fedbrew.core.torch_utils import clone_model_state, zeros_like_model_state

        context = self.context
        server = context.server
        server._num_clients = len(context.client_infos)
        start = self._host_start
        trained: dict[int, dict[str, Tensor]] = {}
        for slots, stack in device_round.trained:
            for row, slot in enumerate(slots):
                trained[slot] = {name: value[row] for name, value in stack.items()}
        results = []
        offset = 0
        for chunk, layout in zip(device_round.chunks, device_round.staged_layouts, strict=True):
            size = _staged_size(layout)
            groups, lists = finished_values(values[offset : offset + size], layout)
            offset += size
            fit = finished_chunk(
                groups,
                lists,
                chunk.layout,
                [self._stand_in(len(members)) for members, *_ in chunk.layout],
                dict(self.metadata),
                self.trainable,
            )
            members, plans = self._record_plans(device_round, chunk.start, chunk.stop)
            built = time.perf_counter()
            evaluations = _client_evaluations(fit, len(members))
            chunk_results = []
            for position, (member, plan) in enumerate(zip(members, plans, strict=True)):
                if member._client_control is None:
                    member._client_control = zeros_like_model_state(start)
                evaluation = evaluations[position]
                state = trained[chunk.start + position]
                chunk_results.append(
                    member.batched_result(
                        FitRequest(round_id=device_round.round_id, client_id=member.client_id),
                        _MemberPlan(
                            program=plan.program,
                            structure=plan.structure,
                            slot=plan.slot,
                            start=start,
                            evaluate=plan.evaluate,
                            eval_rows=plan.eval_rows,
                            train_data=plan.train_data,
                            client_control=clone_model_state(member._client_control),
                            server_control=server._server_control,
                        ),
                        ClientBatchFit(
                            model_state=state,
                            training_outputs=[],
                            eval_outputs=evaluation if isinstance(evaluation, list) else None,
                            optimizer_steps=len(plan.structure),
                            model_state_metadata=dict(self.metadata),
                            trainable_parameters=self.trainable,
                            start=start,
                            eval_metrics=evaluation if isinstance(evaluation, tuple) else None,
                        ),
                    )
                )
            seconds = (chunk.seconds + time.perf_counter() - built) / max(1, len(members))
            for result in chunk_results:
                results.append((result, seconds))
        total = len(results)
        for done, (result, seconds) in enumerate(results, start=1):
            observer.fitted(result, seconds, done, total)
        return server.aggregate_folded(round_info, [result for result, _ in results], host_mean)

    def client_states(self) -> dict[str, Any] | None:
        """Every built client's state, stacked, as a checkpoint holds them: again only when needed.

        The pool's snapshot changes when it builds a client, and a resident
        round's rules keep nothing a round changes, so the last stacking is
        reused until the pool holds another client.
        """

        from fedbrew.core.loop import _stacked_client_states

        pool = self.context.client
        held = (len(getattr(pool, "_clients", pool)), len(getattr(pool, "_saved_states", ())))
        if self.scaffold:
            # SCAFFOLD's rules keep their controls, which a round changes.
            return _stacked_client_states(pool)
        if self._client_states is None or self._client_states[0] != held:
            self._client_states = (held, _stacked_client_states(pool))
        return self._client_states[1]

    def close(self) -> None:
        self.rows = None


class _Requests:
    """What ``batched_stacked_results`` reads of a chunk's requests: how many, and the round."""

    __slots__ = ("round_id", "_size")

    def __init__(self, round_id: int, size: int) -> None:
        self.round_id = round_id
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> _Requests:
        return self


class _Fold:
    """``WeightedStateAccumulator``'s sums over a round's buckets, where they round the same.

    On the device when every bucket holds several clients: each bucket's
    ``weights @ rows`` added into sums on the device, and the sums divided by
    the total weight as a device tensor -- where addition and that division
    round as the CPU's do. The weights and the total come as the round's
    inputs (``RoundPlan``), so a replayed round reads its own. With a bucket
    of one client, whose row the accumulator adds on the CPU with
    ``add_(row, alpha=weight)``, the whole round is folded as the accumulator
    folds it, on the CPU.
    """

    def __init__(self, rounds: ResidentRounds, plan: RoundPlan, inputs: Sequence[Tensor]) -> None:
        self.rounds = rounds
        self.plan = plan
        self.inputs = inputs
        self.host = plan.host_fold
        self.totals: dict[str, Tensor] = {}

    def add(self, bucket: _BucketPlan, stack: Mapping[str, Tensor]) -> None:
        """One bucket's rows, the round's client ``bucket.slots[r]`` at row ``r``."""

        weights = [self.plan.weights[slot] for slot in bucket.slots]
        for key in self.rounds.state_keys:
            tensor = stack[key]
            total = self._total(key, tensor)
            scale = None
            if len(weights) > 1 and not self.host:
                scale = self.inputs[bucket.inputs[("scale", total.dtype)]]
            self._add_rows(total, tensor, weights, scale)

    def add_runs(self, stacks: Sequence[Mapping[str, Tensor]]) -> None:
        """Every bucket's rows, a run of consecutive clients at a time, in the round's order.

        What ``WeightedStateAccumulator.add`` does with rows handed to it one
        by one, as SCAFFOLD's server is handed them: the rows of one stack
        that arrive together are folded together -- the stack itself when
        they are all of it, else those rows selected -- and a run of one row
        is added alone.
        """

        assert self.plan.runs is not None
        for index, rows in self.plan.runs:
            bucket = self.plan.buckets[index]
            weights = [self.plan.weights[bucket.slots[row]] for row in rows]
            stack = stacks[index]
            every_row = rows == list(range(len(bucket.slots)))
            for key in self.rounds.state_keys:
                tensor = stack[key]
                if not every_row:
                    tensor = tensor.index_select(0, torch.tensor(rows, device=tensor.device))
                total = self._total(key, tensor)
                self._add_rows(total, tensor, weights, None)

    def _add_rows(
        self, total: Tensor, tensor: Tensor, weights: list[float], scale: Tensor | None
    ) -> None:
        """``weights @ rows`` added into ``total``; one row as ``add_(row, alpha=weight)``."""

        if len(weights) == 1:
            total.add_(tensor[0].detach().cpu(), alpha=weights[0])
            return
        if scale is None:
            scale = uploaded(torch.tensor(weights, dtype=total.dtype), tensor.device)
        flat = tensor.reshape(len(weights), -1)
        part = (scale @ flat.to(total.dtype)).reshape(total.shape)
        total.add_(part.cpu() if self.host else part)

    def _total(self, key: str, tensor: Tensor) -> Tensor:
        total = self.totals.get(key)
        if total is None:
            dtype = _accumulation_dtype(tensor.dtype)
            device = "cpu" if self.host else tensor.device
            total = self.totals[key] = torch.zeros(tensor.shape[1:], dtype=dtype, device=device)
        return total

    def result(self) -> tuple[dict[str, Tensor], Tensor]:
        """The mean on the model's device, and whether every tensor of it is finite."""

        mean: dict[str, Tensor] = {}
        for key, total in self.totals.items():
            dtype = self.rounds.parameters[key].dtype
            if self.host:
                value = total.div_(self.plan.total_weight).to(dtype)
                mean[key] = value.to(self.rounds.device)
            else:
                divisor = self.inputs[self.plan.inputs[("divisor", total.dtype)]]
                mean[key] = total.div_(divisor).to(dtype)
        finite = torch.stack([torch.isfinite(value).all() for value in mean.values()]).all()
        return mean, finite


def _client_evaluations(fit: Any, size: int) -> list[Any]:
    """Each client's post-fit pass, as ``chunk_fits`` hands it to the rule: folded or its outputs.

    ``(metrics, examples)`` where the task folded the bucket's pass, the
    client's outputs where it did not, None where the pass did not run.
    """

    evaluations: list[Any] = [None] * size
    for bucket in fit.buckets:
        for row, position in enumerate(bucket.positions):
            if bucket.eval_metrics is not None:
                assert bucket.eval_examples is not None
                evaluations[position] = (
                    {name: column[row] for name, column in bucket.eval_metrics.items()},
                    bucket.eval_examples[row],
                )
            elif bucket.eval_outputs is not None:
                evaluations[position] = bucket.eval_outputs[row]
    return evaluations


def _runs(planned: PlannedRound, chunks: list[tuple[int, int]]) -> list[tuple[int, list[int]]]:
    """Each chunk's runs of consecutive clients of one structure, as (chunk, slots)."""

    structures = planned.train.structure
    runs: list[tuple[int, list[int]]] = []
    for index, (start, stop) in enumerate(chunks):
        for slot in range(start, stop):
            if slot > start and structures[slot] == structures[slot - 1]:
                runs[-1][1].append(slot)
            else:
                runs.append((index, [slot]))
    return runs


def _bucket_runs(
    buckets: list[_BucketPlan], runs: list[tuple[int, list[int]]]
) -> list[tuple[int, list[int]]]:
    """``_runs`` as (bucket index, rows of that bucket)."""

    where = {
        slot: (index, row)
        for index, bucket in enumerate(buckets)
        for row, slot in enumerate(bucket.slots)
    }
    placed = []
    for _, slots in runs:
        index = where[slots[0]][0]
        placed.append((index, [where[slot][1] for slot in slots]))
    return placed


def _accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    from fedbrew.core.torch_utils import _accumulation_dtype as accumulation

    return accumulation(dtype)


def _staged_size(layout: StagedLayout) -> int:
    parts, sizes = layout
    return sum(len(keys) * positions * splits for keys, positions, splits, _ in parts) + sum(sizes)


def _train_split(data: Any) -> Any:
    from fedbrew.clients.torch_sgd_client import _get_train_data

    return _get_train_data(data)


def _split_bytes(split: Any) -> int:
    values = split.values() if isinstance(split, Mapping) else split
    return sum(
        int(value.numel() * value.element_size()) for value in values if isinstance(value, Tensor)
    )


def _pool_unsupported(pool: Any, dataset: Any) -> str | None:
    """Why the client pool could change which clients are built behind the round's back, or None.

    A checkpoint lists every client the pool has built. The resident round
    takes a client from the pool when the round would, and holds it from then
    on; a pool that could let it go -- one that releases every client after
    its evaluation, or a shard cache too small for every client's shard --
    would build it again later, when the round no longer asks for it.
    """

    from fedbrew.clients.lazy_pool import LazyClientPool

    if not isinstance(pool, LazyClientPool):
        if isinstance(pool, Mapping):
            return None
        return f"client pool {type(pool).__name__} is not a mapping of built clients"
    if pool._keep_resident is None:
        return "the client pool releases every client after its evaluation"
    holds = getattr(dataset, "caches_every_shard", None)
    if not callable(holds) or not holds():
        return "the dataset's shard cache cannot hold every client's shard"
    return None


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def run_resident_loop(context: Any, rounds: ResidentRounds) -> Any:
    """``run_fl_loop``'s rounds, trained on the device and recorded at each flush.

    Rounds are queued on the device as they come. At a flush round the next
    round is queued too, so the device has work while the host records; then
    the rounds since the last flush are read back in one copy and recorded, in
    order, by the loop's own bookkeeping, and the flush's writes are handed to
    the writer thread (``resident_flush``). A round that stops the run -- a
    divergence verdict, or an aggregate that is not finite -- ends it exactly
    where the per-round loop ends it: the rounds queued after it are dropped.
    """

    from fedbrew.core.loop import _checkpointing_summary, _LongLivedObjects

    loop = _Loop(context, rounds, _LongLivedObjects())
    try:
        loop.run()
        # Only an aggregation refusal leaves rounds unflushed, as in run_fl_loop.
        if loop.unflushed:
            loop.flush()
        loop.writer.wait()
        _warn_on_an_unobserved_metric(context)
        state = context.state
        state.final_payload = context.server_payload
        state.checkpointing = _checkpointing_summary(
            context.output_dir, context.checkpoint_policy, context.checkpoint_tracker
        )
        return state
    finally:
        loop.writer.close()
        loop.long_lived.release()
        rounds.close()


class _Loop:
    """The resident loop's state between rounds: the window, the flush, the writer."""

    def __init__(self, context: Any, rounds: ResidentRounds, long_lived: Any) -> None:
        self.context = context
        self.rounds = rounds
        self.long_lived = long_lived
        self.writer = FlushWriter()
        #: The checkpoints staged since the last flush, and whether rows wait.
        self.staged: WriterStaged | None = None
        self.unflushed = False
        #: The server's model before the next round to record, on the host.
        self.host_model: dict[str, Tensor] = dict(context.server._model_state)

    def run(self) -> None:
        """Queue and record every round, a flush window at a time, until the last or a stop."""

        context = self.context
        window: list[DeviceRound] = []
        ahead: DeviceRound | None = None
        for round_id in range(context.start_round, context.global_rounds + 1):
            try:
                queued = ahead if ahead is not None else self._queue(round_id)
            except Exception:
                # The per-round loop records the rounds before this one first,
                # and stops at one of them if it stops.
                if not self.record(window):
                    raise
                return
            ahead = None
            window.append(queued)
            if round_id % context.flush_every and round_id != context.global_rounds:
                continue
            failed: Exception | None = None
            if round_id < context.global_rounds:
                try:
                    ahead = self._queue(round_id + 1)
                except Exception as error:  # noqa: BLE001 -- raised once this window is recorded
                    failed = error
            if self.record(window):
                return
            window = []
            if failed is not None:
                raise failed

    def _queue(self, round_id: int) -> DeviceRound:
        from fedbrew.core.config import evaluates_round

        context = self.context
        evaluate = evaluates_round(context.fit_schedule, round_id, context.global_rounds)
        return self.rounds.run_round(round_id, evaluate)

    def record(self, window: list[DeviceRound]) -> bool:
        """Record every round of a window in order; whether one of them stopped the run."""

        if not window:
            return False
        for device_round, (values, host_mean) in zip(
            window, self.rounds.read_back(window), strict=True
        ):
            if self._record_round(device_round, values, host_mean):
                return True
        return False

    def _record_round(
        self, device_round: DeviceRound, values: list[float], host_mean: dict[str, Tensor] | None
    ) -> bool:
        """One round, recorded as ``run_fl_loop``'s body records it; whether the run stops here."""

        from fedbrew.core.loop import _FitPhaseTotals, _RoundFitObserver

        context, rounds = self.context, self.rounds
        state, round_id = context.state, device_round.round_id
        round_started = time.perf_counter()
        values, eval_values, central_values, grad_values, finite = _split_values(
            device_round, values
        )
        round_info = RoundInfo(round_id=round_id, total_rounds=context.global_rounds)
        selected = [rounds.roster.client_ids[place] for place in device_round.positions]
        if context.on_client_progress is not None:
            context.on_client_progress(round_id, 0, len(selected), "fit")
        fit_totals = _FitPhaseTotals()
        observer = _RoundFitObserver(state, fit_totals, round_id, context.on_client_progress)
        fit_started = time.perf_counter()
        if selected:
            if not finite:
                return self._refused(device_round, round_info, observer)
            assert host_mean is not None
            context.server_payload = rounds.aggregate(
                device_round, values, round_info, observer, host_mean, start=self.host_model
            )
            _check_weights(device_round, state)
            self.host_model = host_mean
        clock = device_round.clock
        timings = {
            "fit": clock.seconds("start", "trained"),
            "aggregate": time.perf_counter() - fit_started,
        }
        evaluated = _client_evaluation(context, rounds, device_round, eval_values, timings)
        central = _central_evaluation(context, rounds, device_round, central_values, timings)
        if device_round.grad_norm_due:
            central = {**(central or {}), **_grad_norm(context, grad_values, timings)}
        timings["client_eval"] += clock.seconds("trained", "clients")
        timings["global_eval"] += clock.seconds("clients", "evaluated")
        return self._record_rest(
            round_info, selected, (evaluated, central), fit_totals, timings, round_started
        )

    def _refused(self, device_round: DeviceRound, round_info: RoundInfo, observer: Any) -> bool:
        """A round whose aggregate is not finite, run again by the per-round path to refuse it.

        The per-round loop's refusal names the first non-finite client and
        tensor, which only its fold reads; so the round is run once more, from
        the model before it, through that path, which raises it and records
        what it records.
        """

        from fedbrew.core.config import evaluates_round
        from fedbrew.core.loop import _aggregate_round, _fit_requests
        from fedbrew.core.torch_utils import NonFiniteStateError

        context = self.context
        server = context.server
        server._model_state = self.host_model
        requests = _fit_requests(
            server,
            round_info,
            context.client_infos,
            post_fit_evaluation=evaluates_round(
                context.fit_schedule, device_round.round_id, context.global_rounds
            ),
        )
        planner, context.executor.planner = context.executor.planner, None
        try:
            _aggregate_round(
                context.executor,
                context.aggregator,
                server,
                round_info,
                context.client,
                requests,
                observer,
            )
        except NonFiniteStateError as error:
            _diverged(context, device_round.round_id, error)
            return True
        finally:
            context.executor.planner = planner
        raise RuntimeError(
            f"round {device_round.round_id}: the resident round's aggregate is not finite, "
            "and the per-round path's is"
        )

    def _record_rest(
        self,
        round_info: RoundInfo,
        selected: list[str],
        measured: tuple[list[tuple[Any, list[str]]], dict[str, float] | None],
        fit_totals: Any,
        timings: dict[str, float],
        round_started: float,
    ) -> bool:
        """The rest of ``run_fl_loop``'s body: the evaluation's records, the verdict, the flush."""

        _record_evaluation(self.context, round_info, selected, measured)
        return self._record_the_round(round_info, selected, fit_totals, timings, round_started)

    def _record_the_round(
        self,
        round_info: RoundInfo,
        selected: list[str],
        fit_totals: Any,
        timings: dict[str, float],
        round_started: float,
    ) -> bool:
        """The verdict, checkpoints, records and flush of a round, as ``run_fl_loop`` does them."""

        from fedbrew.core.loop import _checkpoint_payload_builder, _flush_due, _update_checkpoints
        from fedbrew.core.state import MetricRecord, RoundState, RoundTimings

        context = self.context
        state, round_id = context.state, round_info.round_id
        metrics = dict(round_info.metrics)
        if isinstance(context.server_payload, dict):
            context.server_payload["metrics"] = metrics
        verdict = context.monitor.update(round_id, metrics)
        flush_due = _flush_due(round_id, context.global_rounds, context.flush_every, verdict)
        checkpoint_started = time.perf_counter()
        self.staged = _update_checkpoints(
            _checkpoint_payload_builder(
                context.server,
                context.client,
                context.server_payload,
                metrics,
                round_id,
                self.rounds.client_states,
            ),
            metrics,
            context.output_dir,
            round_id,
            context.checkpoint_policy,
            context.checkpoint_tracker,
            self.staged or WriterStaged(self.writer),
            write_latest=flush_due,
        )
        record = {
            "round_id": round_id,
            "metrics": metrics,
            "num_clients": len(selected),
            "num_examples": fit_totals.num_examples,
            "timings": RoundTimings(
                fit=timings["fit"],
                aggregate=timings["aggregate"],
                client_eval=timings["client_eval"],
                global_eval=timings["global_eval"],
                checkpoint=time.perf_counter() - checkpoint_started,
                total=time.perf_counter() - round_started + timings["fit"],
            ),
        }
        state.rounds.append(RoundState(**record))
        metric_record = MetricRecord(**record)
        state.metrics_history.append(metric_record)
        if flush_due:
            self.flush()
        self.unflushed = not flush_due
        if round_id == context.start_round:
            self.long_lived.freeze()
        if context.on_round_end is not None:
            context.on_round_end(metric_record)
        return _stops(context, verdict)

    def flush(self) -> None:
        """Hand the rounds since the last flush to the writer: CSVs, run.json, then checkpoints."""

        from fedbrew.core.loop import _submit_flush

        context, staged = self.context, self.staged
        self.staged = None
        _submit_flush(
            self.writer,
            context.state,
            context.output_dir,
            context.per_client_csv,
            context.csv_cursor,
            context.on_round_flush,
            staged,
            context.checkpoint_policy,
        )


def _split_values(
    device_round: DeviceRound, values: list[float]
) -> tuple[list[float], list[float], list[float], list[float], bool]:
    """A round's staged values: the chunks', evaluation's, central pass's, gradient norm's, flag."""

    fit, evaluation, central, grad_norm, flag = device_round.sizes
    rest = fit + evaluation
    after = rest + central
    finite = True if not flag else values[after + grad_norm] != 0.0
    return (
        values[:fit],
        values[fit:rest],
        values[rest:after],
        values[after : after + grad_norm],
        finite,
    )


def _client_evaluation(
    context: Any,
    rounds: ResidentRounds,
    device_round: DeviceRound,
    values: list[float],
    timings: dict[str, float],
) -> list[tuple[Any, list[str]]]:
    """The round's client evaluation: from the device's measurement, or by the evaluator now."""

    started = time.perf_counter()
    stage = device_round.eval_stage
    round_id = device_round.round_id
    if stage is not None:
        evaluated = rounds.evaluation.results(  # type: ignore[union-attr]
            stage,
            values,
            round_id,
            context.server_payload,
            context.evaluation.model_scope,
            context.on_client_progress,
        )
    else:
        infos = {info.client_id: info for info in context.client_infos}
        work = [
            (infos[rounds.roster.client_ids[place]], splits) for place, splits in device_round.work
        ]
        evaluated = context.evaluator.evaluate_clients(
            context.client,
            round_id,
            work,
            context.server_payload,
            context.evaluation.model_scope,
            context.on_client_progress,
        )
    timings["client_eval"] = time.perf_counter() - started
    return evaluated


def _central_evaluation(
    context: Any,
    rounds: ResidentRounds,
    device_round: DeviceRound,
    values: list[float],
    timings: dict[str, float],
) -> dict[str, float] | None:
    """The round's central metrics, if due: from the device's measurement, or by the evaluator."""

    started = time.perf_counter()
    central = None
    if device_round.central_due:
        stage = device_round.central_stage
        if stage is not None:
            central = rounds.evaluation.central_metrics(stage, values)  # type: ignore[union-attr]
        else:
            central = context.evaluator.evaluate_central(context.server, context.dataset)
    timings["global_eval"] = time.perf_counter() - started
    return central


def _grad_norm(context: Any, values: list[float], timings: dict[str, float]) -> dict[str, float]:
    """The round's ``grad_norm_sq``: measured on the device, or by the evaluator now."""

    from fedbrew.core.metrics import GRAD_NORM_COLUMN

    if values:
        return {GRAD_NORM_COLUMN: values[0]}
    started = time.perf_counter()
    measured = context.evaluator.evaluate_grad_norm(context.server, context.dataset)
    timings["global_eval"] += time.perf_counter() - started
    return measured


def _record_evaluation(
    context: Any,
    round_info: RoundInfo,
    selected: list[str],
    measured: tuple[list[tuple[Any, list[str]]], dict[str, float] | None],
) -> None:
    """The evaluation's records and the round's split and central metrics, as ``run_fl_loop``."""

    from fedbrew.core.loop import (
        _aggregate_client_split_metrics,
        _build_client_evaluation_record,
        _scope_split_names,
    )

    evaluated, central = measured
    selected_client_set = set(selected)
    context.state.client_metrics_history.extend(
        _build_client_evaluation_record(
            result,
            participated=result.client_id in selected_client_set,
            evaluated_splits=splits,
            model_scope=context.evaluation.model_scope,
        )
        for result, splits in evaluated
    )
    for split in ("train", "val", "test"):
        subset = [result for result, splits in evaluated if split in splits]
        if not subset:
            continue
        for metric_split in _scope_split_names(context.evaluation.model_scope, [split]):
            round_info.metrics.update(
                _aggregate_client_split_metrics(subset, metric_split, context.statistics)
            )
    if central is not None:
        round_info.metrics.update(central)


def _check_weights(device_round: DeviceRound, state: Any) -> None:
    """The weights the round was folded with are the example counts its records report."""

    records = state.client_update_metrics_history[-len(device_round.positions) :]
    counts = [record.num_examples for record in records]
    if counts != device_round.eval_rows:
        raise RuntimeError(
            f"round {device_round.round_id}: the resident fold's weights "
            f"{device_round.eval_rows[:4]}... are not the example counts {counts[:4]}... "
            "the clients report"
        )


def _diverged(context: Any, round_id: int, error: Exception) -> None:
    """The per-round loop's record of an aggregation refusal."""

    from fedbrew.core.divergence import STATUS_DIVERGED, DivergenceVerdict

    state = context.state
    state.status = STATUS_DIVERGED
    verdict = DivergenceVerdict(
        status=STATUS_DIVERGED,
        detector="non_finite_client_state",
        round_id=round_id,
        metric="client_model_state",
        value=float("nan"),
        threshold=None,
        reason=(
            f"round {round_id} aggregation refused a non-finite client "
            f"state ({error}); the model cannot recover from it"
        ),
    )
    state.termination = verdict.as_dict()
    if context.on_termination is not None:
        context.on_termination(verdict)


def _stops(context: Any, verdict: Any) -> bool:
    """Record a verdict as ``run_fl_loop`` does; whether there was one."""

    if verdict is None:
        return False
    state = context.state
    state.status = verdict.status
    state.termination = verdict.as_dict()
    if context.on_termination is not None:
        context.on_termination(verdict)
    return True


def _warn_on_an_unobserved_metric(context: Any) -> None:
    """``run_fl_loop``'s warning when the divergence metric never appeared."""

    divergence, monitor = context.divergence, context.monitor
    if divergence is not None and divergence.active and not monitor.observed:
        print(
            f"Warning: divergence.metric={monitor.metric!r} was never present in "
            "any round's metrics, so no divergence detector ran. Check the name "
            "against the metrics this config emits.",
            flush=True,
        )
