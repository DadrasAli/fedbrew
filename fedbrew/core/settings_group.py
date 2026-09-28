"""A group of settings run in one process, their clients trained as one batch.

The settings of a group are runs whose configs differ only in numeric
hyperparameters. Each is
``runner.run`` of its own config, unchanged, on its own thread, writing its
own run directory as it would alone. What they share:

- the process, so the imports and the first ``torch.func`` call are paid once;
- the dataset, built by the first setting (``GroupSetting.build_components``);
- each round's batch orders, planned once when every setting's are the same;
- the batched executor: each round, every setting's clients are trained
  together, the settings a second batch axis beside the clients. Each setting
  plans its round as it would alone, into chunks of its own (its units); the
  units of all settings are packed, first units first, into combined chunks
  under the same ``executor_chunk_bytes``, and a bucket takes rows of several
  settings when they are the same unit of the same batch orders -- the same
  clients in the same order, their values their own (``ProgramValues``).
  Each setting's rows are cut back out of the trained stack, a state stack
  shared by several settings copied into its own, so each setting builds and
  folds the results it would alone.

Exactly one setting runs at a time (``_Baton``). A setting hands over when it
waits for the other settings -- to register their round, or to take their
results of a trained chunk -- and when it ends, always to the next live
setting in group order, so the interleaving is the same on every run. At each
hand-over the process-wide generators are saved and the next setting's
restored, so each setting draws what it would draw alone, and its checkpoints
hold the generator state they would hold alone. A setting that ends --
completed, diverged, stalled, refused or raised -- leaves the group; what it
had not taken is dropped, and the others go on.
"""

from __future__ import annotations

import hashlib
import io
import sys
import threading
import time
import traceback
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from fedbrew.clients.batch_orders import RoundOrders
from fedbrew.clients.batched_update import ClientBatchPlan, plan_round
from fedbrew.core.batched_executor import (
    DEFAULT_EXECUTOR_CHUNK_BYTES,
    TrainedChunk,
    _plans,
    _Rows,
    _run_bucket,
    chunk_fit_stacked,
    chunk_fits,
    client_costs,
    client_results,
    cut_chunks,
    select_executor,
    stacked_client_results,
    trained_state_keys,
)
from fedbrew.core.execution import ClientPool, FitObserver
from fedbrew.core.factory import ExperimentComponents, build_components, build_dataset
from fedbrew.core.protocol import FitRequest, FitResult
from fedbrew.core.runtime_setup import capture_rng_state, restore_rng_state
from fedbrew.core.stacked_results import StackedFitResults

#: A setting's unit: (its position in the group, its unit's index in its round).
Part = tuple[int, int]


# ---------------------------------------------------------------------------
# One setting at a time
# ---------------------------------------------------------------------------


class _Baton:
    """Lets exactly one of the group's threads run, handing over in group order.

    Each thread's process-wide generator state is saved when it hands over and
    restored when the baton comes back to it.
    """

    def __init__(self, size: int) -> None:
        self._condition = threading.Condition()
        self._holder = 0
        self._live = list(range(size))
        self._generators: dict[int, dict[str, Any]] = {}

    def wait_first_turn(self, position: int) -> None:
        with self._condition:
            while self._holder != position:
                self._condition.wait()

    def hand_over(self, position: int) -> None:
        """Let the next live setting run, and wait until the baton is back."""

        self._generators[position] = capture_rng_state()
        with self._condition:
            self._holder = self._next(position)
            self._condition.notify_all()
            while self._holder != position:
                self._condition.wait()
        restore_rng_state(self._generators.pop(position))

    def leave(self, position: int) -> None:
        """Leave for good, handing over to the next live setting."""

        with self._condition:
            following = self._next(position)
            self._live.remove(position)
            self._holder = following if self._live else -1
            self._condition.notify_all()

    @property
    def live(self) -> list[int]:
        return list(self._live)

    def _next(self, position: int) -> int:
        later = [other for other in self._live if other > position]
        return later[0] if later else self._live[0]


# ---------------------------------------------------------------------------
# A setting's round, and the group's
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Prep:
    """One setting's round, planned on its own thread as its executor alone would plan it."""

    round_id: int
    members: list[Any]
    requests: list[FitRequest]
    stacked: bool
    task: Any = None
    template: nn.Module | None = None
    plans: list[ClientBatchPlan] = field(default_factory=list)
    orders: tuple[RoundOrders, RoundOrders] | None = None
    units: list[tuple[int, int]] = field(default_factory=list)
    unit_costs: list[int] = field(default_factory=list)
    #: What planning raised, raised again when the setting takes its results,
    #: as its executor alone raises it when the aggregator first asks.
    error: BaseException | None = None


class _Round:
    """The settings' units of one round, packed into combined chunks, and those trained so far."""

    def __init__(self, preps: Mapping[int, _Prep], budget: int) -> None:
        self.preps = dict(preps)
        parts: list[tuple[Part, int]] = []
        deepest = max((len(prep.units) for prep in self.preps.values()), default=0)
        for unit in range(deepest):
            for position, prep in self.preps.items():
                if unit < len(prep.units):
                    parts.append(((position, unit), prep.unit_costs[unit]))
        self.chunks: list[list[Part]] = []
        used = 0
        for part, cost in parts:
            if self.chunks and self.chunks[-1] and used + cost <= budget:
                self.chunks[-1].append(part)
                used += cost
            else:
                self.chunks.append([part])
                used = cost
        self.chunk_of = {part: index for index, chunk in enumerate(self.chunks) for part in chunk}
        self.trained = -1
        self.results: dict[Part, Any] = {}

    def drop(self, position: int) -> None:
        """Forget a setting that left: its results, and its units not yet trained."""

        for part in [part for part in self.results if part[0] == position]:
            del self.results[part]
        for index in range(self.trained + 1, len(self.chunks)):
            self.chunks[index] = [part for part in self.chunks[index] if part[0] != position]


class SettingsGroup:
    """The shared state of a group of settings run in one process."""

    def __init__(self, size: int, configs: Sequence[str | Path], varies: Sequence[str]) -> None:
        self.size = size
        self.baton = _Baton(size)
        self.configs = [str(path) for path in configs]
        self.varies = list(varies)
        self.identity = hashlib.sha256("\0".join(self.configs).encode()).hexdigest()[:12]
        self.chunk_bytes = DEFAULT_EXECUTOR_CHUNK_BYTES
        #: The group's record, shared by every setting's run.json.
        self.largest_chunk_rows = 0
        self._registered: dict[int, _Prep] = {}
        self._rounds: dict[int, _Round] = {}
        self._orders: dict[int, _Prep] = {}
        self._dataset: tuple[Any, dict[str, Any], dict[str, Any]] | None = None
        #: Stacked rows of the last combined chunk, reused by the next chunk
        #: that trains the same splits: with a shared dataset, every
        #: setting's same clients.
        self.rows: dict[tuple[int, ...], _Rows] = {}

    def setting(self, position: int) -> GroupSetting:
        return GroupSetting(self, position)

    # -- the dataset ------------------------------------------------------

    def dataset_for(self, config: Any) -> Any:
        """The group's dataset, if this setting's generators stand where the first's stood."""

        before = capture_rng_state()
        if self._dataset is not None and _same_generators(before, self._dataset[1]):
            dataset, _, after = self._dataset
            restore_rng_state(after)
            return dataset
        dataset = build_dataset(config)
        if self._dataset is None:
            self._dataset = (dataset, before, capture_rng_state())
        return dataset

    # -- a round ----------------------------------------------------------

    def prepare(self, position: int, prep: _Prep) -> None:
        """Plan one setting's round on its own thread, then register it."""

        try:
            prep.task = prep.members[0].task
            prep.template = prep.task.build_model(prep.members[0].model_config)
            prep.plans = _plans(prep.members, prep.requests, prep.template)
            prep.orders = self._round_orders(prep)
            costs = client_costs(prep.task, prep.template, prep.plans, prep.orders[0])
            prep.units = list(cut_chunks(costs, self.chunk_bytes))
            prep.unit_costs = [sum(costs[start:stop]) for start, stop in prep.units]
        except Exception as error:  # raised again when the setting takes its results
            prep.error = error
            prep.units, prep.unit_costs = [], []
        self._registered[position] = prep

    def _round_orders(self, prep: _Prep) -> tuple[RoundOrders, RoundOrders]:
        """The round's batch orders: another setting's, when every plan is drawn alike."""

        reference = self._orders.get(prep.round_id)
        if reference is not None and _drawn_alike(reference.plans, prep.plans):
            for plan, held in zip(prep.plans, reference.plans, strict=True):
                plan.slot, plan.structure, plan.eval_rows = (
                    held.slot,
                    held.structure,
                    held.eval_rows,
                )
            assert reference.orders is not None
            return reference.orders
        orders = plan_round(prep.plans, prep.round_id)
        if reference is None:
            self._orders[prep.round_id] = prep
        return orders

    def round_for(self, position: int, round_id: int) -> _Round:
        """Wait until every live setting has registered this round, then the round's packing."""

        while not all(
            other in self._registered and self._registered[other].round_id >= round_id
            for other in self.baton.live
        ):
            self.baton.hand_over(position)
        if round_id not in self._rounds:
            for old in [old for old in self._rounds if old < round_id]:
                del self._rounds[old]
            self._orders = {key: value for key, value in self._orders.items() if key > round_id}
            preps = {
                other: prep
                for other, prep in self._registered.items()
                if prep.round_id == round_id and prep.error is None
            }
            self._rounds[round_id] = _Round(preps, self.chunk_bytes)
        return self._rounds[round_id]

    def take(self, position: int, round_: _Round, unit: int) -> tuple[Any, float]:
        """This setting's trained share of one unit, training the next chunk when it is due."""

        part = (position, unit)
        index = round_.chunk_of[part]
        while True:
            if round_.trained == index and part in round_.results:
                taken = round_.results.pop(part)
                if isinstance(taken, BaseException):
                    raise taken
                return taken
            if round_.trained < index and not round_.results:
                self._train_next(round_)
                continue
            self.baton.hand_over(position)

    def _train_next(self, round_: _Round) -> None:
        """Train the round's next combined chunk, each setting's share cut out for it."""

        round_.trained += 1
        parts = round_.chunks[round_.trained]
        if not parts:
            return
        started = time.perf_counter()
        try:
            shares = self._train_parts([(round_.preps[p], u) for p, u in parts])
        except Exception as error:
            round_.results = dict.fromkeys(parts, error)
            return
        seconds = time.perf_counter() - started
        rows = sum(len(share[1]) for share in shares)
        self.largest_chunk_rows = max(self.largest_chunk_rows, rows)
        round_.results = {
            part: ((keys, trained), seconds * len(rows_of) / rows)
            for part, (keys, rows_of, trained) in zip(parts, shares, strict=True)
        }

    def _train_parts(
        self, parts: list[tuple[_Prep, int]]
    ) -> list[tuple[list[str], list[int], list[tuple[list[int], dict[str, Tensor], Any, Any]]]]:
        """Train several settings' units as one chunk; per unit, what ``_train_buckets`` returns.

        A bucket holds the rows of every unit that is the same unit of the
        same orders and shares the plan's bucket, unit by unit, so each
        unit's rows are consecutive; a unit's buckets are in the order its
        executor alone forms them.
        """

        first = parts[0][0]
        task, template = first.task, first.template
        assert template is not None
        state_keys = trained_state_keys(task, template)
        buffers = dict(template.named_buffers())
        buckets: dict[tuple[Any, ...], list[tuple[int, int]]] = {}
        orders_of: dict[tuple[Any, ...], tuple[RoundOrders, RoundOrders]] = {}
        order: list[list[tuple[Any, ...]]] = []
        for number, (prep, unit) in enumerate(parts):
            start, stop = prep.units[unit]
            assert prep.orders is not None
            seen: list[tuple[Any, ...]] = []
            for index in range(start, stop):
                key = (prep.plans[index].bucket, id(prep.orders[0]), start, stop)
                buckets.setdefault(key, []).append((number, index))
                orders_of[key] = prep.orders
                if key not in seen:
                    seen.append(key)
            order.append(seen)
        needed = {
            tuple(id(parts[n][0].plans[i].train_data) for n, i in rows) for rows in buckets.values()
        }
        kept = {key: rows for key, rows in self.rows.items() if key in needed}
        self.rows = {}
        trained: dict[tuple[Any, ...], tuple[dict[str, Tensor], Any, Any]] = {}
        for key, rows in buckets.items():
            plans = [parts[number][0].plans[index] for number, index in rows]
            trained[key] = _run_bucket(
                task, template, buffers, plans, orders_of[key], kept, self.rows, _ranges(rows)
            )
        shares = []
        for number, (prep, unit) in enumerate(parts):
            start, _ = prep.units[unit]
            share: list[tuple[list[int], dict[str, Tensor], Any, Any]] = []
            rows_of: list[int] = []
            for key in order[number]:
                rows = buckets[key]
                mine = [row for row, (owner, _) in enumerate(rows) if owner == number]
                positions = [rows[row][1] - start for row in mine]
                owners = sorted({owner for owner, _ in rows})
                share.append(
                    _cut(
                        trained[key],
                        mine[0],
                        mine[-1] + 1,
                        len(rows),
                        positions,
                        owners.index(number),
                    )
                )
                rows_of.extend(positions)
            shares.append((state_keys, rows_of, share))
        return shares

    def leave(self, position: int) -> None:
        """A setting ended: drop what it had not taken, and hand over for good."""

        self._registered.pop(position, None)
        for round_ in self._rounds.values():
            round_.drop(position)
        self.baton.leave(position)


def _ranges(rows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The bucket's rows of each unit, as consecutive ranges, in the order of the units."""

    ranges: list[tuple[int, int]] = []
    for row, (owner, _) in enumerate(rows):
        if ranges and rows[ranges[-1][0]][0] == owner:
            ranges[-1] = (ranges[-1][0], row + 1)
        else:
            ranges.append((row, row + 1))
    return ranges


def _cut(
    trained: tuple[dict[str, Tensor], Any, Any],
    first: int,
    stop: int,
    rows: int,
    positions: list[int],
    part: int,
) -> tuple[list[int], dict[str, Tensor], Any, Any]:
    """One unit's rows of a trained bucket, as its executor alone would hold them.

    A bucket of this unit alone is handed over as it is. Otherwise its state
    stack is copied into the unit's own tensors, which fold as the unit's
    own stack would, its training outputs are sliced, and its post-fit
    pass is the one the bucket measured on its rows alone (``part``).
    """

    stack, training, evaluated = trained
    if first == 0 and stop == rows:
        return positions, stack, training, evaluated
    own = {name: value[first:stop].clone() for name, value in stack.items()}
    outputs, counts = training
    training = (
        [{key: value[first:stop] for key, value in output.items()} for output in outputs],
        counts[first:stop],
    )
    return positions, own, training, None if evaluated is None else evaluated[part]


def _drawn_alike(reference: Sequence[ClientBatchPlan], plans: Sequence[ClientBatchPlan]) -> bool:
    """Whether two settings' plans draw the same batches: same clients, seeds, loops and orders."""

    if len(reference) != len(plans):
        return False
    return all(
        held.replay is None
        and plan.replay is None
        and held.client_id == plan.client_id
        and held.seed == plan.seed
        and held.loop == plan.loop
        and held.train_order == plan.train_order
        and held.eval_order == plan.eval_order
        for held, plan in zip(reference, plans, strict=True)
    )


def _same_generators(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    """Whether two ``capture_rng_state`` records are the same state."""

    if set(first) != set(second) or first.get("python") != second.get("python"):
        return False
    if "numpy" in first:
        left, right = first["numpy"], second["numpy"]
        if left[0] != right[0] or tuple(left[2:]) != tuple(right[2:]):
            return False
        if not bool((left[1] == right[1]).all()):
            return False
    if "torch" in first and not torch.equal(first["torch"], second["torch"]):
        return False
    if "torch_cuda" in first:
        return all(
            torch.equal(left, right)
            for left, right in zip(first["torch_cuda"], second["torch_cuda"], strict=True)
        )
    return True


# ---------------------------------------------------------------------------
# One setting: the runner's hook, and its executor
# ---------------------------------------------------------------------------


class GroupSetting:
    """One setting's place in its group, as ``runner.run(setting=...)`` takes it."""

    def __init__(self, group: SettingsGroup, position: int) -> None:
        self.group = group
        self.position = position
        self.record: dict[str, Any] = {
            "id": group.identity,
            "size": group.size,
            "position": position + 1,
            "configs": group.configs,
            "varies": group.varies,
        }

    def build_components(self, config: Any) -> ExperimentComponents:
        return build_components(config, dataset=self.group.dataset_for(config))

    def select_executor(self, components: Any) -> tuple[Any, dict[str, Any]]:
        """The executor alone would select; a batched one becomes the group's."""

        executor, record = select_executor(components)
        if executor is None:
            return None, record
        self.group.chunk_bytes = executor.chunk_bytes
        self.record["largest_chunk_rows"] = 0
        return GroupExecutor(self, executor.chunk_bytes, record), record


class GroupExecutor:
    """A setting's ClientExecutor: its clients trained with every other setting's."""

    def __init__(self, setting: GroupSetting, chunk_bytes: int, record: dict[str, Any]) -> None:
        self.setting = setting
        self.chunk_bytes = chunk_bytes
        self.record = record

    @property
    def _rows(self) -> dict[tuple[int, ...], _Rows]:
        """The rows the group keeps, which the batched evaluator reuses (``BatchedEvaluator``)."""

        return self.setting.group.rows

    def fit(
        self, clients: ClientPool, requests: Sequence[FitRequest], observer: FitObserver
    ) -> Iterator[FitResult]:
        members, requests = _members(clients, requests)
        prep = self._register(members, requests, stacked=False)
        return client_results(members, requests, observer, self._trained(prep))

    def fit_stacked(
        self, clients: ClientPool, requests: Sequence[FitRequest], observer: FitObserver
    ) -> Iterator[StackedFitResults] | None:
        members, requests = _members(clients, requests)
        for member in {type(member): member for member in members}.values():
            supported = getattr(member, "batched_stacked_supported", None)
            if not callable(supported) or not supported():
                return None
        prep = self._register(members, requests, stacked=True)
        return stacked_client_results(members, requests, observer, self._trained(prep))

    def _register(self, members: list[Any], requests: list[FitRequest], stacked: bool) -> _Prep:
        prep = _Prep(
            round_id=requests[0].round_id, members=members, requests=requests, stacked=stacked
        )
        self.setting.group.prepare(self.setting.position, prep)
        return prep

    def _trained(self, prep: _Prep) -> Iterator[TrainedChunk]:
        group, position = self.setting.group, self.setting.position
        if prep.error is not None:
            raise prep.error
        round_ = group.round_for(position, prep.round_id)
        for unit, (start, stop) in enumerate(prep.units):
            self.record["largest_chunk_clients"] = max(
                self.record.get("largest_chunk_clients", 0), stop - start
            )
            (state_keys, trained), seconds = group.take(position, round_, unit)
            self.setting.record["largest_chunk_rows"] = group.largest_chunk_rows
            built = time.perf_counter()
            if prep.stacked:
                share = chunk_fit_stacked(prep.task, prep.template, state_keys, trained)
            else:
                share = chunk_fits(
                    prep.task, prep.template, prep.plans[start:stop], state_keys, trained
                )
            trained = None
            yield start, stop, prep.plans, share, seconds + time.perf_counter() - built
            share = None


def _members(
    clients: ClientPool, requests: Sequence[FitRequest]
) -> tuple[list[Any], list[FitRequest]]:
    requests = list(requests)
    members = [
        clients[request.client_id] if isinstance(clients, Mapping) else clients
        for request in requests
    ]
    return members, requests


# ---------------------------------------------------------------------------
# Running a group
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SettingOutcome:
    """How one setting's run ended.

    ``completed``, ``diverged`` or ``stalled`` as run.json says, ``refused``,
    or ``crashed``.
    """

    config: str
    status: str
    detail: str = ""


def run_group(configs: Sequence[str | Path], varies: Sequence[str] = ()) -> list[SettingOutcome]:
    """Run every config as one setting of a group, in this process; how each ended.

    Each setting's console output is its own lines, prefixed with its place
    in the group.
    """

    from fedbrew.core import runner
    from fedbrew.core.refusal import RunRefused

    group = SettingsGroup(len(configs), configs, varies)
    outcomes: list[SettingOutcome | None] = [None] * len(configs)
    labels: dict[int, str] = {}

    width = len(str(len(configs)))

    def setting_main(position: int) -> None:
        config = str(configs[position])
        labels[threading.get_ident()] = f"[{position + 1:>{width}}/{len(configs)}] "
        group.baton.wait_first_turn(position)
        try:
            state = runner.run(config, args=None, setting=group.setting(position))
            outcomes[position] = SettingOutcome(config, str(state.status))
        except RunRefused as refusal:
            runner._print_refusal(refusal, _NO_RICH)
            outcomes[position] = SettingOutcome(config, "refused", str(refusal))
        except BaseException as error:  # a setting that raises stops alone
            traceback.print_exc()
            outcomes[position] = SettingOutcome(
                config, "crashed", f"{type(error).__name__}: {error}"
            )
        finally:
            group.leave(position)

    threads = [
        threading.Thread(target=setting_main, args=(position,), daemon=True)
        for position in range(len(configs))
    ]
    with _PrefixedOutput(labels):
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    return [
        outcome or SettingOutcome(str(configs[i]), "crashed") for i, outcome in enumerate(outcomes)
    ]


class _NoRich:
    no_rich = True


_NO_RICH: Any = _NoRich()


class _PrefixedOutput:
    """stdout and stderr, each line prefixed with the place of the setting that wrote it."""

    def __init__(self, labels: dict[int, str]) -> None:
        self.labels = labels
        self._saved: tuple[Any, Any] | None = None

    def __enter__(self) -> _PrefixedOutput:
        self._saved = (sys.stdout, sys.stderr)
        sys.stdout = _PrefixedStream(sys.stdout, self.labels)  # type: ignore[assignment]
        sys.stderr = _PrefixedStream(sys.stderr, self.labels)  # type: ignore[assignment]
        return self

    def __exit__(self, *exc_info: Any) -> None:
        assert self._saved is not None
        for stream in (sys.stdout, sys.stderr):
            if isinstance(stream, _PrefixedStream):
                stream.flush_all()
        sys.stdout, sys.stderr = self._saved


class _PrefixedStream(io.TextIOBase):
    def __init__(self, target: Any, labels: dict[int, str]) -> None:
        self._target = target
        self._labels = labels
        self._partial: dict[int, str] = {}

    def writable(self) -> bool:
        return True

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return str(getattr(self._target, "encoding", None) or "utf-8")

    def isatty(self) -> bool:
        return False

    def write(self, text: str) -> int:
        ident = threading.get_ident()
        buffered = self._partial.pop(ident, "") + text
        *lines, rest = buffered.split("\n")
        label = self._labels.get(ident, "")
        for line in lines:
            self._target.write(f"{label}{line}\n")
        if rest:
            self._partial[ident] = rest
        return len(text)

    def flush(self) -> None:
        self._target.flush()

    def flush_all(self) -> None:
        for ident, rest in self._partial.items():
            self._target.write(f"{self._labels.get(ident, '')}{rest}\n")
        self._partial = {}
        self._target.flush()
