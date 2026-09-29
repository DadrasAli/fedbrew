"""Each round's batch orders, planned ahead of the round loop by worker processes.

A round's batch orders (``fedbrew/clients/batch_orders.py``) depend on the
run's roster and the round's number and on nothing a round trains:

- which clients FedAvg samples is drawn from ``derive_seed(seed,
  "participation", round)`` (``sampled_client_ids``);
- each loader's seed is ``dataloader_seed(seed, round, client, phase)``
  (``loader_seeds``);
- what each loader yields is its task's ``loader_order`` declaration, the
  same every round for the same data and configuration.

So the orders of the rounds to come can be planned while earlier rounds run.
:class:`RoundPlanner` hands them out a round at a time, planned by worker
processes a few rounds ahead, which return them through shared memory
(``torch.multiprocessing``'s queue shares a tensor's storage rather than
copying it). Every order is computed by :func:`plan_roster_round`, which calls
the functions the round itself calls, in the same order, on the same values:
``sampled_client_ids``, ``loader_seeds`` and ``plan_orders``. The orders are
the round's own by construction, and ``tests/test_round_planner.py`` holds
them to ``plan_round``'s, tensor for tensor.

Until a worker has started -- a spawned process imports torch, a few
seconds -- the loop plans each round it asks for itself, with the same
function, rather than wait for one. A worker that cannot start, dies or
raises leaves the planning to this process, with the same function; the run
records it and says so once.
"""

from __future__ import annotations

import os
import queue
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from fedbrew.clients.batch_orders import LocalLoop, RoundOrders, plan_orders
from fedbrew.clients.batched_update import loader_seeds
from fedbrew.servers.fedavg import sampled_client_ids
from fedbrew.tasks.base import LoaderOrder

#: Rounds planned ahead of the one the loop takes, across all workers.
DEFAULT_AHEAD = 4

#: Most workers a run starts; more than the loop can consume is idle memory.
MAX_WORKERS = 4

#: How long the loop waits on a worker's round before planning it itself.
WORKER_TIMEOUT_SEC = 120.0

#: How long the loop waits for a worker to start before planning a round
#: itself: not at all, since the round it plans is the workers' to the bit.
READY_WAIT_SEC = 0.0


@dataclass(frozen=True, slots=True)
class Sampling:
    """FedAvg's participation settings: ``sampled_client_ids``'s arguments besides the round."""

    seed: int
    participation_rate: float
    participation_probability: float | None


@dataclass(slots=True)
class RosterPlan:
    """What every round's orders are planned from: the roster, fixed for the run.

    Per client in roster order: its id, its seed (``experiment.seed``, which
    ``dataloader_seed`` is derived from), what its training loader and its
    post-fit loader yield (unseeded), and the loop its update takes.
    """

    client_ids: tuple[str, ...]
    seeds: tuple[int | None, ...]
    train_orders: tuple[LoaderOrder, ...]
    eval_orders: tuple[LoaderOrder, ...]
    loops: tuple[LocalLoop, ...]
    sampling: Sampling
    #: Each id's roster position.
    where: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.where = {client: place for place, client in enumerate(self.client_ids)}

    def __len__(self) -> int:
        return len(self.client_ids)


@dataclass(slots=True)
class PlannedRound:
    """One round's sampled clients and their orders, as ``plan_round`` computes them.

    ``positions`` are the sampled clients' places in the roster, in request
    order; ``train`` and ``evaluation`` are ``plan_round``'s two orders over
    them, client ``k`` of each being ``positions[k]``.
    """

    round_id: int
    positions: list[int]
    train: RoundOrders
    evaluation: RoundOrders


def sampled_positions(roster: RosterPlan, round_id: int) -> list[int]:
    """The roster positions of the clients FedAvg samples in ``round_id``, in request order."""

    sampling = roster.sampling
    chosen = sampled_client_ids(
        roster.client_ids,
        sampling.seed,
        round_id,
        sampling.participation_rate,
        sampling.participation_probability,
    )
    return [roster.where[client_id] for client_id in chosen]


def plan_roster_round(roster: RosterPlan, round_id: int) -> PlannedRound:
    """``plan_round``'s training and post-fit orders for the round's sampled clients.

    The calls ``plan_round`` makes through ``_phase_orders`` and
    ``round_orders`` for clients whose task declares its orders: the phase
    ``fit`` over each client's training loader and loop, then ``eval`` over its
    post-fit loader for one epoch.
    """

    positions = sampled_positions(roster, round_id)
    owners = [(roster.client_ids[place], roster.seeds[place]) for place in positions]
    train_orders = [roster.train_orders[place] for place in positions]
    eval_orders = [roster.eval_orders[place] for place in positions]
    train = plan_orders(
        train_orders,
        [roster.loops[place] for place in positions],
        loader_seeds(train_orders, owners, round_id, "fit"),
    )
    evaluation = plan_orders(
        eval_orders,
        [LocalLoop(epochs=1)] * len(positions),
        loader_seeds(eval_orders, owners, round_id, "eval"),
    )
    return PlannedRound(round_id, positions, train, evaluation)


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------

_ORDER_FIELDS = ("indices", "lengths", "starts", "steps", "contiguous")


class _Ring:
    """Slots in shared memory a worker writes a planned round into: made once, shared once.

    Sized for the roster: every client sampled, each with its longest loop
    and widest batch. A round's orders fill the front of a slot; what crosses
    the queue is only where (``_header``). The loop copies a slot out as it
    takes the round, and the slot is handed out again.
    """

    def __init__(self, roster: RosterPlan, slots: int) -> None:
        full = _full_orders(roster)
        clients = len(roster)
        self.slots = slots
        self.positions = torch.zeros((slots, clients), dtype=torch.long).share_memory_()
        self.train = _slot_tensors(full[0], slots, clients)
        self.evaluation = _slot_tensors(full[1], slots, clients)

    def write(self, slot: int, planned: PlannedRound) -> tuple[int, ...]:
        """``planned`` into ``slot``; the header that says what of it is the round."""

        count = len(planned.positions)
        self.positions[slot, :count] = torch.tensor(planned.positions, dtype=torch.long)
        header: list[int] = [planned.round_id, slot, count]
        for tensors, orders in ((self.train, planned.train), (self.evaluation, planned.evaluation)):
            steps, width = orders.indices.shape[1], orders.indices.shape[2]
            tensors[0][slot, :count, :steps, :width] = orders.indices
            tensors[1][slot, :count, :steps] = orders.lengths
            tensors[2][slot, :count, :steps] = orders.starts
            tensors[3][slot, :count] = orders.steps
            tensors[4][slot, :count] = orders.contiguous
            header.extend((steps, width))
        return tuple(header)

    def read(
        self, header: tuple[int, ...], structures: tuple[list[Any], list[Any]]
    ) -> PlannedRound:
        """The round a header names, copied out of its slot, with each client's structures."""

        round_id, slot, count, train_steps, train_width, eval_steps, eval_width = header
        positions = self.positions[slot, :count].tolist()
        orders = []
        for tensors, steps, width, structure in (
            (self.train, train_steps, train_width, structures[0]),
            (self.evaluation, eval_steps, eval_width, structures[1]),
        ):
            orders.append(
                RoundOrders(
                    indices=tensors[0][slot, :count, :steps, :width].clone(),
                    lengths=tensors[1][slot, :count, :steps].clone(),
                    starts=tensors[2][slot, :count, :steps].clone(),
                    steps=tensors[3][slot, :count].clone(),
                    structure=[structure[place] for place in positions],
                    contiguous=tensors[4][slot, :count].clone(),
                )
            )
        return PlannedRound(round_id, positions, orders[0], orders[1])


def _full_orders(roster: RosterPlan) -> tuple[RoundOrders, RoundOrders]:
    """Every client's orders at once, for their sizes and structures, which no seed changes."""

    seeds = [0] * len(roster)
    return (
        plan_orders(list(roster.train_orders), list(roster.loops), seeds),
        plan_orders(list(roster.eval_orders), [LocalLoop(epochs=1)] * len(roster), seeds),
    )


def _slot_tensors(full: RoundOrders, slots: int, clients: int) -> tuple[Tensor, ...]:
    steps, width = full.indices.shape[1], full.indices.shape[2]
    return (
        torch.zeros((slots, clients, steps, width), dtype=torch.long).share_memory_(),
        torch.zeros((slots, clients, steps), dtype=torch.long).share_memory_(),
        torch.zeros((slots, clients, steps), dtype=torch.long).share_memory_(),
        torch.zeros((slots, clients), dtype=torch.long).share_memory_(),
        torch.zeros((slots, clients), dtype=torch.bool).share_memory_(),
    )


def _worker(roster: RosterPlan, ring: _Ring, tasks: Any, results: Any) -> None:
    """A planner process: plan each round it is handed into its slot, until it is handed None."""

    torch.set_num_threads(1)
    results.put(("ready",))
    while True:
        task = tasks.get()
        if task is None:
            return
        round_id, slot = task
        try:
            results.put(("round", ring.write(slot, plan_roster_round(roster, round_id))))
        except Exception as error:  # noqa: BLE001 -- reported to the loop, which plans itself
            results.put(("error", round_id, f"{type(error).__name__}: {error}"))


class RoundPlanner:
    """Hands out each round's :class:`PlannedRound`, planned ahead by worker processes.

    ``workers`` 0 plans every round in this process when it is asked for.
    Otherwise every round asked for before a worker has started is planned
    in this process, and so is the first one after; the rounds after that
    are handed to the workers ``ahead`` at a time, and a round already planned
    is taken as it is, one still being planned waited for. The record
    (run.json's ``executor.planner``) says how many workers ran, how many
    rounds the loop planned itself rather than a worker (``in_process``),
    how long it waited on them, and, if planning fell back to this process,
    why.
    """

    def __init__(
        self,
        roster: RosterPlan,
        last_round: int,
        workers: int = 0,
        ahead: int = DEFAULT_AHEAD,
        record: dict[str, Any] | None = None,
    ) -> None:
        self.roster = roster
        self.last_round = int(last_round)
        self.ahead = max(1, int(ahead))
        self.record = record if record is not None else {}
        self.record.update(workers=0, waited_sec=0.0)
        self._received: dict[int, PlannedRound] = {}
        self._next_task: int | None = None
        self._processes: list[Any] = []
        self._tasks: Any = None
        self._results: Any = None
        self._ring: _Ring | None = None
        self._free: list[int] = []
        self._structures: tuple[list[Any], list[Any]] = ([], [])
        #: Whether a worker has said it started.
        self._ready = False
        if workers > 0:
            self._start(workers)

    # -- the loop's side -----------------------------------------------------

    def plan(self, round_id: int) -> PlannedRound:
        """Round ``round_id``'s sampled clients and orders."""

        if not self._processes:
            return plan_roster_round(self.roster, round_id)
        if self._next_task is None:
            # No round handed out yet: this one is planned here, whether or
            # not a worker has started, and once one has, the workers take the
            # rounds after it.
            if self.workers_ready(READY_WAIT_SEC):
                self._next_task = round_id + 1
                self._submit(round_id)
            self.record["in_process"] += 1
            return plan_roster_round(self.roster, round_id)
        self._submit(round_id)
        planned = self._received.pop(round_id, None)
        if planned is None:
            planned = self._wait(round_id)
        return planned

    def workers_ready(self, timeout: float = 0.0) -> bool:
        """Whether a worker has started, waiting up to ``timeout`` seconds for one to say so.

        Before any round is handed out, a worker's word that it started is the
        only message the loop can receive.
        """

        deadline = time.monotonic() + timeout
        while not self._ready and self._processes:
            left = deadline - time.monotonic()
            try:
                message = (
                    self._results.get(timeout=left) if left > 0 else self._results.get_nowait()
                )
            except queue.Empty:
                return False
            except Exception:  # noqa: BLE001 -- a worker died mid-handover; the loop plans on
                return False
            self._ready = message[0] == "ready"
        return self._ready

    def close(self) -> None:
        """Stop the workers; planning continues in this process if asked again."""

        processes, self._processes = self._processes, []
        for _ in processes:
            try:
                self._tasks.put(None)
            except Exception:  # noqa: BLE001 -- a queue already closed has no reader
                break
        for process in processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        self._received.clear()

    # -- the workers -----------------------------------------------------------

    def _start(self, workers: int) -> None:
        try:
            ring = self._ring = _Ring(self.roster, self.ahead + 2)
            full = _full_orders(self.roster)
            self._structures = (full[0].structure, full[1].structure)
            self._free = list(range(ring.slots))
            context = torch.multiprocessing.get_context("spawn")
            self._tasks = context.Queue()
            self._results = context.Queue()
            with _main_unseen():
                for _ in range(workers):
                    process = context.Process(
                        target=_worker,
                        args=(self.roster, ring, self._tasks, self._results),
                        daemon=True,
                    )
                    process.start()
                    self._processes.append(process)
        except Exception as error:  # noqa: BLE001 -- the run plans in process instead
            self._fall_back(f"the planner workers could not start: {type(error).__name__}: {error}")
            return
        self.record.update(workers=len(self._processes), in_process=0)

    def _submit(self, round_id: int) -> None:
        """Hand the workers every round up to ``round_id + ahead`` not yet handed out."""

        assert self._next_task is not None
        stop = min(self.last_round, round_id + self.ahead)
        while self._next_task <= stop and self._free:
            self._tasks.put((self._next_task, self._free.pop(0)))
            self._next_task += 1

    def _wait(self, round_id: int) -> PlannedRound:
        started = time.perf_counter()
        try:
            while True:
                try:
                    message = self._results.get(timeout=WORKER_TIMEOUT_SEC)
                except queue.Empty:
                    return self._fall_back(
                        f"no planner worker returned round {round_id} in {WORKER_TIMEOUT_SEC:g} s",
                        round_id,
                    )
                except Exception as error:  # noqa: BLE001 -- a worker died mid-handover
                    return self._fall_back(
                        f"round {round_id}'s plan could not be received: "
                        f"{type(error).__name__}: {error}",
                        round_id,
                    )
                if message[0] == "ready":
                    continue
                if message[0] == "error":
                    return self._fall_back(
                        f"a planner worker raised on round {message[1]}: {message[2]}", round_id
                    )
                assert self._ring is not None
                header = message[1]
                planned = self._ring.read(header, self._structures)
                # Copied out: the slot can take another round.
                self._free.append(header[1])
                self._submit(round_id)
                if planned.round_id == round_id:
                    return planned
                self._received[planned.round_id] = planned
        finally:
            self.record["waited_sec"] = self.record["waited_sec"] + time.perf_counter() - started

    def _fall_back(self, reason: str, round_id: int | None = None) -> PlannedRound:
        """Plan in this process from now on, record why, and plan ``round_id`` if given."""

        self.close()
        self.record["fallback"] = reason[:300]
        print(f"planner: {reason}; planning in this process", file=sys.stderr, flush=True)
        return plan_roster_round(self.roster, round_id) if round_id is not None else None  # type: ignore[return-value]


def planned_for(roster: RosterPlan, planned: PlannedRound, plans: Sequence[Any]) -> bool:
    """Whether ``planned`` is the round of these plans: the same clients, loaders, loops and seeds.

    A round's plans come from its clients' rules; the planner's from the
    roster read at the start. They agree while the run's clients and their
    data are the ones it started with, which this checks rather than assumes.
    """

    if len(planned.positions) != len(plans):
        return False
    return all(
        plan.client_id == roster.client_ids[place]
        and plan.train_order == roster.train_orders[place]
        and plan.eval_order == roster.eval_orders[place]
        and plan.loop == roster.loops[place]
        and plan.seed == roster.seeds[place]
        for plan, place in zip(plans, planned.positions, strict=True)
    )


@contextmanager
def _main_unseen() -> Iterator[None]:
    """Start processes that do not run the parent's ``__main__`` again.

    A spawned process imports the parent's main module before its target, so
    a script that trains at import -- any driver without an ``if __name__ ==
    "__main__"`` guard -- would train again in every worker. A worker needs
    nothing from it: its target is this module's ``_worker``.
    """

    main = sys.modules.get("__main__")
    saved = {name: getattr(main, name) for name in ("__file__", "__spec__") if hasattr(main, name)}
    try:
        if main is not None:
            if "__file__" in saved:
                del main.__file__
            main.__spec__ = None
        yield
    finally:
        for name, value in saved.items():
            setattr(main, name, value)


def auto_workers() -> int:
    """The workers a run starts: its CPUs less two (the loop, and feeding the device), at most 4."""

    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    return max(0, min(MAX_WORKERS, available - 2))


# ---------------------------------------------------------------------------
# A run's roster
# ---------------------------------------------------------------------------


def roster_plan(components: Any) -> tuple[RosterPlan | None, str | None]:
    """The run's :class:`RosterPlan`, or None and why its rounds cannot be planned ahead.

    Planned ahead only where every round's orders are a function of the
    roster: FedAvg's own sampling, a rule whose loop and loaders are its
    class's (``batched_loop``, ``_train_loader_config``), one configuration for
    every client -- the factory builds each from the run's one config -- and a
    task that declares what its loaders yield. Each client's data is read
    once, here, for its training split's row count.
    """

    from fedbrew.clients.fedavg_client import FedAvgClient
    from fedbrew.clients.torch_sgd_client import TorchSGDClient, _get_train_data
    from fedbrew.servers.fedavg import FedAvgServer

    reason = _unplannable(components, FedAvgServer, (FedAvgClient, TorchSGDClient))
    if reason is not None:
        return None, reason
    clients, dataset, server = components.clients, components.dataset, components.server
    client_ids = tuple(str(client_id) for client_id in dataset.list_clients())
    representative = clients[client_ids[0]]
    task = representative.task
    train_config = representative._train_loader_config(1, seeded=False)
    eval_config = representative._eval_loader_config(1, seeded=False)
    loop = representative.batched_loop()
    train_orders, eval_orders = [], []
    for client_id in client_ids:
        data = _get_train_data(dataset.get_client_data(client_id))
        train_orders.append(task.loader_order(data, train_config))
        eval_orders.append(task.loader_order(data, eval_config))
    if any(order is None for order in (*train_orders, *eval_orders)):
        return None, "the task declares no order for a client's loader"
    size = len(client_ids)
    return (
        RosterPlan(
            client_ids=client_ids,
            seeds=(representative.base_seed,) * size,
            train_orders=tuple(train_orders),
            eval_orders=tuple(eval_orders),
            loops=(loop,) * size,
            sampling=Sampling(
                seed=int(server.seed),
                participation_rate=float(server.participation_rate or 1.0),
                participation_probability=server.participation_probability,
            ),
        ),
        None,
    )


def _unplannable(components: Any, server_class: type, rule_classes: tuple[type, ...]) -> str | None:
    """Why a run's rounds cannot be planned from its roster, or None."""

    server = components.server
    cls = type(server)
    if not isinstance(server, server_class) or any(
        getattr(cls, name) is not getattr(server_class, name)
        for name in ("sample_clients", "configure_round")
    ):
        return f"server {cls.__name__} does not sample as FedAvg samples"
    client_ids = list(components.dataset.list_clients())
    if not client_ids:
        return "the run has no clients"
    representative = components.clients[client_ids[0]]
    rule = type(representative)
    if not any(rule.batched_plan is base.batched_plan for base in rule_classes):
        return f"update rule {rule.__name__} plans its update its own way"
    if not callable(getattr(representative.task, "loader_order", None)):
        return f"task {type(representative.task).__name__} declares no loader order"
    if representative.base_seed is None:
        return "experiment.seed is unset"
    return None
