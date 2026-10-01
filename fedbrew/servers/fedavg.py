"""FedAvg server strategy for model-state aggregation."""

from __future__ import annotations

import copy
import math
import random
import weakref
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from fedbrew.core.checkpointing import refuse_a_reconfigured_resume
from fedbrew.core.federated_state import (
    payload_model_state_scope,
    validate_federated_state_metadata,
    validate_state_matches,
)
from fedbrew.core.metrics import filter_metrics
from fedbrew.core.protocol import ClientInfo, EvalResult, FitRequest, FitResult, RoundInfo
from fedbrew.core.refusal import RunRefused
from fedbrew.core.seeding import derive_seed
from fedbrew.core.stacked_results import StackedFitResults, StackedResults
from fedbrew.core.torch_utils import (
    StackedRow,
    WeightedStateAccumulator,
    clone_model_state,
    copy_state_into,
)
from fedbrew.servers.base import ServerStrategy
from fedbrew.tasks.base import SupportsDatasetEvaluation, TaskAdapter

#: How much each client's model state counts in the aggregate.
#:   examples  weight by the client's example count (FedAvg as published)
#:   uniform   weight every participating client equally, 1/|S_t|
#: The papers behind the newer client rules (Delta-SGD, FedLALR) both
#: aggregate uniformly, so this is a knob rather than a rewrite: "examples"
#: stays the default and no existing baseline moves.
SUPPORTED_AGGREGATION_WEIGHTING = {"examples", "uniform"}


class WeightedMetricAccumulator:
    """Fold example-weighted client metrics without buffering the results.

    Each metric keeps its own weight total, so a metric only some clients
    report is averaged over just those clients rather than being diluted by the
    ones that never sent it.
    """

    def __init__(self) -> None:
        """Start with no metrics accumulated."""

        self._totals: dict[str, float] = {}
        self._weights: dict[str, int] = {}

    def add(self, metrics: Mapping[str, float], num_examples: int) -> None:
        """Add one client's metrics weighted by its example count.

        Args:
            metrics: Metric name to scalar value for one client. Names need not
                match those of any other client.
            num_examples: The client's example count, in examples, used as the
                weight. Zero contributes nothing and leaves the metric's
                divisor unchanged.
        """

        for name, value in metrics.items():
            self._totals[name] = self._totals.get(name, 0.0) + value * num_examples
            self._weights[name] = self._weights.get(name, 0) + num_examples

    def add_columns(
        self,
        columns: Mapping[str, Sequence[float]],
        counts: Sequence[int],
        reported: Mapping[str, Sequence[bool]] | None = None,
    ) -> None:
        """Add several clients' metrics, name by name, as ``add`` adds them client by client.

        ``columns[name][p]`` is client ``p``'s value and ``counts[p]`` its
        weight; ``reported[name]``, where given, says which clients report
        ``name``. A name's total is the same sum over the same clients in the
        same order as ``add`` makes it, so the result is the same to the bit.
        """

        for name, values in columns.items():
            mask = (reported or {}).get(name)
            total = self._totals.get(name)
            weight = self._weights.get(name, 0)
            for position, (value, count) in enumerate(zip(values, counts, strict=True)):
                if mask is not None and not mask[position]:
                    continue
                total = (0.0 if total is None else total) + value * count
                weight += count
            if total is not None:
                self._totals[name] = total
                self._weights[name] = weight

    def result(self) -> dict[str, float]:
        """Return the weighted mean of every accumulated metric.

        Returns:
            Metric name to ``sum(value * num_examples) / sum(num_examples)``,
            taken over the clients that reported that name. A metric whose
            weights sum to zero is omitted rather than raising.
        """

        return {
            name: total / self._weights[name]
            for name, total in self._totals.items()
            if self._weights[name]
        }


class FedAvgServer(ServerStrategy):
    """Server strategy that averages client model states by example count.

    Implements the streaming aggregation path, so peak memory is one model
    state regardless of how many clients participate: each incoming result is
    folded into a running weighted sum and then becomes unreachable.
    """

    streaming_aggregation = True

    #: Keys of :meth:`save_state` whose value only means anything alongside
    #: per-client state restored from the *same* checkpoint, mapped to the
    #: client-state key each is coupled to. Empty here, because FedAvg's
    #: clients carry nothing across rounds that a resume has to reconstruct.
    #: :func:`fedbrew.core.loop._refuse_a_half_restored_resume` reads it.
    coupled_client_state: Mapping[str, str] = {}

    def __init__(
        self,
        participation_rate: float | None,
        seed: int,
        task: TaskAdapter | None = None,
        model_config: Mapping[str, Any] | None = None,
        metrics: list[str] | None = None,
        aggregation_weighting: str = "examples",
        participation_probability: float | None = None,
    ) -> None:
        """Configure sampling, aggregation weighting and the model source.

        Args:
            participation_rate: Fraction of clients sampled per round, in
                (0, 1]: ``ceil(rate x clients)`` of them, at least one. 1.0 is
                full participation. None when ``participation_probability`` is
                set.
            seed: Base seed for client sampling. The per-round selection is
                derived from it and the round id, so the same seed replays the
                same client sequence.
            task: Task adapter used to build and read the model state. Required
                before :meth:`initialize`; may be None only for a strategy
                constructed purely to aggregate a supplied state.
            model_config: The ``model`` config block, forwarded to
                ``task.build_model``.
            metrics: Round metric names to record. Copied, not aliased.
            aggregation_weighting: ``"examples"`` to weight each client by its
                example count (FedAvg as published), or ``"uniform"`` to weight
                every participating client 1/|S_t|.
            participation_probability: Bernoulli participation instead: each
                client joins each round independently with this probability, in
                (0, 1]. The count varies by round and can be zero, and the loop
                skips aggregation on a round that selects no client. None when
                ``participation_rate`` is set.

        Raises:
            ValueError: If ``aggregation_weighting`` is neither supported value,
                or unless exactly one of the two participation arguments is set
                and in (0, 1].
        """

        self.task = task
        self.model_config = dict(model_config or {})
        self.participation_rate, self.participation_probability = _participation(
            participation_rate, participation_probability
        )
        self.seed = seed
        self.metrics = list(metrics or [])
        self.aggregation_weighting = _normalize_aggregation_weighting(aggregation_weighting)
        self._model_state: dict[str, Any] | None = None
        self._model_state_scope: str | None = None
        self._model_state_metadata: dict[str, Any] | None = None
        self._state_validated = False
        #: The stack whose rows this aggregation has checked against the
        #: state (_compatible_model_state).
        self._matched: weakref.ref[Any] | None = None

    def initialize(self) -> dict[str, Any]:
        """Initialize a global model state from the configured task."""

        if self.task is None:
            raise ValueError("FedAvgServer requires a task to initialize")
        if not self._state_validated:
            model = self.task.build_model(self.model_config)
            expected_metadata = self.task.federated_model_state_metadata(model)
            expected_scope = str(expected_metadata["model_state_scope"])
            if self._model_state is None:
                self._model_state = self.task.get_federated_model_state(model)
            else:
                received_scope = self._model_state_scope or "full"
                validate_federated_state_metadata(
                    expected_metadata,
                    self._model_state_metadata,
                    received_scope=received_scope,
                    context="server/checkpoint state",
                )
                self.task.load_federated_model_state(model, self._model_state)
                self._model_state = self.task.get_federated_model_state(model)
            self._model_state_scope = expected_scope
            self._model_state_metadata = expected_metadata
            self._state_validated = True
        return self._federated_payload()

    def sample_clients(
        self,
        clients: Sequence[ClientInfo],
        round_id: int = 0,
    ) -> list[ClientInfo]:
        """Select participating clients for a round.

        The draw is a function of the seed, the round and the client *ids* --
        not of the order the roster arrived in; ``sampled_client_ids`` draws
        it, and says why each half of that matters.
        """

        if not clients:
            return []
        if self.participation_probability is None and self.participation_rate >= 1.0:
            return list(clients)
        by_id = _clients_by_id(clients)
        chosen = sampled_client_ids(
            list(by_id),
            self.seed,
            round_id,
            self.participation_rate,
            self.participation_probability,
        )
        return [by_id[client_id] for client_id in chosen]

    def configure_round(
        self,
        round_info: RoundInfo,
        clients: Sequence[ClientInfo],
    ) -> Sequence[FitRequest]:
        """Create fit requests carrying the current global model state."""

        if self._model_state is None:
            self.initialize()
        if self._model_state is None:
            raise ValueError("model state was not initialized")

        # One shared read-only broadcast payload rather than a per-client deep
        # copy: every client clones the state before loading it, so cloning here
        # too costs one full model copy per selected client and nothing else.
        payload = self._federated_payload()
        return [
            FitRequest(
                round_id=round_info.round_id,
                client_id=client.client_id,
                payload=payload,
                total_rounds=round_info.total_rounds,
            )
            for client in self.sample_clients(clients, round_info.round_id)
        ]

    def aggregate(
        self,
        round_info: RoundInfo,
        results: Sequence[FitResult],
    ) -> dict[str, Any]:
        """Aggregate client model states and weighted metrics."""

        return self.aggregate_stream(round_info, results)

    def aggregate_stream(
        self,
        round_info: RoundInfo,
        results: Iterable[FitResult],
    ) -> dict[str, Any]:
        """Aggregate fit results in one pass without buffering client states."""

        averaged_state, metrics = self._accumulate_fit_results(results)
        self._model_state = averaged_state
        metrics = filter_metrics(metrics, self.metrics)
        round_info.metrics.update(metrics)
        return self._federated_payload(metrics=metrics)

    # -- the fold and the update from it ----------------------------------------
    #
    # What a strategy does with its clients' states, as functions the resident
    # round (fedbrew/core/resident.py) can run where the states are: on its
    # device, so the next round starts from the updated model without a copy.
    # The fold's weights are the server's (fold_weights); what it does with the
    # fold is update_from_fold. FedAvg adopts the mean; a strategy that updates
    # from it overrides the update's three methods (FedOptServer), and its
    # aggregate_stream makes the same update on the host's tensors, from the
    # same function.

    def fold_weights(self, counts: Sequence[int]) -> list[float]:
        """Each client's weight in the fold, from the example counts its results report."""

        if self.aggregation_weighting == "uniform":
            return [1.0] * len(counts)
        return [float(count) for count in counts]

    def update_from_fold(
        self,
        model: Mapping[str, Any],
        fold: Mapping[str, Any],
        carried: Mapping[str, dict[str, Any]] | None,
        round_info: RoundInfo,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]] | None]:
        """The model after a round whose clients' mean is ``fold``, and what it carries on.

        A pure function: it reads and writes no server state, and touches no
        device but its arguments'. ``carried`` is what the last round returned
        (named state of the model's shape, such as moments), None before the
        first update; ``round_info`` names the round. FedAvg's model is the
        mean itself, and carries nothing.
        """

        del model, carried, round_info
        return dict(fold), None

    def carried_state(self) -> dict[str, dict[str, Any]] | None:
        """What the server holds of ``update_from_fold``'s carried state; None for none."""

        return None

    def adopt_update(
        self,
        model: dict[str, Any],
        carried: Mapping[str, dict[str, Any]] | None,
        round_info: RoundInfo,
    ) -> dict[str, float]:
        """Take the model and carried state of a round ``update_from_fold`` computed elsewhere.

        Returns the server's own metrics of the round, which join the clients'
        before the run's one filter; FedAvg has none.
        """

        del carried, round_info
        self._model_state = model
        return {}

    def _accumulate_fit_results(
        self,
        results: Iterable[FitResult],
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """Fold fit results into a weighted mean state and weighted metrics.

        Consumes ``results`` exactly once so callers may pass a generator that
        discards each client's model state right after it is folded in.
        """

        if self._model_state is None or self._model_state_metadata is None:
            self.initialize()
        if self._model_state_metadata is None:
            raise ValueError("server model state metadata was not initialized")

        if isinstance(results, StackedResults) and self._folds_stacks():
            return self._accumulate_stacks(results.stacks())
        accumulator = WeightedStateAccumulator()
        metric_accumulator = WeightedMetricAccumulator()
        num_results = 0
        self._matched = None
        for result in results:
            model_state = self._compatible_model_state(result)
            accumulator.add(model_state, self._result_weight(result), source=result.client_id)
            metric_accumulator.add(result.metrics, result.num_examples)
            num_results += 1

        if not num_results:
            raise ValueError("FedAvg aggregate requires at least one result")
        return accumulator.result(), metric_accumulator.result()

    def _folds_stacks(self) -> bool:
        """Whether a round's stacks may be folded whole: this class's check and weight are used.

        A subclass that checks or weighs a result its own way is handed the
        results one by one (``StackedResults``), so it sees what it always saw.
        """

        cls = type(self)
        return (
            cls._compatible_model_state is FedAvgServer._compatible_model_state
            and cls._result_weight is FedAvgServer._result_weight
        )

    def _accumulate_stacks(
        self, stacks: Iterable[StackedFitResults]
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """``_accumulate_fit_results`` over stacked results, a stack at a time.

        The same checks, weights and sums as folding each stack's results in
        order: the state and metadata are checked once per stack, which is
        what the per-result check does for rows of one stack; the metrics are
        summed name by name over the clients in order, to the bit; the states
        are folded a stack at a time, as the rows of one stack always were.
        """

        assert self._model_state is not None and self._model_state_metadata is not None
        accumulator = WeightedStateAccumulator()
        metric_accumulator = WeightedMetricAccumulator()
        num_results = 0
        for stacked in stacks:
            if not len(stacked):
                continue
            self._compatible_stack(stacked)
            counts = stacked.counts()
            weights = self.fold_weights(counts)
            accumulator.add_stacked(stacked.states, weights, stacked.client_ids)
            columns, reported = stacked.metric_columns()
            metric_accumulator.add_columns(columns, counts, reported)
            num_results += len(stacked)
        if not num_results:
            raise ValueError("FedAvg aggregate requires at least one result")
        return accumulator.result(), metric_accumulator.result()

    def _compatible_stack(self, stacked: StackedFitResults) -> None:
        """``_compatible_model_state`` for every client of a stack at once.

        The metadata is one object's copies for every client, so it is
        checked once, as the first client's; the keys and shapes once per
        state stack, as its first client's in request order.
        """

        assert self._model_state is not None and self._model_state_metadata is not None
        context = f"fit result from client {stacked.client_ids[0]!r}"
        received_metadata = stacked.payload.get("model_state_metadata")
        validate_federated_state_metadata(
            self._model_state_metadata,
            received_metadata if isinstance(received_metadata, Mapping) else None,
            received_scope=payload_model_state_scope(stacked.payload, context=context),
            context=context,
        )
        for stack, positions in sorted(stacked.states, key=lambda part: min(part[1])):
            first = stacked.client_ids[min(positions)]
            validate_state_matches(
                self._model_state, stack.row(0), context=f"fit result from client {first!r}"
            )

    def _compatible_model_state(self, result: FitResult) -> dict[str, Any]:
        """Return ``result``'s model state, refusing one this server cannot fold.

        The one compatibility check every strategy built on this class runs on
        every fit result, before anything is accumulated: the state scope and,
        for adapters, the base-model and adapter identity
        (`validate_federated_state_metadata`), then the keys and shapes of the
        server's own state (`validate_state_matches`). SCAFFOLD and FedLALR
        used to fold results without it, so an adapter-scoped result was
        refused by FedAvg and averaged into a full-model server by both of
        them. FINDINGS.csv POST-F28.

        Raises:
            ValueError: If the payload has no model state, or it is
                incompatible with this server's.
        """

        if self._model_state is None or self._model_state_metadata is None:
            raise ValueError("server model state metadata was not initialized")
        context = f"fit result from client {result.client_id!r}"
        model_state = result.payload.get("model_state")
        if not isinstance(model_state, dict):
            raise ValueError("fit result payload must contain model_state")
        received_metadata = result.payload.get("model_state_metadata")
        validate_federated_state_metadata(
            self._model_state_metadata,
            received_metadata if isinstance(received_metadata, Mapping) else None,
            received_scope=payload_model_state_scope(result.payload, context=context),
            context=context,
        )
        # Rows of one stack share their keys and shapes, so within one
        # aggregation the first row checked is every row's check.
        stack = model_state.stack if isinstance(model_state, StackedRow) else None
        if stack is None or self._matched is None or self._matched() is not stack:
            validate_state_matches(self._model_state, model_state, context=context)
            if stack is not None:
                self._matched = weakref.ref(stack)
        return model_state

    def _result_weight(self, result: FitResult) -> float:
        """Return the aggregation weight for one client's model state.

        Only the model update is affected. Client-reported metrics stay
        example-weighted in every mode: how much a client's parameters count is
        an algorithm choice, while a metric's weighting is what makes the
        reported number a population mean.
        """

        if self.aggregation_weighting == "uniform":
            return 1.0
        return float(result.num_examples)

    def evaluate_global(self, global_data: Any, model: Any = None) -> dict[str, float]:
        """Evaluate the current global model on global data if available.

        ``model`` is one the caller keeps (the batched evaluator's): the state
        is copied into it in place, rather than a model being built and the
        state cloned into it. The same numbers either way.
        """

        if global_data is None:
            return {}
        if self._model_state is None:
            self.initialize()
        if self._model_state is None or self.task is None:
            return {}

        resident = model is not None
        if model is None:
            model = self.task.build_model(self.model_config)
        expected_metadata = self.task.federated_model_state_metadata(model)
        validate_federated_state_metadata(
            expected_metadata,
            self._model_state_metadata,
            received_scope=self._model_state_scope or "full",
            context="server evaluation state",
        )
        if resident:
            copy_state_into(model, self._model_state)
        else:
            self.task.load_federated_model_state(model, self._model_state)
        # A task without the optional whole-dataset evaluator produces no
        # central_test_* columns rather than failing -- twelve task doubles in
        # tests/ stop at the five abstract methods. The check is the protocol
        # rather than a getattr on a string, so the capability is declared in
        # one place and both servers read the same declaration.
        if not isinstance(self.task, SupportsDatasetEvaluation):
            return {}

        metrics = self.task.evaluate_model(model, global_data)
        return {f"global_{name}": value for name, value in metrics.items()}

    def evaluate(
        self,
        round_info: RoundInfo,
        results: Sequence[EvalResult],
    ) -> dict[str, float]:
        """Aggregate weighted evaluation metrics."""

        accumulator = WeightedMetricAccumulator()
        for result in results:
            accumulator.add(result.metrics, result.num_examples)
        metrics = filter_metrics(accumulator.result(), self.metrics)
        round_info.metrics.update(metrics)
        return metrics

    def save_state(self) -> dict[str, Any]:
        """Return a snapshot of server state."""

        return {
            "model_state": clone_model_state(self._model_state or {}),
            "model_state_scope": self._model_state_scope or "full",
            "model_state_metadata": copy.deepcopy(
                self._model_state_metadata or {"model_state_scope": "full"}
            ),
            # No per-round history: it made every checkpoint grow with the
            # round count, and nothing restored from it read it. The run's
            # history is round_metrics.csv, which a resume replays.
            "metrics": list(self.metrics),
            "aggregation_weighting": self.aggregation_weighting,
            # Compared on resume and restored by nothing. A setting absent here
            # is one no resume can compare, so an edited value would be taken
            # silently while run.json recorded it for the whole run. Both
            # sampling keys are written, the unused one as None: which scheme a
            # run used is itself the setting, so a resume that swapped schemes
            # has to be refused as loudly as one that moved a value.
            "participation_rate": self.participation_rate,
            "participation_probability": self.participation_probability,
            "seed": self.seed,
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore a snapshot of server state.

        Raises:
            ValueError: If the checkpoint's `aggregation_weighting`,
                `participation_rate`, `participation_probability` or `seed` is
                not this run's. `metrics` is exempt -- which columns a run
                writes is bookkeeping, and the per-client CSV cursor already
                handles a changed header.
        """

        refuse_a_reconfigured_resume(
            f"{type(self).__name__.replace('Server', '').lower()} server",
            state,
            {
                "aggregation_weighting": self.aggregation_weighting,
                "participation_rate": self.participation_rate,
                "participation_probability": self.participation_probability,
                "seed": self.seed,
            },
        )
        model_state = state.get("model_state")
        if isinstance(model_state, dict):
            self._model_state = clone_model_state(model_state)
            self._model_state_scope = payload_model_state_scope(
                state,
                context="checkpoint server_state",
            )
            metadata = state.get("model_state_metadata")
            if self._model_state_scope == "adapter" and not isinstance(metadata, Mapping):
                raise RunRefused("checkpoint adapter server_state requires model_state_metadata")
            self._model_state_metadata = (
                copy.deepcopy(dict(metadata)) if isinstance(metadata, Mapping) else None
            )
            self._state_validated = False
        # A checkpoint written before the history was dropped still carries
        # it under "round_metrics"; it is ignored, as it always went unread.
        metrics = state.get("metrics", self.metrics)
        if isinstance(metrics, list):
            self.metrics = [str(metric) for metric in metrics]
        weighting = state.get("aggregation_weighting")
        if weighting is not None:
            self.aggregation_weighting = _normalize_aggregation_weighting(weighting)

    def _federated_payload(
        self,
        *,
        metrics: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        if (
            self._model_state is None
            or self._model_state_scope is None
            or self._model_state_metadata is None
        ):
            raise ValueError("server model state was not initialized")
        payload: dict[str, Any] = {
            "model_state": clone_model_state(self._model_state),
            "model_state_scope": self._model_state_scope,
            "model_state_metadata": copy.deepcopy(self._model_state_metadata),
        }
        if metrics is not None:
            payload["metrics"] = dict(metrics)
        return payload


def _participation(
    rate: float | None,
    probability: float | None,
) -> tuple[float | None, float | None]:
    """Refuse anything but exactly one participation scheme, in (0, 1]."""

    if (rate is None) == (probability is None):
        raise ValueError(
            "exactly one of participation_rate and participation_probability must be set"
        )
    for name, value in (("participation_rate", rate), ("participation_probability", probability)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value <= 1
        ):
            raise ValueError(f"{name} must be in (0, 1], got {value!r}")
    return rate, probability


def sampled_client_ids(
    client_ids: Sequence[str],
    seed: int,
    round_id: int,
    participation_rate: float,
    participation_probability: float | None,
) -> list[str]:
    """The ids ``FedAvgServer.sample_clients`` selects from a roster of distinct ids, in its order.

    A function of the seed, the round and the ids alone -- nothing a round
    trains -- so a planner can draw any round's clients ahead of it
    (``fedbrew/core/round_planner.py``) with the very draws the server makes.

    The draw is over the ids sorted by name, not the order the roster arrived
    in, for the reasons ``sample_clients`` states. Under
    ``participation_probability`` each id is kept independently and the kept
    ids are returned in roster order, so probability 1.0 selects exactly what
    rate 1.0 does, in the same order; under a fixed ``participation_rate``
    they are returned in the order drawn.
    """

    if not client_ids:
        return []
    if participation_probability is None and participation_rate >= 1.0:
        return list(client_ids)
    # Hashed, not added. Random(seed + round_id) makes seeds s and s+1 one
    # sequence read from two offsets: seed 43 sees at round r exactly the
    # clients seed 42 saw at round r+1, for every round but the last. A
    # replicate set built that way shares its participation schedule, so the
    # spread across it leaves out the part of run-to-run variance that comes
    # from which clients take part.
    rng = random.Random(derive_seed(seed, "participation", round_id))
    if participation_probability is not None:
        chosen = {
            client_id
            for client_id in sorted(client_ids)
            if rng.random() < participation_probability
        }
        return [client_id for client_id in client_ids if client_id in chosen]
    sample_size = max(1, math.ceil(len(client_ids) * participation_rate))
    # Sorted, for the reason loop._sampled_client_infos states about its own
    # draw: so the schedule does not depend on the order the dataset happens
    # to list clients in. That order is clients.jsonl's line order, which
    # both generators write sorted -- so this changes nothing about a dataset
    # either of them produced. What it removes is the coupling: FEMNIST's
    # _client_id rewrites any writer id that is not filename-safe to
    # writer_<sha256[:16]>, and one such writer is enough to make the file
    # order differ from the id order, at which point every round's
    # participants change with nothing to signal it. Measured on a 3597-writer
    # roster in that state: 0 of 36 clients in common per round at
    # participation_rate 0.01.
    return rng.sample(sorted(client_ids), sample_size)


def _clients_by_id(clients: Sequence[ClientInfo]) -> dict[str, ClientInfo]:
    by_id = {client.client_id: client for client in clients}
    if len(by_id) != len(clients):
        # Keying by id makes a duplicate id silently shrink the population,
        # which would then draw fewer clients than participation asks for --
        # or raise "Sample larger than population" from inside random, which
        # names neither the roster nor the id.
        raise ValueError(
            f"client roster has {len(clients)} entries under "
            f"{len(by_id)} distinct client_ids; ids must be unique"
        )
    return by_id


def _normalize_aggregation_weighting(value: Any) -> str:
    """Validate and normalize an aggregation weighting mode."""

    if not isinstance(value, str) or value not in SUPPORTED_AGGREGATION_WEIGHTING:
        raise ValueError(
            "aggregation_weighting must be one of: "
            + ", ".join(sorted(SUPPORTED_AGGREGATION_WEIGHTING))
        )
    return value
