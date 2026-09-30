"""SCAFFOLD server strategy with global control variates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from fedbrew.core.metrics import filter_metrics
from fedbrew.core.protocol import ClientInfo, FitRequest, FitResult, RoundInfo
from fedbrew.core.torch_utils import (
    StateDict,
    WeightedStateAccumulator,
    add_model_states,
    as_cpu_tensor,
    clone_model_state,
    copy_state_into,
    load_model_state,
    refuse_non_finite_state,
    scale_model_state,
    squared_l2_norm_model_state,
    validate_matching_keys,
    zeros_like_model_state,
)
from fedbrew.servers.fedavg import FedAvgServer, WeightedMetricAccumulator
from fedbrew.tasks.base import SupportsDatasetEvaluation, TaskAdapter


class ScaffoldServer(FedAvgServer):
    """SCAFFOLD server using FedAvg sampling with a server control variate.

    Carries one extra model-shaped tensor, the server control variate ``c``,
    broadcast alongside the weights and updated from the client control deltas.
    **This doubles the per-round communication volume in both directions**
    relative to a FedAvg arm: two model-shaped states out, two back. Any
    cost comparison against FedAvg has to account for that.
    """

    #: ``c`` is defined as the mean of the clients' ``c_i``, and the algorithm
    #: corrects each local step by ``c - c_i``. The two halves are therefore
    #: only meaningful together: a checkpoint that restores ``c`` while every
    #: ``c_i`` resets to zero makes the correction ``+c`` for every client, for
    #: the rest of the run, and neither side ever notices.
    coupled_client_state = {"server_control": "client_control"}

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
        """Configure FedAvg sampling and defer the control variate.

        Args are exactly FedAvgServer's and mean the same things. The server
        control variate is not built here: it is allocated as zeros in
        :meth:`initialize`, once the model state exists to shape it.
        """

        super().__init__(
            task=task,
            model_config=model_config,
            participation_rate=participation_rate,
            seed=seed,
            metrics=metrics,
            aggregation_weighting=aggregation_weighting,
            participation_probability=participation_probability,
        )
        self._server_control: StateDict | None = None
        self._num_clients = 0

    def initialize(self) -> dict[str, Any]:
        """Initialize global model and server control variate."""

        if self.task is None:
            raise ValueError("ScaffoldServer requires a task to initialize")
        # Delegate the model state to FedAvg rather than building it here: it is
        # what sets _model_state_scope / _model_state_metadata, and
        # _federated_payload (used by aggregate_stream every round) refuses to
        # emit a payload while either is None. Building the state locally left
        # both unset and blew up on the first aggregation, fresh or resumed.
        payload = super().initialize()
        if self._server_control is None:
            self._server_control = zeros_like_model_state(self._model_state)
        payload["server_control"] = clone_model_state(self._server_control)
        return payload

    def configure_round(
        self,
        round_info: RoundInfo,
        clients: Sequence[ClientInfo],
    ) -> Sequence[FitRequest]:
        """Create SCAFFOLD fit requests for selected clients."""

        if self._model_state is None or self._server_control is None:
            self.initialize()
        if self._model_state is None or self._server_control is None:
            raise ValueError("SCAFFOLD state was not initialized")

        self._num_clients = len(clients)
        selected_clients = self.sample_clients(clients, round_info.round_id)
        # Shared read-only broadcast payload; clients clone before loading.
        payload = {
            "model_state": clone_model_state(self._model_state),
            "server_control": clone_model_state(self._server_control),
            "total_num_clients": self._num_clients,
        }
        return [
            FitRequest(
                round_id=round_info.round_id,
                client_id=client.client_id,
                payload=payload,
                total_rounds=round_info.total_rounds,
            )
            for client in selected_clients
        ]

    def aggregate(
        self,
        round_info: RoundInfo,
        results: Sequence[FitResult],
    ) -> dict[str, Any]:
        """Aggregate client models and update the server control variate."""

        return self.aggregate_stream(round_info, results)

    def aggregate_stream(
        self,
        round_info: RoundInfo,
        results: Iterable[FitResult],
    ) -> dict[str, Any]:
        """Fold client models and control deltas in a single pass.

        Requires the client-roster size N, which reaches the server through
        :meth:`configure_round` or :meth:`load_state` and never through the
        results: the paper's ``c <- c + (1/N) sum(dc_i)`` is over the whole
        roster, and the round's own ``|S|`` is a different number by a factor
        of the participation rate.

        Raises:
            ValueError: If no results arrive, or if N is unknown.
        """

        return self._aggregate(round_info, results, None)

    def aggregate_folded(
        self,
        round_info: RoundInfo,
        results: Iterable[FitResult],
        mean: StateDict,
    ) -> dict[str, Any]:
        """``aggregate_stream`` of results whose models were already folded into ``mean``.

        The resident round (``fedbrew/core/resident.py``) folds a round's
        models on the device as ``WeightedStateAccumulator`` folds them, and
        refuses a round whose mean, control deltas or updated control are not
        finite by running it again through ``aggregate_stream``; everything
        else -- the control deltas' sum, ``c``'s update, the metrics and the
        payload -- is this method's, the same code as ``aggregate_stream``'s.
        """

        return self._aggregate(round_info, results, mean)

    def _aggregate(
        self,
        round_info: RoundInfo,
        results: Iterable[FitResult],
        mean: StateDict | None,
    ) -> dict[str, Any]:
        if self._model_state is None or self._server_control is None:
            self.initialize()
        if self._model_state is None or self._server_control is None:
            raise ValueError("SCAFFOLD state was not initialized")

        model_accumulator = WeightedStateAccumulator() if mean is None else None
        metric_accumulator = WeightedMetricAccumulator()
        summed_control_delta: StateDict | None = None
        total_control_delta_norm = 0.0
        num_results = 0
        for result in results:
            control_delta = result.payload.get("control_delta")
            if model_accumulator is not None:
                model_state = self._compatible_model_state(result)
                if not isinstance(control_delta, dict):
                    raise ValueError("SCAFFOLD fit result payload must contain control_delta")
                model_accumulator.add(
                    model_state, self._result_weight(result), source=result.client_id
                )
            elif not isinstance(control_delta, dict):
                raise ValueError("SCAFFOLD fit result payload must contain control_delta")
            # The model state is checked by the accumulator; the control delta
            # is summed outside it and was not checked at all, so a finite
            # model beside a NaN delta put NaN into server_control, where it
            # stays for the rest of the run and is checkpointed. POST-F27.
            refuse_non_finite_state(
                control_delta, f"control_delta from client {result.client_id!r}"
            )
            metric_accumulator.add(result.metrics, result.num_examples)
            if summed_control_delta is None:
                summed_control_delta = zeros_like_model_state(control_delta)
            summed_control_delta = _added_into(summed_control_delta, control_delta)
            total_control_delta_norm += squared_l2_norm_model_state(control_delta) ** 0.5
            num_results += 1

        if not num_results or summed_control_delta is None:
            raise ValueError("SCAFFOLD aggregate requires at least one result")
        if self._num_clients <= 0:
            # Was `self._num_clients = num_results`, which substitutes |S| for
            # N in the only place N appears. The paper's server control update
            # is c <- c + (1/N) sum(dc_i) over the whole roster; with |S| it
            # becomes (1/|S|) sum(dc_i), an update N/|S| too large -- 100x at
            # FEMNIST's participation_rate of 0.01 -- and the result is a
            # plausible number in the right units that nothing contradicts.
            #
            # Nothing reached it: loop.py:200 calls configure_round before
            # loop.py:206 calls aggregate_stream, and configure_round sets N
            # from the roster. So the branch existed only to turn a caller
            # error into a wrong answer. It now says what the caller skipped.
            raise ValueError(
                "SCAFFOLD does not know the client-roster size N: aggregate "
                "was called before configure_round, or after load_state on a "
                "checkpoint written without num_clients. N is the whole "
                "roster, not this round's sample, and is not recoverable from "
                "the results -- call configure_round first."
            )

        # Both halves are computed and checked before either is assigned, so a
        # refused round leaves the model and c exactly as they were: finite
        # deltas can still sum, or add to c, past the float range.
        new_model_state = model_accumulator.result() if model_accumulator is not None else mean
        refuse_non_finite_state(summed_control_delta, "summed control_delta")
        new_server_control = add_model_states(
            self._server_control,
            scale_model_state(summed_control_delta, 1.0 / self._num_clients),
        )
        refuse_non_finite_state(new_server_control, "updated server_control")
        self._model_state = new_model_state
        self._server_control = new_server_control

        # The server's own diagnostics first, then the run's one filter
        # (reporting.fit_metrics) over the whole round, so a column is kept or
        # dropped by the same list wherever it came from. Matches
        # servers/fedlalr.py.
        metrics = metric_accumulator.result()
        metrics.update(
            {
                "server_control_norm": squared_l2_norm_model_state(self._server_control) ** 0.5,
                "mean_client_control_delta_norm": total_control_delta_norm / num_results,
            }
        )
        metrics = filter_metrics(metrics, self.metrics)
        round_info.metrics.update(metrics)
        # Go through _federated_payload so adapter-scoped runs keep
        # model_state_scope / model_state_metadata, then add the control variate
        # that SCAFFOLD clients additionally need.
        payload = self._federated_payload(metrics=metrics)
        payload["server_control"] = clone_model_state(self._server_control)
        return payload

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
        if resident:
            copy_state_into(model, self._model_state)
        else:
            load_model_state(model, self._model_state)
        # A task without the optional whole-dataset evaluator produces no
        # central_test_* columns rather than failing -- twelve task doubles in
        # tests/ stop at the five abstract methods. The check is the protocol
        # rather than a getattr on a string, so the capability is declared in
        # one place and both servers read the same declaration.
        if not isinstance(self.task, SupportsDatasetEvaluation):
            return {}

        metrics = self.task.evaluate_model(model, global_data)
        return {f"global_{name}": value for name, value in metrics.items()}

    def save_state(self) -> dict[str, Any]:
        """Return a checkpointable SCAFFOLD server snapshot."""

        # super() carries model_state plus model_state_scope/metadata, so this
        # is a complete snapshot for any caller. The checkpoint writer drops
        # those three before writing, because it already holds them at the top
        # level; a resumed run gets them from there.
        state = super().save_state()
        state["server_control"] = clone_model_state(self._server_control or {})
        state["num_clients"] = self._num_clients
        return state

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore SCAFFOLD server state from checkpoint."""

        super().load_state(state)
        server_control = state.get("server_control")
        if isinstance(server_control, dict):
            self._server_control = clone_model_state(server_control)
        elif self._model_state is not None:
            self._server_control = zeros_like_model_state(self._model_state)
        self._num_clients = int(state.get("num_clients", self._num_clients))


def _added_into(total: StateDict, state: Mapping[str, Any]) -> StateDict:
    """``add_model_states(total, state)``, the sum written into ``total``, which the caller owns.

    The same additions: in place where the two tensors share a dtype, and
    ``add_model_states``' own, promoting, where they do not. A new sum per
    client was most of the fold's cost for a thousand clients' deltas.
    """

    validate_matching_keys(total, state)
    for key in total:
        right = as_cpu_tensor(key, state[key])
        left = total[key]
        if left.dtype == right.dtype:
            left.add_(right)
        else:
            total[key] = left + right
    return total
