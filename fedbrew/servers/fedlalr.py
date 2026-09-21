"""FedLALR server: averages the model, the momentum and the second moment.

Implements the server half of Algorithm 1 of Sun et al., "FedLALR:
Client-Specific Adaptive Learning Rates Achieve Linear Speedup for Non-IID
Data" (arXiv:2309.09719).

Unlike FedAdam, the adaptivity lives on the clients: each one runs AMSGrad
locally and so has its own per-parameter learning rate 1/sqrt(v_hat) for the
duration of a round. The server is the synchronization point that keeps those
rates from drifting apart forever -- it averages all three quantities and
broadcasts them back:

    x_{t+1} = (1/m) sum_i x_{t,K+1,i}
    m_t     = (1/m) sum_i m_{t,K,i}
    v_hat_t = (1/m) sum_i v_hat_{t,K,i}

Global initialization is m_{-1} = 0 and v_hat_{-1} = epsilon^2 (Algorithm 1
input line). The epsilon^2 floor is what keeps 1/sqrt(v_hat) finite: v_hat is
a running maximum seeded from it, and an average of values each at least
epsilon^2 is itself at least epsilon^2, so the bound survives every round.
That is why the client's update has no epsilon in its denominator.

The paper's Corollaries 2 and 3 describe variants -- not sending momenta, and
aggregating v_hat by maximum instead of mean. Neither is implemented here;
this is Algorithm 1.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from torch import Tensor

from fedbrew.core.checkpointing import refuse_a_reconfigured_resume
from fedbrew.core.metrics import filter_metrics
from fedbrew.core.protocol import ClientInfo, FitRequest, FitResult, RoundInfo
from fedbrew.core.torch_utils import (
    StateDict,
    WeightedStateAccumulator,
    clone_model_state,
    squared_l2_norm_model_state,
)
from fedbrew.servers.fedavg import FedAvgServer, WeightedMetricAccumulator
from fedbrew.tasks.base import TaskAdapter


class FedLALRServer(FedAvgServer):
    """Synchronize the model, momentum and second moment every round.

    Clients run local AMSGrad, so the optimizer state is part of what has to
    stay in sync: the server averages ``x``, ``m`` and ``v_hat`` together and
    broadcasts all three. **That is three model-shaped states per direction per
    round, 3x a FedAvg arm's volume** -- the dominant cost of the method and
    the thing to state in any comparison against it.

    The paper aggregates uniformly rather than by example count; preflight
    warns when ``aggregation_weighting`` is left at ``examples``.
    """

    def __init__(
        self,
        epsilon: float,
        participation_rate: float | None,
        seed: int,
        task: TaskAdapter | None = None,
        model_config: Mapping[str, Any] | None = None,
        metrics: list[str] | None = None,
        aggregation_weighting: str = "examples",
        participation_probability: float | None = None,
    ) -> None:
        """Configure FedAvg sampling plus the AMSGrad stabiliser.

        Args:
            epsilon: The AMSGrad denominator floor, in gradient units. Must be
                finite and positive. Read from ``client.epsilon`` rather than a
                separate server knob, because it is the same number by
                definition and two knobs that must agree eventually will not.
                It also seeds the second moment: ``v_hat_{-1} = epsilon ** 2``.
            participation_rate: As FedAvgServer.
            seed: As FedAvgServer.
            task: As FedAvgServer.
            model_config: As FedAvgServer.
            metrics: As FedAvgServer.
            aggregation_weighting: As FedAvgServer. ``uniform`` matches the
                paper.
            participation_probability: As FedAvgServer.

        Raises:
            ValueError: If ``epsilon`` is not finite and positive.
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
        # Read from client.epsilon rather than a second server knob: it is the
        # same number by definition, and two knobs that must agree eventually
        # will not.
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        self.epsilon = float(epsilon)
        self._momentum: StateDict | None = None
        self._second_moment: StateDict | None = None

    def initialize(self) -> dict[str, Any]:
        """Initialize the model, m_{-1} = 0 and v_hat_{-1} = epsilon^2."""

        if self.task is None:
            raise ValueError("FedLALRServer requires a task to initialize")
        # Delegate the model state to FedAvg: it is what sets
        # _model_state_scope / _model_state_metadata, without which
        # _federated_payload refuses to emit a payload.
        payload = super().initialize()
        self._ensure_optimizer_state()
        return self._with_optimizer_state(payload)

    def configure_round(
        self,
        round_info: RoundInfo,
        clients: Sequence[ClientInfo],
    ) -> Sequence[FitRequest]:
        """Broadcast x_t, m_{t-1} and v_hat_{t-1} to the selected clients."""

        if self._model_state is None:
            self.initialize()
        self._ensure_optimizer_state()
        # One shared read-only payload, as in FedAvg: clients clone before use.
        payload = self._with_optimizer_state(self._federated_payload())
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
        """Average the model, momentum and second moment."""

        return self.aggregate_stream(round_info, results)

    def aggregate_stream(
        self,
        round_info: RoundInfo,
        results: Iterable[FitResult],
    ) -> dict[str, Any]:
        """Fold all three synchronized states in a single pass."""

        if self._model_state is None or self._model_state_metadata is None:
            self.initialize()
        if self._model_state_metadata is None:
            raise ValueError("server model state metadata was not initialized")

        model_accumulator = WeightedStateAccumulator()
        momentum_accumulator = WeightedStateAccumulator()
        second_moment_accumulator = WeightedStateAccumulator()
        metric_accumulator = WeightedMetricAccumulator()
        # Per-client rates are what separates FedLALR from a shared server
        # step, so the spread across clients is the number to watch. Collected as one float
        # per client rather than a state, so it costs nothing.
        effective_learning_rates: list[float] = []
        num_results = 0

        for result in results:
            weight = self._result_weight(result)
            model_accumulator.add(self._compatible_model_state(result), weight)
            momentum_accumulator.add(
                _required_state(result, "momentum_state"),
                weight,
            )
            second_moment_accumulator.add(
                _required_state(result, "second_moment_state"),
                weight,
            )
            metric_accumulator.add(result.metrics, result.num_examples)
            rate = result.metrics.get("effective_learning_rate_coordinate_mean")
            if rate is not None:
                effective_learning_rates.append(float(rate))
            num_results += 1

        if not num_results:
            raise ValueError("FedLALR aggregate requires at least one result")

        self._model_state = model_accumulator.result()
        self._momentum = momentum_accumulator.result()
        self._second_moment = second_moment_accumulator.result()

        # Filter first, then add the server's own diagnostics, so server.metrics
        # governs the client-reported metrics and only those. Matches
        # servers/scaffold.py and the client-side convention in
        # torch_sgd_client.py, where the algorithm extras are likewise added
        # after client.metrics has been applied. Filtering these would let a
        # metrics list silently drop a column checkpointing.best_metric names.
        metrics = filter_metrics(metric_accumulator.result(), self.metrics)
        metrics.update(
            {
                "momentum_norm": squared_l2_norm_model_state(self._momentum) ** 0.5,
                "second_moment_norm": (squared_l2_norm_model_state(self._second_moment) ** 0.5),
            }
        )
        metrics.update(_dispersion_metrics(effective_learning_rates))
        round_info.metrics.update(metrics)
        self._round_metrics.append(metrics)
        return self._with_optimizer_state(self._federated_payload(metrics=metrics))

    def save_state(self) -> dict[str, Any]:
        """Return a checkpointable FedLALR server snapshot."""

        state = super().save_state()
        state["momentum_state"] = clone_model_state(self._momentum or {})
        state["second_moment_state"] = clone_model_state(self._second_moment or {})
        state["epsilon"] = self.epsilon
        return state

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore FedLALR server state from a checkpoint.

        Raises:
            ValueError: If the checkpoint disagrees with this run's config.
        """

        refuse_a_reconfigured_resume("fedlalr server", state, {"epsilon": self.epsilon})
        super().load_state(state)
        epsilon = state.get("epsilon")
        if epsilon is not None:
            if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
                raise ValueError("epsilon must be finite and positive")
            self.epsilon = float(epsilon)
        momentum = state.get("momentum_state")
        self._momentum = (
            clone_model_state(momentum) if isinstance(momentum, dict) and momentum else None
        )
        second_moment = state.get("second_moment_state")
        self._second_moment = (
            clone_model_state(second_moment)
            if isinstance(second_moment, dict) and second_moment
            else None
        )

    def _ensure_optimizer_state(self) -> None:
        """Build m and v_hat once the model state exists."""

        if self._model_state is None:
            raise ValueError("FedLALR state was not initialized")
        if self._momentum is None:
            self._momentum = _floating_state_like(self._model_state, 0.0)
        if self._second_moment is None:
            self._second_moment = _floating_state_like(
                self._model_state,
                self.epsilon**2,
            )

    def _with_optimizer_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._momentum is None or self._second_moment is None:
            raise ValueError("FedLALR optimizer state was not initialized")
        payload["momentum_state"] = clone_model_state(self._momentum)
        payload["second_moment_state"] = clone_model_state(self._second_moment)
        return payload


def _required_state(result: FitResult, name: str) -> Mapping[str, Any]:
    state = result.payload.get(name)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"client {result.client_id!r} fit payload must contain {name}")
    return state


def _floating_state_like(state: Mapping[str, Any], value: float) -> StateDict:
    """Return a constant CPU state over the floating-point entries only.

    Integer buffers are skipped rather than filled: epsilon**2 truncates to 0
    in an integer dtype, and the client would then divide by sqrt(0). Skipping
    them also keeps the key set identical to what the client sends back, which
    is what lets the accumulators above match states across clients.
    """

    result: StateDict = {}
    for key, tensor in state.items():
        if not isinstance(tensor, Tensor) or not tensor.is_floating_point():
            continue
        result[key] = tensor.detach().cpu().new_full(tensor.shape, float(value))
    if not result:
        raise ValueError("model state contains no floating-point tensors")
    return result


def _dispersion_metrics(values: Sequence[float]) -> dict[str, float]:
    """Summarise how far the clients' learning rates drifted apart this round.

    FedLALR's whole claim is that per-client rates should differ; a round where
    they are identical is a round where the method reduced to FedAvg-with-Adam.

    ``values`` holds one number per participating client that reported it:
    that client's ``effective_learning_rate_coordinate_mean``. Each client
    counts once, whatever its example count, and the std is the population
    one, dividing by the number of clients.
    """

    if not values:
        return {}
    count = float(len(values))
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    return {
        "effective_learning_rate_across_clients_mean": mean,
        "effective_learning_rate_across_clients_std": math.sqrt(variance),
        "effective_learning_rate_across_clients_min": min(values),
        "effective_learning_rate_across_clients_max": max(values),
    }
