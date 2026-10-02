"""FedLADA server: the server step eta_g, the averaged second moment, the amended direction.

Implements the server half of FedLADA, the third algorithm of Section 3 of Sun
et al., "Efficient Federated Learning via Local Adaptive Amended Optimizer
with Linear Speedup" (arXiv:2308.00522), as AdaFed's port reads it
(``AdaFed/adafed/fl.py``). Each round, from the sampled clients' final models
``x_i``, the second moments ``v_hat_i`` they ended on, and their amended
directions ``(x_t - x_i) / (alpha_l K_i)``:

    x_bar   = (1/m) sum_i x_i
    x_{t+1} = x_t + eta_g (x_bar - x_t)
    g_a     = (1/m) sum_i (x_t - x_i) / (alpha_l K_i)  = (x_t - x_bar) / (alpha_l K)
    v       = (1/m) sum_i v_hat_i

and broadcasts ``x_{t+1}``, ``v`` and ``g_a``. ``g_a`` is the average
direction the clients moved in, per local step at the local rate: each client
amends its local AMSGrad step with it (``clients/torch_fedlada_client.py``).
Initialization is ``v = epsilon^2`` and ``g_a = 0``, so round 1 is local
AMSGrad scaled by ``alpha``.

The clients compute their own term of ``g_a`` from the steps they took, so a
client of another local-step count contributes its own direction; with every
client at ``K`` steps the average is the port's ``(x_t - x_bar) / (alpha_l K)``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from fedbrew.core.checkpointing import refuse_a_reconfigured_resume
from fedbrew.core.metrics import filter_metrics
from fedbrew.core.protocol import ClientInfo, FitRequest, FitResult, RoundInfo
from fedbrew.core.torch_utils import (
    StateDict,
    WeightedStateAccumulator,
    as_state_tensor,
    clone_model_state,
    refuse_non_finite_state,
    squared_l2_norm_model_state,
    validate_matching_keys,
)
from fedbrew.servers.fedavg import FedAvgServer, WeightedMetricAccumulator
from fedbrew.servers.fedlalr import _floating_state_like, _required_state
from fedbrew.tasks.base import TaskAdapter

#: The payload keys of the two broadcast states, and of the clients' returns.
SECOND_MOMENT_KEY = "second_moment_state"
AMENDED_DIRECTION_KEY = "amended_direction_state"


class FedLADAServer(FedAvgServer):
    """Step the model by ``eta_g`` toward the clients' mean; average ``v_hat`` and ``g_a``.

    Three model-shaped states cross the wire each way per round: ``x``, ``v``
    and ``g_a`` down, ``x_i``, ``v_hat_i`` and the client's term of ``g_a`` up.
    The paper averages uniformly; preflight notes ``aggregation_weighting:
    examples``.
    """

    def __init__(
        self,
        epsilon: float,
        server_learning_rate: float,
        participation_rate: float | None,
        seed: int,
        task: TaskAdapter | None = None,
        model_config: Mapping[str, Any] | None = None,
        metrics: list[str] | None = None,
        aggregation_weighting: str = "examples",
        participation_probability: float | None = None,
    ) -> None:
        """Configure FedAvg sampling, the server step and the second moment's start.

        Args:
            epsilon: ``client.epsilon``, read from the client block because it
                is the client's AMSGrad floor: ``v = epsilon ** 2`` before the
                first round. Finite and positive.
            server_learning_rate: ``eta_g``, finite and positive.
            participation_rate: As FedAvgServer.
            seed: As FedAvgServer.
            task: As FedAvgServer.
            model_config: As FedAvgServer.
            metrics: As FedAvgServer.
            aggregation_weighting: As FedAvgServer.
            participation_probability: As FedAvgServer.

        Raises:
            ValueError: If ``epsilon`` or ``server_learning_rate`` is not finite
                and positive.
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
        self.epsilon = _positive(epsilon, "epsilon")
        self.server_learning_rate = _positive(server_learning_rate, "server_learning_rate")
        self._second_moment: StateDict | None = None
        self._amended_direction: StateDict | None = None

    def initialize(self) -> dict[str, Any]:
        """Initialize the model, ``v = epsilon^2`` and ``g_a = 0``."""

        if self.task is None:
            raise ValueError("FedLADAServer requires a task to initialize")
        payload = super().initialize()
        self._ensure_state()
        return self._with_state(payload)

    def configure_round(
        self,
        round_info: RoundInfo,
        clients: Sequence[ClientInfo],
    ) -> Sequence[FitRequest]:
        """Broadcast ``x_t``, ``v`` and ``g_a`` to the sampled clients."""

        if self._model_state is None:
            self.initialize()
        self._ensure_state()
        payload = self._with_state(self._federated_payload())
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
        """Fold the round's results: see :meth:`aggregate_stream`."""

        return self.aggregate_stream(round_info, results)

    def aggregate_stream(
        self,
        round_info: RoundInfo,
        results: Iterable[FitResult],
    ) -> dict[str, Any]:
        """Average ``x_i``, ``v_hat_i`` and the clients' ``g_a`` terms; step by ``eta_g``."""

        if self._model_state is None or self._model_state_metadata is None:
            self.initialize()
        if self._model_state is None or self._model_state_metadata is None:
            raise ValueError("server model state metadata was not initialized")

        model_accumulator = WeightedStateAccumulator()
        second_moment_accumulator = WeightedStateAccumulator()
        direction_accumulator = WeightedStateAccumulator()
        metric_accumulator = WeightedMetricAccumulator()
        num_results = 0
        for result in results:
            weight = self._result_weight(result)
            model_accumulator.add(
                self._compatible_model_state(result), weight, source=result.client_id
            )
            second_moment_accumulator.add(
                _required_state(result, SECOND_MOMENT_KEY), weight, source=result.client_id
            )
            direction_accumulator.add(
                _required_state(result, AMENDED_DIRECTION_KEY), weight, source=result.client_id
            )
            metric_accumulator.add(result.metrics, result.num_examples)
            num_results += 1
        if not num_results:
            raise ValueError("FedLADA aggregate requires at least one result")

        # Every state is computed, and checked, before any is assigned: a
        # refused round leaves the server as it was.
        mean_model = model_accumulator.result()
        second_moment = second_moment_accumulator.result()
        direction = direction_accumulator.result()
        model_state = server_step(self._model_state, mean_model, self.server_learning_rate)
        refuse_non_finite_state(model_state, "FedLADA server model")
        self._model_state = model_state
        self._second_moment = second_moment
        self._amended_direction = direction

        metrics = metric_accumulator.result()
        metrics.update(
            {
                "second_moment_norm": squared_l2_norm_model_state(second_moment) ** 0.5,
                "amended_direction_norm": squared_l2_norm_model_state(direction) ** 0.5,
            }
        )
        metrics = filter_metrics(metrics, self.metrics)
        round_info.metrics.update(metrics)
        return self._with_state(self._federated_payload(metrics=metrics))

    def save_state(self) -> dict[str, Any]:
        """Return a checkpointable FedLADA server snapshot."""

        state = super().save_state()
        state[SECOND_MOMENT_KEY] = clone_model_state(self._second_moment or {})
        state[AMENDED_DIRECTION_KEY] = clone_model_state(self._amended_direction or {})
        state["epsilon"] = self.epsilon
        state["server_learning_rate"] = self.server_learning_rate
        return state

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore FedLADA server state; refuse a checkpoint of other settings."""

        refuse_a_reconfigured_resume(
            "fedlada server",
            state,
            {"epsilon": self.epsilon, "server_learning_rate": self.server_learning_rate},
        )
        super().load_state(state)
        second_moment = state.get(SECOND_MOMENT_KEY)
        self._second_moment = (
            clone_model_state(second_moment)
            if isinstance(second_moment, dict) and second_moment
            else None
        )
        direction = state.get(AMENDED_DIRECTION_KEY)
        self._amended_direction = (
            clone_model_state(direction) if isinstance(direction, dict) and direction else None
        )

    def _ensure_state(self) -> None:
        if self._model_state is None:
            raise ValueError("FedLADA state was not initialized")
        if self._second_moment is None:
            self._second_moment = _floating_state_like(self._model_state, self.epsilon**2)
        if self._amended_direction is None:
            self._amended_direction = _floating_state_like(self._model_state, 0.0)

    def _with_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._second_moment is None or self._amended_direction is None:
            raise ValueError("FedLADA state was not initialized")
        payload[SECOND_MOMENT_KEY] = clone_model_state(self._second_moment)
        payload[AMENDED_DIRECTION_KEY] = clone_model_state(self._amended_direction)
        return payload


def server_step(
    model: Mapping[str, Any], mean_model: Mapping[str, Any], server_learning_rate: float
) -> StateDict:
    """``x_t + eta_g (x_bar - x_t)``, on the floating tensors; the others are the mean's."""

    validate_matching_keys(model, mean_model)
    stepped: StateDict = {}
    for key in mean_model:
        mean = as_state_tensor(key, mean_model[key])
        if not mean.is_floating_point():
            stepped[key] = mean
            continue
        current = as_state_tensor(key, model[key]).to(mean.device)
        stepped[key] = current + server_learning_rate * (mean - current)
    return stepped


def _positive(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number
