"""FAFED server: synchronise the momentum and second moment, and take the round's last step.

Implements the server half of FAFED, Algorithm 2 of Wu et al., "Faster Adaptive
Federated Learning" (AAAI 2023, arXiv:2212.00974), with q = K and a constant
step and alpha, as AdaFed's port reads it (``AdaFed/adafed/fl.py``).

Before round 1 every client sends its gradient ``g_0`` at ``x_0`` on an initial
batch (``initial_requests``, ``absorb_initial``: the loop asks before round 1),
and the server starts the moments from them:

    m = (1/m) sum_i g_0,i      v = (1/m) sum_i g_0,i^2

Each round it broadcasts ``x_t``, ``m`` and ``v``; each client steps ``K - 1``
times and returns its last iterate ``x_i``, its ``m_i`` and its ``v_i``
(``clients/torch_fafed_client.py``); the server averages them and takes the
round's last step with the synchronised denominator:

    m = (1/m) sum_i m_i,  v = (1/m) sum_i v_i,  den = sqrt(v) + rho
    x_{t+1} = (1/m) sum_i x_i - eta m / den

Every client takes part every round: a client's momentum tracking reads its
own previous iterate, which would go stale while it sat out, and the paper has
no partial participation. ``participation_rate`` or
``participation_probability`` must be 1.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch

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
)
from fedbrew.servers.fedavg import FedAvgServer, WeightedMetricAccumulator
from fedbrew.servers.fedlalr import _required_state
from fedbrew.tasks.base import TaskAdapter

MOMENTUM_KEY = "momentum_state"
SECOND_MOMENT_KEY = "second_moment_state"
INITIAL_GRADIENT_KEY = "initial_gradient_state"
#: The flag of the initial request, and of round 1's (whose clients first step by ``-eta m``).
INITIAL_FLAG = "fafed_initial"
FIRST_ROUND_FLAG = "fafed_first_round"


class FAFEDServer(FedAvgServer):
    """Start the moments from the clients' initial gradients; average and step each round.

    Three model-shaped states cross the wire each way per round: ``x``, ``m``
    and ``v``. The paper averages uniformly; preflight notes ``examples``.
    """

    #: The server's momentum is the mean of the clients' tracked momenta, each
    #: built from the client's previous iterate: restoring the one without the
    #: others would track from a reset point (loop._refuse_a_half_restored_resume).
    coupled_client_state = {MOMENTUM_KEY: "previous_iterate"}

    def __init__(
        self,
        learning_rate: float,
        fafed_rho: float,
        participation_rate: float | None,
        seed: int,
        task: TaskAdapter | None = None,
        model_config: Mapping[str, Any] | None = None,
        metrics: list[str] | None = None,
        aggregation_weighting: str = "examples",
        participation_probability: float | None = None,
    ) -> None:
        """Configure the step the server takes and the denominator's offset.

        Args:
            learning_rate: ``client.learning_rate``, the step ``eta`` of every
                local step and of the server's last one. Finite and positive.
            fafed_rho: ``client.fafed_rho``, ``rho`` in ``sqrt(v) + rho``.
                Finite and positive.
            participation_rate: As FedAvgServer; 1 or None.
            seed: As FedAvgServer.
            task: As FedAvgServer.
            model_config: As FedAvgServer.
            metrics: As FedAvgServer.
            aggregation_weighting: As FedAvgServer.
            participation_probability: As FedAvgServer; 1 or None.

        Raises:
            ValueError: If a value is out of range, or participation is partial.
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
        for value in (participation_rate, participation_probability):
            if value is not None and float(value) != 1.0:
                raise ValueError("fafed takes every client every round: participation must be 1")
        self.learning_rate = _positive(learning_rate, "learning_rate")
        self.fafed_rho = _positive(fafed_rho, "fafed_rho")
        self._momentum: StateDict | None = None
        self._second_moment: StateDict | None = None
        #: Rounds aggregated so far; round 1's clients take the initial step.
        self._rounds = 0

    # -- before round 1 -------------------------------------------------------

    def initial_requests(self, clients: Sequence[ClientInfo]) -> Sequence[FitRequest]:
        """Ask every client for its gradient at ``x_0`` on its initial batch."""

        if self._model_state is None:
            self.initialize()
        payload = self._federated_payload()
        payload[INITIAL_FLAG] = True
        return [
            FitRequest(round_id=0, client_id=client.client_id, payload=payload)
            for client in clients
        ]

    def absorb_initial(self, results: Iterable[FitResult]) -> None:
        """``m`` and ``v`` from the clients' initial gradients: their mean and mean square."""

        gradients = WeightedStateAccumulator()
        squares = WeightedStateAccumulator()
        count = 0
        for result in results:
            weight = self._result_weight(result)
            gradient = _required_state(result, INITIAL_GRADIENT_KEY)
            gradients.add(gradient, weight, source=result.client_id)
            squares.add(
                {key: as_state_tensor(key, value).square() for key, value in gradient.items()},
                weight,
                source=result.client_id,
            )
            count += 1
        if not count:
            raise ValueError("FAFED's initial pass requires at least one client")
        self._momentum = gradients.result()
        self._second_moment = squares.result()

    # -- every round ------------------------------------------------------------

    def configure_round(
        self,
        round_info: RoundInfo,
        clients: Sequence[ClientInfo],
    ) -> Sequence[FitRequest]:
        """Broadcast ``x_t``, ``m`` and ``v``; round 1 says it is the first."""

        if self._model_state is None:
            self.initialize()
        payload = self._with_moments(self._federated_payload())
        payload[FIRST_ROUND_FLAG] = self._rounds == 0
        return [
            FitRequest(
                round_id=round_info.round_id,
                client_id=client.client_id,
                payload=payload,
                total_rounds=round_info.total_rounds,
            )
            for client in self.sample_clients(clients, round_info.round_id)
        ]

    def aggregate(self, round_info: RoundInfo, results: Sequence[FitResult]) -> dict[str, Any]:
        """Fold the round's results: see :meth:`aggregate_stream`."""

        return self.aggregate_stream(round_info, results)

    def aggregate_stream(
        self,
        round_info: RoundInfo,
        results: Iterable[FitResult],
    ) -> dict[str, Any]:
        """Average ``x_i``, ``m_i`` and ``v_i``; ``x_{t+1} = x_bar - eta m / (sqrt(v) + rho)``."""

        if self._model_state is None or self._model_state_metadata is None:
            self.initialize()
        if self._model_state is None or self._model_state_metadata is None:
            raise ValueError("server model state metadata was not initialized")

        model_accumulator = WeightedStateAccumulator()
        momentum_accumulator = WeightedStateAccumulator()
        second_moment_accumulator = WeightedStateAccumulator()
        metric_accumulator = WeightedMetricAccumulator()
        num_results = 0
        for result in results:
            weight = self._result_weight(result)
            model_accumulator.add(
                self._compatible_model_state(result), weight, source=result.client_id
            )
            momentum_accumulator.add(
                _required_state(result, MOMENTUM_KEY), weight, source=result.client_id
            )
            second_moment_accumulator.add(
                _required_state(result, SECOND_MOMENT_KEY), weight, source=result.client_id
            )
            metric_accumulator.add(result.metrics, result.num_examples)
            num_results += 1
        if not num_results:
            raise ValueError("FAFED aggregate requires at least one result")

        mean_model = model_accumulator.result()
        momentum = momentum_accumulator.result()
        second_moment = second_moment_accumulator.result()
        model_state = last_step(
            mean_model, momentum, second_moment, self.learning_rate, self.fafed_rho
        )
        refuse_non_finite_state(model_state, "FAFED server model")
        self._model_state = model_state
        self._momentum = momentum
        self._second_moment = second_moment
        self._rounds += 1

        metrics = metric_accumulator.result()
        metrics.update(
            {
                "momentum_norm": squared_l2_norm_model_state(momentum) ** 0.5,
                "second_moment_norm": squared_l2_norm_model_state(second_moment) ** 0.5,
            }
        )
        metrics = filter_metrics(metrics, self.metrics)
        round_info.metrics.update(metrics)
        return self._with_moments(self._federated_payload(metrics=metrics))

    def save_state(self) -> dict[str, Any]:
        """Return a checkpointable FAFED server snapshot."""

        state = super().save_state()
        state[MOMENTUM_KEY] = clone_model_state(self._momentum or {})
        state[SECOND_MOMENT_KEY] = clone_model_state(self._second_moment or {})
        state["fafed_rounds"] = self._rounds
        state["learning_rate"] = self.learning_rate
        state["fafed_rho"] = self.fafed_rho
        return state

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore FAFED server state; refuse a checkpoint of other settings."""

        refuse_a_reconfigured_resume(
            "fafed server",
            state,
            {"learning_rate": self.learning_rate, "fafed_rho": self.fafed_rho},
        )
        super().load_state(state)
        momentum = state.get(MOMENTUM_KEY)
        self._momentum = (
            clone_model_state(momentum) if isinstance(momentum, dict) and momentum else None
        )
        second_moment = state.get(SECOND_MOMENT_KEY)
        self._second_moment = (
            clone_model_state(second_moment)
            if isinstance(second_moment, dict) and second_moment
            else None
        )
        self._rounds = int(state.get("fafed_rounds", 0))

    def _with_moments(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._momentum is None or self._second_moment is None:
            raise ValueError(
                "FAFED's moments were not initialized: its initial pass, before round 1, "
                "did not run"
            )
        payload[MOMENTUM_KEY] = clone_model_state(self._momentum)
        payload[SECOND_MOMENT_KEY] = clone_model_state(self._second_moment)
        return payload


def denominator(second_moment: torch.Tensor, rho: float) -> torch.Tensor:
    """``sqrt(v) + rho``: the denominator of every FAFED step."""

    return second_moment.sqrt() + rho


def last_step(
    mean_model: Mapping[str, Any],
    momentum: Mapping[str, Any],
    second_moment: Mapping[str, Any],
    learning_rate: float,
    rho: float,
) -> StateDict:
    """``x_bar - eta m / (sqrt(v) + rho)`` where the moments hold a tensor; else the mean."""

    stepped: StateDict = {}
    for key, value in mean_model.items():
        mean = as_state_tensor(key, value)
        if key not in momentum:
            stepped[key] = mean
            continue
        m = as_state_tensor(key, momentum[key]).to(mean.device)
        v = as_state_tensor(key, second_moment[key]).to(mean.device)
        stepped[key] = mean - m * learning_rate / denominator(v, rho)
    return stepped


def _positive(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number
