"""FedLADA client: local AMSGrad amended by the server's average direction.

Implements the client half of FedLADA, the third algorithm of Section 3 of
Sun et al., "Efficient Federated Learning via Local Adaptive Amended Optimizer
with Linear Speedup" (arXiv:2308.00522), as AdaFed's port reads it
(``AdaFed/adafed/fl.py``). Each round the client starts from the broadcast
``x_t``, from ``m = 0`` and from ``v = v_hat = v`` (the server's average of the
clients' last ``v_hat``), and per local step, on the step's gradient ``g``:

    m     = beta1 m + (1 - beta1) g
    v     = beta2 v + (1 - beta2) g^2
    v_hat = max(v_hat, v)
    x     = x - alpha_l (alpha m / sqrt(v_hat) + (1 - alpha) g_a)

with ``g_a`` the server's amended direction (``servers/fedlada.py``),
``alpha`` = ``client.lada_alpha`` and ``alpha_l`` = ``client.learning_rate``.
No epsilon in the denominator: ``v_hat`` starts from the server's ``v``, which
is ``epsilon^2`` before round 1 and an average of running maxima after, so it
never falls below ``epsilon^2`` (FedLALR's argument, ``torch_fedlalr_client.py``).
The moment updates are FedLALR's, in its operation order.

It returns ``x_i``, ``v_hat_i`` and its term of the next ``g_a``,
``(x_t - x_i) / (alpha_l K_i)`` over its ``K_i`` local steps. Nothing is kept
between rounds.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from fedbrew.clients.local_update_modes import (
    FULL_GRADIENT_UPDATE_MODE,
    _GradientOnlyOptimizer,
    full_gradient_into_grad,
    own_loop_update_mode,
)
from fedbrew.clients.torch_fedlalr_client import (
    _broadcast_state,
    _to_cpu_state,
    _unit_interval_float,
)
from fedbrew.clients.torch_sgd_client import TorchSGDClient, _get_train_data
from fedbrew.core.checkpointing import refuse_a_reconfigured_resume
from fedbrew.core.federated_state import model_state_size, refuse_adapter_state
from fedbrew.core.metrics import filter_metrics
from fedbrew.core.protocol import FitRequest, FitResult
from fedbrew.servers.fedlada import AMENDED_DIRECTION_KEY, SECOND_MOMENT_KEY
from fedbrew.tasks.base import take_train_step


class TorchFedLADAClient(TorchSGDClient):
    """Run local AMSGrad amended by ``g_a`` from the broadcast second moment.

    Each of the ``local_iterations`` iterations is one pass of the client's
    training loader (``sequential_epoch``; one with-replacement batch under
    ``client.sampling: with_replacement``) or one step on the whole split's
    gradient (``full_gradient``), as for FedLALR.
    """

    def __init__(
        self,
        *args: Any,
        beta1: float,
        beta2: float,
        epsilon: float,
        lada_alpha: float,
        update_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure the amended local AMSGrad.

        Args:
            *args: Forwarded to the base client positionally.
            beta1: First-moment decay in [0, 1).
            beta2: Second-moment decay in [0, 1).
            epsilon: The AMSGrad floor the server seeds ``v`` with
                (``epsilon ** 2``); finite and positive. Kept here, and
                compared on resume, because it is the client's setting.
            lada_alpha: ``alpha`` in [0, 1], the weight of the local adaptive
                direction against the amended one.
            update_mode: ``sequential_epoch`` (also when unset) or
                ``full_gradient``.
            **kwargs: Forwarded to the base client by keyword.
        """

        super().__init__(*args, **kwargs)
        self.beta1 = _unit_interval_float(beta1, "beta1")
        self.beta2 = _unit_interval_float(beta2, "beta2")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be a finite positive number")
        self.epsilon = float(epsilon)
        self.lada_alpha = _closed_unit_float(lada_alpha, "lada_alpha")
        self.update_mode = own_loop_update_mode(update_mode)
        if self.momentum not in (None, 0.0):
            raise ValueError("fedlada carries its own momentum; set momentum: 0.0")
        if self.weight_decay not in (None, 0.0):
            raise ValueError("fedlada requires weight_decay: 0.0")
        if self.nesterov:
            raise ValueError("fedlada requires nesterov: false")
        if self.max_local_steps is not None:
            raise ValueError("fedlada does not support max_local_steps")
        if self.learning_rate_schedule not in (None, "constant"):
            raise ValueError("fedlada adapts its own rate; use no schedule")

    def fit(self, request: FitRequest) -> FitResult:
        """One round of amended local AMSGrad; return ``x_i``, ``v_hat_i`` and the ``g_a`` term."""

        train_data = _get_train_data(self.client_data)
        model = self.task.build_model(self.model_config)
        refuse_adapter_state(self, "fedlada", model)
        self._load_federated_payload(model, request.payload, context="fit request")
        if getattr(self.task, "_scaler", None) is not None:
            raise ValueError("fedlada does not support numerics.use_amp: true")

        device = self.task.device
        start = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        # v_hat and v from the server's v; m from zero, every round.
        second_moment_hat = _broadcast_state(request.payload, SECOND_MOMENT_KEY, device)
        second_moment = {name: tensor.clone() for name, tensor in second_moment_hat.items()}
        momentum = {name: torch.zeros_like(tensor) for name, tensor in second_moment_hat.items()}
        direction = _broadcast_state(request.payload, AMENDED_DIRECTION_KEY, device)

        train_loader = self._train_loader(train_data, request.round_id)
        training_outputs: list[Mapping[str, float]] = []
        local_steps = 0
        for _ in range(self.local_iterations):
            if self.update_mode == FULL_GRADIENT_UPDATE_MODE:
                training_outputs.extend(
                    full_gradient_into_grad(
                        task=self.task,
                        model=model,
                        train_loader=train_loader,
                        client_id=self.client_id,
                    )
                )
                self._step(model, momentum, second_moment, second_moment_hat, direction)
                local_steps += 1
                continue
            epoch_had_batch = False
            for batch in train_loader:
                epoch_had_batch = True
                optimizer = _GradientOnlyOptimizer(model.parameters())
                training_outputs.append(take_train_step(self.task, model, batch, optimizer))
                self._step(model, momentum, second_moment, second_moment_hat, direction)
                local_steps += 1
            if not epoch_had_batch:
                raise ValueError(f"client {self.client_id!r} has no training batches")
        if local_steps == 0:
            raise ValueError("fedlada local_steps must be positive")

        # This client's term of the next g_a: its direction per step at the local rate.
        per_step = float(self.learning_rate) * local_steps
        amended = {
            name: (start[name] - parameter.detach()) / per_step
            for name, parameter in model.named_parameters()
            if name in start
        }

        metrics, evaluated_num_examples = self._post_fit_evaluation(model, train_data, request)
        num_examples = self.task.federated_aggregation_weight(
            training_outputs, evaluated_num_examples
        )
        if num_examples < 0:
            raise ValueError("task federated aggregation weight must be non-negative")

        model_state = self.task.get_federated_model_state(model)
        model_state_metadata = self.task.federated_model_state_metadata(model)
        communicated_parameters, communicated_bytes = model_state_size(model_state)
        for state in (second_moment_hat, amended):
            state_parameters, state_bytes = model_state_size(state)
            communicated_parameters += state_parameters
            communicated_bytes += state_bytes
        metrics.update(
            {
                "local_steps": float(local_steps),
                "optimizer_steps": float(local_steps),
                "communicated_parameters": float(communicated_parameters),
                "communicated_bytes": float(communicated_bytes),
            }
        )
        return FitResult(
            round_id=request.round_id,
            client_id=self.client_id,
            num_examples=num_examples,
            payload={
                "model_state": model_state,
                "model_state_scope": str(model_state_metadata["model_state_scope"]),
                "model_state_metadata": model_state_metadata,
                SECOND_MOMENT_KEY: _to_cpu_state(second_moment_hat),
                AMENDED_DIRECTION_KEY: _to_cpu_state(amended),
            },
            metrics=filter_metrics(metrics, self.metrics),
        )

    @torch.no_grad()
    def _step(
        self,
        model: nn.Module,
        momentum: dict[str, Tensor],
        second_moment: dict[str, Tensor],
        second_moment_hat: dict[str, Tensor],
        direction: dict[str, Tensor],
    ) -> None:
        """One amended AMSGrad step in place, in the port's operation order."""

        alpha, beta1, beta2 = self.lada_alpha, self.beta1, self.beta2
        rate = float(self.learning_rate)
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            if name not in momentum or name not in direction:
                raise ValueError(f"missing FedLADA state for parameter: {name}")
            gradient = parameter.grad.detach()
            momentum[name].mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            second_moment[name].mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
            torch.maximum(second_moment_hat[name], second_moment[name], out=second_moment_hat[name])
            step = alpha * momentum[name] / second_moment_hat[name].sqrt()
            step = step + (1.0 - alpha) * direction[name]
            parameter.sub_(step * rate)

    def get_state(self) -> dict[str, Any]:
        """Checkpointable configuration; no optimizer state, all of it is rebroadcast."""

        state = super().get_state()
        state.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "lada_alpha": self.lada_alpha,
                "update_mode": self.update_mode,
            }
        )
        return state

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore configuration; refuse a checkpoint of other settings."""

        refuse_a_reconfigured_resume(
            "fedlada client",
            state,
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "lada_alpha": self.lada_alpha,
                "update_mode": self.update_mode,
            },
        )
        super().load_state(state)


def _closed_unit_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number in [0, 1]")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be a number in [0, 1]")
    return normalized
