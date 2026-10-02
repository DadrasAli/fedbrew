"""FAFED client: momentum tracking with a second gradient at the previous iterate.

Implements the client half of FAFED, Algorithm 2 of Wu et al., "Faster Adaptive
Federated Learning" (AAAI 2023, arXiv:2212.00974), with q = K and a constant
step ``eta`` and ``alpha``, as AdaFed's port reads it (``AdaFed/adafed/fl.py``).

Before round 1 (the server's initial request) the client returns its gradient
``g_0`` at ``x_0`` on an initial batch -- every row under ``full_gradient``,
otherwise one batch of ``min(n, batch_size * local_iterations)`` rows drawn as
its training loader draws -- and keeps ``x_0`` as its previous iterate.

Each round it starts from the broadcast ``x_t``, ``m`` and ``v``, with
``den = sqrt(v) + rho``. In round 1 it first steps ``x = x_0 - eta m``
(Algorithm 2's line 3 as printed, without ``A_0^-1``: round 1 applies one more
update than the others). Then per local step, on one batch, with ``g`` the
gradient at ``x`` and ``g_p`` the gradient at the previous iterate on the same
batch:

    m    = g + (1 - alpha) (m - g_p)
    v    = beta v + (1 - beta) g^2
    prev = x
    x    = x - eta m / den            (every step but the round's last)

The round's last step is the server's, with the synchronised ``m`` and ``den``
(``servers/fafed.py``); the client returns its last iterate, ``m`` and ``v``.
The previous iterate is the client's own and is kept across rounds, in its
checkpointed state. Two gradients per local step.
"""

from __future__ import annotations

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
from fedbrew.clients.sampling import WITH_REPLACEMENT, ReplacementLoader, split_row_count
from fedbrew.clients.torch_fedlada_client import _closed_unit_float
from fedbrew.clients.torch_fedlalr_client import _broadcast_state, _to_cpu_state
from fedbrew.clients.torch_sgd_client import TorchSGDClient, _get_train_data
from fedbrew.core.checkpointing import refuse_a_reconfigured_resume
from fedbrew.core.federated_state import model_state_size, refuse_adapter_state
from fedbrew.core.metrics import filter_metrics
from fedbrew.core.protocol import FitRequest, FitResult
from fedbrew.core.torch_utils import clone_model_state
from fedbrew.servers.fafed import (
    FIRST_ROUND_FLAG,
    INITIAL_FLAG,
    INITIAL_GRADIENT_KEY,
    MOMENTUM_KEY,
    SECOND_MOMENT_KEY,
    denominator,
)
from fedbrew.tasks.base import take_train_step

#: The dataloader phase of the initial batch, so its draw is its own (``dataloader_seed``).
INITIAL_PHASE = "fafed_initial"


class TorchFAFEDClient(TorchSGDClient):
    """FAFED's local momentum tracking; its previous iterate is kept between rounds."""

    def __init__(
        self,
        *args: Any,
        beta2: float,
        fafed_alpha: float,
        fafed_rho: float,
        update_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure FAFED's local loop.

        Args:
            *args: Forwarded to the base client positionally.
            beta2: ``beta`` in ``v = beta v + (1 - beta) g^2``, in [0, 1).
            fafed_alpha: ``alpha`` in the momentum tracking, in [0, 1].
            fafed_rho: ``rho`` in ``sqrt(v) + rho``, finite and positive.
            update_mode: ``sequential_epoch`` (also when unset) or ``full_gradient``.
            **kwargs: Forwarded to the base client by keyword.
        """

        super().__init__(*args, **kwargs)
        self.beta2 = _half_open_unit(beta2, "beta2")
        self.fafed_alpha = _closed_unit_float(fafed_alpha, "fafed_alpha")
        rho = float(fafed_rho)
        if not torch.isfinite(torch.tensor(rho)) or rho <= 0.0:
            raise ValueError("fafed_rho must be a finite positive number")
        self.fafed_rho = rho
        self.update_mode = own_loop_update_mode(update_mode)
        if self.momentum not in (None, 0.0):
            raise ValueError("fafed carries its own momentum; set momentum: 0.0")
        if self.weight_decay not in (None, 0.0):
            raise ValueError("fafed requires weight_decay: 0.0")
        if self.nesterov:
            raise ValueError("fafed requires nesterov: false")
        if self.max_local_steps is not None:
            raise ValueError("fafed does not support max_local_steps")
        if self.learning_rate_schedule not in (None, "constant"):
            raise ValueError("fafed takes a constant step; use no schedule")
        #: The previous local iterate, by parameter name; set by the initial pass.
        self._previous: dict[str, Tensor] | None = None

    def fit(self, request: FitRequest) -> FitResult:
        """The initial gradient, or one round of FAFED's local loop."""

        train_data = _get_train_data(self.client_data)
        model = self.task.build_model(self.model_config)
        refuse_adapter_state(self, "fafed", model)
        self._load_federated_payload(model, request.payload, context="fit request")
        if getattr(self.task, "_scaler", None) is not None:
            raise ValueError("fafed does not support numerics.use_amp: true")
        if request.payload.get(INITIAL_FLAG):
            return self._initial_result(model, train_data, request)
        return self._round_result(model, train_data, request)

    # -- before round 1 ---------------------------------------------------------

    def _initial_result(self, model: nn.Module, train_data: Any, request: FitRequest) -> FitResult:
        """``g_0`` at ``x_0`` on the initial batch; ``x_0`` kept as the previous iterate."""

        if self.update_mode == FULL_GRADIENT_UPDATE_MODE:
            full_gradient_into_grad(
                task=self.task,
                model=model,
                train_loader=self._train_loader(train_data, request.round_id),
                client_id=self.client_id,
            )
        else:
            batch = next(iter(self._initial_loader(train_data)))
            take_train_step(self.task, model, batch, _GradientOnlyOptimizer(model.parameters()))
        gradient = _gradients(model)
        self._previous = _parameters(model)
        return FitResult(
            round_id=request.round_id,
            client_id=self.client_id,
            # The roster's count, which the loop gives every client (setup).
            num_examples=max(int(getattr(self, "_num_examples", 0) or 0), 1),
            payload={INITIAL_GRADIENT_KEY: _to_cpu_state(gradient)},
            metrics={},
        )

    def _initial_loader(self, train_data: Any) -> Any:
        """One batch of ``min(n, batch_size * local_iterations)`` rows, drawn as training draws."""

        size = self.batch_size * self.local_iterations
        seed = self._loader_seed(0, INITIAL_PHASE)
        if self.train_sampling == WITH_REPLACEMENT:
            size = min(split_row_count(self.task, train_data), size)
            return ReplacementLoader(tuple(self.task.split_rows(train_data)), size, seed)
        # The task's loader's first batch: min(n, size) rows, as drop_last is off.
        config = {"batch_size": size, "shuffle": self.train_shuffle, "drop_last": False}
        if seed is not None:
            config["seed"] = seed
        return self.task.build_dataloader(train_data, config)

    # -- a round -----------------------------------------------------------------

    def _round_result(self, model: nn.Module, train_data: Any, request: FitRequest) -> FitResult:
        if self._previous is None:
            raise ValueError(
                f"fafed client {self.client_id!r} has no previous iterate: FAFED's initial "
                "pass, before round 1, did not reach it, and a resume restored no client state"
            )
        device = self.task.device
        momentum = _broadcast_state(request.payload, MOMENTUM_KEY, device)
        second_moment = _broadcast_state(request.payload, SECOND_MOMENT_KEY, device)
        den = {name: denominator(tensor, self.fafed_rho) for name, tensor in second_moment.items()}
        rate = float(self.learning_rate)
        previous = self.task.build_model(self.model_config)
        self._load_federated_payload(previous, request.payload, context="fit request")
        if request.payload.get(FIRST_ROUND_FLAG):
            # Algorithm 2's line 3 as printed: x_1 = x_0 - eta m_0, without A_0^-1.
            self._previous = _parameters(model)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in momentum:
                        parameter.sub_(momentum[name] * rate)
        _copy_into(previous, self._previous)

        train_loader = self._train_loader(train_data, request.round_id)
        training_outputs, step = self._local_steps(
            model, previous, train_loader, momentum, second_moment, den
        )
        self._previous = _parameters(previous)

        metrics, evaluated_num_examples = self._post_fit_evaluation(model, train_data, request)
        num_examples = self.task.federated_aggregation_weight(
            training_outputs, evaluated_num_examples
        )
        if num_examples < 0:
            raise ValueError("task federated aggregation weight must be non-negative")
        model_state = self.task.get_federated_model_state(model)
        model_state_metadata = self.task.federated_model_state_metadata(model)
        communicated_parameters, communicated_bytes = model_state_size(model_state)
        for state in (momentum, second_moment):
            state_parameters, state_bytes = model_state_size(state)
            communicated_parameters += state_parameters
            communicated_bytes += state_bytes
        metrics.update(
            {
                "local_steps": float(step),
                "optimizer_steps": float(step),
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
                MOMENTUM_KEY: _to_cpu_state(momentum),
                SECOND_MOMENT_KEY: _to_cpu_state(second_moment),
            },
            metrics=filter_metrics(metrics, self.metrics),
        )

    def _local_steps(
        self,
        model: nn.Module,
        previous: nn.Module,
        train_loader: Any,
        momentum: dict[str, Tensor],
        second_moment: dict[str, Tensor],
        den: dict[str, Tensor],
    ) -> tuple[list[Mapping[str, float]], int]:
        """The round's local steps, two gradients each; the outputs and the step count."""

        rate = float(self.learning_rate)
        full = self.update_mode == FULL_GRADIENT_UPDATE_MODE
        total = self.local_iterations * (1 if full else len(train_loader))
        training_outputs: list[Mapping[str, float]] = []
        step = 0
        for _ in range(self.local_iterations):
            if full:
                training_outputs.extend(
                    full_gradient_into_grad(
                        task=self.task, model=model, train_loader=train_loader,
                        client_id=self.client_id,
                    )
                )  # fmt: skip
                full_gradient_into_grad(
                    task=self.task, model=previous, train_loader=train_loader,
                    client_id=self.client_id,
                )  # fmt: skip
                step += 1
                self._step(model, previous, momentum, second_moment, den, rate, step < total)
                continue
            had_batch = False
            for batch in train_loader:
                had_batch = True
                training_outputs.append(
                    take_train_step(
                        self.task, model, batch, _GradientOnlyOptimizer(model.parameters())
                    )
                )
                take_train_step(
                    self.task, previous, batch, _GradientOnlyOptimizer(previous.parameters())
                )
                step += 1
                self._step(model, previous, momentum, second_moment, den, rate, step < total)
            if not had_batch:
                raise ValueError(f"client {self.client_id!r} has no training batches")
        return training_outputs, step

    @torch.no_grad()
    def _step(
        self,
        model: nn.Module,
        previous: nn.Module,
        momentum: dict[str, Tensor],
        second_moment: dict[str, Tensor],
        den: dict[str, Tensor],
        rate: float,
        moves: bool,
    ) -> None:
        """Track the momentum, update ``v``, move ``prev`` to ``x``, and step unless the last."""

        alpha, beta = self.fafed_alpha, self.beta2
        previous_parameters = dict(previous.named_parameters())
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            if name not in momentum:
                raise ValueError(f"missing FAFED state for parameter: {name}")
            gradient = parameter.grad.detach()
            behind = previous_parameters[name]
            if behind.grad is None:
                raise ValueError(f"no gradient at the previous iterate for parameter: {name}")
            momentum[name].sub_(behind.grad.detach()).mul_(1.0 - alpha).add_(gradient)
            second_moment[name].mul_(beta).add_(gradient * gradient * (1.0 - beta))
            behind.copy_(parameter)
            if moves:
                parameter.sub_(momentum[name] * rate / den[name])

    def get_state(self) -> dict[str, Any]:
        """Settings, and the previous iterate: the one thing FAFED keeps between rounds."""

        state = super().get_state()
        state.update(
            {
                "beta2": self.beta2,
                "fafed_alpha": self.fafed_alpha,
                "fafed_rho": self.fafed_rho,
                "update_mode": self.update_mode,
                "previous_iterate": clone_model_state(self._previous or {}),
            }
        )
        return state

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore settings and the previous iterate; refuse a checkpoint of other settings."""

        refuse_a_reconfigured_resume(
            "fafed client",
            state,
            {
                "beta2": self.beta2,
                "fafed_alpha": self.fafed_alpha,
                "fafed_rho": self.fafed_rho,
                "update_mode": self.update_mode,
            },
        )
        super().load_state(state)
        previous = state.get("previous_iterate")
        if isinstance(previous, dict) and previous:
            self._previous = clone_model_state(previous)


def _gradients(model: nn.Module) -> dict[str, Tensor]:
    gradients: dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            gradients[name] = parameter.grad.detach().clone()
    return gradients


def _parameters(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


@torch.no_grad()
def _copy_into(model: nn.Module, values: Mapping[str, Tensor]) -> None:
    for name, parameter in model.named_parameters():
        if name in values:
            parameter.copy_(values[name].to(parameter.device))


def _half_open_unit(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number in [0, 1)")
    normalized = float(value)
    if not 0.0 <= normalized < 1.0:
        raise ValueError(f"{name} must be a number in [0, 1)")
    return normalized
