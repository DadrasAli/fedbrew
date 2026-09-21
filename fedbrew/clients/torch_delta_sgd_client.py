"""Delta-SGD client: an auto-tuned step size from the local smoothness.

Implements the client of Kim et al., "Adaptive Federated Learning with
Auto-Tuned Clients" (arXiv:2306.11201). The server is plain FedAvg, so the
whole algorithm lives here.

The step size is not configured: it is measured. Each local step compares how
far the parameters moved with how much the gradient changed, which estimates
the local smoothness, and sets the next step size from that. eta and theta
reset to eta_0 / theta_0 at the start of every round (Algorithm 1 lines 6-7),
so unlike SCAFFOLD this client keeps nothing model-sized between rounds and
costs no per-client memory at 3597 writers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from fedbrew.clients.local_update_modes import (
    DELTA_SGD_UPDATE_MODES,
    SUPPORTED_FROZEN_WEIGHTING,
    normalize_choice,
    run_delta_sgd_update_mode,
)
from fedbrew.clients.torch_sgd_client import TorchSGDClient, _get_train_data
from fedbrew.core.checkpointing import refuse_a_reconfigured_resume
from fedbrew.core.federated_state import model_state_size
from fedbrew.core.metrics import filter_metrics
from fedbrew.core.protocol import FitRequest, FitResult

#: Paper defaults, used unchanged across every experiment in arXiv:2306.11201.
DEFAULT_ETA_0 = 0.2
DEFAULT_THETA_0 = 1.0
DEFAULT_GAMMA = 2.0
DEFAULT_DELTA = 0.1


class TorchDeltaSGDClient(TorchSGDClient):
    """Train locally with the Delta-SGD auto-tuned step size.

    ``local_iterations`` counts iterations of the local loop, and the selected
    ``update_mode`` is what one iteration does:

    - ``single_batch``: one mini-batch step.
    - ``sequential_epoch``: one complete chained mini-batch epoch.
    - ``frozen_batch_gradients``: one full pass whose batch gradients are all
      evaluated at the same starting model, then one combined update.
    - ``full_gradient``: one update on the exact gradient of the whole train
      split.
    """

    def __init__(
        self,
        *args: Any,
        update_mode: str,
        frozen_gradient_weighting: str,
        eta_0: float,
        theta_0: float,
        gamma: float,
        delta: float,
        eta_max: float | None = None,
        max_grad_norm: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure the auto-tuned step-size rule.

        Args:
            *args: Forwarded to :class:`TorchSGDClient` positionally.
            update_mode: As :class:`~fedbrew.clients.fedavg_client.FedAvgClient`.
            frozen_gradient_weighting: As FedAvgClient.
            eta_0: Initial step size for each round, in model-parameter units
                per step. Reset at the start of every round (Algorithm 1 lines
                6-7), not carried across them. Paper default 0.2.
            theta_0: Initial step-size growth ratio, dimensionless. Also reset
                every round. Paper default 1.0.
            gamma: Growth bound on the step size between consecutive steps,
                dimensionless. Paper default 2.0.
            delta: Damping on the smoothness estimate, dimensionless. Paper
                default 0.1.
            eta_max: Optional ceiling on the step size, in the same units as
                ``eta_0``. **Not part of the published rule** -- the paper has
                none and the AdGD fallback already covers a vanishing
                denominator -- so leave it unset to run the algorithm as
                published. Set, it is a diagnostic backstop, and a non-zero
                ``clamped_steps`` in the update metrics means the clamp rather
                than the local smoothness is choosing the step size.
            max_grad_norm: Gradient-norm clipping threshold. Positive when set.
            **kwargs: Forwarded to :class:`TorchSGDClient` by keyword.

        Raises:
            ValueError: If any of ``eta_0``, ``theta_0``, ``gamma``, ``delta``,
                ``eta_max`` or ``max_grad_norm`` is not finite and positive, or
                if a setting that would also modify the step size is supplied:
                non-zero ``momentum`` or ``weight_decay``, ``nesterov``,
                ``max_local_steps``, or a non-constant learning-rate schedule.
                These are refused rather than ignored, because the whole point
                of the rule is that it chooses the step size. The factory never
                forwards them for this rule, so the checks below fire only on
                direct construction; a *config* that sets one is refused by
                ``validate_config`` against ``UNHONOURED_CLIENT_OPTIONS``.
        """

        super().__init__(*args, **kwargs)

        self.update_mode = normalize_choice(
            update_mode,
            DELTA_SGD_UPDATE_MODES,
            "update_mode",
        )
        self.frozen_gradient_weighting = normalize_choice(
            frozen_gradient_weighting,
            SUPPORTED_FROZEN_WEIGHTING,
            "frozen_gradient_weighting",
        )

        self._eta_0 = _positive_finite_float(eta_0, "eta_0")
        self._theta_0 = _positive_finite_float(theta_0, "theta_0")
        self._gamma = _positive_finite_float(gamma, "gamma")
        self._delta = _positive_finite_float(delta, "delta")

        # Optional ceiling. The paper has none, and the reference AdGD fallback
        # already covers the vanishing-denominator case, so this is a diagnostic
        # backstop rather than part of the rule: leave it unset to run the
        # algorithm as published.
        if eta_max is not None:
            eta_max = _positive_finite_float(eta_max, "eta_max")
        self.eta_max = eta_max

        if max_grad_norm is not None and not max_grad_norm > 0.0:
            raise ValueError("max_grad_norm must be positive when set")
        self.max_grad_norm = max_grad_norm

        # The rule supplies the step size, so anything that would also modify
        # it is rejected rather than silently ignored. Reachable by direct
        # construction only: the factory hands this rule none of these, so from
        # a config every attribute below is the base class's default and the
        # refusal that bites is UNHONOURED_CLIENT_OPTIONS in config.py.
        for name in ("momentum", "weight_decay"):
            value = getattr(self, name)
            if value not in (None, 0.0):
                raise ValueError(f"delta_sgd requires {name}: 0.0")
        if self.nesterov:
            raise ValueError("delta_sgd requires nesterov: false")
        if self.max_local_steps is not None:
            raise ValueError("delta_sgd does not support max_local_steps")
        if self.learning_rate_schedule not in (None, "constant"):
            raise ValueError("delta_sgd tunes its own step size; use no schedule")

    def fit(self, request: FitRequest) -> FitResult:
        """Run one round of auto-tuned local training."""

        train_data = _get_train_data(self.client_data)
        model = self.task.build_model(self.model_config)
        self._load_federated_payload(model, request.payload, context="fit request")

        train_loader = self.task.build_dataloader(
            train_data,
            self._train_loader_config(request.round_id),
        )
        update_result = run_delta_sgd_update_mode(
            task=self.task,
            model=model,
            train_loader=train_loader,
            local_iterations=self.local_iterations,
            update_mode=self.update_mode,
            frozen_gradient_weighting=self.frozen_gradient_weighting,
            client_id=self.client_id,
            eta_0=self._eta_0,
            theta_0=self._theta_0,
            gamma=self._gamma,
            delta=self._delta,
            eta_max=self.eta_max,
            max_grad_norm=self.max_grad_norm,
        )

        metrics, evaluated_num_examples = self._evaluate_model(
            model,
            train_data,
            round_id=request.round_id,
            prefix="fit_",
        )
        num_examples = self.task.federated_aggregation_weight(
            update_result.training_outputs,
            evaluated_num_examples,
        )
        if num_examples < 0:
            raise ValueError("task federated aggregation weight must be non-negative")

        model_state = self.task.get_federated_model_state(model)
        model_state_metadata = self.task.federated_model_state_metadata(model)
        model_state_scope = str(model_state_metadata["model_state_scope"])
        communicated_parameters, communicated_bytes = model_state_size(model_state)
        trainable_parameters = sum(
            int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
        )
        active_target_tokens = sum(
            float(output.get("total", 0.0)) for output in update_result.training_outputs
        )

        metrics.update(
            {
                "optimizer_steps": float(update_result.optimizer_steps),
                "active_target_tokens": float(active_target_tokens),
                "trainable_parameters": float(trainable_parameters),
                "communicated_parameters": float(communicated_parameters),
                "communicated_bytes": float(communicated_bytes),
            }
        )
        metrics.update(_step_size_metrics(update_result, self._eta_0))

        return FitResult(
            round_id=request.round_id,
            client_id=self.client_id,
            num_examples=num_examples,
            payload={
                "model_state": model_state,
                "model_state_scope": model_state_scope,
                "model_state_metadata": model_state_metadata,
            },
            metrics=filter_metrics(metrics, self.metrics),
        )

    def get_state(self) -> dict[str, Any]:
        """Return checkpointable client configuration.

        No adaptive state appears here on purpose: eta and theta are reset every
        round, so there is nothing about them to carry across a resume.
        """

        state = super().get_state()
        state.update(
            {
                "update_mode": self.update_mode,
                "frozen_gradient_weighting": self.frozen_gradient_weighting,
                "eta_0": self._eta_0,
                "theta_0": self._theta_0,
                "gamma": self._gamma,
                "delta": self._delta,
                "eta_max": self.eta_max,
                # Compared on resume and restored by nothing, as in FedAvgClient.
                "max_grad_norm": self.max_grad_norm,
            }
        )
        return state

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore checkpointable client configuration.

        Raises:
            ValueError: If the checkpoint disagrees with this run's config.
        """

        refuse_a_reconfigured_resume(
            "delta_sgd client",
            state,
            {
                "update_mode": self.update_mode,
                "frozen_gradient_weighting": self.frozen_gradient_weighting,
                "eta_0": self._eta_0,
                "theta_0": self._theta_0,
                "gamma": self._gamma,
                "delta": self._delta,
                "eta_max": self.eta_max,
                "max_grad_norm": self.max_grad_norm,
            },
        )
        super().load_state(state)
        self.update_mode = normalize_choice(
            str(state.get("update_mode", self.update_mode)),
            DELTA_SGD_UPDATE_MODES,
            "update_mode",
        )
        self.frozen_gradient_weighting = normalize_choice(
            str(
                state.get(
                    "frozen_gradient_weighting",
                    self.frozen_gradient_weighting,
                )
            ),
            SUPPORTED_FROZEN_WEIGHTING,
            "frozen_gradient_weighting",
        )
        self._eta_0 = _positive_finite_float(state.get("eta_0", self._eta_0), "eta_0")
        self._theta_0 = _positive_finite_float(state.get("theta_0", self._theta_0), "theta_0")
        self._gamma = _positive_finite_float(state.get("gamma", self._gamma), "gamma")
        self._delta = _positive_finite_float(state.get("delta", self._delta), "delta")
        eta_max = state.get("eta_max", self.eta_max)
        self.eta_max = None if eta_max is None else _positive_finite_float(eta_max, "eta_max")


def _step_size_metrics(update_result: Any, eta_0: float) -> dict[str, float]:
    """Summarise the round's step-size trace.

    This is the whole point of the algorithm, so it has to be visible in the
    CSV: a run where the mean step size never leaves eta_0 is one where the
    auto-tuner did nothing.
    """

    step_sizes: Sequence[float] = update_result.step_sizes
    if not step_sizes:
        raise ValueError("delta_sgd produced no local steps")
    num_steps = float(len(step_sizes))
    return {
        "client_eta_0": float(eta_0),
        "client_step_size_mean": float(sum(step_sizes) / num_steps),
        "client_step_size_min": float(min(step_sizes)),
        "client_step_size_max": float(max(step_sizes)),
        "client_step_size_final": float(step_sizes[-1]),
        "step_size_clamp_fraction": float(update_result.clamped_steps) / num_steps,
        "undefined_curvature_fraction": (
            float(update_result.undefined_curvature_steps) / num_steps
        ),
    }


def _positive_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite positive number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return normalized
