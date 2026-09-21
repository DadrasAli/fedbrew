"""PyTorch FedProx client implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import optim

from fedbrew.clients.local_update_modes import (
    FULL_GRADIENT_UPDATE_MODE,
    full_gradient_into_grad,
    own_loop_update_mode,
)
from fedbrew.clients.torch_sgd_client import TorchSGDClient, _get_train_data
from fedbrew.core.checkpointing import refuse_a_reconfigured_resume
from fedbrew.core.federated_state import model_state_size, refuse_adapter_state
from fedbrew.core.metrics import filter_metrics
from fedbrew.core.protocol import ClientInfo, FitRequest, FitResult
from fedbrew.core.torch_utils import clone_model_state, get_model_state, load_model_state
from fedbrew.tasks.base import TaskAdapter


class TorchFedProxClient(TorchSGDClient[TaskAdapter]):
    """Train a PyTorch model locally with the FedProx proximal objective."""

    def __init__(
        self,
        client_id: str,
        task: TaskAdapter,
        model_config: dict[str, Any],
        client_data: Any,
        local_iterations: int,
        batch_size: int,
        learning_rate: float,
        proximal_mu: float,
        eval_batch_size: int | None = None,
        device: str = "cpu",
        metrics: list[str] | None = None,
        base_seed: int | None = None,
        train_shuffle: bool = True,
        eval_shuffle: bool = False,
        drop_last: bool = False,
        update_mode: str | None = None,
    ) -> None:
        """Configure local SGD plus the FedProx proximal term.

        Args:
            proximal_mu: Weight of the ``mu / 2 * ||w - w_global||^2`` penalty
                added to the local objective, in loss units per squared
                model-parameter unit. 0.0 reduces this client exactly to
                FedAvg. Larger values pull the local solution towards the
                broadcast model and damp client drift.
            update_mode: ``sequential_epoch`` (also when unset): each of the
                ``local_iterations`` iterations is one pass, one step per
                batch. ``full_gradient``: each is one step on the exact
                gradient of the whole train split, plus the proximal term.

        Every other argument is :class:`TorchSGDClient`'s and means the same
        thing. The proximal term costs one extra model-sized reference held for
        the duration of the round -- the round's starting weights -- but adds
        nothing to what is communicated.
        """
        super().__init__(
            client_id=client_id,
            task=task,
            model_config=model_config,
            client_data=client_data,
            local_iterations=local_iterations,
            batch_size=batch_size,
            learning_rate=learning_rate,
            eval_batch_size=eval_batch_size,
            device=device,
            metrics=metrics,
            base_seed=base_seed,
            train_shuffle=train_shuffle,
            eval_shuffle=eval_shuffle,
            drop_last=drop_last,
        )
        if proximal_mu < 0:
            raise ValueError("proximal_mu must be non-negative")
        self.proximal_mu = float(proximal_mu)
        self.update_mode = own_loop_update_mode(update_mode)

    def setup(self, client_info: ClientInfo) -> None:
        """Record client metadata supplied by the loop."""

        super().setup(client_info)

    def fit(self, request: FitRequest) -> FitResult:
        """Run local FedProx training from the provided global model state."""

        global_state = request.payload.get("model_state")
        if not isinstance(global_state, dict):
            raise ValueError("fit request payload must contain model_state")

        # Composes with runtime.use_amp: GradScaler.step unscales .grad before
        # delegating to a wrapped optimizer, so _FedProxCorrectingOptimizer.step
        # corrects true-scale gradients. This used to raise here and at config
        # load; both went when it was measured, and
        # tests/test_amp_composes_with_wrapped_optimizers.py records the
        # measurement.

        train_data = _get_train_data(self.client_data)
        model = self.task.build_model(self.model_config)
        refuse_adapter_state(self, "fedprox", model)
        load_model_state(model, clone_model_state(global_state))
        reference_parameters = _reference_parameters(model)
        optimizer = _FedProxCorrectingOptimizer(
            optim.SGD(model.parameters(), lr=self.learning_rate),
            model,
            reference_parameters,
            self.proximal_mu,
        )
        train_loader = self.task.build_dataloader(
            train_data,
            self._train_loader_config(request.round_id),
        )

        optimizer_steps = 0
        for _ in range(self.local_iterations):
            if self.update_mode == FULL_GRADIENT_UPDATE_MODE:
                full_gradient_into_grad(
                    task=self.task, model=model, train_loader=train_loader, client_id=self.client_id
                )
                optimizer.step()
                optimizer_steps += 1
                continue
            for batch in train_loader:
                self.task.train_step(model, batch, optimizer)
                optimizer_steps += 1
        self._require_training_batches(optimizer_steps)

        base_metrics, num_examples = self._evaluate_model(
            model,
            train_data,
            metrics=[],
            round_id=request.round_id,
            prefix="fit_",
        )
        proximal_loss = _proximal_loss_value(
            model,
            reference_parameters,
            self.proximal_mu,
        )
        model_state = get_model_state(model)
        # One model's worth per round, the same as FedAvg: the proximal term
        # is computed locally against the state the client was already sent
        # and nothing extra crosses the wire. Reported anyway, because a
        # missing column reads as "no cost" when arms are joined.
        communicated_parameters, communicated_bytes = model_state_size(model_state)

        all_metrics = dict(base_metrics)
        all_metrics["fit_proximal_loss"] = proximal_loss
        all_metrics["fit_total_loss"] = float(all_metrics.get("fit_loss", 0.0)) + proximal_loss
        all_metrics["communicated_parameters"] = float(communicated_parameters)
        all_metrics["communicated_bytes"] = float(communicated_bytes)
        metrics = filter_metrics(all_metrics, self.metrics)
        return FitResult(
            round_id=request.round_id,
            client_id=self.client_id,
            num_examples=num_examples,
            payload={"model_state": model_state},
            metrics=metrics,
        )

    def get_state(self) -> dict[str, Any]:
        """Return serializable client metadata."""

        state = super().get_state()
        state["proximal_mu"] = self.proximal_mu
        state["update_mode"] = self.update_mode
        return state

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore serializable client metadata.

        Raises:
            ValueError: If the checkpoint disagrees with this run's config.
        """

        refuse_a_reconfigured_resume(
            "fedprox client",
            state,
            {"proximal_mu": self.proximal_mu, "update_mode": self.update_mode},
        )
        super().load_state(state)
        if "proximal_mu" in state:
            self.proximal_mu = float(state["proximal_mu"])


class _FedProxCorrectingOptimizer:
    """Wraps a real optimizer; adds the proximal term's closed-form gradient
    -- ``mu * (w - w0)`` -- to ``.grad`` right before delegating to it,
    instead of building a differentiable penalty and backpropagating
    through it.

    So FedProx composes with ``task.train_step`` the same way
    ``_ScaffoldCorrectingOptimizer`` does, for the same reason: every task's
    ``train_step`` already accepts an optimizer, and this is one.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        model: torch.nn.Module,
        reference_parameters: list[torch.Tensor],
        proximal_mu: float,
    ) -> None:
        self._optimizer = optimizer
        self._model = model
        self._reference_parameters = reference_parameters
        self._proximal_mu = proximal_mu

    @property
    def param_groups(self) -> Any:
        return self._optimizer.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._optimizer.zero_grad(set_to_none)

    def step(self, closure: Any | None = None) -> None:
        if self._proximal_mu:
            _apply_proximal_gradient_correction(
                self._model,
                self._reference_parameters,
                self._proximal_mu,
            )
        self._optimizer.step(closure)


def _apply_proximal_gradient_correction(
    model: torch.nn.Module,
    reference_parameters: list[torch.Tensor],
    proximal_mu: float,
) -> None:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    with torch.no_grad():
        for parameter, reference in zip(trainable, reference_parameters, strict=True):
            if parameter.grad is None:
                continue
            parameter.grad.add_(parameter - reference.to(parameter.device), alpha=proximal_mu)


def _reference_parameters(model: torch.nn.Module) -> list[torch.Tensor]:
    with torch.no_grad():
        return [
            parameter.detach().clone()
            for parameter in model.parameters()
            if parameter.requires_grad
        ]


def _proximal_loss_tensor(
    model: torch.nn.Module,
    reference_parameters: list[torch.Tensor],
    proximal_mu: float,
) -> torch.Tensor:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        return torch.tensor(0.0)
    penalty = parameters[0].new_tensor(0.0)
    if proximal_mu == 0.0:
        return penalty
    for parameter, reference in zip(parameters, reference_parameters, strict=True):
        penalty = penalty + torch.sum((parameter - reference.to(parameter.device)) ** 2)
    return 0.5 * proximal_mu * penalty


def _proximal_loss_value(
    model: torch.nn.Module,
    reference_parameters: list[torch.Tensor],
    proximal_mu: float,
) -> float:
    return float(
        _proximal_loss_tensor(model, reference_parameters, proximal_mu).detach().cpu().item()
    )
