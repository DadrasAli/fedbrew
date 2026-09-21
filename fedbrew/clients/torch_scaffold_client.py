"""PyTorch SCAFFOLD client implementation."""

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
from fedbrew.core.torch_utils import (
    StateDict,
    add_model_states,
    clone_model_state,
    get_model_state,
    load_model_state,
    scale_model_state,
    squared_l2_norm_model_state,
    subtract_model_states,
    zeros_like_model_state,
)
from fedbrew.tasks.base import TaskAdapter


class TorchScaffoldClient(TorchSGDClient[TaskAdapter]):
    """Train a PyTorch model locally with SCAFFOLD gradient correction."""

    def __init__(
        self,
        client_id: str,
        task: TaskAdapter,
        model_config: dict[str, Any],
        client_data: Any,
        local_iterations: int,
        batch_size: int,
        learning_rate: float,
        eval_batch_size: int | None = None,
        device: str = "cpu",
        metrics: list[str] | None = None,
        base_seed: int | None = None,
        train_shuffle: bool = True,
        eval_shuffle: bool = False,
        drop_last: bool = False,
        update_mode: str | None = None,
    ) -> None:
        """Configure local SGD with SCAFFOLD control-variate correction.

        Every argument is :class:`TorchSGDClient`'s and means the same thing;
        SCAFFOLD adds no hyperparameter of its own. What it adds is state: each
        local gradient is corrected by ``c_global - c_local``, so the client
        holds one model-shaped control variate persistently and exchanges
        another with the server each round. **That is 2x a FedAvg arm's
        per-round communication volume, and one extra model-sized tensor per
        client held between rounds** -- the memory cost matters at thousands of
        clients. The control variate is refreshed by the paper's Option II,
        ``c_i - c + (x - y_i) / (K * learning_rate)``, which reuses the local
        steps rather than taking a fresh gradient at ``x``.

        ``update_mode`` is ``sequential_epoch`` (also when unset), one corrected
        step per batch over ``local_iterations`` passes, or ``full_gradient``,
        one corrected step per iteration on the exact gradient of the whole
        train split. K in Option II is the number of steps taken either way.
        """
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive for SCAFFOLD")
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
        self._client_control: StateDict | None = None
        self.update_mode = own_loop_update_mode(update_mode)

    def setup(self, client_info: ClientInfo) -> None:
        """Record client metadata supplied by the loop."""

        super().setup(client_info)

    def fit(self, request: FitRequest) -> FitResult:
        """Run local SCAFFOLD training from the provided global state."""

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive for SCAFFOLD")

        global_state = request.payload.get("model_state")
        if not isinstance(global_state, dict):
            raise ValueError("fit request payload must contain model_state")
        server_control = request.payload.get("server_control")
        if not isinstance(server_control, dict):
            raise ValueError("SCAFFOLD fit request payload must contain server_control")

        if self._client_control is None:
            self._client_control = zeros_like_model_state(global_state)
        old_client_control = clone_model_state(self._client_control)

        # Composes with runtime.use_amp: GradScaler.step unscales .grad before
        # delegating to a wrapped optimizer, so _ScaffoldCorrectingOptimizer.step
        # corrects true-scale gradients. This used to raise here and at config
        # load; both went when it was measured, and
        # tests/test_amp_composes_with_wrapped_optimizers.py records the
        # measurement.

        train_data = _get_train_data(self.client_data)
        model = self.task.build_model(self.model_config)
        refuse_adapter_state(self, "scaffold", model)
        load_model_state(model, clone_model_state(global_state))
        optimizer = _ScaffoldCorrectingOptimizer(
            optim.SGD(model.parameters(), lr=self.learning_rate),
            model,
            old_client_control,
            server_control,
        )
        train_loader = self.task.build_dataloader(
            train_data,
            self._train_loader_config(request.round_id),
        )

        local_steps = 0
        for _ in range(self.local_iterations):
            if self.update_mode == FULL_GRADIENT_UPDATE_MODE:
                full_gradient_into_grad(
                    task=self.task, model=model, train_loader=train_loader, client_id=self.client_id
                )
                optimizer.step()
                local_steps += 1
                continue
            for batch in train_loader:
                self.task.train_step(model, batch, optimizer)
                local_steps += 1

        if local_steps == 0:
            raise ValueError("SCAFFOLD local_steps must be positive")

        local_state = get_model_state(model)
        correction = scale_model_state(
            subtract_model_states(global_state, local_state),
            1.0 / (local_steps * self.learning_rate),
        )
        new_client_control = add_model_states(
            subtract_model_states(old_client_control, server_control),
            correction,
        )
        control_delta = subtract_model_states(new_client_control, old_client_control)
        self._client_control = new_client_control

        base_metrics, num_examples = self._evaluate_model(
            model,
            train_data,
            metrics=[],
            round_id=request.round_id,
            prefix="fit_",
        )
        communicated_parameters, communicated_bytes = model_state_size(local_state)
        # SCAFFOLD uploads the model and the control-variate delta together,
        # and the server sends its own control variate back down beside the
        # model, so a round moves two model-shaped states in each direction.
        # Counting model_state alone -- the one-line pattern every other
        # client uses -- would report exactly half the real volume, which is
        # plausible enough to survive review. Same reason FedLALR adds its
        # momentum and second-moment states here.
        delta_parameters, delta_bytes = model_state_size(control_delta)
        communicated_parameters += delta_parameters
        communicated_bytes += delta_bytes

        all_metrics = dict(base_metrics)
        all_metrics.update(
            {
                "control_delta_norm": squared_l2_norm_model_state(control_delta) ** 0.5,
                "client_control_norm": squared_l2_norm_model_state(self._client_control) ** 0.5,
                "local_steps": float(local_steps),
                "communicated_parameters": float(communicated_parameters),
                "communicated_bytes": float(communicated_bytes),
            }
        )
        metrics = filter_metrics(all_metrics, self.metrics)
        return FitResult(
            round_id=request.round_id,
            client_id=self.client_id,
            num_examples=num_examples,
            payload={
                "model_state": local_state,
                "control_delta": clone_model_state(control_delta),
            },
            metrics=metrics,
        )

    def get_state(self) -> dict[str, Any]:
        """Return serializable SCAFFOLD client metadata."""

        state = super().get_state()
        state["client_control"] = clone_model_state(self._client_control or {})
        state["update_mode"] = self.update_mode
        return state

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore SCAFFOLD client metadata and control variate.

        Raises:
            ValueError: If the checkpoint disagrees with this run's config.
        """

        refuse_a_reconfigured_resume("scaffold client", state, {"update_mode": self.update_mode})
        super().load_state(state)
        client_control = state.get("client_control")
        if isinstance(client_control, dict) and client_control:
            self._client_control = clone_model_state(client_control)


class _ScaffoldCorrectingOptimizer:
    """Wraps a real optimizer; corrects ``.grad`` by ``server - client``
    right before delegating to it.

    So SCAFFOLD composes with ``task.train_step`` -- and, through it, with
    whatever backward/step path the task implements, AMP included -- the
    same way ``local_update_modes.py``'s ``_ClippingOptimizer`` composes
    gradient clipping with it, instead of reimplementing
    forward/loss/backward by hand against two of the task's private
    attributes. Every task's ``train_step`` already accepts an optimizer;
    this is one, from the optimizer's point of view.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        model: torch.nn.Module,
        client_control: Mapping[str, Any],
        server_control: Mapping[str, Any],
    ) -> None:
        self._optimizer = optimizer
        self._model = model
        self._client_control = client_control
        self._server_control = server_control

    @property
    def param_groups(self) -> Any:
        return self._optimizer.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._optimizer.zero_grad(set_to_none)

    def step(self, closure: Any | None = None) -> None:
        _apply_scaffold_gradient_correction(
            self._model,
            self._client_control,
            self._server_control,
        )
        self._optimizer.step(closure)


def _apply_scaffold_gradient_correction(
    model: torch.nn.Module,
    client_control: Mapping[str, Any],
    server_control: Mapping[str, Any],
) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name not in client_control or name not in server_control:
            raise ValueError(f"missing SCAFFOLD control variate for parameter: {name}")
        client_value = _control_tensor(client_control[name], name, parameter.device)
        server_value = _control_tensor(server_control[name], name, parameter.device)
        parameter.grad = parameter.grad - client_value + server_value


def _control_tensor(value: Any, name: str, device: torch.device) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"control variate for {name} is not a tensor")
    return value.detach().to(device)
