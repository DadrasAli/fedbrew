"""FedAvg client with selectable shared SGD local-update modes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from fedbrew.clients.batch_orders import LocalLoop
from fedbrew.clients.batched_update import (
    ClientBatchFit,
    ClientBatchPlan,
    LocalProgram,
    OptimizerSpec,
    no_training_batches,
    sgd_mode_updates,
)
from fedbrew.clients.local_update_modes import (
    SUPPORTED_FROZEN_WEIGHTING,
    SUPPORTED_UPDATE_MODES,
    normalize_choice,
    run_sgd_update_mode,
)
from fedbrew.clients.torch_sgd_client import (
    TorchSGDClient,
    _get_train_data,
    trainable_parameter_count,
)
from fedbrew.core.checkpointing import refuse_a_reconfigured_resume
from fedbrew.core.protocol import FitRequest, FitResult

#: How each update mode forms the gradient one update steps on.
_COMBINE_OF_MODE = {
    "single_batch": "batch",
    "sequential_epoch": "batch",
    "frozen_batch_gradients": "frozen",
    "full_gradient": "full",
}


class FedAvgClient(TorchSGDClient):
    """FedAvg client on the shared SGD local-update engine.

    The learning rate is fixed or round-scheduled through the normal FedAvg
    configuration. The server still performs standard sample-weighted FedAvg.

    ``local_iterations`` counts iterations of the local loop, and the selected
    mode is what one iteration does:

    - ``single_batch``: one mini-batch SGD step.
    - ``sequential_epoch``: one complete chained mini-batch SGD epoch.
    - ``frozen_batch_gradients``: one full pass with all gradients evaluated
      at the same starting model, followed by one combined update.
    - ``full_gradient``: one step on the exact gradient of the task's training
      loss over the whole train split, each batch weighted by its share of
      ``task.train_loss_denominator``.
    """

    def __init__(
        self,
        *args: Any,
        update_mode: str,
        frozen_gradient_weighting: str,
        max_grad_norm: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure the local-update mode on top of TorchSGDClient.

        Args:
            *args: Forwarded to :class:`TorchSGDClient` positionally.
            update_mode: How ``local_iterations`` is executed -- ``single_batch``
                (one mini-batch step), ``sequential_epoch`` (one chained epoch),
                ``frozen_batch_gradients`` (one pass whose batch gradients
                are all taken at the iteration's starting model, then one
                combined update) or ``full_gradient`` (one step on the exact
                gradient of the whole train split). Case- and
                whitespace-insensitive.
            frozen_gradient_weighting: How those frozen batch gradients combine
                -- ``examples`` (weight by batch size), ``uniform`` (weight
                each batch equally) or ``sum`` (add them). Only consulted for
                ``frozen_batch_gradients``; ``full_gradient`` reads none.
            max_grad_norm: Gradient-norm clipping threshold, in gradient-norm
                units. Positive when set; None disables clipping.
            **kwargs: Forwarded to :class:`TorchSGDClient` by keyword.

        Raises:
            ValueError: If either choice is unrecognised or ``max_grad_norm``
                is non-positive.
        """
        super().__init__(*args, **kwargs)

        self.update_mode = normalize_choice(
            update_mode,
            SUPPORTED_UPDATE_MODES,
            "update_mode",
        )
        self.frozen_gradient_weighting = normalize_choice(
            frozen_gradient_weighting,
            SUPPORTED_FROZEN_WEIGHTING,
            "frozen_gradient_weighting",
        )

        # The shared engine currently implements plain SGD. Reject options that
        # would otherwise be silently ignored and make the comparison unfair.
        if self.momentum != 0.0:
            raise ValueError("fedavg_client currently requires momentum: 0.0")
        if self.weight_decay != 0.0:
            raise ValueError("fedavg_client currently requires weight_decay: 0.0")
        if self.nesterov:
            raise ValueError("fedavg_client currently requires nesterov: false")
        if self.max_local_steps is not None:
            raise ValueError("fedavg_client does not support max_local_steps")

        if max_grad_norm is not None and not max_grad_norm > 0.0:
            raise ValueError("max_grad_norm must be positive when set")
        self.max_grad_norm = max_grad_norm

    def fit(self, request: FitRequest) -> FitResult:
        """Run the selected local mode with the configured FedAvg learning rate."""

        train_data = _get_train_data(self.client_data)
        model = self.task.build_model(self.model_config)
        self._load_federated_payload(model, request.payload, context="fit request")

        learning_rate_used = self._round_learning_rate(request.round_id)
        train_loader = self.task.build_dataloader(
            train_data,
            self._train_loader_config(request.round_id),
        )
        update_result = run_sgd_update_mode(
            task=self.task,
            model=model,
            train_loader=train_loader,
            local_iterations=self.local_iterations,
            learning_rate=learning_rate_used,
            update_mode=self.update_mode,
            frozen_gradient_weighting=self.frozen_gradient_weighting,
            client_id=self.client_id,
            max_grad_norm=self.max_grad_norm,
        )

        metrics, evaluated_num_examples = self._post_fit_evaluation(model, train_data, request)
        return self._fit_result(
            request,
            metrics=metrics,
            evaluated_num_examples=evaluated_num_examples,
            training_outputs=update_result.training_outputs,
            optimizer_steps=update_result.optimizer_steps,
            model_state=self.task.get_federated_model_state(model),
            model_state_metadata=self.task.federated_model_state_metadata(model),
            trainable_parameters=trainable_parameter_count(model),
            extra_metrics={"client_learning_rate": float(learning_rate_used)},
        )

    #: The batched executor runs this rule's update: every mode, with or
    #: without max_grad_norm (fedbrew/clients/batched_update.py).
    _batched_rule = "fedavg"

    def batched_program(self, request: FitRequest) -> LocalProgram:
        """What one update of ``run_sgd_update_mode`` does this round, checked as it checks."""

        learning_rate = self._round_learning_rate(request.round_id)
        if self.local_iterations <= 0:
            raise ValueError("local_iterations must be positive")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        combine = _COMBINE_OF_MODE[self.update_mode]
        return LocalProgram(
            optimizer=OptimizerSpec("sgd", lr=learning_rate),
            combine=combine,
            max_grad_norm=self.max_grad_norm,
            weighting=self.frozen_gradient_weighting if combine == "frozen" else None,
        )

    def batched_plan(
        self,
        request: FitRequest,
        model: torch.nn.Module,
        start: Mapping[str, Any] | None = None,
    ) -> ClientBatchPlan:
        """This round's local update, as the batched executor runs it (``fit``)."""

        return self._batch_plan(
            request,
            self.batched_program(request),
            self.batched_start(request, model) if start is None else start,
            LocalLoop(
                epochs=self.local_iterations,
                per_update="batch" if _COMBINE_OF_MODE[self.update_mode] == "batch" else "epoch",
                single_batch=self.update_mode == "single_batch",
            ),
            lambda: no_training_batches(self.client_id),
            lambda loader: sgd_mode_updates(
                loader,
                local_iterations=self.local_iterations,
                update_mode=self.update_mode,
                client_id=self.client_id,
            ),
        )

    def batched_result(
        self, request: FitRequest, plan: ClientBatchPlan, fit: ClientBatchFit
    ) -> FitResult:
        """The FitResult ``fit`` returns, from the batched executor's share for this client."""

        metrics, evaluated_num_examples = self._batched_post_fit(plan, fit)
        return self._fit_result(
            request,
            metrics=metrics,
            evaluated_num_examples=evaluated_num_examples,
            training_outputs=fit.training_outputs,
            optimizer_steps=fit.optimizer_steps,
            model_state=fit.model_state,
            model_state_metadata=fit.model_state_metadata,
            trainable_parameters=fit.trainable_parameters,
            extra_metrics={"client_learning_rate": float(plan.program.optimizer.lr)},
        )

    def get_state(self) -> dict[str, Any]:
        """Return checkpointable client configuration and metadata."""

        state = super().get_state()
        state.update(
            {
                "update_mode": self.update_mode,
                "frozen_gradient_weighting": self.frozen_gradient_weighting,
                # Compared on resume and restored by nothing. A setting absent
                # here is one no resume can compare, so an edited threshold
                # would be taken silently while run.json recorded it.
                "max_grad_norm": self.max_grad_norm,
            }
        )
        return state

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore checkpointable client configuration and metadata.

        Raises:
            ValueError: If the checkpoint disagrees with this run's config.
        """

        refuse_a_reconfigured_resume(
            "fedavg client",
            state,
            {
                "update_mode": self.update_mode,
                "frozen_gradient_weighting": self.frozen_gradient_weighting,
                "max_grad_norm": self.max_grad_norm,
            },
        )
        super().load_state(state)
        self.update_mode = normalize_choice(
            str(state.get("update_mode", self.update_mode)),
            SUPPORTED_UPDATE_MODES,
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
