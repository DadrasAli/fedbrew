"""PyTorch local AdamW client implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import optim

from fedbrew.clients.torch_sgd_client import TorchSGDClient
from fedbrew.core.checkpointing import refuse_a_reconfigured_resume
from fedbrew.tasks.base import TaskAdapter


class TorchAdamWClient(TorchSGDClient[TaskAdapter]):
    """Train a task-provided PyTorch model locally with AdamW."""

    def __init__(
        self,
        client_id: str,
        task: TaskAdapter,
        model_config: dict[str, Any],
        client_data: Any,
        local_iterations: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        beta1: float,
        beta2: float,
        epsilon: float,
        learning_rate_schedule: str,
        min_learning_rate: float,
        total_rounds: int,
        eval_batch_size: int | None = None,
        device: str = "cpu",
        metrics: list[str] | None = None,
        base_seed: int | None = None,
        train_shuffle: bool = True,
        eval_shuffle: bool = False,
        drop_last: bool = False,
        max_local_steps: int | None = None,
        update_mode: str | None = None,
    ) -> None:
        """Configure local AdamW instead of SGD.

        Args:
            client_id: As :class:`TorchSGDClient`.
            task: As :class:`TorchSGDClient`.
            model_config: As :class:`TorchSGDClient`.
            client_data: As :class:`TorchSGDClient`.
            local_iterations: Iterations of the local loop per round, each
                one full pass over the train split, or one step under
                ``update_mode: full_gradient``.
            batch_size: Training mini-batch size, in examples.
            learning_rate: AdamW step size, in model-parameter units per step.
            weight_decay: Decoupled weight decay coefficient, non-negative.
                Decoupled means it is applied to the parameters directly rather
                than added to the gradient, so it does not enter the moment
                estimates.
            beta1: First-moment decay in [0, 1).
            beta2: Second-moment decay in [0, 1).
            epsilon: Denominator floor, in gradient units. Positive.
            learning_rate_schedule: ``"constant"`` or ``"cosine"``, decaying
                across rounds rather than within them.
            min_learning_rate: Schedule floor, in [0, learning_rate].
            total_rounds: Total rounds in the run, used to place the current
                round on the cosine curve.
            eval_batch_size: As :class:`TorchSGDClient`. Declared here because
                the factory passes it to every client rule unconditionally, so
                a subclass that omits it does not fall back to a default -- it
                raises TypeError before the first round.
            device: As :class:`TorchSGDClient`.
            metrics: As :class:`TorchSGDClient`.
            base_seed: As :class:`TorchSGDClient`.
            train_shuffle: As :class:`TorchSGDClient`.
            eval_shuffle: As :class:`TorchSGDClient`.
            drop_last: As :class:`TorchSGDClient`.
            max_local_steps: Hard cap on optimizer steps per round. Under
                ``full_gradient`` an iteration is one step, so it caps them.
            update_mode: As :class:`TorchSGDClient`: ``sequential_epoch``
                (also when unset) or ``full_gradient``, one AdamW step per
                iteration on the exact gradient of the whole train split.

        The optimizer is rebuilt every round, so the moment estimates do not
        persist across rounds: each round starts AdamW from zero state. Nothing
        AdamW-specific is communicated.
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
            learning_rate_schedule=learning_rate_schedule,
            min_learning_rate=min_learning_rate,
            total_rounds=total_rounds,
            eval_shuffle=eval_shuffle,
            drop_last=drop_last,
            weight_decay=weight_decay,
            max_local_steps=max_local_steps,
            update_mode=update_mode,
        )
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1)")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        # TorchSGDClient stores this as float | None (None means "no decay" for
        # plain SGD); AdamW takes weight_decay as a required argument, so this
        # subclass narrows it to a plain float.
        self.weight_decay: float = float(weight_decay)

    def get_state(self) -> dict[str, Any]:
        """Return serializable client metadata."""

        state = super().get_state()
        state.update(
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
        )
        return state

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore serializable client metadata.

        Raises:
            ValueError: If the checkpoint disagrees with this run's config.
        """

        refuse_a_reconfigured_resume(
            "local_adamw client",
            state,
            {"beta1": self.beta1, "beta2": self.beta2, "epsilon": self.epsilon},
        )
        super().load_state(state)
        if "beta1" in state:
            self.beta1 = float(state["beta1"])
        if "beta2" in state:
            self.beta2 = float(state["beta2"])
        if "epsilon" in state:
            self.epsilon = float(state["epsilon"])

    def _build_optimizer(
        self,
        model: torch.nn.Module,
        round_id: int,
    ) -> optim.Optimizer:
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError("local_adamw cannot train a model with no trainable parameters")
        return optim.AdamW(
            trainable_parameters,
            lr=self._round_learning_rate(round_id),
            betas=(self.beta1, self.beta2),
            eps=self.epsilon,
            weight_decay=self.weight_decay,
        )
