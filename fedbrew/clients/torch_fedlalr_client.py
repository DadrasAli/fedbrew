"""FedLALR client: local AMSGrad with a client-specific learning rate.

Implements the client half of Algorithm 1 of Sun et al., "FedLALR:
Client-Specific Adaptive Learning Rates Achieve Linear Speedup for Non-IID
Data" (arXiv:2309.09719). Per local step k on client i:

    g            = grad f(x_{t,k,i})
    m_{t,k,i}    = beta1 * m_{t,k-1,i} + (1 - beta1) * g
    v_{t,k,i}    = beta2 * v_{t,k-1,i} + (1 - beta2) * g^2
    v_hat_{t,k,i}= max(v_hat_{t,k-1,i}, v_{t,k,i})
    x_{t,k+1,i}  = x_{t,k,i} - alpha * m_{t,k,i} / sqrt(v_hat_{t,k,i})

At the start of round t the client seeds all three from the broadcast
(Algorithm 1 line 3):

    m_{t,0,i} = m_{t-1},   v_{t,0,i} = v_hat_{t,0,i} = v_hat_{t-1}

Note that v -- the plain second moment -- is seeded from the synchronized
v_hat and then stays local for the round; only v_hat travels back to the
server. That is what makes the learning rate client-specific: every client
starts a round from the same rate and diverges from it according to its own
gradients, which on non-IID data is the point.

There is no epsilon in the denominator. v_hat is a running maximum seeded from
v_hat_{-1} = epsilon^2 and averaged across clients, so it never falls below
epsilon^2 and 1/sqrt(v_hat) is bounded by 1/epsilon.

Because every quantity is reseeded from the broadcast each round, this client
keeps nothing model-sized between rounds -- unlike SCAFFOLD, whose per-writer
control variates dominate memory on FEMNIST.
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
from fedbrew.clients.torch_sgd_client import TorchSGDClient, _get_train_data
from fedbrew.core.checkpointing import refuse_a_reconfigured_resume
from fedbrew.core.federated_state import model_state_size, refuse_adapter_state
from fedbrew.core.metrics import filter_metrics
from fedbrew.core.protocol import FitRequest, FitResult

#: beta1 and epsilon are the paper's CIFAR settings (arXiv:2309.09719, Section
#: V-A; it uses beta1 0.8 on Shakespeare). beta2 is not the paper's: it reports
#: 0.995 on CIFAR and 0.998 on Shakespeare, and 0.999 is the AMSGrad convention.
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.999
DEFAULT_EPSILON = 1e-8


class TorchFedLALRClient(TorchSGDClient):
    """Run local AMSGrad from the broadcast momentum and second moment.

    Each of the ``local_iterations`` iterations is one full local pass,
    matching the FedProx and SCAFFOLD clients, so the FedAvg comparison arm is
    like-for-like under ``update_mode: sequential_epoch``.
    ``client.learning_rate`` is the paper's alpha.
    """

    def __init__(
        self,
        *args: Any,
        beta1: float,
        beta2: float,
        epsilon: float,
        update_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure local AMSGrad from the broadcast optimizer state.

        Args:
            *args: Forwarded to the base client positionally.
            beta1: First-moment decay in [0, 1).
            beta2: Second-moment decay in [0, 1).
            epsilon: Denominator floor, in gradient units. Must be finite and
                positive, and must be the same value the server was given --
                the server reads this key rather than defining its own, because
                two knobs that must agree eventually will not.
            update_mode: ``sequential_epoch`` (also when unset), one AMSGrad
                step per batch over ``local_iterations`` passes, or
                ``full_gradient``, one per iteration on the exact gradient of
                the whole train split.
            **kwargs: Forwarded to the base client by keyword.

        Unlike the AdamW client, the moment estimates are *not* local scratch:
        they are received from the server, advanced locally, and sent back to
        be averaged. That is what makes the per-round volume 3x a FedAvg arm's.
        """
        super().__init__(*args, **kwargs)

        self.beta1 = _unit_interval_float(beta1, "beta1")
        self.beta2 = _unit_interval_float(beta2, "beta2")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be a finite positive number")
        self.epsilon = float(epsilon)
        self.update_mode = own_loop_update_mode(update_mode)

        # The AMSGrad update supplies the step, so anything that would also
        # modify it is rejected rather than silently ignored. Reachable by
        # direct construction only: the factory hands this rule none of these,
        # so from a config every attribute below is the base class's default
        # and the refusal that bites is UNHONOURED_CLIENT_OPTIONS in config.py.
        # That table also covers max_grad_norm, which this client has no
        # parameter for at all -- fedlalr + max_grad_norm used to train
        # unclipped while run.json recorded the clip.
        if self.momentum not in (None, 0.0):
            raise ValueError("fedlalr carries its own momentum; set momentum: 0.0")
        if self.weight_decay not in (None, 0.0):
            raise ValueError("fedlalr requires weight_decay: 0.0")
        if self.nesterov:
            raise ValueError("fedlalr requires nesterov: false")
        if self.max_local_steps is not None:
            raise ValueError("fedlalr does not support max_local_steps")
        if self.learning_rate_schedule not in (None, "constant"):
            raise ValueError("fedlalr adapts its own rate; use no schedule")

    def fit(self, request: FitRequest) -> FitResult:
        """Run one round of local AMSGrad and return all three states."""

        train_data = _get_train_data(self.client_data)
        model = self.task.build_model(self.model_config)
        refuse_adapter_state(self, "fedlalr", model)
        self._load_federated_payload(model, request.payload, context="fit request")
        # The rule reads raw gradients off .grad, which GradScaler.step
        # consumes instead of leaving there.
        if getattr(self.task, "_scaler", None) is not None:
            raise ValueError("fedlalr does not support runtime.use_amp: true")

        device = self.task.device
        momentum = _broadcast_state(request.payload, "momentum_state", device)
        # v_{t,0,i} = v_hat_{t,0,i} = v_hat_{t-1}: both start from the
        # synchronized second moment, then v stays local for the round.
        second_moment_hat = _broadcast_state(request.payload, "second_moment_state", device)
        second_moment = {name: tensor.clone() for name, tensor in second_moment_hat.items()}

        train_loader = self.task.build_dataloader(
            train_data,
            self._train_loader_config(request.round_id),
        )
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
                _amsgrad_step(
                    model,
                    momentum,
                    second_moment,
                    second_moment_hat,
                    learning_rate=self.learning_rate,
                    beta1=self.beta1,
                    beta2=self.beta2,
                )
                local_steps += 1
                continue
            epoch_had_batch = False
            for batch in train_loader:
                epoch_had_batch = True
                optimizer = _GradientOnlyOptimizer(model.parameters())
                training_outputs.append(
                    self.task.train_step(model, batch, optimizer)  # type: ignore[arg-type]
                )
                _amsgrad_step(
                    model,
                    momentum,
                    second_moment,
                    second_moment_hat,
                    learning_rate=self.learning_rate,
                    beta1=self.beta1,
                    beta2=self.beta2,
                )
                local_steps += 1
            if not epoch_had_batch:
                raise ValueError(f"client {self.client_id!r} has no training batches")

        if local_steps == 0:
            raise ValueError("fedlalr local_steps must be positive")

        metrics, evaluated_num_examples = self._evaluate_model(
            model,
            train_data,
            round_id=request.round_id,
            prefix="fit_",
        )
        num_examples = self.task.federated_aggregation_weight(
            training_outputs,
            evaluated_num_examples,
        )
        if num_examples < 0:
            raise ValueError("task federated aggregation weight must be non-negative")

        model_state = self.task.get_federated_model_state(model)
        model_state_metadata = self.task.federated_model_state_metadata(model)
        model_state_scope = str(model_state_metadata["model_state_scope"])
        communicated_parameters, communicated_bytes = model_state_size(model_state)
        # x, m and v_hat all cross the wire, so the communicated volume is not
        # the model size. Report what is actually sent.
        for state in (momentum, second_moment_hat):
            state_parameters, state_bytes = model_state_size(state)
            communicated_parameters += state_parameters
            communicated_bytes += state_bytes

        metrics.update(
            {
                "local_steps": float(local_steps),
                "optimizer_steps": float(local_steps),
                "communicated_parameters": float(communicated_parameters),
                "communicated_bytes": float(communicated_bytes),
                "client_alpha": float(self.learning_rate),
            }
        )
        metrics.update(_learning_rate_metrics(second_moment_hat, self.learning_rate))

        return FitResult(
            round_id=request.round_id,
            client_id=self.client_id,
            num_examples=num_examples,
            payload={
                "model_state": model_state,
                "model_state_scope": model_state_scope,
                "model_state_metadata": model_state_metadata,
                "momentum_state": _to_cpu_state(momentum),
                "second_moment_state": _to_cpu_state(second_moment_hat),
            },
            metrics=filter_metrics(metrics, self.metrics),
        )

    def get_state(self) -> dict[str, Any]:
        """Return checkpointable client configuration.

        No optimizer state appears here on purpose: m, v and v_hat are all
        reseeded from the server broadcast every round, so there is nothing
        per-client to carry across a resume.
        """

        state = super().get_state()
        state.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "update_mode": self.update_mode,
            }
        )
        return state

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore checkpointable client configuration.

        Raises:
            ValueError: If the checkpoint disagrees with this run's config.
        """

        refuse_a_reconfigured_resume(
            "fedlalr client",
            state,
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "update_mode": self.update_mode,
            },
        )
        super().load_state(state)
        self.beta1 = _unit_interval_float(state.get("beta1", self.beta1), "beta1")
        self.beta2 = _unit_interval_float(state.get("beta2", self.beta2), "beta2")
        epsilon = float(state.get("epsilon", self.epsilon))
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be a finite positive number")
        self.epsilon = epsilon


@torch.no_grad()
def _amsgrad_step(
    model: nn.Module,
    momentum: dict[str, Tensor],
    second_moment: dict[str, Tensor],
    second_moment_hat: dict[str, Tensor],
    *,
    learning_rate: float,
    beta1: float,
    beta2: float,
) -> None:
    """Apply one local AMSGrad step in place.

    Every buffer is updated in place: allocating three model-sized tensors per
    local step would dominate a 20-epoch FEMNIST round.
    """

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if name not in momentum or name not in second_moment_hat:
            raise ValueError(f"missing FedLALR optimizer state for parameter: {name}")
        gradient = parameter.grad.detach()

        momentum[name].mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        second_moment[name].mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
        torch.maximum(second_moment_hat[name], second_moment[name], out=second_moment_hat[name])
        # No epsilon: v_hat is floored at epsilon**2 by its initialization and
        # by being a running maximum, so the square root is bounded away from
        # zero. See the module docstring.
        parameter.addcdiv_(
            momentum[name],
            second_moment_hat[name].sqrt(),
            value=-float(learning_rate),
        )


def _broadcast_state(
    payload: Mapping[str, Any],
    name: str,
    device: torch.device,
) -> dict[str, Tensor]:
    state = payload.get(name)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"fit request payload must contain {name}")
    resolved: dict[str, Tensor] = {}
    for key, value in state.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} entry {key!r} is not a tensor")
        # Clone: the server hands one shared read-only payload to every client
        # in the round, and the loop below writes in place.
        resolved[key] = value.detach().to(device).clone()
    return resolved


def _to_cpu_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in state.items()}


def _learning_rate_metrics(
    second_moment_hat: Mapping[str, Tensor],
    learning_rate: float,
) -> dict[str, float]:
    """Summarise this client's per-coordinate learning rate alpha / sqrt(v_hat).

    Over every coordinate of every floating tensor in ``v_hat``: the mean (sum
    over coordinates / number of coordinates), the minimum and the maximum.
    Named ``effective_learning_rate_coordinate_*`` because they are statistics
    over one client's coordinates. The server turns each client's mean into
    ``effective_learning_rate_across_clients_*``, the spread that shows whether
    the rates actually became client-specific. The two families shared the
    ``client_effective_learning_rate_*`` names until FINDINGS.csv POST-F14.
    """

    total = 0.0
    count = 0
    smallest = math.inf
    largest = 0.0
    for tensor in second_moment_hat.values():
        rate = float(learning_rate) / tensor.detach().float().sqrt()
        total += float(rate.sum())
        count += rate.numel()
        smallest = min(smallest, float(rate.min()))
        largest = max(largest, float(rate.max()))
    if not count:
        raise ValueError("fedlalr second moment state is empty")
    return {
        "effective_learning_rate_coordinate_mean": total / count,
        "effective_learning_rate_coordinate_min": smallest,
        "effective_learning_rate_coordinate_max": largest,
    }


def _unit_interval_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number in [0, 1)")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized < 1.0:
        raise ValueError(f"{name} must be a number in [0, 1)")
    return normalized
