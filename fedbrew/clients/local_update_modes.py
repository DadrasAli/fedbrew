"""Shared SGD local-update modes."""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn, optim

from fedbrew.core.torch_utils import OptimizerLike
from fedbrew.tasks.base import TaskAdapter, batch_example_count, loss_averages_over_examples

#: The smallest learning rate this module will run a step at.
#:
#: `run_sgd_update_mode` refuses a non-positive rate, because a configured zero
#: is a misconfiguration: the round would do nothing and nothing would say so.
#: A *decaying schedule* reaching zero is not that. `min_learning_rate: 0.0` is
#: in the range `config.py` documents, and a cosine schedule that reaches zero
#: at the horizon is the standard form -- so `_round_learning_rate` returns
#: exactly 0.0 on the final round, and every such run used to raise here after
#: spending all of its compute, leaving `global_rounds - 1` rounds on disk with
#: `run.json` still saying `status: running`.
#:
#: The floor belongs here, where the requirement is, rather than in the config.
#: This is the smallest positive float, so it satisfies the guard and changes
#: no run that did not previously raise: `p - lr * gradient` is `p` in float64
#: for any gradient below about 5e291, which is the no-op step the schedule
#: asked for. It is not a chosen magnitude -- it is the smallest one the guard
#: accepts, so it introduces no step size anybody configured.
#: `torch_sgd_client._round_learning_rate` clamps to it. FINDINGS.csv POST-F01.
MIN_POSITIVE_LEARNING_RATE = sys.float_info.min

#: One optimizer step on the exact gradient of the task's training loss over the
#: client's whole train split. Named rather than reached by a batch size that
#: happens to cover the split, so a config says what it computes.
FULL_GRADIENT_UPDATE_MODE = "full_gradient"

SUPPORTED_UPDATE_MODES = {
    "single_batch",
    "sequential_epoch",
    "frozen_batch_gradients",
    FULL_GRADIENT_UPDATE_MODE,
}
#: delta_sgd takes every mode. Under ``full_gradient`` its step-size rule sees
#: the exact gradient of the client's objective, so its smoothness estimate is
#: exact rather than taken between two different batches.
DELTA_SGD_UPDATE_MODES = set(SUPPORTED_UPDATE_MODES)
SUPPORTED_FROZEN_WEIGHTING = {"examples", "uniform", "sum"}
#: The modes of a rule that runs its own local loop (fedprox, scaffold,
#: fedlalr, and local_sgd and local_adamw in the base client's): that loop, one
#: step of the rule's update per batch, and one step of it per iteration on the
#: whole split's gradient (`full_gradient_into_grad`).
OWN_LOOP_UPDATE_MODES = {"sequential_epoch", FULL_GRADIENT_UPDATE_MODE}


def own_loop_update_mode(update_mode: str | None) -> str:
    """An own-loop rule's mode, where unset is the loop it always ran: sequential_epoch."""

    return normalize_choice(update_mode or "sequential_epoch", OWN_LOOP_UPDATE_MODES, "update_mode")


@dataclass(slots=True)
class LocalUpdateResult:
    """Outputs and applied optimizer-step count from one local update."""

    training_outputs: list[Mapping[str, float]]
    optimizer_steps: int


@dataclass(slots=True)
class DeltaSGDUpdateResult:
    """Outputs, step count and step-size trace from one Delta-SGD update."""

    training_outputs: list[Mapping[str, float]]
    optimizer_steps: int
    #: The step size actually applied at each local step, in order.
    step_sizes: list[float] = field(default_factory=list)
    #: Steps where ``eta_max`` bound the rule. Non-zero means the clamp, not the
    #: local smoothness, is choosing the step size.
    clamped_steps: int = 0
    #: Steps where the smoothness estimate was undefined because the parameter
    #: or gradient difference vanished, and only the growth term applied.
    undefined_curvature_steps: int = 0


def run_sgd_update_mode(
    *,
    task: TaskAdapter,
    model: nn.Module,
    train_loader: Iterable[Any],
    local_iterations: int,
    learning_rate: float,
    update_mode: str,
    frozen_gradient_weighting: str | None = None,
    client_id: str,
    max_grad_norm: float | None = None,
) -> LocalUpdateResult:
    """Run one of the shared SGD local-update modes.

    ``local_iterations`` counts iterations of the local loop, and the mode is
    what one iteration does:

    - ``single_batch``: one mini-batch SGD step. ``local_iterations`` updates.
    - ``sequential_epoch``: one complete chained mini-batch SGD epoch, one step
      per batch. ``local_iterations`` times the batch count.
    - ``frozen_batch_gradients``: compute all batch gradients at the
      iteration's starting model, combine them, then apply one parameter
      update. ``local_iterations`` updates.
    - ``full_gradient``: one parameter update on the exact gradient of the
      task's training loss over the whole train split -- the gradient
      ``train_step`` would take on one batch holding every sample. Computed
      batch by batch at the iteration's starting model, each batch weighted
      by its share of ``task.train_loss_denominator``, so ``batch_size``
      decides only how much is in memory at once. ``local_iterations``
      updates.

    So one ``local_iterations`` value is not one amount of local work across
    modes: FINDINGS.csv POST-F15.

    ``frozen_gradient_weighting`` is how ``frozen_batch_gradients`` combines
    its batch gradients, and no other mode reads it, so it is required under
    that mode alone. It used to be required and checked under all four, which
    made a caller that runs only the other three state a weighting that
    changes nothing. A value given under another mode is still checked: a
    caller holding one that is not a weighting has a bug either way.
    FINDINGS.csv POST-F21.

    ``max_grad_norm`` bounds the norm of the gradient each applied update uses,
    but *which* gradient, and how often, follows from the mode -- so the same
    threshold does not bound the same quantity across them:

    - ``single_batch`` and ``sequential_epoch`` clip **each batch gradient**,
      once per optimizer step, through ``_ClippingOptimizer``. An epoch over N
      batches moves at most ``N * learning_rate * max_grad_norm``.
    - ``frozen_batch_gradients`` and ``full_gradient`` clip the **combined
      epoch gradient** once, after the batch gradients have been weighted
      and summed. The same epoch
      moves at most ``learning_rate * max_grad_norm`` -- a factor of N less,
      and under ``frozen_gradient_weighting: sum`` the pre-clip quantity is N
      times larger too, so the clip engages far sooner.

    So two arms sharing a ``max_grad_norm`` across two different
    ``update_mode`` settings are not on a common bound. Per mode the behaviour
    is coherent: one clip per applied update. FINDINGS.csv P03-F05.

    Clipping composes with ``runtime.use_amp``; ``frozen_batch_gradients`` and
    ``full_gradient`` do not. ``_ClippingOptimizer`` wraps a real optimizer and exposes its
    ``param_groups``, which is all ``GradScaler.unscale_`` needs;
    ``_GradientOnlyOptimizer`` wraps none and exposes none. Both were refused
    until the difference was measured. ``validate_config`` refuses the one that
    cannot work; see ``config.amp_unsupported_sgd_engine_setting``.
    """

    mode = normalize_choice(update_mode, SUPPORTED_UPDATE_MODES, "update_mode")
    weighting = (
        None
        if frozen_gradient_weighting is None
        else normalize_choice(
            frozen_gradient_weighting,
            SUPPORTED_FROZEN_WEIGHTING,
            "frozen_gradient_weighting",
        )
    )

    if local_iterations <= 0:
        raise ValueError("local_iterations must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if max_grad_norm is not None and not max_grad_norm > 0.0:
        raise ValueError("max_grad_norm must be positive when set")

    if mode == "single_batch":
        return _run_single_batch(
            task=task,
            model=model,
            train_loader=train_loader,
            local_iterations=local_iterations,
            learning_rate=learning_rate,
            client_id=client_id,
            max_grad_norm=max_grad_norm,
        )

    if mode == "sequential_epoch":
        return _run_sequential_epochs(
            task=task,
            model=model,
            train_loader=train_loader,
            local_iterations=local_iterations,
            learning_rate=learning_rate,
            client_id=client_id,
            max_grad_norm=max_grad_norm,
        )

    if mode == FULL_GRADIENT_UPDATE_MODE:
        return _run_full_gradient(
            task=task,
            model=model,
            train_loader=train_loader,
            local_iterations=local_iterations,
            learning_rate=learning_rate,
            client_id=client_id,
            max_grad_norm=max_grad_norm,
        )

    if weighting is None:
        raise ValueError(
            "update_mode: frozen_batch_gradients combines its batch gradients by "
            "frozen_gradient_weighting, and none was given; choose one of: "
            f"{', '.join(sorted(SUPPORTED_FROZEN_WEIGHTING))}"
        )
    return _run_frozen_gradient_epochs(
        task=task,
        model=model,
        train_loader=train_loader,
        local_iterations=local_iterations,
        learning_rate=learning_rate,
        frozen_gradient_weighting=weighting,
        client_id=client_id,
        max_grad_norm=max_grad_norm,
    )


def run_delta_sgd_update_mode(
    *,
    task: TaskAdapter,
    model: nn.Module,
    train_loader: Iterable[Any],
    local_iterations: int,
    update_mode: str,
    frozen_gradient_weighting: str,
    client_id: str,
    eta_0: float,
    theta_0: float,
    gamma: float,
    delta: float,
    eta_max: float | None = None,
    max_grad_norm: float | None = None,
) -> DeltaSGDUpdateResult:
    """Run a local update whose step size auto-tunes to the local smoothness.

    Implements Algorithm 1 of Kim et al., "Adaptive Federated Learning with
    Auto-Tuned Clients" (arXiv:2306.11201), with the delta that the paper's
    Section 4 adds to the second condition. At local step k, having arrived at
    x_k from x_{k-1}:

        eta_k   = min{ gamma * ||x_k - x_{k-1}|| / (2 ||g_k - g_{k-1}||),
                       sqrt(1 + delta * theta_{k-1}) * eta_{k-1} }
        theta_k = eta_k / eta_{k-1}
        x_{k+1} = x_k - eta_k * g_k

    eta and theta are reset to eta_0 / theta_0 at the start of every round
    (Algorithm 1 lines 6-7), so nothing model-sized persists between rounds and
    the client stays stateless.

    Cost is one backward pass per step, exactly as plain SGD: g_k is measured
    once at x_k, used to pick eta_k, and then reused as the update direction.

    ``local_iterations`` counts iterations of the local loop and the selected
    ``update_mode`` is what one iteration does, as for the FedAvg arms, which
    keeps the comparison with them like-for-like:

    - ``single_batch``: one mini-batch step.
    - ``sequential_epoch``: one complete chained mini-batch epoch.
    - ``frozen_batch_gradients``: all batch gradients evaluated at the same
      starting model, combined into one update. The step-size rule then sees
      one step per epoch, with the combined gradient as g_k.
    - ``full_gradient``: one step on the exact gradient of the whole train
      split, computed as the FedAvg arms' ``full_gradient`` computes it. g_k is
      then the client objective's gradient at x_k, and the smoothness estimate
      is exact.

    eta resets to eta_0 every round, so a round of one local step -- one
    iteration of ``full_gradient`` or ``single_batch`` -- steps at eta_0 and is
    FedAvg at that learning rate. The rule adapts from a round's second step.
    """

    mode = normalize_choice(update_mode, DELTA_SGD_UPDATE_MODES, "update_mode")
    weighting = normalize_choice(
        frozen_gradient_weighting,
        SUPPORTED_FROZEN_WEIGHTING,
        "frozen_gradient_weighting",
    )

    if local_iterations <= 0:
        raise ValueError("local_iterations must be positive")
    if max_grad_norm is not None and not max_grad_norm > 0.0:
        raise ValueError("max_grad_norm must be positive when set")
    # The rule needs the raw gradient tensors, which the AMP path consumes
    # inside GradScaler.step instead of leaving on .grad. Fail loudly rather
    # than silently training on scaled gradients.
    if getattr(task, "_scaler", None) is not None:
        raise ValueError("delta_sgd does not support runtime.use_amp: true")

    stepper = _DeltaSGDStepper(
        eta_0=eta_0,
        theta_0=theta_0,
        gamma=gamma,
        delta=delta,
        eta_max=eta_max,
    )
    outputs: list[Mapping[str, float]] = []

    if mode == "frozen_batch_gradients":
        _refuse_frozen_off_examples(task, client_id)
        for _ in range(local_iterations):
            batches = list(train_loader)
            if not batches:
                raise ValueError(f"client {client_id!r} has no training batches")
            epoch_examples = _frozen_epoch_examples(batches)
            combined = _zero_parameter_like(model)
            for batch in batches:
                gradient_optimizer = _GradientOnlyOptimizer(model.parameters())
                outputs.append(
                    task.train_step(
                        model,
                        batch,
                        gradient_optimizer,
                    )
                )
                _accumulate_parameter_gradients(
                    model,
                    combined,
                    _gradient_weight(
                        batch=batch,
                        num_batches=len(batches),
                        epoch_examples=epoch_examples,
                        weighting=weighting,
                    ),
                )
            _apply_delta_sgd_step(model, combined, stepper, max_grad_norm)
        return _delta_sgd_result(outputs, stepper)

    if mode == FULL_GRADIENT_UPDATE_MODE:
        for _ in range(local_iterations):
            gradient = _whole_split_gradient(task, model, train_loader, client_id, outputs)
            _apply_delta_sgd_step(model, gradient, stepper, max_grad_norm)
        return _delta_sgd_result(outputs, stepper)

    if mode == "single_batch":
        batch_iterator = iter(train_loader)
        for _ in range(local_iterations):
            batch, batch_iterator = _next_batch(
                train_loader,
                batch_iterator,
                client_id=client_id,
            )
            outputs.append(_delta_sgd_batch_step(task, model, batch, stepper, max_grad_norm))
        return _delta_sgd_result(outputs, stepper)

    for _ in range(local_iterations):
        epoch_had_batch = False
        for batch in train_loader:
            epoch_had_batch = True
            outputs.append(_delta_sgd_batch_step(task, model, batch, stepper, max_grad_norm))
        if not epoch_had_batch:
            raise ValueError(f"client {client_id!r} has no training batches")
    return _delta_sgd_result(outputs, stepper)


def _delta_sgd_batch_step(
    task: TaskAdapter,
    model: nn.Module,
    batch: Any,
    stepper: _DeltaSGDStepper,
    max_grad_norm: float | None,
) -> Mapping[str, float]:
    """Measure the gradient on one batch, pick a step size, and apply it."""

    gradient_optimizer = _GradientOnlyOptimizer(model.parameters())
    output = task.train_step(model, batch, gradient_optimizer)  # type: ignore[arg-type]
    _apply_delta_sgd_step(model, _parameter_gradients(model), stepper, max_grad_norm)
    return output


def _apply_delta_sgd_step(
    model: nn.Module,
    gradient: dict[str, Tensor],
    stepper: _DeltaSGDStepper,
    max_grad_norm: float | None,
) -> None:
    """Clip, choose eta from the clipped gradient, then take the step.

    Clipping happens first so the step-size rule sees the gradient that is
    actually applied. Otherwise the smoothness estimate would describe a step
    the model never takes.
    """

    if max_grad_norm is not None:
        _clip_accumulated_update(gradient, max_grad_norm)
    _apply_parameter_update(model, gradient, stepper.step_size_for(gradient))


def _delta_sgd_result(
    outputs: list[Mapping[str, float]],
    stepper: _DeltaSGDStepper,
) -> DeltaSGDUpdateResult:
    return DeltaSGDUpdateResult(
        training_outputs=outputs,
        optimizer_steps=len(stepper.step_sizes),
        step_sizes=list(stepper.step_sizes),
        clamped_steps=stepper.clamped_steps,
        undefined_curvature_steps=stepper.undefined_curvature_steps,
    )


class _DeltaSGDStepper:
    """Carry eta and theta across the local steps of one round."""

    def __init__(
        self,
        *,
        eta_0: float,
        theta_0: float,
        gamma: float,
        delta: float,
        eta_max: float | None,
    ) -> None:
        for name, value in (("eta_0", eta_0), ("theta_0", theta_0), ("gamma", gamma)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive number")
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError("delta must be a finite positive number")
        if eta_max is not None and (not math.isfinite(eta_max) or eta_max <= 0.0):
            raise ValueError("eta_max must be a finite positive number when set")

        self.eta = float(eta_0)
        self.theta = float(theta_0)
        self._gamma = float(gamma)
        self._delta = float(delta)
        self._eta_max = None if eta_max is None else float(eta_max)
        self._previous_gradient: dict[str, Tensor] | None = None
        self.step_sizes: list[float] = []
        self.clamped_steps = 0
        self.undefined_curvature_steps = 0

    def step_size_for(self, gradient: Mapping[str, Tensor]) -> float:
        """Return the step size to apply where ``gradient`` was measured."""

        previous = self._previous_gradient
        if previous is not None:
            # ||x_k - x_{k-1}|| is derived, not measured: the update that
            # produced the current iterate was exactly -eta * previous gradient,
            # so the distance is eta * ||previous gradient||. Exact in every
            # mode here, and it saves cloning the parameters on every step.
            iterate_distance = self.eta * _state_norm(previous)
            gradient_distance = _state_distance(gradient, previous)
            growth = math.sqrt(1.0 + self._delta * self.theta) * self.eta
            if iterate_distance > 0.0 and gradient_distance > 0.0:
                eta_next = min(
                    self._gamma * iterate_distance / (2.0 * gradient_distance),
                    growth,
                )
            else:
                # Malitsky & Mishchenko's reference implementation falls back to
                # the growth term when either difference vanishes: the local
                # smoothness is undefined there, not infinite. The paper leaves
                # this case unspecified.
                eta_next = growth
                self.undefined_curvature_steps += 1
            if self._eta_max is not None and eta_next > self._eta_max:
                eta_next = self._eta_max
                self.clamped_steps += 1
            if not math.isfinite(eta_next) or eta_next <= 0.0:
                raise ValueError("delta_sgd step size must stay finite and positive")
            # theta is the ratio to the step size this one replaces, so it has
            # to be taken before eta is reassigned.
            self.theta = eta_next / self.eta
            self.eta = eta_next

        self._previous_gradient = {name: value.detach().clone() for name, value in gradient.items()}
        self.step_sizes.append(self.eta)
        return self.eta


def _parameter_gradients(model: nn.Module) -> dict[str, Tensor]:
    """Read every trainable gradient, filling absent ones with zeros.

    Every step must produce the same key set, or the norms below cannot be
    differenced against the previous step.
    """

    gradients: dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            gradients[name] = torch.zeros_like(parameter.detach())
        else:
            gradients[name] = parameter.grad.detach().clone()
    if not gradients:
        raise ValueError("delta_sgd requires at least one trainable parameter")
    return gradients


def _state_norm(state: Mapping[str, Tensor]) -> float:
    total = sum(float((tensor.detach() ** 2).sum()) for tensor in state.values())
    return math.sqrt(total)


def _state_distance(a: Mapping[str, Tensor], b: Mapping[str, Tensor]) -> float:
    if set(a) != set(b):
        raise ValueError("delta_sgd gradient keys changed between local steps")
    total = sum(float(((a[name].detach() - b[name].detach()) ** 2).sum()) for name in a)
    return math.sqrt(total)


def _run_single_batch(
    *,
    task: TaskAdapter,
    model: nn.Module,
    train_loader: Iterable[Any],
    local_iterations: int,
    learning_rate: float,
    client_id: str,
    max_grad_norm: float | None = None,
) -> LocalUpdateResult:
    optimizer = _clipped(optim.SGD(model.parameters(), lr=learning_rate), model, max_grad_norm)
    batch_iterator = iter(train_loader)
    outputs: list[Mapping[str, float]] = []

    for _ in range(local_iterations):
        batch, batch_iterator = _next_batch(
            train_loader,
            batch_iterator,
            client_id=client_id,
        )
        outputs.append(task.train_step(model, batch, optimizer))

    return LocalUpdateResult(outputs, optimizer_steps=local_iterations)


def _run_sequential_epochs(
    *,
    task: TaskAdapter,
    model: nn.Module,
    train_loader: Iterable[Any],
    local_iterations: int,
    learning_rate: float,
    client_id: str,
    max_grad_norm: float | None = None,
) -> LocalUpdateResult:
    optimizer = _clipped(optim.SGD(model.parameters(), lr=learning_rate), model, max_grad_norm)
    outputs: list[Mapping[str, float]] = []
    optimizer_steps = 0

    for _ in range(local_iterations):
        epoch_had_batch = False
        for batch in train_loader:
            epoch_had_batch = True
            outputs.append(task.train_step(model, batch, optimizer))
            optimizer_steps += 1
        if not epoch_had_batch:
            raise ValueError(f"client {client_id!r} has no training batches")

    return LocalUpdateResult(outputs, optimizer_steps=optimizer_steps)


def _run_frozen_gradient_epochs(
    *,
    task: TaskAdapter,
    model: nn.Module,
    train_loader: Iterable[Any],
    local_iterations: int,
    learning_rate: float,
    frozen_gradient_weighting: str,
    client_id: str,
    max_grad_norm: float | None = None,
) -> LocalUpdateResult:
    _refuse_frozen_off_examples(task, client_id)
    outputs: list[Mapping[str, float]] = []

    for _ in range(local_iterations):
        batches = list(train_loader)
        if not batches:
            raise ValueError(f"client {client_id!r} has no training batches")

        epoch_examples = _frozen_epoch_examples(batches)
        accumulated = _zero_parameter_like(model)
        for batch in batches:
            gradient_optimizer = _GradientOnlyOptimizer(model.parameters())
            outputs.append(
                task.train_step(
                    model,
                    batch,
                    gradient_optimizer,
                )
            )
            weight = _gradient_weight(
                batch=batch,
                num_batches=len(batches),
                epoch_examples=epoch_examples,
                weighting=frozen_gradient_weighting,
            )
            _accumulate_parameter_gradients(model, accumulated, weight)

        if max_grad_norm is not None:
            _clip_accumulated_update(accumulated, max_grad_norm)
        _apply_parameter_update(model, accumulated, learning_rate)

    return LocalUpdateResult(outputs, optimizer_steps=local_iterations)


def _refuse_frozen_off_examples(task: TaskAdapter, client_id: str) -> None:
    """Refuse `frozen_batch_gradients` on a task whose loss is not an example mean.

    The mode weights batch gradients by example count (or equally, or not at
    all), which is the gradient of the pass only when each batch's loss is a
    mean over its examples. On the causal-LM task, a mean over active target
    tokens, the combined update was 53.5% from that gradient on two batches of
    2 and 8 tokens: FINDINGS.csv POST-F19. `validate_config` refuses the
    built-in tasks this covers at load; this is the backstop for the rest, an
    extension task among them, whose class is not known until it is built.
    Raised before the first gradient, so nothing is applied.
    """

    if not loss_averages_over_examples(task):
        raise ValueError(
            f"client {client_id!r}: update_mode: frozen_batch_gradients combines batch "
            f"gradients by example count, and {type(task).__name__} averages its training "
            "loss over something else (it overrides train_loss_denominator), so the "
            "combined update would not be the gradient of the pass (FINDINGS.csv "
            "POST-F19). Use update_mode: full_gradient, which weights by the task's own "
            "count."
        )


def _run_full_gradient(
    *,
    task: TaskAdapter,
    model: nn.Module,
    train_loader: Iterable[Any],
    local_iterations: int,
    learning_rate: float,
    client_id: str,
    max_grad_norm: float | None = None,
) -> LocalUpdateResult:
    """One update per iteration on the gradient of the whole train split.

    A batch's gradient is the gradient of its own mean loss, over the
    `train_loss_denominator` it reports: examples for a loss that averages
    over examples, active target tokens for the causal-LM loss. The whole
    split's mean loss is the denominator-weighted mean of the batch losses, so
    weighting each batch gradient by its denominator and dividing by their sum
    is that loss's gradient exactly, whatever the batch sizes or order.
    Weighting by examples instead is exact only for the first kind of task:
    FINDINGS.csv POST-F19, which is `frozen_batch_gradients`' weighting.

    The loader is iterated, not listed, so only one batch is in memory at a
    time; the denominators are summed as it goes and the division comes last.
    """

    outputs: list[Mapping[str, float]] = []

    for _ in range(local_iterations):
        accumulated = _whole_split_gradient(task, model, train_loader, client_id, outputs)
        if max_grad_norm is not None:
            _clip_accumulated_update(accumulated, max_grad_norm)
        _apply_parameter_update(model, accumulated, learning_rate)

    return LocalUpdateResult(outputs, optimizer_steps=local_iterations)


def full_gradient_into_grad(
    *,
    task: TaskAdapter,
    model: nn.Module,
    train_loader: Iterable[Any],
    client_id: str,
) -> list[Mapping[str, float]]:
    """Leave the gradient of the whole train split in ``.grad``, for a rule's own step.

    One ``full_gradient`` iteration without the update: every trainable
    parameter that the loss reaches holds the exact gradient of the task's
    training loss over the whole split, computed as `_run_full_gradient`
    computes it, and nothing has moved. A rule whose step is not plain SGD --
    FedProx's proximal term, SCAFFOLD's ``c - c_i``, FedLALR's AMSGrad -- then
    takes that step from ``.grad`` exactly as it does after a batch. A
    parameter the loss never reaches keeps ``grad`` None, as after a batch.

    Returns the batches' ``train_step`` outputs, in order.
    """

    outputs: list[Mapping[str, float]] = []
    accumulated = _whole_split_gradient(task, model, train_loader, client_id, outputs)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            parameter.grad = accumulated[name]
    return outputs


def _whole_split_gradient(
    task: TaskAdapter,
    model: nn.Module,
    train_loader: Iterable[Any],
    client_id: str,
    outputs: list[Mapping[str, float]],
) -> dict[str, Tensor]:
    """The gradient of the training loss over the whole split, at the current model.

    Appends each batch's ``train_step`` output to ``outputs``.
    """

    accumulated = _zero_parameter_like(model)
    denominator = 0.0
    batches = 0
    for batch in train_loader:
        output = task.train_step(model, batch, _GradientOnlyOptimizer(model.parameters()))
        outputs.append(output)
        weight = float(task.train_loss_denominator(batch, output))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"client {client_id!r}: train_loss_denominator returned {weight!r}; "
                "it must be a finite count, zero or more"
            )
        _accumulate_parameter_gradients(model, accumulated, weight)
        denominator += weight
        batches += 1
    if not batches:
        raise ValueError(f"client {client_id!r} has no training batches")
    if denominator <= 0.0:
        raise ValueError(
            f"client {client_id!r}: the train split holds nothing its loss averages "
            "over, so it has no gradient to step on"
        )
    for tensor in accumulated.values():
        tensor.div_(denominator)
    return accumulated


def _clipped(
    optimizer: optim.Optimizer,
    model: nn.Module,
    max_grad_norm: float | None,
) -> OptimizerLike:
    """Wrap an optimizer so gradients are clipped before every step.

    Returns the argument unwrapped when there is nothing to clip, so the return
    type is the wider of the two: `OptimizerLike` covers both, and every caller
    only ever hands the result to `task.train_step`.
    """

    if max_grad_norm is None:
        return optimizer
    return _ClippingOptimizer(optimizer, model, max_grad_norm)


class _ClippingOptimizer:
    """Delegate to an optimizer, clipping the gradient norm before each step."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        max_grad_norm: float,
    ) -> None:
        self._optimizer = optimizer
        self._model = model
        self._max_grad_norm = float(max_grad_norm)

    @property
    def param_groups(self) -> Any:
        return self._optimizer.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._optimizer.zero_grad(set_to_none)

    def step(self, closure: Any | None = None) -> None:
        torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter in self._model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ],
            self._max_grad_norm,
        )
        self._optimizer.step(closure)


def _clip_accumulated_update(
    accumulated: dict[str, Tensor],
    max_grad_norm: float,
) -> None:
    """Scale a combined frozen-mode update down to the clipping threshold."""

    total = torch.sqrt(sum((tensor.detach() ** 2).sum() for tensor in accumulated.values()))
    if float(total) > max_grad_norm:
        scale = max_grad_norm / (float(total) + 1e-6)
        for tensor in accumulated.values():
            tensor.mul_(scale)


def _frozen_epoch_examples(batches: Sequence[Any]) -> float:
    """Examples in the batches this epoch will actually combine.

    The "examples" weighting means the example-weighted mean gradient over
    what was differentiated, so the denominator has to be the sum over these
    batches and nothing else. A client-level total is wrong whenever the two
    differ -- it includes the eval split, and it ignores drop_last and any cap
    on the loader -- and the weights then sum to less than one, silently
    scaling the step.
    """

    return max(1.0, sum(batch_example_count(batch) for batch in batches))


def _gradient_weight(
    *,
    batch: Any,
    num_batches: int,
    epoch_examples: float,
    weighting: str,
) -> float:
    if weighting == "uniform":
        return 1.0 / float(num_batches)
    if weighting == "sum":
        return 1.0
    return batch_example_count(batch) / epoch_examples


def _zero_parameter_like(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: torch.zeros_like(parameter.detach(), device=parameter.device)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _accumulate_parameter_gradients(
    model: nn.Module,
    accumulated: dict[str, Tensor],
    weight: float,
) -> None:
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        accumulated[name] = accumulated[name] + parameter.grad.detach() * float(weight)


@torch.no_grad()
def _apply_parameter_update(
    model: nn.Module,
    accumulated: Mapping[str, Tensor],
    learning_rate: float,
) -> None:
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        update = accumulated.get(name)
        if update is not None:
            parameter.add_(update.to(parameter.device), alpha=-float(learning_rate))


def _next_batch(
    train_loader: Iterable[Any],
    batch_iterator: Iterator[Any],
    *,
    client_id: str,
) -> tuple[Any, Iterator[Any]]:
    try:
        return next(batch_iterator), batch_iterator
    except StopIteration:
        batch_iterator = iter(train_loader)
        try:
            return next(batch_iterator), batch_iterator
        except StopIteration as exc:
            raise ValueError(f"client {client_id!r} has no training batches") from exc


def normalize_choice(value: str, allowed: set[str], name: str) -> str:
    """Lower-case and strip a config choice, then check it against ``allowed``.

    Args:
        value: The configured string.
        allowed: The permitted values, already lower-case.
        name: The setting's name, used in the error message.

    Returns:
        The normalised value, so callers store one spelling regardless of how
        the config was written.

    Raises:
        ValueError: If the normalised value is not in ``allowed``. Rejecting
            rather than falling back to a default is the point: an unrecognised
            choice would otherwise run a different update rule than the config
            names, and run.json would record the config's spelling.
    """

    normalized = value.lower().strip()
    if normalized not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(allowed))}")
    return normalized


class _GradientOnlyOptimizer:
    """Optimizer-like object that computes gradients without parameter updates."""

    def __init__(self, parameters: Iterable[nn.Parameter]) -> None:
        self._parameters = list(parameters)

    def zero_grad(self, set_to_none: bool = False) -> None:
        for parameter in self._parameters:
            if parameter.grad is None:
                continue
            if set_to_none:
                parameter.grad = None
            else:
                parameter.grad.detach_()
                parameter.grad.zero_()

    def step(self, closure: Any | None = None) -> None:
        if closure is not None:
            closure()
