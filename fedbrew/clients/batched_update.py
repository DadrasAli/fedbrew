"""A client's local update, described so the batched executor can run many at once.

The sequential path runs a rule's local update by handing ``task.train_step``
an optimizer and iterating a loader. The batched executor
(``fedbrew/core/batched_executor.py``) runs the same update for every sampled
client together, through ``torch.func`` over a stack of their parameters, so it
needs the update in pieces it can stack:

- a :class:`LocalProgram`: what one applied update does -- which gradient it
  steps on (one batch's, or a combination of a pass's batches), which
  correction and clipping it applies, and which optimizer step -- the same for
  every client of a round;
- a :class:`ClientBatchPlan` per client: the batches each of its applied
  updates consumes, as row indices, drawn exactly as its sequential loop draws
  them (:func:`sgd_mode_updates`, :func:`own_loop_updates`), and where it
  starts from;
- the step arithmetic, as functions of one client's tensors
  (:func:`apply_update` and what it calls), written with the operations
  ``torch.optim.SGD``, ``torch.optim.AdamW``, ``clip_grad_norm_`` and this
  package's update modes use, in their order. ``torch.func.vmap`` runs them
  over the client dimension; without it they are the sequential arithmetic bit
  for bit, which is what the one-client tolerance test checks.

A rule opts in by declaring ``_batched_rule`` on its own class and
implementing ``batched_plan`` and ``batched_result``
(:class:`fedbrew.clients.torch_sgd_client.TorchSGDClient`). A subclass that
does not declare it inherits nothing batched, so an extension rule built on a
batchable one runs sequentially.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from fedbrew.clients.batch_orders import LocalLoop, RoundOrders, plan_orders
from fedbrew.clients.local_update_modes import FULL_GRADIENT_UPDATE_MODE, _next_batch
from fedbrew.core.seeding import dataloader_seeds
from fedbrew.tasks.base import LoaderOrder

#: How the gradient of one applied update is formed.
#: ``batch``: one batch's. ``frozen``: the pass's batch gradients at one
#: model, weighted and added (``update_mode: frozen_batch_gradients``).
#: ``full``: the same weighted by each batch's loss denominator and divided by
#: their sum (``update_mode: full_gradient``).
COMBINES = ("batch", "frozen", "full")


@dataclass(frozen=True, slots=True)
class OptimizerSpec:
    """The optimizer step, with the hyperparameters of this round.

    ``sgd`` is ``torch.optim.SGD``'s single-tensor step (dampening 0),
    ``adamw`` ``torch.optim.AdamW``'s; both start from no state every round,
    as ``reused_optimizer`` hands them out.
    """

    kind: str
    lr: float
    momentum: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    @property
    def state_slots(self) -> int:
        """Model-sized tensors of state it keeps per client."""

        if self.kind == "adamw":
            return 2
        return 1 if self.momentum != 0.0 else 0


@dataclass(frozen=True, slots=True)
class LocalProgram:
    """What one applied update does, the same for every client of a round.

    ``max_grad_norm`` clips the batch gradient before each step under
    ``combine: batch`` (``clip_grad_norm_``, as ``_ClippingOptimizer``) and the
    combined gradient once under the other two (``_clip_accumulated_update``).
    ``proximal_mu`` adds FedProx's ``mu (w - w0)`` to the gradient, and
    ``scaffold`` SCAFFOLD's ``c - c_i``, before the optimizer step.
    ``weighting`` is ``frozen_gradient_weighting`` under ``combine: frozen``.
    """

    optimizer: OptimizerSpec
    combine: str = "batch"
    max_grad_norm: float | None = None
    proximal_mu: float = 0.0
    scaffold: bool = False
    weighting: str | None = None


@dataclass(slots=True)
class ClientBatchPlan:
    """One client's local update of one round, as the batched executor runs it.

    What the rule declares: its program, the loop its update takes over its
    loader (``loop``), what its training and post-fit loaders yield
    (``train_order``, ``eval_order``; seeded when the round is planned), and
    how it refuses a split that yields no batch (``refuse``). A task that
    declares no order has its loaders replayed instead (``replay``: each
    update's batches and the post-fit pass's, as row indices).

    What the round's planning fills in (:func:`plan_round`): ``slot``, this
    client's row in the round's orders; ``structure``, how many batches each
    of its applied updates consumes; ``eval_rows``, the rows its post-fit
    pass takes, which are the example count that pass reports.
    """

    program: LocalProgram
    train_data: Any
    loop: LocalLoop
    train_order: LoaderOrder | None
    eval_order: LoaderOrder | None
    evaluate: bool
    start: Mapping[str, Tensor]
    client_id: str
    seed: int | None
    refuse: Callable[[], Exception]
    replay: Callable[[], tuple[list[list[Tensor]], list[Tensor]]] | None = None
    client_control: Mapping[str, Tensor] | None = None
    server_control: Mapping[str, Tensor] | None = None
    slot: int = -1
    structure: tuple[int, ...] = ()
    eval_rows: int = 0

    @property
    def bucket(self) -> tuple[Any, ...]:
        """What two clients must share to be stepped together."""

        return (self.program, self.structure, self.evaluate)


@dataclass(slots=True)
class ClientBatchFit:
    """What the batched executor hands back to the rule, for one client."""

    #: The trained model state, rows of the chunk's stack.
    model_state: dict[str, Tensor]
    #: What ``train_step`` would have returned for each batch, in order.
    training_outputs: list[dict[str, float]]
    #: What ``eval_step`` would have returned for each post-fit batch; None
    #: on a round the pass is skipped.
    eval_outputs: list[dict[str, float]] | None
    optimizer_steps: int
    #: ``task.federated_model_state_metadata`` of the model, this client's copy.
    model_state_metadata: dict[str, Any]
    trainable_parameters: int
    #: The start state, for a rule whose result is measured against it.
    start: Mapping[str, Tensor] = field(default_factory=dict)
    #: ``task.compute_metrics`` of the post-fit outputs and their example
    #: count, when the task computed them for the whole stack at once
    #: (``stacked_metrics``); ``eval_outputs`` is then None.
    eval_metrics: tuple[dict[str, float], int] | None = None


@dataclass(slots=True)
class ClientEvalPlan:
    """One client's evaluation of one round, as the batched evaluator runs it.

    Per requested split, in order: the split's data, or None for a ``val``
    split the client does not have (reported as zero examples); what its
    evaluation loader yields (``orders``, seeded when the round is planned),
    or, for a task that declares no order, the loader's batches as row
    indices (``batches``). ``refusal`` is what the client's own ``evaluate``
    would raise, raised when its result is built.
    """

    splits: list[str]
    client_id: str
    seed: int | None
    data: list[Any] = field(default_factory=list)
    orders: list[LoaderOrder | None] = field(default_factory=list)
    batches: list[list[Tensor] | None] = field(default_factory=list)
    refusal: Exception | None = None

    def row_count(self, position: int) -> int:
        """The rows split ``position`` holds."""

        order = self.orders[position]
        if order is not None:
            return order.rows
        return sum(len(batch) for batch in self.batches[position] or [])


def data_versions(data: Any) -> tuple[int, ...]:
    """The version counters of a split's tensors, which move on any in-place edit."""

    if isinstance(data, Mapping):
        return tuple(value._version for value in data.values() if isinstance(value, Tensor))
    return (data._version,) if isinstance(data, Tensor) else ()


# ---------------------------------------------------------------------------
# The batches, drawn as the sequential loops draw them
# ---------------------------------------------------------------------------


def _no_training_batches(client_id: str) -> ValueError:
    return ValueError(f"client {client_id!r} has no training batches")


def sgd_mode_updates(
    loader: Iterable[Any],
    *,
    local_iterations: int,
    update_mode: str,
    client_id: str,
) -> list[list[Any]]:
    """The batches each update of ``run_sgd_update_mode`` consumes, in order.

    Iterates ``loader`` exactly as ``_run_single_batch``,
    ``_run_sequential_epochs``, ``_run_frozen_gradient_epochs`` and
    ``_run_full_gradient`` do -- the same ``iter`` calls, the same early stop --
    so a loader that draws as it starts draws the same numbers, and refuses a
    client without batches with the same message.
    """

    updates: list[list[Any]] = []
    if update_mode == "single_batch":
        batch_iterator = iter(loader)
        for _ in range(local_iterations):
            batch, batch_iterator = _next_batch(loader, batch_iterator, client_id=client_id)
            updates.append([batch])
        return updates
    for _ in range(local_iterations):
        batches = list(loader)
        if not batches:
            raise _no_training_batches(client_id)
        if update_mode == "sequential_epoch":
            updates.extend([batch] for batch in batches)
        else:
            updates.append(batches)
    return updates


def own_loop_updates(
    loader: Iterable[Any],
    *,
    local_iterations: int,
    update_mode: str,
    max_local_steps: int | None,
    client_id: str,
) -> list[list[Any]]:
    """The batches each step of a rule's own loop consumes, in order.

    ``TorchSGDClient.fit``'s loop, which FedProx and SCAFFOLD repeat without a
    step cap: one step per batch, or under ``full_gradient`` one per iteration
    on the whole pass, stopping at ``max_local_steps``. A client with no
    batches is left with no updates for the rule to refuse in its own words,
    except under ``full_gradient``, whose pass refuses first
    (``_whole_split_gradient``).
    """

    updates: list[list[Any]] = []
    for _ in range(local_iterations):
        if update_mode == FULL_GRADIENT_UPDATE_MODE:
            batches = list(loader)
            if not batches:
                raise _no_training_batches(client_id)
            updates.append(batches)
        else:
            for batch in loader:
                updates.append([batch])
                if max_local_steps is not None and len(updates) >= max_local_steps:
                    break
        if max_local_steps is not None and len(updates) >= max_local_steps:
            break
    return updates


def no_training_batches(client_id: str) -> ValueError:
    """The refusal ``sgd_mode_updates`` and a full-gradient pass raise for an empty loader."""

    return _no_training_batches(client_id)


def update_weights(lengths: Tensor, structure: tuple[int, ...], program: LocalProgram) -> Tensor:
    """Each batch's weight in its update's combined gradient, for clients sharing ``structure``.

    ``lengths[c, t]`` is client ``c``'s ``t``-th batch's rows. ``full``: the
    batch's loss denominator, its row count for a task whose loss averages over
    examples (``_whole_split_gradient``); ``frozen``: ``_gradient_weight``
    under ``frozen_gradient_weighting``. The same IEEE float64 arithmetic as
    the Python floats there: sums of whole numbers, and one division.
    """

    rows = lengths.to(torch.float64)
    weights = torch.zeros_like(rows)
    if program.combine == "batch":
        return weights
    start = 0
    for count in structure:
        part = rows[:, start : start + count]
        if program.combine == "full":
            weights[:, start : start + count] = part
        elif program.weighting == "uniform":
            weights[:, start : start + count] = 1.0 / float(count)
        elif program.weighting == "sum":
            weights[:, start : start + count] = 1.0
        else:
            epoch_examples = torch.clamp(part.sum(dim=1, keepdim=True), min=1.0)
            weights[:, start : start + count] = part / epoch_examples
        start += count
    return weights


def plan_round(plans: Sequence[ClientBatchPlan], round_id: int) -> tuple[RoundOrders, RoundOrders]:
    """Every client's training and post-fit batches for the round, planned together.

    Fills each plan's ``slot``, ``structure`` and ``eval_rows``, and raises the
    refusal of the first client, in request order, whose loader yields no
    batch -- before any client runs, in its rule's words.
    """

    for slot, plan in enumerate(plans):
        plan.slot = slot
    train = _phase_orders(
        plans,
        round_id,
        "fit",
        [plan.train_order for plan in plans],
        [plan.loop for plan in plans],
        lambda replayed: replayed[0],
    )
    steps = train.steps.tolist()
    for plan in plans:
        if not steps[plan.slot]:
            raise plan.refuse()
        plan.structure = train.structure[plan.slot]
    evaluation = _phase_orders(
        plans,
        round_id,
        "eval",
        [plan.eval_order for plan in plans],
        [LocalLoop(epochs=1)] * len(plans),
        lambda replayed: [[batch] for batch in replayed[1]],
    )
    rows = evaluation.lengths.sum(dim=1).tolist()
    for plan in plans:
        plan.eval_rows = rows[plan.slot]
    return train, evaluation


def _phase_orders(
    plans: Sequence[ClientBatchPlan],
    round_id: int,
    phase: str,
    orders: Sequence[LoaderOrder | None],
    loops: Sequence[LocalLoop],
    pick: Callable[[tuple[list[list[Tensor]], list[Tensor]]], list[list[Tensor]]],
) -> RoundOrders:
    """One phase's orders: the declared ones seeded in bulk and computed, the rest replayed."""

    owners = [(plan.client_id, plan.seed) for plan in plans]
    return round_orders(
        orders,
        loader_seeds(orders, owners, round_id, phase),
        loops,
        lambda index: pick(plans[index].replay()),  # type: ignore[misc]
    )


def loader_seeds(
    orders: Sequence[LoaderOrder | None],
    owners: Sequence[tuple[str, int | None]],
    round_id: int,
    phase: str,
) -> list[int | None]:
    """The seed of each shuffled order's loader, derived for every client in one pass.

    ``owners[k]`` is the client whose loader ``orders[k]`` is, and its run
    seed; the seed is ``dataloader_seed(seed, round_id, client, phase)``, as
    the client's own loader configuration carries it. None where no seed is
    drawn from.
    """

    seeds: list[int | None] = [None] * len(orders)
    shuffled = [index for index, order in enumerate(orders) if order is not None and order.shuffle]
    for base in {owners[index][1] for index in shuffled} - {None}:
        group = [index for index in shuffled if owners[index][1] == base]
        derived = dataloader_seeds(
            int(base),  # type: ignore[arg-type]
            int(round_id),
            [owners[index][0] for index in group],
            phase,
        )
        for index, seed in zip(group, derived, strict=True):
            seeds[index] = seed
    return seeds


def round_orders(
    orders: Sequence[LoaderOrder | None],
    seeds: Sequence[int | None],
    loops: Sequence[LocalLoop],
    replay: Callable[[int], list[list[Tensor]]],
) -> RoundOrders:
    """The declared orders computed together, and each undeclared one replayed (``replay``)."""

    declared = [index for index, order in enumerate(orders) if order is not None]
    if len(declared) == len(orders):
        return plan_orders(orders, loops, seeds)  # type: ignore[arg-type]
    replayed = [index for index, order in enumerate(orders) if order is None]
    parts, groups = [], []
    if declared:
        parts.append(
            plan_orders(
                [orders[index] for index in declared],  # type: ignore[misc]
                [loops[index] for index in declared],
                [seeds[index] for index in declared],
            )
        )
        groups.append(declared)
    parts.append(RoundOrders.from_updates([replay(index) for index in replayed]))
    groups.append(replayed)
    return RoundOrders.merge(parts, groups, len(orders))


# ---------------------------------------------------------------------------
# The step arithmetic, for one client's tensors
# ---------------------------------------------------------------------------


def initial_optimizer_state(spec: OptimizerSpec, params: Mapping[str, Tensor]) -> dict[str, Any]:
    """The state an optimizer starts a round with: none for SGD, zero moments for AdamW."""

    if spec.kind == "adamw":
        return {
            "exp_avg": {name: torch.zeros_like(value) for name, value in params.items()},
            "exp_avg_sq": {name: torch.zeros_like(value) for name, value in params.items()},
        }
    return {}


def accumulate(
    accumulated: Mapping[str, Tensor] | None,
    gradients: Mapping[str, Tensor],
    weight: float | Tensor,
) -> dict[str, Tensor]:
    """``accumulated + gradient * weight`` per tensor, as ``_accumulate_parameter_gradients``.

    ``None`` is the zero state ``_zero_parameter_like`` starts from, added to
    rather than skipped so the first sum rounds as the sequential one does.
    """

    return {
        name: (torch.zeros_like(gradient) if accumulated is None else accumulated[name])
        + gradient * weight
        for name, gradient in gradients.items()
    }


def divide(accumulated: Mapping[str, Tensor], denominator: float | Tensor) -> dict[str, Tensor]:
    """The whole split's gradient: ``_whole_split_gradient``'s ``div_(denominator)``."""

    return {name: value.div(denominator) for name, value in accumulated.items()}


def clip_batch_gradients(gradients: Mapping[str, Tensor], max_norm: float) -> dict[str, Tensor]:
    """``torch.nn.utils.clip_grad_norm_`` on one batch's gradients, out of place.

    Each tensor's 2-norm, the norm of those, ``max_norm / (norm + 1e-6)``
    clamped at 1, and every gradient scaled by it: its single-device path,
    which on the CPU computes each tensor's norm as ``linalg.vector_norm``.
    """

    norms = [torch.linalg.vector_norm(gradient, 2.0) for gradient in gradients.values()]
    total = torch.linalg.vector_norm(torch.stack(norms), 2.0)
    coefficient = torch.clamp(float(max_norm) / (total + 1e-6), max=1.0)
    return {name: gradient.mul(coefficient) for name, gradient in gradients.items()}


def clip_combined(accumulated: Mapping[str, Tensor], max_norm: float) -> dict[str, Tensor]:
    """``_clip_accumulated_update``, without its host round trip.

    The norm is compared, and the scale computed, in float64 as the Python
    floats there are; scaling by 1.0 where it does not clip changes nothing.
    """

    total = torch.sqrt(sum((value**2).sum() for value in accumulated.values()))
    wide = total.to(torch.float64)
    # A true division, as Python's: `float / tensor` is torch's reciprocal
    # times the float, which rounds twice.
    scale = torch.where(
        wide > float(max_norm), torch.div(wide.new_tensor(float(max_norm)), wide + 1e-6), 1.0
    )
    return {name: value.mul(scale.to(value.dtype)) for name, value in accumulated.items()}


def apply_update(
    program: LocalProgram,
    params: Mapping[str, Tensor],
    gradients: Mapping[str, Tensor],
    state: Mapping[str, Any],
    step: int,
    reference: Mapping[str, Tensor] | None = None,
    client_control: Mapping[str, Tensor] | None = None,
    server_control: Mapping[str, Tensor] | None = None,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """One applied update from its gradient: correction, clipping, optimizer step.

    ``step`` counts applied updates from 1. The order is the sequential one:
    FedProx's and SCAFFOLD's optimizer wrappers correct ``.grad`` inside
    ``step()``; ``_ClippingOptimizer`` clips inside it; the combined modes clip
    the combined gradient before stepping on it.
    """

    grads = dict(gradients)
    if program.combine != "batch" and program.max_grad_norm is not None:
        grads = clip_combined(grads, program.max_grad_norm)
    if program.proximal_mu:
        assert reference is not None
        grads = {
            name: gradient.add(params[name] - reference[name], alpha=program.proximal_mu)
            for name, gradient in grads.items()
        }
    if program.scaffold:
        assert client_control is not None and server_control is not None
        grads = {
            name: gradient - client_control[name] + server_control[name]
            for name, gradient in grads.items()
        }
    if program.combine == "batch" and program.max_grad_norm is not None:
        grads = clip_batch_gradients(grads, program.max_grad_norm)
    if program.optimizer.kind == "adamw":
        return _adamw_step(program.optimizer, params, grads, state, step)
    return _sgd_step(program.optimizer, params, grads, state, step)


def _sgd_step(
    spec: OptimizerSpec,
    params: Mapping[str, Tensor],
    gradients: Mapping[str, Tensor],
    state: Mapping[str, Any],
    step: int,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """``torch.optim.SGD``'s ``_single_tensor_sgd``, dampening 0, out of place."""

    new_params: dict[str, Tensor] = {}
    buffers: dict[str, Tensor] = {}
    for name, param in params.items():
        grad = gradients[name]
        if spec.weight_decay != 0:
            grad = grad.add(param, alpha=spec.weight_decay)
        if spec.momentum != 0:
            if step == 1:
                buffer = torch.clone(grad)
            else:
                buffer = state["momentum"][name].mul(spec.momentum).add(grad, alpha=1)
            buffers[name] = buffer
            grad = grad.add(buffer, alpha=spec.momentum) if spec.nesterov else buffer
        new_params[name] = param.add(grad, alpha=-spec.lr)
    return new_params, ({"momentum": buffers} if buffers else {})


def _adamw_step(
    spec: OptimizerSpec,
    params: Mapping[str, Tensor],
    gradients: Mapping[str, Tensor],
    state: Mapping[str, Any],
    step: int,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """``torch.optim.AdamW``'s ``_single_tensor_adamw``, no amsgrad, out of place.

    The step count is the Python float torch reads off its step tensor, so the
    bias corrections are the same Python arithmetic.
    """

    count = float(step)
    bias_correction1 = 1 - spec.beta1**count
    bias_correction2 = 1 - spec.beta2**count
    step_size = spec.lr / bias_correction1
    bias_correction2_sqrt = bias_correction2**0.5
    new_params: dict[str, Tensor] = {}
    exp_avgs: dict[str, Tensor] = {}
    exp_avg_sqs: dict[str, Tensor] = {}
    for name, param in params.items():
        grad = gradients[name]
        param = param.mul(1 - spec.lr * spec.weight_decay)
        exp_avg = state["exp_avg"][name].lerp(grad, 1 - spec.beta1)
        exp_avg_sq = (
            state["exp_avg_sq"][name].mul(spec.beta2).addcmul(grad, grad, value=1 - spec.beta2)
        )
        denominator = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add(spec.eps)
        new_params[name] = param.addcdiv(exp_avg, denominator, value=-step_size)
        exp_avgs[name] = exp_avg
        exp_avg_sqs[name] = exp_avg_sq
    return new_params, {"exp_avg": exp_avgs, "exp_avg_sq": exp_avg_sqs}
