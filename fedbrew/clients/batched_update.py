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

A program's numeric values -- the learning rate, momentum, weight decay,
FedProx's mu, the clipping norm -- are not part of what clients must share to
be stepped together, only its shape is (:attr:`LocalProgram.shape`): the step
arithmetic reads them as each client's own tensors (:class:`ProgramValues`),
so clients of different runs with different values, the settings of a group
(``fedbrew/core/settings_group.py``), are rows of one stack. On the CPU each
scalar form torch's optimizers use has a tensor form that rounds the same:
``add(b, alpha=s)`` is one fused multiply-add, as ``addcmul(a, b, s)`` is, and
``addcdiv(b, d, value=s)`` is ``addcdiv(a, b * s, d)``. On CUDA they differ in
the last bit (chapter 11 §9), so there a batched client is the sequential one
to the executor's tolerance.

A rule opts in by declaring ``_batched_rule`` on its own class and
implementing ``batched_plan`` and ``batched_result``
(:class:`fedbrew.clients.torch_sgd_client.TorchSGDClient`). A subclass that
does not declare it inherits nothing batched, so an extension rule built on a
batchable one runs sequentially.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
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


# Not slotted: a compiled step guards on its program, and torch's guards hold a
# weak reference to it, which a slotted class cannot give.
@dataclass(frozen=True)
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


@dataclass(frozen=True)
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

    @property
    def shape(self) -> tuple[Any, ...]:
        """What clients stepped together must share: every branch the step takes, no value.

        Which terms the step has -- momentum, weight decay, FedProx's
        correction, clipping, each present or not -- and everything read as a
        Python number rather than per client (AdamW's betas and eps).
        """

        spec = self.optimizer
        return (
            spec.kind,
            spec.momentum != 0.0,
            spec.weight_decay != 0.0,
            spec.nesterov,
            spec.beta1,
            spec.beta2,
            spec.eps,
            self.combine,
            self.max_grad_norm is not None,
            bool(self.proximal_mu),
            self.scaffold,
            self.weighting,
        )


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
        """What two clients must share to be stepped together: their values need not match."""

        return (self.program.shape, self.structure, self.evaluate)


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
    covered = sum(structure)
    if program.combine == "full" and covered == rows.shape[1]:
        # Every batch is some update's, and its weight is its rows: the rows.
        return rows
    if program.weighting in ("uniform", "sum") and program.combine != "full":
        # 1/count per column of its update's, or 1: the same division, or none.
        counts = torch.tensor(structure, dtype=torch.float64)
        per_step = counts.repeat_interleave(torch.tensor(structure))
        weights[:, :covered] = 1.0 / per_step if program.weighting == "uniform" else 1.0
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


def update_denominators(weights: Tensor, structure: tuple[int, ...]) -> Tensor:
    """Each client's sum of each update's batch weights, ``(clients, updates)``.

    Where every update takes the same number of batches, one sum over the
    weights seen as ``(clients, updates, count)``; a sum reduces each run of
    contiguous weights alike, so it is each update's own ``sum`` of its slice.
    """

    if structure and len(set(structure)) == 1:
        count = structure[0]
        covered = weights[:, : count * len(structure)]
        return covered.reshape(weights.shape[0], len(structure), count).sum(dim=2)
    totals = []
    first = 0
    for count in structure:
        totals.append(weights[:, first : first + count].sum(dim=1))
        first += count
    return torch.stack(totals, dim=1)


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
    _adopt_training(plans, train)
    evaluation = _phase_orders(
        plans,
        round_id,
        "eval",
        [plan.eval_order for plan in plans],
        [LocalLoop(epochs=1)] * len(plans),
        lambda replayed: [[batch] for batch in replayed[1]],
    )
    _adopt_evaluation(plans, evaluation)
    return train, evaluation


def adopt_orders(
    plans: Sequence[ClientBatchPlan], train: RoundOrders, evaluation: RoundOrders
) -> tuple[RoundOrders, RoundOrders]:
    """``plan_round``'s effect on ``plans`` for orders planned elsewhere (``round_planner``).

    ``train`` and ``evaluation`` must be what ``plan_round`` would compute for
    these plans, client ``k`` of each being ``plans[k]``: the slots, structures,
    post-fit row counts and refusal are filled and raised as it does.
    """

    for slot, plan in enumerate(plans):
        plan.slot = slot
    _adopt_training(plans, train)
    _adopt_evaluation(plans, evaluation)
    return train, evaluation


def _adopt_training(plans: Sequence[ClientBatchPlan], train: RoundOrders) -> None:
    """Each plan's structure, and the first client with no training batch refused."""

    steps = train.steps.tolist()
    for plan in plans:
        if not steps[plan.slot]:
            raise plan.refuse()
        plan.structure = train.structure[plan.slot]


def _adopt_evaluation(plans: Sequence[ClientBatchPlan], evaluation: RoundOrders) -> None:
    """Each plan's post-fit row count: the example count its pass reports."""

    rows = evaluation.lengths.sum(dim=1).tolist()
    for plan in plans:
        plan.eval_rows = rows[plan.slot]


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


class ProgramValues:
    """The numeric values of several clients' programs, one tensor per value over the clients.

    ``programs`` share a :attr:`LocalProgram.shape`. Each value is computed as
    the scalar step computes it, in Python floats, and held in the dtype the
    step applies it in -- ``dtype``, the parameters', as torch casts a scalar
    to the tensor it scales -- except the clipping norm the combined modes
    compare in float64. AdamW's step size depends on the step, so it is held
    for each of ``steps`` steps; with ``per_step``, so is its second-moment
    correction, which a compiled step, told only whether it is the first,
    cannot compute from the step's number.
    """

    def __init__(
        self,
        programs: Sequence[LocalProgram],
        steps: int,
        dtype: torch.dtype,
        device: torch.device | str,
        per_step: bool = False,
    ) -> None:
        first = programs[0]

        def column(values: list[float], kind: torch.dtype = dtype) -> Tensor:
            # Cast where it is built, then placed: the cast rounds the same on
            # either device, and the copy does not wait on the device's queue.
            from fedbrew.core.torch_utils import uploaded

            return uploaded(torch.tensor(values, dtype=torch.float64).to(dtype=kind), device)

        specs = [program.optimizer for program in programs]
        self._values: dict[str, Tensor] = {}
        self._step_sizes: Tensor | None = None
        self._corrections: Tensor | None = None
        if first.optimizer.kind == "adamw" and per_step:
            self._corrections = column(
                [
                    [(1 - spec.beta2 ** float(step)) ** 0.5 for spec in specs]
                    for step in range(1, steps + 1)
                ]
            )
        if first.optimizer.kind == "adamw":
            self._values["decay"] = column([1 - spec.lr * spec.weight_decay for spec in specs])
            self._step_sizes = column(
                [
                    [-(spec.lr / (1 - spec.beta1 ** float(step))) for spec in specs]
                    for step in range(1, steps + 1)
                ]
            )
        else:
            self._values["negated_lr"] = column([-spec.lr for spec in specs])
            if first.optimizer.momentum != 0.0:
                self._values["momentum"] = column([spec.momentum for spec in specs])
            if first.optimizer.weight_decay != 0.0:
                self._values["weight_decay"] = column([spec.weight_decay for spec in specs])
        if first.proximal_mu:
            self._values["proximal_mu"] = column([program.proximal_mu for program in programs])
        if first.max_grad_norm is not None:
            norms = [float(program.max_grad_norm) for program in programs]  # type: ignore[arg-type]
            self._values["max_grad_norm"] = column(norms)
            self._values["max_grad_norm_wide"] = column(norms, torch.float64)

    def at(self, step: int) -> dict[str, Tensor]:
        """Every value the ``step``-th applied update reads (from 1), per client."""

        return values_at(self.parts(), step)

    def parts(self) -> tuple[dict[str, Tensor], Tensor | None, Tensor | None]:
        """What ``at`` reads: the values every step shares, and the per-step ones, or None."""

        return self._values, self._step_sizes, self._corrections


def values_at(
    parts: tuple[dict[str, Tensor], Tensor | None, Tensor | None], step: int
) -> dict[str, Tensor]:
    """``ProgramValues.at`` from its ``parts``, as a compiled loop reads it."""

    values, step_sizes, corrections = parts
    if step_sizes is None:
        return values
    at = {**values, "step_size": step_sizes[step - 1]}
    if corrections is not None:
        at["bias_correction2_sqrt"] = corrections[step - 1]
    return at


def _per_client(value: Tensor, like: Tensor) -> Tensor:
    """A client's value, in ``like``'s dtype, shaped to scale it element by element.

    Under vmap ``value`` is one client's scalar; on a stack stepped as a whole
    (``_run_summed``'s unclipped step) it is one value per row of ``like``.
    """

    key = (like.dim(), like.dtype)
    held = _held_shapes(value)
    if held is not None and key in held:
        return held[key]
    shaped = value.to(like.dtype)
    shaped = shaped.reshape(tuple(shaped.shape) + (1,) * (like.dim() - shaped.dim()))
    if held is not None:
        held[key] = shaped
    return shaped


def _held_shapes(value: Tensor) -> dict[Any, Tensor] | None:
    """The shapes ``_per_client`` made of an eager tensor of values, kept on it; None if not kept.

    A program's values are the same tensors every step of a round, and each
    step cast and reshaped them again. Not under vmap or compilation, whose
    tensors are the call's own.
    """

    if torch.compiler.is_compiling() or torch._C._functorch.is_functorch_wrapped_tensor(value):
        return None
    held = value.__dict__.get("_per_client_shapes")
    if held is None:
        held = value.__dict__["_per_client_shapes"] = {}
    return held


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
        + gradient * _stacked(weight, gradient)
        for name, gradient in gradients.items()
    }


def divide(accumulated: Mapping[str, Tensor], denominator: float | Tensor) -> dict[str, Tensor]:
    """The whole split's gradient: ``_whole_split_gradient``'s ``div_(denominator)``."""

    return {name: value.div(_stacked(denominator, value)) for name, value in accumulated.items()}


def _stacked(value: float | Tensor, like: Tensor) -> float | Tensor:
    """A value per client, on a stack stepped as a whole, shaped to scale ``like``'s rows.

    Under vmap -- one client's scalar, logically 0-dim -- or as a Python
    number it is returned as it is. One value per row of a stack is cast to
    ``like``'s dtype first, as vmap's promotion casts a client's scalar to the
    result's, so the stacked product or quotient is the vmapped one.
    """

    if isinstance(value, Tensor) and value.dim() == 1 and like.dim() >= 1:
        return _per_client(value, like)
    return value


def clip_batch_gradients(gradients: Mapping[str, Tensor], max_norm: Tensor) -> dict[str, Tensor]:
    """``torch.nn.utils.clip_grad_norm_`` on one batch's gradients, out of place.

    Each tensor's 2-norm, the norm of those, ``max_norm / (norm + 1e-6)``
    clamped at 1, and every gradient scaled by it: its single-device path,
    which on the CPU computes each tensor's norm as ``linalg.vector_norm``.
    ``max_norm`` is the client's, in the gradients' dtype; its division is
    torch's ``float / tensor``, the reciprocal times the float.
    """

    norms = [torch.linalg.vector_norm(gradient, 2.0) for gradient in gradients.values()]
    total = torch.linalg.vector_norm(torch.stack(norms), 2.0)
    coefficient = torch.clamp(torch.reciprocal(total + 1e-6) * max_norm.to(total.dtype), max=1.0)
    return {name: gradient.mul(coefficient) for name, gradient in gradients.items()}


def clip_combined(accumulated: Mapping[str, Tensor], max_norm: Tensor) -> dict[str, Tensor]:
    """``_clip_accumulated_update``, without its host round trip.

    The norm is compared, and the scale computed, in float64 as the Python
    floats there are; scaling by 1.0 where it does not clip changes nothing.
    ``max_norm`` is the client's, in float64.
    """

    total = torch.sqrt(sum((value**2).sum() for value in accumulated.values()))
    wide = total.to(torch.float64)
    # A true division, as Python's: `float / tensor` is torch's reciprocal
    # times the float, which rounds twice.
    scale = torch.where(wide > max_norm, torch.div(max_norm, wide + 1e-6), 1.0)
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
    *,
    values: Mapping[str, Tensor],
    in_place: bool = False,
    params_owned: bool = False,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """One applied update from its gradient: correction, clipping, optimizer step.

    ``step`` counts applied updates from 1. The order is the sequential one:
    FedProx's and SCAFFOLD's optimizer wrappers correct ``.grad`` inside
    ``step()``; ``_ClippingOptimizer`` clips inside it; the combined modes clip
    the combined gradient before stepping on it. ``program`` gives the
    update's shape, and ``values`` (``ProgramValues.at(step)``) the client's
    numbers.

    ``in_place``: the gradients and the optimizer state are the caller's to
    overwrite, and each operation on them is made in place -- the same
    operation, into the tensor it would have replaced; ``params_owned``: so
    are the parameters. On a stack of many clients a new tensor per operation
    is most of the step's cost on the CPU (a 1000-client stack of the MNIST
    MLP's first layer: 290 against 33 ms for ``g - c_i + c``, measured
    2026-09-30). Under vmap neither is asked for.
    """

    grads = dict(gradients)
    if program.combine != "batch" and program.max_grad_norm is not None:
        grads = clip_combined(grads, values["max_grad_norm_wide"])
    if program.proximal_mu:
        assert reference is not None
        grads = {
            name: (gradient.addcmul_ if in_place else partial(torch.addcmul, gradient))(
                params[name] - reference[name],
                _per_client(values["proximal_mu"], gradient),
            )
            for name, gradient in grads.items()
        }
    if program.scaffold:
        assert client_control is not None and server_control is not None
        grads = {
            name: (
                gradient.sub_(client_control[name]).add_(server_control[name])
                if in_place
                else gradient - client_control[name] + server_control[name]
            )
            for name, gradient in grads.items()
        }
    if program.combine == "batch" and program.max_grad_norm is not None:
        grads = clip_batch_gradients(grads, values["max_grad_norm"])
    if program.optimizer.kind == "adamw":
        return _adamw_step(
            program.optimizer, params, grads, state, step, values, in_place, params_owned
        )
    return _sgd_step(program.optimizer, params, grads, state, step, values, in_place, params_owned)


def _sgd_step(
    spec: OptimizerSpec,
    params: Mapping[str, Tensor],
    gradients: Mapping[str, Tensor],
    state: Mapping[str, Any],
    step: int,
    values: Mapping[str, Tensor],
    in_place: bool = False,
    params_owned: bool = False,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """``torch.optim.SGD``'s ``_single_tensor_sgd``, dampening 0: out of place, or in place.

    Its ``add(b, alpha=s)`` is ``addcmul(a, b, s)`` with the client's ``s``.
    ``in_place`` and ``params_owned`` as ``apply_update``'s: each operation
    the same, written into the tensor it would have replaced.
    """

    new_params: dict[str, Tensor] = {}
    buffers: dict[str, Tensor] = {}
    for name, param in params.items():
        grad = gradients[name]
        if spec.weight_decay != 0:
            decay = _per_client(values["weight_decay"], param)
            grad = grad.addcmul_(param, decay) if in_place else torch.addcmul(grad, param, decay)
        if spec.momentum != 0:
            momentum = _per_client(values["momentum"], grad)
            if step == 1:
                buffer = torch.clone(grad)
            elif in_place:
                buffer = state["momentum"][name].mul_(momentum).add_(grad, alpha=1)
            else:
                buffer = state["momentum"][name].mul(momentum).add(grad, alpha=1)
            buffers[name] = buffer
            if spec.nesterov:
                grad = (
                    grad.addcmul_(buffer, momentum)
                    if in_place
                    else torch.addcmul(grad, buffer, momentum)
                )
            else:
                grad = buffer
        rate = _per_client(values["negated_lr"], param)
        new_params[name] = (
            param.addcmul_(grad, rate) if params_owned else torch.addcmul(param, grad, rate)
        )
    return new_params, ({"momentum": buffers} if buffers else {})


def _adamw_step(
    spec: OptimizerSpec,
    params: Mapping[str, Tensor],
    gradients: Mapping[str, Tensor],
    state: Mapping[str, Any],
    step: int,
    values: Mapping[str, Tensor],
    in_place: bool = False,
    params_owned: bool = False,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """``torch.optim.AdamW``'s ``_single_tensor_adamw``, no amsgrad: out of place, or in place.

    The step count is the Python float torch reads off its step tensor, so the
    bias corrections are the same Python arithmetic. The client's decay
    ``1 - lr * weight_decay`` and step size ``-lr / bias_correction1`` are
    computed so too (``ProgramValues``); ``addcdiv(b, d, value=s)`` is
    ``addcdiv(a, b * s, d)``.
    """

    count = float(step)
    bias_correction2 = 1 - spec.beta2**count
    bias_correction2_sqrt = bias_correction2**0.5
    new_params: dict[str, Tensor] = {}
    exp_avgs: dict[str, Tensor] = {}
    exp_avg_sqs: dict[str, Tensor] = {}
    for name, param in params.items():
        grad = gradients[name]
        decay = _per_client(values["decay"], param)
        param = param.mul_(decay) if params_owned else param.mul(decay)
        if in_place:
            exp_avg = state["exp_avg"][name].lerp_(grad, 1 - spec.beta1)
            exp_avg_sq = (
                state["exp_avg_sq"][name]
                .mul_(spec.beta2)
                .addcmul_(grad, grad, value=1 - spec.beta2)
            )
        else:
            exp_avg = state["exp_avg"][name].lerp(grad, 1 - spec.beta1)
            exp_avg_sq = (
                state["exp_avg_sq"][name].mul(spec.beta2).addcmul(grad, grad, value=1 - spec.beta2)
            )
        correction: Any = bias_correction2_sqrt
        if "bias_correction2_sqrt" in values:
            correction = _per_client(values["bias_correction2_sqrt"], exp_avg_sq)
        denominator = (exp_avg_sq.sqrt() / correction).add(spec.eps)
        scaled = exp_avg * _per_client(values["step_size"], exp_avg)
        new_params[name] = (
            param.addcdiv_(scaled, denominator)
            if params_owned
            else torch.addcdiv(param, scaled, denominator)
        )
        exp_avgs[name] = exp_avg
        exp_avg_sqs[name] = exp_avg_sq
    return new_params, {"exp_avg": exp_avgs, "exp_avg_sq": exp_avg_sqs}
