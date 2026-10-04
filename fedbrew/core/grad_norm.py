"""The gradient norm of the global objective: ``evaluation.grad_norm``.

``grad_norm_sq`` is ``||grad F(x)||^2`` at the global model ``x``, where F is
the task's training loss over every client's train split, each batch's loss
weighted by the count it is a mean over (``TaskAdapter.objective_loss``, the
``train_loss_denominator`` ``update_mode: full_gradient`` weights by). So F is
the pooled mean: examples for classification and the shipped examples, whose
parameter-only terms -- fed-lasso's penalty -- enter every batch once and so
F once; active target tokens for the causal-LM task. The gradient is taken in
the model's trainable parameters, in eval mode.

Where F carries an l1 term (``TaskAdapter.objective_l1``) it is not
differentiable at a zero coordinate, and the column is the squared norm of
F's minimum-norm subgradient instead: at ``x_j != 0`` the gradient, at
``x_j == 0`` the smooth part's gradient soft-thresholded at ``lam``,
``sign(g_j) max(|g_j| - lam, 0)`` (``minimum_norm_gradient``).

Three paths compute it, to summation order:

- ``SequentialGradNorm``, the reference: the server's model, each client's
  train split through the task's own loader, one backward per batch;
- ``flat_chunk_gradient`` for the batched evaluator and the resident round:
  every client's train rows, cut into chunks of consecutive rows across
  clients, one ``functional_loss`` and one backward per chunk, on the device.

None of them draws from a random generator the run reads, or changes the
model the run trains: a run with the column on computes every other column
as it would with it off. ``tests/test_grad_norm.py`` holds all of it.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn

from fedbrew.core.metrics import GRAD_NORM_COLUMN, GRAD_NORM_DIRECTION

__all__ = ["GRAD_NORM_COLUMN", "GRAD_NORM_DIRECTION"]


def minimum_norm_gradient(
    gradient: Mapping[str, Tensor],
    params: Mapping[str, Tensor],
    l1: Mapping[str, float],
) -> dict[str, Tensor]:
    """F's minimum-norm subgradient, from autograd's gradient and F's l1 terms.

    Autograd's subgradient of ``lam |x_j|`` is ``lam sign(x_j)``, 0 at
    ``x_j == 0``: there ``gradient`` is the smooth part's alone, and the
    subdifferential is ``g_j + [-lam, lam]``, whose smallest element is the
    soft-threshold of ``g_j`` at ``lam``. Elsewhere F is differentiable and the
    gradient is kept.
    """

    minimum = dict(gradient)
    for name, level in l1.items():
        if name not in minimum:
            continue
        g = minimum[name]
        at_zero = params[name].detach() == 0
        thresholded = torch.sign(g) * torch.clamp(g.abs() - float(level), min=0.0)
        minimum[name] = torch.where(at_zero, thresholded, g)
    return minimum


def squared_norm(gradient: Mapping[str, Tensor]) -> Tensor:
    """``sum_j g_j^2`` over every tensor, in float64, as a 0-d tensor on the tensors' device."""

    total: Tensor | None = None
    for g in gradient.values():
        value = torch.sum(torch.square(g.to(torch.float64)))
        total = value if total is None else total + value
    if total is None:
        return torch.zeros((), dtype=torch.float64)
    return total


class WeightedGradient:
    """A sum of loss gradients, each weighted by its count, and their mean.

    Summed in float64: F's gradient is a mean over every client's rows, and
    ``grad_norm_sq`` is compared against zero.
    """

    def __init__(self, params: Mapping[str, Tensor]) -> None:
        self.params = dict(params)
        self.sums = {
            name: torch.zeros(param.shape, dtype=torch.float64, device=param.device)
            for name, param in self.params.items()
        }
        self.total = 0.0

    def add(self, loss: Tensor, count: float) -> None:
        """Add ``count`` times the gradient of ``loss``; nothing for a count of zero."""

        if count == 0.0:
            return
        if count < 0.0 or count != count:
            raise ValueError(f"an objective batch reported the count {count!r}")
        grads = torch.autograd.grad(loss, list(self.params.values()), allow_unused=True)
        self.add_gradient(dict(zip(self.params, grads, strict=True)), count)

    def add_gradient(self, gradient: Mapping[str, Tensor | None], count: float) -> None:
        """Add ``count`` times a loss's gradient, already taken; nothing for a count of zero."""

        if count == 0.0:
            return
        if count < 0.0 or count != count:
            raise ValueError(f"an objective batch reported the count {count!r}")
        for name, total in self.sums.items():
            grad = gradient.get(name)
            if grad is not None:
                total.add_(grad.to(torch.float64), alpha=count)
        self.total += count

    def mean(self) -> dict[str, Tensor]:
        if self.total <= 0.0:
            raise ValueError(
                "evaluation.grad_norm: the clients' train splits hold nothing the task's "
                "loss averages over, so F has no gradient"
            )
        return {name: total / self.total for name, total in self.sums.items()}


def grad_norm_sq(
    gradient: WeightedGradient, params: Mapping[str, Tensor], l1: Mapping[str, float]
) -> Tensor:
    """``||g||^2`` of F's mean gradient, or of its minimum-norm subgradient under ``l1``."""

    return squared_norm(minimum_norm_gradient(gradient.mean(), params, l1))


#: The relative bound two computations of a sum are first held to.
RELATIVE_TOLERANCE = 1e-12
#: Units of the terms' precision (``torch.finfo(dtype).eps``) a sum in any
#: order is held to, times the sum of its terms' magnitudes: a small multiple,
#: the same for every task and run (FINDINGS.md, POST-F38).
ROUNDING_UNITS = 8


def gradient_terms(
    task: Any,
    template: nn.Module,
    params: Mapping[str, Tensor],
    buffers: Mapping[str, Tensor],
    rows: tuple[Tensor, ...],
    chunk: int = 4096,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """F's gradient at ``params`` and the magnitudes of the terms it sums, in float64.

    F is the mean over ``rows`` of ``f_i``, the task's ``functional_loss`` of
    row ``i`` alone (a parameter-only term, as fed-lasso's penalty, enters
    every ``f_i`` once and so F once), and its gradient the mean of the
    rows' gradients: returned is that mean and, per coordinate, the mean of
    their magnitudes, ``S_j = (1/n) sum_i |d f_i / d x_j|``. Taken as
    :func:`flat_chunk_gradient` takes its leaves, one row's gradient at a
    time by ``torch.func``, ``chunk`` rows at once.
    """

    parameters = dict(template.named_parameters())
    fixed = {name: value.detach() for name, value in params.items() if name in parameters}
    held = {
        **buffers,
        **{name: value for name, value in params.items() if name not in parameters},
    }
    wanted = {name: fixed[name] for name, param in parameters.items() if param.requires_grad}

    def row_loss(trainable: dict[str, Tensor], *row: Tensor) -> Tensor:
        batch = tuple(tensor.unsqueeze(0) for tensor in row)
        loss, _ = task.functional_loss(template, {**fixed, **trainable}, held, batch, None)
        return loss

    per_row = torch.func.vmap(torch.func.grad(row_loss), in_dims=(None, *([0] * len(rows))))
    sums = {
        name: torch.zeros(value.shape, dtype=torch.float64, device=value.device)
        for name, value in wanted.items()
    }
    magnitudes = {name: torch.zeros_like(total) for name, total in sums.items()}
    count = len(rows[0])
    with measuring(template, next(iter(fixed.values())).device):
        for first in range(0, count, max(1, int(chunk))):
            grads = per_row(wanted, *(tensor[first : first + chunk] for tensor in rows))
            for name, grad in grads.items():
                sums[name] += grad.to(torch.float64).sum(dim=0)
                magnitudes[name] += grad.to(torch.float64).abs().sum(dim=0)
    return (
        {name: total / count for name, total in sums.items()},
        {name: total / count for name, total in magnitudes.items()},
    )


def rounding_tolerance(
    task: Any,
    template: nn.Module,
    params: Mapping[str, Tensor],
    buffers: Mapping[str, Tensor],
    rows: tuple[Tensor, ...],
) -> float:
    """How far two computations of ``grad_norm_sq`` at ``params`` may differ by rounding alone.

    Each coordinate of F's gradient is a sum of the rows' terms
    (:func:`gradient_terms`); summed in any order, it is held within
    ``d_j = ROUNDING_UNITS * eps * S_j`` of the exact sum, ``eps`` the
    precision of the parameter it is the gradient of and ``S_j`` its terms'
    magnitudes. Two such gradients are within ``d`` of one, so their squared
    norms within ``2 sum_j d_j (2 |g_j| + d_j)`` of each other, ``g`` F's
    gradient -- its minimum-norm subgradient where F carries an l1 term,
    whose soft-threshold moves no coordinate further than ``d_j``.
    """

    gradient, magnitudes = gradient_terms(task, template, params, buffers, rows)
    minimum = minimum_norm_gradient(gradient, params, task.objective_l1(template))
    total = 0.0
    for name, g in minimum.items():
        eps = torch.finfo(params[name].dtype).eps
        bound = ROUNDING_UNITS * eps * magnitudes[name]
        total += float(torch.sum(2.0 * bound * (2.0 * g.abs() + bound)))
    return total


def within_rounding(value: float, reference: float, tolerance: float) -> bool:
    """Whether ``value`` is ``reference`` to rounding (FINDINGS.md, POST-F38).

    Within ``RELATIVE_TOLERANCE`` of the larger, or within ``tolerance``: a
    sum's computations in two orders, held to a small multiple of machine
    epsilon times the sum of its terms' magnitudes (:func:`rounding_tolerance`
    for ``grad_norm_sq``), which bounds them where the sum is near its own
    rounding and no relative bound can.
    """

    gap = abs(value - reference)
    return gap <= RELATIVE_TOLERANCE * max(abs(value), abs(reference)) or gap <= tolerance


@contextmanager
def measuring(model: nn.Module, device: torch.device | str | None = None) -> Any:
    """The model in eval mode, and the random generators as they were, for the block.

    Restores the model's mode after it, and every generator a forward pass
    could draw from -- the CPU's, and the device's when it is CUDA -- so the
    measurement consumes nothing the run reads afterwards.
    """

    devices: list[Any] = []
    if device is not None and torch.device(device).type == "cuda":
        devices = [torch.device(device)]
    was_training = model.training
    with torch.random.fork_rng(devices=devices):
        model.eval()
        try:
            with torch.enable_grad():
                yield
        finally:
            model.train(was_training)


def trainable(model: nn.Module) -> dict[str, Tensor]:
    """The model's parameters F's gradient is taken in: the trainable ones."""

    return {name: param for name, param in model.named_parameters() if param.requires_grad}


def _train_split(data: Any) -> Any:
    from fedbrew.clients.torch_sgd_client import _get_train_data

    return _get_train_data(data)


class SequentialGradNorm:
    """The reference pass: the server's model, every client's train split through the task's loader.

    The model is the server's, copied once into one of its own: a task that
    caches its models hands every caller the same instance.
    """

    def __init__(self) -> None:
        self._model: nn.Module | None = None

    def measure(self, server: Any, dataset: Any) -> dict[str, float]:
        task = server.task
        model = self._global_model(server)
        params = trainable(model)
        device = next(iter(params.values())).device if params else None
        config: dict[str, Any] = {"shuffle": False}
        eval_batch_size = getattr(task, "eval_batch_size", None)
        if isinstance(eval_batch_size, int) and eval_batch_size > 0:
            config["batch_size"] = eval_batch_size
        with measuring(model, device):
            gradient = WeightedGradient(params)
            for client_id in dataset.list_clients():
                train = _train_split(dataset.get_client_data(client_id))
                for batch in task.build_dataloader(train, dict(config)):
                    loss, count = task.objective_loss(model, batch)
                    gradient.add(loss, float(count))
            value = grad_norm_sq(gradient, params, task.objective_l1(model))
        return {GRAD_NORM_COLUMN: float(value)}

    def _global_model(self, server: Any) -> nn.Module:
        self._model = server_model(server, self._model)
        return self._model


def server_model(server: Any, model: nn.Module | None) -> nn.Module:
    """``model`` holding the server's state, checked as the server checks it; made the first time.

    A copy of the task's model of its own: a task that caches its models
    hands every caller the same instance.
    """

    from fedbrew.core.federated_state import validate_federated_state_metadata

    task = server.task
    if server._model_state is None:
        server.initialize()
    if model is None:
        model = copy.deepcopy(task.build_model(server.model_config))
    validate_federated_state_metadata(
        task.federated_model_state_metadata(model),
        server._model_state_metadata,
        received_scope=server._model_state_scope or "full",
        context="server grad-norm state",
    )
    task.load_federated_model_state(model, server._model_state)
    return model


def chunk_pieces(lengths: Sequence[int], cap: int) -> list[list[tuple[int, int, int]]]:
    """Every row of ``lengths`` splits, cut into chunks of at most ``cap`` consecutive rows.

    A piece is (split, first row, stop); a chunk's pieces run on across splits,
    so one chunk may hold many clients' rows and one client's rows may span
    two chunks.
    """

    cap = max(1, int(cap))
    chunks: list[list[tuple[int, int, int]]] = []
    current: list[tuple[int, int, int]] = []
    room = cap
    for split, length in enumerate(lengths):
        first = 0
        while first < length:
            take = min(room, length - first)
            current.append((split, first, first + take))
            first += take
            room -= take
            if room == 0:
                chunks.append(current)
                current, room = [], cap
    if current:
        chunks.append(current)
    return chunks


def chunk_rows(task: Any, row_bytes: int, chunk_bytes: int) -> int:
    """How many rows one forward of the gradient takes: ``chunk_bytes`` of rows, at most.

    And at most the task's ``eval_batch_size`` when it has one, which is what
    its evaluation passes hold at once.
    """

    rows = max(1, int(chunk_bytes) // max(1, 2 * int(row_bytes)))
    eval_batch_size = getattr(task, "eval_batch_size", None)
    if isinstance(eval_batch_size, int) and eval_batch_size > 0:
        rows = min(rows, eval_batch_size)
    return rows


def flat_chunk_gradient(
    task: Any,
    template: nn.Module,
    params: Mapping[str, Tensor],
    buffers: Mapping[str, Tensor],
    chunks: Iterable[tuple[Tensor, ...]],
) -> Tensor:
    """``grad_norm_sq`` at ``params`` over ``chunks`` of every client's train rows, on their device.

    Each chunk is a batch of the task's rows (``BatchableTask.split_rows``),
    and ``functional_loss`` its mean loss over them: weighted by its row count
    and summed, that is F. ``params`` are every parameter of ``template``; the
    gradient is taken in the ones ``template`` trains. Returns a 0-d float64
    tensor, which the caller reads when it reads its round's other values.
    ``params`` may carry the model state's buffers too, as a round's mean does.
    """

    parameters = dict(template.named_parameters())
    leaves = {
        name: value.detach().requires_grad_(parameters[name].requires_grad)
        for name, value in params.items()
        if name in parameters
    }
    # A model state carries the buffers beside the parameters.
    held = {**buffers, **{name: value for name, value in params.items() if name not in parameters}}
    wanted = {name: value for name, value in leaves.items() if value.requires_grad}
    device = next(iter(leaves.values())).device
    with measuring(template, device):
        gradient = WeightedGradient(wanted)
        for rows in chunks:
            loss, _ = task.functional_loss(template, leaves, held, rows, None)
            gradient.add(loss, float(len(rows[0])))
        return grad_norm_sq(gradient, wanted, task.objective_l1(template)).detach()


class FusedPass:
    """F and its gradient in one pass: the central pass, its loss differentiated.

    Where the central pass is measured in parts -- ``functional_eval`` over the
    global rows in the batches its loader cuts, then the task's metrics of
    those steps (``CentralPassInParts``, or the classification task's
    ``compute_metrics``) -- on rows that are every client's train rows, its
    loss is F itself: each batch's mean, pooled by rows. Taking that loss's
    gradient as the pass runs gives ``grad_norm_sq`` with the central metrics,
    from one forward and one backward, where the gradient's own pass would
    run a second forward over the same rows. The central metrics are the
    pass's own, bit for bit; the gradient is F's, summed over the central
    batches -- weighted by their rows, as ``flat_chunk_gradient`` weighs its
    chunks -- rather than the gradient pass's chunks.

    Where the task gives F and its gradient in closed form
    (``closed_form_eval``, :class:`~fedbrew.tasks.base.BatchableTask`) and the
    run asks for it (``evaluation.grad_norm.gradient_form: closed_form``, the
    default), each batch is measured by it instead, with no graph and no
    backward: its outputs, loss included, and its gradient, from one
    computation over the batch's prepared rows (``closed_form_batch``), a stack
    of one. The loss and the gradient are then the closed form's arithmetic,
    the same to rounding (FINDINGS.md, POST-F38); every other output is the
    iterate's, as ``functional_eval`` gives it.

    Made by :func:`fused_pass`, which says where it cannot be.
    """

    def __init__(
        self,
        task: Any,
        rows: tuple[Tensor, ...],
        batch_size: int,
        in_parts: bool,
        closed: Any = None,
    ) -> None:
        self.task = task
        self.in_parts = in_parts
        size = max(1, int(batch_size))
        self.batches = [
            tuple(tensor[first : first + size] for tensor in rows)
            for first in range(0, len(rows[0]), size)
        ]
        #: The model the closed form prepares its rows with, or None for autograd's pass.
        self.closed = closed is not None
        if closed is not None:
            from fedbrew.tasks.base import closed_form_batch

            self.batches = [closed_form_batch(task, closed, batch) for batch in self.batches]
        self._model: nn.Module | None = None

    def share_rows(self, stack: tuple[Tensor, ...]) -> bool:
        """Read the closed form's one batch from ``stack`` where it holds those rows: whether so.

        ``stack`` is prepared rows with a leading client dimension and no
        padding, contiguous, as the resident round trains on them: where they
        are this pass's rows -- the same values in the same order, dtype and
        device -- the batch becomes the stack seen as one client, the same
        shape, strides and alignment, and the pass's own copy is let go. One
        copy of the rows is then read each round where two were, and every
        value is the same.
        """

        from fedbrew.core.torch_utils import same_tensors

        if not self.closed or len(self.batches) != 1:
            return False
        (batch,) = self.batches
        if len(stack) != len(batch) or not all(tensor.is_contiguous() for tensor in stack):
            return False
        views = tuple(tensor.reshape(1, -1, *tensor.shape[2:]) for tensor in stack)
        if not same_tensors(views, batch):
            return False
        self.batches = [views]
        return True

    def measure(
        self, template: nn.Module, params: Mapping[str, Tensor], buffers: Mapping[str, Tensor]
    ) -> tuple[list[dict[str, Tensor]], dict[str, Tensor], Tensor]:
        """The central pass's steps and terms at ``params``, and ``grad_norm_sq`` there.

        As ``flat_chunk_gradient`` takes its leaves: every parameter of
        ``template``, the gradient in the ones it trains; ``params`` may
        carry the model state's buffers. Every value stays on the device.
        """

        if self.closed:
            return self._closed_form(template, params, buffers)
        parameters = dict(template.named_parameters())
        leaves = {
            name: value.detach().requires_grad_(parameters[name].requires_grad)
            for name, value in params.items()
            if name in parameters
        }
        held = {
            **buffers,
            **{name: value for name, value in params.items() if name not in parameters},
        }
        wanted = {name: value for name, value in leaves.items() if value.requires_grad}
        device = next(iter(leaves.values())).device
        outputs: list[dict[str, Tensor]] = []
        with measuring(template, device):
            gradient = WeightedGradient(wanted)
            for batch in self.batches:
                measured = self.task.functional_eval(template, leaves, held, batch, None)
                gradient.add(measured["loss"], float(len(batch[0])))
                outputs.append({key: value.detach() for key, value in measured.items()})
            value = grad_norm_sq(gradient, wanted, self.task.objective_l1(template)).detach()
        fixed = {name: leaf.detach() for name, leaf in leaves.items()}
        return outputs, self._terms(template, {**fixed, **held}), value

    def _closed_form(
        self, template: nn.Module, params: Mapping[str, Tensor], buffers: Mapping[str, Tensor]
    ) -> tuple[list[dict[str, Tensor]], dict[str, Tensor], Tensor]:
        """``measure`` from the task's closed form: each batch a stack of one, no graph."""

        parameters = dict(template.named_parameters())
        fixed = {name: value.detach() for name, value in params.items() if name in parameters}
        held = {
            **buffers,
            **{name: value for name, value in params.items() if name not in parameters},
        }
        wanted = {name: fixed[name] for name, value in parameters.items() if value.requires_grad}
        stacked = {name: value.unsqueeze(0) for name, value in fixed.items()}
        device = next(iter(fixed.values())).device
        outputs: list[dict[str, Tensor]] = []
        with measuring(template, device), torch.no_grad():
            gradient = WeightedGradient(wanted)
            for batch in self.batches:
                grads, measured = self.task.closed_form_eval(template, stacked, held, batch, None)
                gradient.add_gradient(
                    {name: grads[name][0] for name in wanted if name in grads},
                    float(batch[0].shape[1]),
                )
                outputs.append({key: value[0] for key, value in measured.items()})
            value = grad_norm_sq(gradient, wanted, self.task.objective_l1(template))
        return outputs, self._terms(template, {**fixed, **held}), value

    def _terms(self, template: nn.Module, params: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """The central pass's terms of the model alone, at ``params``, where it is in parts."""

        if not self.in_parts:
            return {}
        with torch.no_grad():
            return dict(self.task.central_terms(template, params))

    def measure_round(self, server: Any) -> dict[str, float]:
        """The server's model's ``central_test_*`` metrics and ``grad_norm_sq``, as floats.

        The central pass ``evaluate_global`` runs -- the task's own metrics of
        its steps, read as floats, and its terms -- in this pass, on a model of
        this pass's own that the server's state is copied into.
        """

        from fedbrew.core.loop import _central_test_metrics

        model = self._model = server_model(server, self._model)
        state = dict(model.state_dict())
        buffers = dict(model.named_buffers())
        outputs, terms, value = self.measure(model, state, buffers)
        floats = [{key: float(item) for key, item in output.items()} for output in outputs]
        if self.in_parts:
            metrics = self.task.central_metrics(
                floats, {name: float(item) for name, item in terms.items()}
            )
        else:
            metrics = self.task.compute_metrics(floats)
        central = _central_test_metrics({f"global_{name}": item for name, item in metrics.items()})
        return {**central, GRAD_NORM_COLUMN: float(value)}


def fused_pass(
    evaluation: Any, server: Any, dataset: Any
) -> tuple[FusedPass | None, dict[str, Any]]:
    """The run's fused pass and the record of it, or None and why each round takes two.

    The record is None where the run measures no ``grad_norm_sq``. A pass is
    made where the config asks for it (``evaluation.grad_norm.fused``), the
    central pass is measured too, the server's ``evaluate_global`` is the
    task's ``evaluate_model`` of the global rows (``central_pass_is_the_tasks``),
    and that is a pass in parts (``CentralPassInParts``, whose loader neither
    shuffles nor draws nor drops a batch, or the classification task's), the
    global rows are every client's train rows (the same rows, in any order),
    and the task's ``functional_eval`` gives its loss with its graph -- or the
    pass takes the closed form (``evaluation.grad_norm.gradient_form``), which
    the record names beside the pass (``gradient``).
    """

    from fedbrew.core.config import parse_evaluation_schedule

    if parse_evaluation_schedule(evaluation.grad_norm.every, "evaluation.grad_norm") is None:
        return None, {}
    asked = "fused" if evaluation.grad_norm.fused else "separate"
    reason = _unfused(evaluation, server, dataset) if asked == "fused" else "asked for"
    if isinstance(reason, FusedPass):
        return reason, {"pass": "fused", "asked": asked, **_gradient_record(evaluation, reason)}
    return None, {"pass": "separate", "asked": asked, "reason": reason}


def _gradient_record(evaluation: Any, fused: FusedPass) -> dict[str, str]:
    """Which form the fused pass takes F and its gradient in, and why not the closed form."""

    if fused.closed:
        return {"gradient": "closed_form"}
    if evaluation.grad_norm.gradient_form != "closed_form":
        return {"gradient": "autograd", "gradient_reason": "asked for"}
    return {
        "gradient": "autograd",
        "gradient_reason": "the task gives no closed form of F and its gradient (closed_form_eval)",
    }


def _unfused(evaluation: Any, server: Any, dataset: Any) -> FusedPass | str:
    """The fused pass, or why there is none."""

    from fedbrew.core.config import parse_evaluation_schedule
    from fedbrew.servers.fedavg import central_pass_is_the_tasks
    from fedbrew.tasks.base import BatchableTask, CentralPassInParts
    from fedbrew.tasks.classification.torch_classification import TorchClassificationTask

    if parse_evaluation_schedule(evaluation.central_test.every, "evaluation.central_test") is None:
        return "the central pass is not measured"
    task = getattr(server, "task", None)
    if not isinstance(task, BatchableTask):
        return "the task measures no batch of its own (BatchableTask)"
    if not central_pass_is_the_tasks(server):
        return "the server's central pass is its own"
    classification = type(task).evaluate_model is TorchClassificationTask.evaluate_model
    in_parts = isinstance(task, CentralPassInParts)
    if not (classification or in_parts):
        return "the task's central pass is not measured in parts (CentralPassInParts)"
    try:
        data = dataset.get_global_data()
    except (FileNotFoundError, KeyError):
        data = None
    if data is None:
        return "the dataset has no global rows"
    rows = tuple(task.split_rows(data))
    batch_size = _central_batch_size(task, data, classification)
    if isinstance(batch_size, str):
        return batch_size
    train = [
        task.split_rows(_train_split(dataset.get_client_data(client)))
        for client in dataset.list_clients()
    ]
    if not _same_rows(rows, train):
        return "the central pass's rows are not every client's train rows"
    closed = _closed_form_model(evaluation, task, server)
    fused = FusedPass(task, rows, batch_size, in_parts, closed=closed)
    if closed is None and not _loss_keeps_its_graph(fused, server):
        return "the task's functional_eval gives its loss without its graph"
    return fused


def _closed_form_model(evaluation: Any, task: Any, server: Any) -> nn.Module | None:
    """The model a closed-form fused pass prepares its rows with, or None for autograd's.

    Where the config asks for the closed form (``gradient_form: closed_form``,
    the default) and the task gives one of F and its gradient (``closed_form_eval``).
    """

    if evaluation.grad_norm.gradient_form != "closed_form":
        return None
    if not callable(getattr(task, "closed_form_eval", None)):
        return None
    return task.build_model(server.model_config)


def _central_batch_size(task: Any, data: Any, classification: bool) -> int | str:
    """The rows each batch of the central pass holds, or why its loader cannot be cut so."""

    if classification:
        return int(task.eval_batch_size)
    order = task.loader_order(data, task.central_loader_config())
    if order.shuffle or order.replacement or order.drop_last:
        return "the central pass's loader shuffles, draws or drops a batch"
    return int(order.batch_size)


def _same_rows(rows: tuple[Tensor, ...], parts: Sequence[tuple[Tensor, ...]]) -> bool:
    """Whether ``rows`` are the rows of ``parts`` together: the same rows, in any order."""

    if not parts or any(len(part) != len(rows) for part in parts):
        return False
    pooled = tuple(torch.cat([part[index] for part in parts]) for index in range(len(rows)))
    if any(a.shape != b.shape or a.device != b.device for a, b in zip(pooled, rows, strict=True)):
        return False
    if all(torch.equal(a, b) for a, b in zip(pooled, rows, strict=True)):
        return True

    def keyed(tensors: tuple[Tensor, ...]) -> Tensor:
        return torch.cat([t.reshape(len(t), -1).to(torch.float64) for t in tensors], dim=1)

    ours, theirs = (
        torch.unique(keyed(tensors), dim=0, return_counts=True) for tensors in (rows, pooled)
    )
    return all(torch.equal(a, b) for a, b in zip(ours, theirs, strict=True))


def _loss_keeps_its_graph(fused: FusedPass, server: Any) -> bool:
    """Whether the task's ``functional_eval`` loss carries the graph a gradient is taken through."""

    task = fused.task
    model = task.build_model(server.model_config)
    parameters = dict(model.named_parameters())
    leaves = {
        name: value.detach().clone().requires_grad_(value.requires_grad)
        for name, value in parameters.items()
    }
    if not any(leaf.requires_grad for leaf in leaves.values()):
        return False
    batch = tuple(tensor[:1] for tensor in fused.batches[0])
    with measuring(model, next(iter(leaves.values())).device):
        loss = task.functional_eval(model, leaves, dict(model.named_buffers()), batch, None)["loss"]
    return bool(loss.requires_grad)
