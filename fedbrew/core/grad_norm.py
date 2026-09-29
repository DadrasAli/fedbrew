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
        for (name, _), grad in zip(self.params.items(), grads, strict=True):
            if grad is not None:
                self.sums[name].add_(grad.to(torch.float64), alpha=count)
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
        from fedbrew.core.federated_state import validate_federated_state_metadata

        task = server.task
        if server._model_state is None:
            server.initialize()
        if self._model is None:
            self._model = copy.deepcopy(task.build_model(server.model_config))
        validate_federated_state_metadata(
            task.federated_model_state_metadata(self._model),
            server._model_state_metadata,
            received_scope=server._model_state_scope or "full",
            context="server grad-norm state",
        )
        task.load_federated_model_state(self._model, server._model_state)
        return self._model


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
