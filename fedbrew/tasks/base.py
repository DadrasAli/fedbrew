"""Abstract task adapter contract for benchmark workloads."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, cast, runtime_checkable

import torch
from torch import Tensor, nn

from fedbrew.core.federated_state import model_state_size
from fedbrew.core.torch_utils import OptimizerLike, get_model_state, load_model_state
from fedbrew.models.group_norm import group_norm_reductions


def model_config_key(model_config: Mapping[str, Any]) -> str:
    """Return a stable cache key for a resolved model configuration.

    Shared by every task that caches models under reuse_model, so two tasks
    cannot disagree about when two configs describe the same architecture.
    """

    return json.dumps(dict(model_config), sort_keys=True, default=repr)


@runtime_checkable
class SupportsDatasetEvaluation(Protocol):
    """A task that can score a model over a whole dataset in one call.

    Optional, and optional on purpose. `eval_step` is per batch and every task
    must have it; `evaluate_model` scores a dataset the caller already holds,
    which only the central test set needs. Twelve task doubles in `tests/`
    implement the five abstract methods and stop there, so making this a sixth
    would break them all to state something two servers already treat as
    optional.

    What was missing is the statement. `TaskAdapter` declared nothing, and
    three callers read the absence three ways: `FedAvgServer.evaluate_global`
    and `ScaffoldServer.evaluate_global` looked it up with
    `getattr(self.task, "evaluate_model", None)` and returned no metrics when
    it was not there, `fedbrew/cli/eval_base_model.py` called it outright, and
    `docs/12` §3.5 listed the five methods a new task implements without
    mentioning it -- so a task written from the chapter loses every
    `central_test_*` column and nothing says why.

    Both registered tasks implement it and so does `examples/pl-1d`. This
    protocol is what the three callers now check, so the optionality is a
    claim rather than a string lookup.
    """

    def evaluate_model(self, model: Any, data: Any) -> dict[str, float]:
        """Score `model` over the whole of `data`."""


@runtime_checkable
class BatchableTask(Protocol):
    """A task the batched executor can train and measure many clients of at once.

    Optional, as :class:`SupportsDatasetEvaluation` is: a task without these
    four methods runs sequentially, and ``runtime.performance.executor:
    batched`` falls back to that with a notice (chapter 11 §9).

    The executor holds a split as its rows, gathers each batch as those rows at
    the batch's indices, and runs ``functional_loss`` and ``functional_eval``
    through ``torch.func`` over a stack of clients' parameters. What the task
    declares by implementing them:

    - a batch of ``build_dataloader`` is exactly ``split_rows`` at the indices
      ``row_batches`` yields for it, in the same order and after the same
      random draws, and holds as many examples as it holds rows, so
      ``evaluation_total`` and the example count are its row count;
    - ``functional_loss`` and ``functional_eval`` compute what ``train_step``
      and ``eval_step`` compute, with the same operations, from ``params``
      rather than from the model's own tensors: they draw nothing from a
      random generator, move nothing to the host, and change none of their
      inputs, so ``torch.func.vmap`` can run them over a leading client
      dimension and, without one, they are ``train_step``'s and
      ``eval_step``'s arithmetic bit for bit;
    - the training loss is a mean over the batch's rows, and every trainable
      parameter enters it.

    ``mask`` is None for a batch whose rows are all real. For a batch padded to
    a longer one it holds 1.0 for each real row and 0.0 for each padded row,
    and the result is the unpadded batch's, up to summation order.

    One more is optional:
    ``closed_form_gradient(model, params, buffers, batch, mask, outputs=True)``
    gives the gradient of ``functional_loss`` in closed form for a whole stack
    at once -- ``params``, ``batch`` and ``mask`` each with a leading client
    dimension -- as ``(gradients, outputs)``: the gradients stacked like
    ``params``, and the outputs ``functional_loss`` returns, each stacked over
    the clients. With ``outputs=False`` -- how a batched training step asks,
    which reads no output but ``total`` -- only ``total`` is returned, where
    the task has one, and the gradients are the same tensors. It is the same
    gradient to rounding: an ``l1`` term takes ``lam * sign(x)``, 0 at exactly
    0, as autograd does. ``stacked_row_weights`` and ``stacked_row_mean`` are
    ``row_mean``'s derivative and value per client. Two optional aids make
    it cheaper a step, each a claim that it changes no bit:
    ``closed_form_rows(model, rows)`` gives, from a split's rows as
    ``split_rows`` gives them -- one split's, or a stack's with a leading
    client dimension -- the rows ``closed_form_gradient`` reads instead,
    computed row by row, so that a batch of them is the same rows computed
    from the batch; a stack's are computed once and gathered every step
    (fed-logistic-l1's signed rows ``-b a``). A task that declares it is
    always handed batches of them. And a ``workspace`` keyword -- a dict kept
    for one stack's steps, whose tensors the form may write in place
    (``scratch``) -- lets it take no new memory a step; the gradients it
    returns may be workspace tensors, read before the next call, and its
    outputs may not. A task that gives it
    trains on it by default, under either executor
    (``runtime.performance.gradient_form``; ``autograd`` asks for autograd).
    The sequential executor then takes each step as :func:`closed_form_train_step`
    does, so such a task's ``train_step`` must be what that function computes
    with autograd: zero the gradients, backpropagate ``functional_loss`` at the
    model's own parameters on the batch as the loader yields it -- a tuple of
    ``split_rows``' tensors, moved to the model's device -- step the optimizer,
    and return ``functional_loss``'s outputs as floats.

    And ``stacked_eval(model, params, buffers, batch, mask)``, optional
    too: ``functional_eval`` for a whole stack at once -- ``params``, ``batch``
    and ``mask`` each with a leading client dimension -- every output stacked
    over the clients, the tensors ``torch.func.vmap(functional_eval)`` gives,
    bit for bit. A stack measured with each client's own parameters (the
    post-fit pass) takes it in place of vmap, whose cost a call -- and the
    torch._dynamo import of its first -- it saves.

    Two more are optional. ``loader_order(data, config)`` declares what the
    loader yields (:class:`LoaderOrder`), so a round's orders are planned
    together rather than replayed per client. ``stacked_metrics(outputs,
    counts)`` folds many splits' ``functional_eval`` outputs -- per position,
    a tensor per key over the splits, of which split ``k`` has ``counts[k]``
    -- into each split's ``compute_metrics`` and example count, as tensors,
    so a stack's metrics are computed on its device; its padding positions
    hold empty batches' outputs and must be ignored. And a class attribute,
    ``batched_gradient``: ``"vmap_grad"`` (the default) takes a stack's
    gradients as ``vmap(grad(functional_loss))``, ``"summed"`` as one vmapped
    forward and one backward through the per-client losses' sum. Both give
    each client its own gradient; which is faster depends on the model, so a
    task declares the one it measured.
    """

    def split_rows(self, data: Any) -> tuple[Tensor, ...]:
        """A split's rows as the loader yields them: on the task's device, in its dtypes."""

    def row_batches(self, rows: int, config: Mapping[str, Any]) -> Iterable[Tensor]:
        """What ``build_dataloader(data, config)`` yields for a split of ``rows`` rows, as indices.

        Re-iterable as the loader is, and each iteration draws what the
        loader's draws, from the same generator.
        """

    def functional_loss(
        self,
        model: nn.Module,
        params: Mapping[str, Tensor],
        buffers: Mapping[str, Tensor],
        batch: tuple[Tensor, ...],
        mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """The loss ``train_step`` backpropagates, and what it returns, as tensors."""

    def functional_eval(
        self,
        model: nn.Module,
        params: Mapping[str, Tensor],
        buffers: Mapping[str, Tensor],
        batch: tuple[Tensor, ...],
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """What ``eval_step`` returns for the batch, key for key, as tensors."""


@runtime_checkable
class CentralPassInParts(Protocol):
    """A task whose central pass is made of parts a resident round can measure on its device.

    Optional, beside :class:`BatchableTask`, whose ``functional_eval`` and
    ``loader_order`` it reads. Its ``evaluate_model(model, data)`` is
    :func:`evaluate_in_parts`: ``eval_step`` over ``data`` as
    ``build_dataloader(data, central_loader_config())`` yields it, the
    ``central_terms`` of the model alone, and ``central_metrics`` of the
    steps' outputs and the terms, every value read as a float. A resident
    round measures the same parts where it trains -- the steps with
    ``functional_eval`` on the global rows cut as ``loader_order`` declares,
    the terms at the round's model -- reads them back with its other values,
    and calls the same ``central_metrics`` at the flush: the same operations
    on the same numbers, so the same floats. Only a loader that neither
    shuffles nor drops a batch is measured there; any other runs as
    ``evaluate_model``, at the flush.
    """

    def central_loader_config(self) -> Mapping[str, Any] | None:
        """The loader config ``evaluate_model`` cuts the global rows with."""

    def central_terms(
        self, model: nn.Module, params: Mapping[str, Tensor] | None
    ) -> dict[str, Tensor]:
        """The pass's values of the model alone, as 0-d tensors: at ``params``, or its own."""

    def central_metrics(
        self, outputs: Sequence[Mapping[str, float]], terms: Mapping[str, float]
    ) -> dict[str, float]:
        """What ``evaluate_model`` returns, from the steps' ``eval_step`` outputs and the terms."""


def evaluate_in_parts(task: Any, model: Any, data: Any) -> dict[str, float]:
    """``evaluate_model`` of a :class:`CentralPassInParts` task, from its parts, on the host."""

    loader = task.build_dataloader(data, task.central_loader_config())
    outputs = [task.eval_step(model, batch) for batch in loader]
    with torch.no_grad():
        terms = {name: float(value) for name, value in task.central_terms(model, None).items()}
    return task.central_metrics(outputs, terms)


@dataclass(frozen=True, slots=True)
class LoaderOrder:
    """What a task's loader yields for a split, declared so it can be computed without it.

    Optional beside :class:`BatchableTask`: a task that returns one from
    ``loader_order(data, config)`` has every client's batch order for a round
    computed together (``fedbrew/clients/batch_orders.py``); one that does not
    has each client's loader replayed through ``row_batches``. A declaration
    is a claim about ``build_dataloader(data, config)``, and
    ``tests/test_batch_orders.py`` holds each shipped task to it:

    - it yields ``rows`` rows in batches of ``batch_size``, in order when
      ``shuffle`` is off, the last batch short unless ``drop_last`` drops it
      -- which a ``keep_single_batch`` loader does only when the split makes
      more than one batch;
    - shuffled, it permutes the rows with ``torch.randperm`` on a
      ``torch.Generator`` seeded with ``seed``: once per epoch after one int64
      base-seed draw, and once more after the last batch, when ``per_epoch``
      (torch's ``DataLoader``); otherwise once, when it is built, and every
      iteration yields that order;
    - with ``replacement`` (and ``shuffle``) it is an iid oracle instead:
      every iteration yields one batch of ``batch_size`` rows drawn uniformly
      with replacement, ``torch.randint(rows, (batch_size,))`` on a
      ``torch.Generator`` seeded with ``seed`` once, when it is built. The
      batch may hold more rows than the split.
    """

    rows: int
    batch_size: int
    shuffle: bool
    drop_last: bool
    seed: int | None
    per_epoch: bool
    keep_single_batch: bool = False
    replacement: bool = False


def listed_loader_order(rows: int, config: Mapping[str, Any] | bool | None) -> LoaderOrder:
    """The order of a loader that permutes once and cuts a list: the linear examples'.

    ``batch_size`` defaults to the whole split, ``shuffle`` and ``drop_last``
    to off, and a bool config is ``shuffle``.
    """

    values = {"shuffle": config} if isinstance(config, bool) else dict(config or {})
    seed = values.get("seed")
    return LoaderOrder(
        rows=rows,
        batch_size=max(1, int(values.get("batch_size", rows) or rows)),
        shuffle=bool(values.get("shuffle", False)),
        drop_last=bool(values.get("drop_last", False)),
        seed=None if seed is None else int(seed),
        per_epoch=False,
        keep_single_batch=True,
    )


def row_mean(values: Tensor, mask: Tensor | None = None) -> Tensor:
    """The mean of ``values`` over a batch's rows, or over its real rows under ``mask``.

    ``values`` holds one number per row. Without a mask this is ``values.mean()``
    itself, so a task's loss written with it is its ``train_step`` loss bit for
    bit; with one (:class:`BatchableTask`), padded rows count for nothing.
    """

    if mask is None:
        return values.mean()
    return (values * mask).sum() / mask.sum()


def stacked_row_weights(values: Tensor, mask: Tensor | None = None) -> Tensor:
    """Each row's weight in its client's ``row_mean``, for a stack of ``(clients, rows, ...)``.

    ``row_mean``'s derivative with respect to each row's value: ``1 / rows``,
    or under ``mask`` the row's mask over its client's real rows, ``(clients,
    rows)``. In ``values``' dtype and device; only their shape is read.
    """

    if mask is None:
        rows = values.shape[1]
        return torch.full(values.shape[:2], 1.0 / rows, dtype=values.dtype, device=values.device)
    mask = mask.to(values.dtype)
    return mask / mask.sum(dim=1, keepdim=True)


def stacked_row_mean(values: Tensor, mask: Tensor | None = None) -> Tensor:
    """``row_mean`` of each client's rows, for a stack of ``(clients, rows)``: ``(clients,)``."""

    if mask is None:
        return values.mean(dim=1)
    mask = mask.to(values.dtype)
    return (values * mask).sum(dim=1) / mask.sum(dim=1)


def row_count(rows: Tensor, mask: Tensor | None = None) -> Tensor:
    """How many real rows a batch holds, as a float64 tensor: its ``total``."""

    if mask is None:
        # Filled on the device: ``torch.tensor(..., device=)`` would copy the
        # number from the host, which waits for the device's queue and cannot
        # be recorded in a CUDA graph (fedbrew/core/resident_graphs.py).
        return torch.full((), float(len(rows)), dtype=torch.float64, device=rows.device)
    return mask.sum().to(torch.float64)


def row_numbers(rows: int) -> Tensor:
    """``0 .. rows - 1`` as float64: a split whose rows are their own numbers.

    Handed to a task's ``build_dataloader`` in place of the rows, it makes the
    loader name the rows each batch holds: its shuffle, batching and
    ``drop_last`` are the loader's own. Exact below 2**53 rows.
    """

    return torch.arange(rows, dtype=torch.float64)


def batch_row_numbers(numbered: Tensor) -> Tensor:
    """The row numbers a batch of :func:`row_numbers` holds, as a CPU index tensor."""

    return numbered.reshape(len(numbered), -1)[:, 0].to(device="cpu", dtype=torch.long)


def batch_example_count(batch: Any) -> float:
    """How many examples a batch holds: the leading dimension of its inputs.

    Reads a mapping's ``X``, ``x`` or ``input_ids``, or a sequence's first
    element, and falls back to 1.0 when neither has a length.
    """

    if isinstance(batch, Mapping):
        features = batch.get("X", batch.get("x", batch.get("input_ids")))
        if hasattr(features, "shape") and len(features.shape) > 0:
            return float(features.shape[0])
        if hasattr(features, "__len__"):
            return float(len(features))

    if isinstance(batch, tuple | list) and batch:
        first = batch[0]
        if hasattr(first, "shape") and len(first.shape) > 0:
            return float(first.shape[0])
        if hasattr(first, "__len__"):
            return float(len(first))

    return 1.0


#: Which side of a task metric is better: smaller, larger, or neither -- a
#: diagnostic read against a target (a support size against the true one) or
#: describing where the iterate is rather than how good it is.
METRIC_DIRECTIONS = frozenset({"min", "max", "none"})


@dataclass(frozen=True, slots=True)
class ReportedMetrics:
    """The declared metrics one run reports, on its clients' passes and on its central pass.

    For a task whose columns depend on the run -- on the problem its model
    block poses, or on what its data carries -- or whose central pass measures
    what a client's cannot: a function of the resolved config returning this is
    registered beside ``METRICS`` (``registry.tasks.register(..., reported=...)``),
    and the plan header lists ``fit_<name>`` and the client splits' aggregates
    of ``client``, and ``central_test_<name>`` of ``central``. Both are names
    from ``METRICS``. A task that registers no such function reports every
    declared name on both.
    """

    client: tuple[str, ...]
    central: tuple[str, ...]


class TaskAdapter(ABC):
    """Task-specific bridge used by generic FL orchestration code."""

    #: What ``compute_metrics`` reports, each name with the direction that is
    #: better: ``"min"``, ``"max"``, or ``"none"`` for neither. Registered with the task
    #: (``registry.tasks.register(..., metrics=...)``), and read there, without
    #: building the task, by the plan header: a run writes ``fit_<name>`` and
    #: ``central_test_<name>`` for each, and the client splits' aggregates of the
    #: ones among ``loss`` and ``accuracy`` (the client evaluation path keeps
    #: those two). None for a task that declares nothing, which the header then
    #: takes to report loss and accuracy.
    METRICS: ClassVar[Mapping[str, str] | None] = None

    #: What each name in ``METRICS`` measures, as a noun phrase that reads in
    #: the middle of a sentence ("cross-entropy", "the client objective
    #: ½xᵀAx − b_iᵀx"). Registered beside ``METRICS``
    #: (``registry.tasks.register(..., glosses=...)``) and read by the plan
    #: header's column glosses, which described every ``loss`` as a
    #: cross-entropy before a task could say what its own is. A name left out
    #: is glossed by its own words.
    METRIC_GLOSSES: ClassVar[Mapping[str, str] | None] = None

    #: Whether the run trains this task on its closed-form gradient
    #: (``closed_form_gradient``, :class:`BatchableTask`): set by the run when
    #: it chooses its gradient form, read by :func:`take_train_step`.
    closed_form_steps: bool = False

    #: What ``grad_norm_sq`` measures for this task (``evaluation.grad_norm``),
    #: as a noun phrase: the squared norm of the gradient of the task's global
    #: objective F at the global model, and what F is. Registered beside
    #: ``METRICS`` (``registry.tasks.register(..., grad_norm=...)``). None for a
    #: task that cannot take F's gradient, which refuses the key at load. A task
    #: that sets it implements ``objective_loss``, and ``objective_l1`` when F
    #: carries an l1 term.
    GRAD_NORM_GLOSS: ClassVar[str | None] = None

    @abstractmethod
    def build_model(self, config: Mapping[str, Any]) -> Any:
        """Build a task-specific model object."""

    @abstractmethod
    def build_dataloader(self, data: Any, config: Mapping[str, Any]) -> Any:
        """Build a task-specific dataloader object."""

    @abstractmethod
    def train_step(
        self,
        model: Any,
        batch: Any,
        optimizer: OptimizerLike | None = None,
    ) -> dict[str, float]:
        """Run one task-specific training step with an optional optimizer."""

    @abstractmethod
    def eval_step(self, model: Any, batch: Any) -> dict[str, float]:
        """Run one task-specific evaluation step."""

    @abstractmethod
    def compute_metrics(self, outputs: Sequence[Any]) -> dict[str, float]:
        """Compute task-specific metrics from step outputs."""

    def get_federated_model_state(self, model: nn.Module) -> dict[str, Any]:
        """Extract the complete model state used by legacy/full-state workloads."""

        return get_model_state(model)

    def load_federated_model_state(
        self,
        model: nn.Module,
        state: Mapping[str, Any],
    ) -> None:
        """Load the complete model state used by legacy/full-state workloads."""

        load_model_state(model, state)

    def federated_model_state_metadata(self, model: nn.Module) -> dict[str, Any]:
        """Describe the default complete-state communication contract."""

        state = self.get_federated_model_state(model)
        communicated_parameters, communicated_bytes = model_state_size(state)
        metadata: dict[str, Any] = {
            "model_state_scope": "full",
            "total_parameters": sum(int(parameter.numel()) for parameter in model.parameters()),
            "trainable_parameters": sum(
                int(parameter.numel())
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "communicated_parameters": communicated_parameters,
            "communicated_bytes": communicated_bytes,
        }
        # Present only when the answer differs from the question, so run.json
        # stays quiet for a model whose widths all divide model.group_norm_groups
        # and says so for one whose do not. Measured off the built model rather
        # than re-derived from the config, which is the only way it cannot drift
        # from the layers that exist. P10-F32.
        reductions = group_norm_reductions(model)
        if reductions:
            metadata["group_norm_reductions"] = reductions
        return metadata

    def evaluation_total(self, batch: Any) -> float | None:
        """The "total" eval_step reports for ``batch``, read off the batch; None if it cannot be.

        A client whose post-fit pass is skipped (evaluation.fit.every) still
        needs the pass's example count, its aggregation weight. A task whose
        count depends on the batch alone returns it here, and the forward pass
        is saved; None, the default, has eval_step run for the batch, so the
        count is the same either way.
        """

        del batch
        return None

    def federated_aggregation_weight(
        self,
        training_outputs: Sequence[Mapping[str, float]],
        evaluated_num_examples: int,
    ) -> int:
        """Return a client's aggregation weight without changing legacy semantics."""

        del training_outputs
        return int(evaluated_num_examples)

    def train_loss_denominator(self, batch: Any, output: Mapping[str, float]) -> float:
        """The count `train_step`'s loss on `batch` is a mean over.

        `update_mode: full_gradient` combines batch gradients into the
        gradient of one batch holding the whole split, and that is exact only
        when each batch's gradient is weighted by its share of this count. The
        default is the batch's example count, which is right for a loss that
        averages over examples: classification's cross-entropy and every
        shipped example's objective, whose parameter-only terms enter every
        batch once and so once in the combination too. A task whose loss
        averages over something else overrides this: the causal-LM task's is a
        mean over active target tokens. Weighting by examples there is 53.5%
        from the gradient of the pass on two batches of 2 and 8 tokens
        (FINDINGS.csv POST-F19).

        Args:
            batch: The batch `train_step` was handed.
            output: What `train_step` returned for it.
        """

        del output
        return batch_example_count(batch)

    def objective_loss(self, model: Any, batch: Any) -> tuple[Tensor, float]:
        """The training loss on ``batch`` at the model's parameters, and what it is a mean over.

        ``evaluation.grad_norm`` sums these over every client's train split,
        each loss weighted by its count, and differentiates: F is that
        weighted mean, so its gradient is exact whatever the batches. The
        loss is ``train_step``'s, as a tensor autograd can differentiate --
        its parameter terms included, once per batch -- taken as the model
        is: the caller has put it in eval mode, and nothing here may draw
        from a random generator or step an optimizer. The count is
        ``train_loss_denominator``'s.

        Only a task that declares ``GRAD_NORM_GLOSS`` is asked.
        """

        del model, batch
        raise NotImplementedError(
            f"{type(self).__name__} declares no gradient of its objective (GRAD_NORM_GLOSS)"
        )

    def objective_l1(self, model: Any) -> Mapping[str, float]:
        """The l1 terms of F: each parameter name ``lam * ||p||_1`` is on, with its ``lam``.

        Empty for a smooth F. Where F has one, ``grad_norm_sq`` is the squared
        norm of F's minimum-norm subgradient: autograd's subgradient of
        ``|p_j|`` is 0 at ``p_j == 0``, so there the smooth gradient is
        soft-thresholded at ``lam`` instead (``fedbrew/core/grad_norm.py``).
        """

        del model
        return {}


def loss_averages_over_examples(task: TaskAdapter | type[TaskAdapter]) -> bool:
    """Whether a task keeps the default `train_loss_denominator`, the example count.

    Read off the class, not off a batch, so the answer cannot depend on what a
    batch happens to hold: a causal-LM batch whose active tokens equal its
    sequence count would otherwise pass one check and fail the next.
    """

    cls = task if isinstance(task, type) else type(task)
    return cls.train_loss_denominator is TaskAdapter.train_loss_denominator


def take_train_step(task: Any, model: Any, batch: Any, optimizer: Any) -> dict[str, float]:
    """One local training step of ``task``: its own ``train_step``, or its closed form's.

    Every rule's local loop takes its steps through here. A run whose gradient
    form is the task's closed form (``closed_form_steps``) steps on
    ``closed_form_gradient``; any other takes the task's ``train_step``.
    """

    if getattr(task, "closed_form_steps", False) and optimizer is not None:
        return closed_form_train_step(task, model, batch, optimizer)
    return cast(dict[str, float], task.train_step(model, batch, optimizer))


def closed_form_train_step(task: Any, model: Any, batch: Any, optimizer: Any) -> dict[str, float]:
    """``train_step`` with the gradient taken from ``closed_form_gradient`` (BatchableTask).

    The batch is the loader's tuple of tensors, moved to the model's device,
    and the model's parameters a stack of one client. Each parameter's
    ``grad`` is set to its closed-form gradient where ``loss.backward()`` would
    have put autograd's, and the optimizer steps on it; what comes back is
    ``functional_loss``'s outputs, as ``train_step`` returns them.
    """

    model.train()
    optimizer.zero_grad(set_to_none=True)
    parameters = dict(model.named_parameters())
    device = next(iter(parameters.values())).device
    with torch.no_grad():
        grads, outputs = task.closed_form_gradient(
            model,
            {name: parameter.detach().unsqueeze(0) for name, parameter in parameters.items()},
            dict(model.named_buffers()),
            closed_form_batch(task, model, tuple(tensor.to(device) for tensor in batch)),
            None,
            outputs=True,
        )
    for name, parameter in parameters.items():
        parameter.grad = grads[name].squeeze(0)
    optimizer.step()
    return {name: float(value.reshape(-1)[0]) for name, value in outputs.items()}


def closed_form_batch(task: Any, model: Any, batch: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    """One split's batch as ``closed_form_gradient`` reads it: a stack of one, prepared.

    Its ``closed_form_rows`` where the task declares them (:class:`BatchableTask`).
    """

    rows = tuple(tensor.unsqueeze(0) for tensor in batch)
    prepare = getattr(task, "closed_form_rows", None)
    return tuple(prepare(model, rows)) if callable(prepare) else rows


def scratch(
    workspace: dict[str, Tensor] | None, name: str, shape: Sequence[int], like: Tensor
) -> Tensor | None:
    """A tensor of ``shape`` kept under ``name`` in a closed form's workspace, for ``out=``.

    Made on first use, in ``like``'s dtype and device, and made again when the
    shape changes; None without a workspace, which ``out=None`` reads as a
    new tensor, so a form written with it runs either way (:class:`BatchableTask`).
    """

    if workspace is None:
        return None
    held = workspace.get(name)
    if held is None or held.shape != tuple(shape) or held.dtype != like.dtype:
        held = workspace[name] = torch.empty(tuple(shape), dtype=like.dtype, device=like.device)
    return held
