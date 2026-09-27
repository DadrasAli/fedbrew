"""Abstract task adapter contract for benchmark workloads."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

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

    Two more are optional. ``loader_order(data, config)`` declares what the
    loader yields (:class:`LoaderOrder`), so a round's orders are planned
    together rather than replayed per client. ``stacked_metrics(outputs,
    counts)`` folds many splits' ``functional_eval`` outputs -- per position,
    a tensor per key over the splits, of which split ``k`` has ``counts[k]``
    -- into each split's ``compute_metrics`` and example count, as tensors,
    so a stack's metrics are computed on its device; its padding positions
    hold empty batches' outputs and must be ignored.
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
      iteration yields that order.
    """

    rows: int
    batch_size: int
    shuffle: bool
    drop_last: bool
    seed: int | None
    per_epoch: bool
    keep_single_batch: bool = False


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


def row_count(rows: Tensor, mask: Tensor | None = None) -> Tensor:
    """How many real rows a batch holds, as a float64 tensor: its ``total``."""

    if mask is None:
        return torch.tensor(float(len(rows)), dtype=torch.float64, device=rows.device)
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


class TaskAdapter(ABC):
    """Task-specific bridge used by generic FL orchestration code."""

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


def loss_averages_over_examples(task: TaskAdapter | type[TaskAdapter]) -> bool:
    """Whether a task keeps the default `train_loss_denominator`, the example count.

    Read off the class, not off a batch, so the answer cannot depend on what a
    batch happens to hold: a causal-LM batch whose active tokens equal its
    sequence count would otherwise pass one check and fail the next.
    """

    cls = task if isinstance(task, type) else type(task)
    return cls.train_loss_denominator is TaskAdapter.train_loss_denominator
