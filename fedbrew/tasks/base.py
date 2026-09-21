"""Abstract task adapter contract for benchmark workloads."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from torch import nn

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
