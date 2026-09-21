"""FedAvg + local fine-tuning: a personalization baseline.

Training is FedAvg, unchanged -- `fit` is inherited verbatim, and the server is
the ordinary `fedavg` strategy. The personalization happens entirely at
evaluation time: the client takes the global model it was just sent,
fine-tunes it on its OWN training data for a few epochs, and reports how that
fine-tuned model scores. The fine-tuned model is then discarded.

So the rule keeps no per-client state between rounds. A method that trains a
personal model for every client stores one per client instead, about 40 GB
over FEMNIST's 3597 writers at this model's size.

It also has a property that matters in a cross-device regime: a client is
sampled about five times in a 500-round run at participation_rate 0.01, but it
can be fine-tuned at every evaluation regardless of whether it ever trained.

**Fine-tuning reads the train split and nothing else.** Never the split being
measured. Personalizing on val or test data and then reporting a score on it
would not be a weak result, it would be a wrong one, so the split is
hard-wired here rather than configurable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from fedbrew.clients.fedavg_client import FedAvgClient
from fedbrew.clients.local_update_modes import run_sgd_update_mode
from fedbrew.clients.torch_sgd_client import _eval_request_splits, _get_train_data
from fedbrew.core.checkpointing import refuse_a_reconfigured_resume
from fedbrew.core.config import PERSONAL_SPLIT_PREFIX
from fedbrew.core.protocol import EvalRequest, EvalResult
from fedbrew.core.torch_utils import forget_resident_state

#: Scopes that ask for the aggregated server model.
_GLOBAL_SCOPES = {"global", "both"}
#: Scopes that ask for this client's fine-tuned model.
_PERSONAL_SCOPES = {"personal", "both"}


class FedAvgFTClient(FedAvgClient):
    """FedAvg training; evaluation on a locally fine-tuned copy of the model.

    ``finetune_epochs`` chained local epochs are run before the personalized
    measurement. The mode is fixed to sequential_epoch rather than following
    the client's training ``update_mode``: fine-tuning is a separate procedure
    from the federated local update, and a baseline whose meaning shifted with
    an unrelated training knob would not be comparable across arms.
    """

    def __init__(
        self,
        *args: Any,
        finetune_epochs: int,
        finetune_learning_rate: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure the post-aggregation fine-tuning pass.

        Training is FedAvg's, unchanged; what differs is evaluation, which
        happens on a locally fine-tuned *copy* of the global model. The copy is
        never sent to the server, so this changes the reported numbers and not
        the aggregate -- which is what makes it a personalization baseline
        rather than a different federated algorithm.

        Args:
            *args: Forwarded to the base client positionally.
            finetune_epochs: Passes over the client's train split before
                evaluating. 0 makes this client identical to FedAvg.
            finetune_learning_rate: Step size for those passes, in
                model-parameter units per step. None reuses the training
                ``learning_rate``.
            **kwargs: Forwarded to the base client by keyword.
        """
        super().__init__(*args, **kwargs)

        if isinstance(finetune_epochs, bool) or not isinstance(finetune_epochs, int):
            raise ValueError("finetune_epochs must be a positive integer")
        if finetune_epochs <= 0:
            raise ValueError("finetune_epochs must be a positive integer")
        self.finetune_epochs = int(finetune_epochs)

        if finetune_learning_rate is None:
            finetune_learning_rate = self.learning_rate
        if not float(finetune_learning_rate) > 0.0:
            raise ValueError("finetune_learning_rate must be positive")
        self.finetune_learning_rate = float(finetune_learning_rate)

    def evaluate(self, request: EvalRequest) -> EvalResult:
        """Evaluate the global model, the fine-tuned model, or both."""

        # Same isolation the base class applies, and it matters more here: the
        # personal pass runs real SGD in train() mode on the evaluation path,
        # so without the fork every fine-tuned client advanced the stream the
        # next round trains from.
        with self.isolated_evaluation_rng(request):
            return self._evaluate_scoped(request)

    def _evaluate_scoped(self, request: EvalRequest) -> EvalResult:
        model_scope = str(request.payload.get("model_scope", "global"))
        requested_splits = _eval_request_splits(request)
        if requested_splits is None:
            # The single-split path reports one unprefixed metric set, which
            # has nowhere to put a second scope's numbers.
            if model_scope != "global":
                raise ValueError(
                    f"fedavg_ft needs an explicit split list to report model_scope={model_scope!r}"
                )
            return super().evaluate(request)

        model = self.task.build_model(self.model_config)
        model_state_metadata = self._load_federated_payload(
            model,
            request.payload,
            context="evaluation request",
            mutates=False,
        )

        metrics: dict[str, float] = {}
        counts: dict[str, int] = {}

        # Order is load-bearing. With runtime.performance.reuse_model on --
        # the default, and what every FEMNIST config sets -- build_model
        # returns ONE cached module per architecture, so the fine-tuning below
        # mutates the same object the global pass reads. Measuring the global
        # model first and only then fine-tuning in place keeps both numbers
        # honest and allocates no second model.
        if model_scope in _GLOBAL_SCOPES:
            global_metrics, global_counts = self._split_metrics_and_counts(
                model,
                request,
                requested_splits,
            )
            metrics.update(global_metrics)
            counts.update(global_counts)

        if model_scope in _PERSONAL_SCOPES:
            # Deliberately not reported as a metric: the round CSV only carries
            # aggregated {split}_{loss,accuracy} columns, so a finetune_steps
            # metric would name a column that never appears. The count is
            # deterministic anyway -- finetune_epochs x batches per client.
            finetune_steps = self._finetune(model, request)
            personal_metrics, personal_counts = self._split_metrics_and_counts(
                model,
                request,
                requested_splits,
                split_prefix=PERSONAL_SPLIT_PREFIX,
            )
            metrics.update(personal_metrics)
            counts.update(personal_counts)
            if finetune_steps <= 0:
                raise ValueError(
                    f"client {self.client_id!r} personalized evaluation ran no fine-tuning steps"
                )

        return EvalResult(
            round_id=request.round_id,
            client_id=self.client_id,
            num_examples=sum(counts.values()),
            metrics=metrics,
            payload={
                "model_scope": model_scope,
                "num_examples_by_split": counts,
                "model_state_scope": str(model_state_metadata["model_state_scope"]),
                "model_state_metadata": model_state_metadata,
            },
        )

    def get_state(self) -> dict[str, Any]:
        """Return checkpointable client configuration.

        Nothing personalized appears here: the fine-tuned model is rebuilt from
        the current global model at every evaluation and thrown away, so this
        client costs no per-client memory at all.
        """

        state = super().get_state()
        state.update(
            {
                "finetune_epochs": self.finetune_epochs,
                "finetune_learning_rate": self.finetune_learning_rate,
            }
        )
        return state

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore checkpointable client configuration.

        Raises:
            ValueError: If the checkpoint disagrees with this run's config.
        """

        refuse_a_reconfigured_resume(
            "fedavg_ft client",
            state,
            {
                "finetune_epochs": self.finetune_epochs,
                "finetune_learning_rate": self.finetune_learning_rate,
            },
        )
        super().load_state(state)
        epochs = state.get("finetune_epochs", self.finetune_epochs)
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("finetune_epochs must be a positive integer")
        self.finetune_epochs = int(epochs)
        learning_rate = float(state.get("finetune_learning_rate", self.finetune_learning_rate))
        if not learning_rate > 0.0:
            raise ValueError("finetune_learning_rate must be positive")
        self.finetune_learning_rate = learning_rate

    def _finetune(self, model: torch.nn.Module, request: EvalRequest) -> int:
        """Adapt ``model`` in place to this client, on its train split only.

        The split is not a parameter and must never become one: fine-tuning on
        val or test and then reporting a score there would be leakage, not a
        weak baseline.
        """

        # This is the one place in the codebase that mutates a model which was
        # loaded with mutates=False, so it is the one place that has to retract
        # the residency claim that load left behind. Without this, the next
        # client's evaluation request -- same cached module under
        # reuse_model=True, same payload object from the loop -- matches
        # `resident is model_state` and skips its load, and every client after
        # the first is measured on the previous client's fine-tuned weights.
        forget_resident_state(model)

        # The draws this makes stay out of the training stream because
        # evaluate() runs the whole call inside isolated_evaluation_rng.
        #
        # `phase="finetune"`, not the default `"fit"`. Both passes read the same
        # train split in the same round, so under one phase name they took one
        # seed and one batch order -- measured identical, 24 examples in 6
        # batches, element for element. And both start from the same weights,
        # the round's global model, so with a sequential_epoch fit pass -- where
        # both counts are passes -- and `finetune_epochs <= local_iterations`
        # the personal model was a replay of the fit pass's first epochs rather
        # than an adaptation drawn independently of them. The seed derivation
        # has taken a phase since it was written; this pass just never named
        # its own. FINDINGS.csv P09-F09.
        train_data = _get_train_data(self.client_data)
        train_loader = self.task.build_dataloader(
            train_data,
            self._train_loader_config(request.round_id, phase="finetune"),
        )
        result = run_sgd_update_mode(
            task=self.task,
            model=model,
            train_loader=train_loader,
            local_iterations=self.finetune_epochs,
            learning_rate=self.finetune_learning_rate,
            update_mode="sequential_epoch",
            client_id=self.client_id,
            max_grad_norm=self.max_grad_norm,
        )
        return result.optimizer_steps
