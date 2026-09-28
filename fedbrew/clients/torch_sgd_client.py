"""PyTorch local SGD client implementation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Generic, TypeVar, cast

import torch
from torch import optim

from fedbrew.clients.base import ClientUpdate
from fedbrew.clients.batch_orders import LocalLoop
from fedbrew.clients.batched_update import (
    ClientBatchFit,
    ClientBatchPlan,
    ClientEvalPlan,
    LocalProgram,
    OptimizerSpec,
    data_versions,
    no_training_batches,
    own_loop_updates,
)
from fedbrew.clients.local_update_modes import (
    FULL_GRADIENT_UPDATE_MODE,
    MIN_POSITIVE_LEARNING_RATE,
    full_gradient_into_grad,
    own_loop_update_mode,
    release_optimizer,
    reused_optimizer,
)
from fedbrew.core.checkpointing import (
    refuse_a_pre_rename_client_state,
    refuse_a_reconfigured_resume,
)
from fedbrew.core.federated_state import (
    model_state_size,
    payload_model_state_scope,
    validate_federated_state_metadata,
)
from fedbrew.core.metrics import filter_metrics
from fedbrew.core.protocol import (
    ClientInfo,
    EvalRequest,
    EvalResult,
    FitRequest,
    FitResult,
)
from fedbrew.core.runtime_setup import dataloader_seed
from fedbrew.core.seeding import client_seed
from fedbrew.core.stacked_results import MetricColumns, StackedFitResults
from fedbrew.core.torch_utils import (
    RESIDENT_STATE_ATTR,
    forget_resident_state,
)
from fedbrew.tasks.base import BatchableTask, TaskAdapter, loss_averages_over_examples

TaskT = TypeVar("TaskT", bound=TaskAdapter)


class _SeededFork:
    """torch.random.fork_rng, re-seeded once inside the forked stream."""

    def __init__(self, fork: Any, seed: int | None) -> None:
        self._fork = fork
        self._seed = seed

    def __enter__(self) -> None:
        self._fork.__enter__()
        if self._seed is not None:
            torch.manual_seed(self._seed)

    def __exit__(self, *exc: Any) -> bool:
        return bool(self._fork.__exit__(*exc))


class TorchSGDClient(ClientUpdate, Generic[TaskT]):
    """Train a PyTorch model locally on one client's data.

    The base local-update rule every other fixed-learning-rate client derives
    from. One call to ``fit`` runs ``local_iterations`` iterations of the local
    loop, each one pass over the client's train split with plain SGD (or, under
    ``update_mode: full_gradient``, one step on the exact gradient of the whole
    split), then returns the resulting model state; the server does the
    averaging.
    """

    def __init__(
        self,
        client_id: str,
        task: TaskT,
        model_config: dict[str, Any],
        client_data: Any,
        local_iterations: int,
        batch_size: int,
        learning_rate: float,
        eval_batch_size: int | None = None,
        device: str = "cpu",
        metrics: list[str] | None = None,
        base_seed: int | None = None,
        train_shuffle: bool = True,
        eval_shuffle: bool = False,
        drop_last: bool = False,
        momentum: float | None = None,
        weight_decay: float | None = None,
        nesterov: bool | None = None,
        learning_rate_schedule: str | None = None,
        min_learning_rate: float | None = None,
        total_rounds: int | None = None,
        max_local_steps: int | None = None,
        update_mode: str | None = None,
    ) -> None:
        """Configure one client's local optimizer and data loading.

        Args:
            client_id: Stable identifier for this client. Combined with the
                round id and ``base_seed`` to derive every RNG stream the
                client consumes, so it determines batch order.
            task: Task adapter that builds the model and computes loss and
                metrics.
            model_config: The ``model`` config block, copied on entry.
            client_data: This client's shards, carrying the train / val / test
                splits.
            local_iterations: Iterations of the local loop per round. For
                this rule each is one full pass over the train split; a
                subclass with an ``update_mode`` redefines it.
            batch_size: Training mini-batch size, in examples.
            learning_rate: SGD step size, in model-parameter units per step.
                Must be positive.
            eval_batch_size: Batch size for gradient-free evaluation passes, in
                examples. Defaults to ``batch_size``. Larger is faster and
                changes only floating-point summation order -- metrics stay
                example-weighted -- so results differ in the last bits only.
            device: Torch device string the model and batches live on.
            metrics: Update metric names to record. Copied, not aliased.
            base_seed: Run seed. None leaves the process RNG untouched, which
                makes the client's batch order non-reproducible.
            train_shuffle: Shuffle the train split each epoch.
            eval_shuffle: Shuffle evaluation batches. Off by default; it cannot
                change an example-weighted metric, only its summation order.
            drop_last: Drop a final partial training batch.
            momentum: SGD momentum in [0, 1), or None for none.
            weight_decay: L2 penalty, non-negative, or None for none.
            nesterov: Use Nesterov momentum. Requires positive ``momentum``.
            learning_rate_schedule: ``"constant"`` or ``"cosine"``. Cosine
                decays across rounds, not within them, so it needs
                ``total_rounds``.
            min_learning_rate: Floor for the schedule, in [0, learning_rate].
            total_rounds: Total rounds in the run, used only by the cosine
                schedule to place the current round on the curve.
            max_local_steps: Hard cap on optimizer steps per round, regardless
                of ``local_iterations``. Positive when set. Under
                ``full_gradient`` an iteration is one step, so it caps them.
            update_mode: ``sequential_epoch`` (also when unset): each iteration
                is one pass, one optimizer step per batch. ``full_gradient``:
                each is one optimizer step on the exact gradient of the whole
                train split. A subclass with modes of its own redefines it.

        Raises:
            ValueError: If any of the bounds above is violated.

        Not every rule honours every option, and the factory only hands each
        one to the rules that do. A config setting an option its rule would
        never receive is refused by ``validate_config``, per rule, against
        ``UNHONOURED_CLIENT_OPTIONS`` -- on the run path rather than in
        preflight, which only ``--validate-only`` reaches -- so no run.json
        claims a setting that never ran. A derived subclass's own ``__init__``
        check covers direct construction, where the caller passes the argument
        itself; it cannot see a config key the factory dropped.
        """

        self.client_id = client_id
        self.task = task
        self.model_config = dict(model_config)
        self.client_data = client_data
        self.local_iterations = local_iterations
        self.batch_size = batch_size
        # Evaluation is gradient-free, so a larger batch changes throughput but
        # not the reported metrics, which stay example-weighted.
        if eval_batch_size is not None and (
            isinstance(eval_batch_size, bool) or eval_batch_size <= 0
        ):
            raise ValueError("eval_batch_size must be positive when set")
        self.eval_batch_size = int(eval_batch_size or batch_size)
        self.learning_rate = learning_rate
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if momentum is not None and not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if weight_decay is not None and weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if nesterov is True and (momentum is None or momentum <= 0.0):
            raise ValueError("nesterov requires positive momentum")
        if learning_rate_schedule is not None and learning_rate_schedule not in {
            "constant",
            "cosine",
        }:
            raise ValueError("learning_rate_schedule must be constant or cosine")
        if min_learning_rate is not None and (
            min_learning_rate < 0.0 or min_learning_rate > self.learning_rate
        ):
            raise ValueError("min_learning_rate must be in [0, learning_rate]")
        if total_rounds is not None and total_rounds <= 0:
            raise ValueError("total_rounds must be positive when set")
        if max_local_steps is not None and (
            isinstance(max_local_steps, bool) or max_local_steps <= 0
        ):
            raise ValueError("max_local_steps must be positive when set")
        self.momentum = None if momentum is None else float(momentum)
        self.weight_decay = None if weight_decay is None else float(weight_decay)
        self.nesterov = nesterov
        self.learning_rate_schedule = learning_rate_schedule
        self.min_learning_rate = None if min_learning_rate is None else float(min_learning_rate)
        self.total_rounds = total_rounds
        self.max_local_steps = max_local_steps
        self.update_mode = own_loop_update_mode(update_mode)

        self.device = device
        self.metrics = list(metrics or [])
        self.base_seed = base_seed
        self.train_shuffle = bool(train_shuffle)
        self.eval_shuffle = bool(eval_shuffle)
        self.drop_last = bool(drop_last)
        self._num_examples = _infer_num_examples(client_data)
        # The message below is about training batches, so it needs the train
        # split, not the client's total across every split.
        self._num_train_examples = _infer_train_num_examples(client_data)

    def setup(self, client_info: ClientInfo) -> None:
        """Record client metadata supplied by the loop."""

        if client_info.client_id == self.client_id and client_info.num_examples > 0:
            self._num_examples = client_info.num_examples

    def fit(self, request: FitRequest) -> FitResult:
        """Run local SGD from the provided global model state."""

        train_data = _get_train_data(self.client_data)
        model = self.task.build_model(self.model_config)
        self._load_federated_payload(model, request.payload, context="fit request")
        optimizer = self._build_optimizer(model, request.round_id)
        train_loader = self.task.build_dataloader(
            train_data,
            self._train_loader_config(request.round_id),
        )

        training_outputs: list[Mapping[str, float]] = []
        optimizer_steps = 0
        try:
            for _ in range(self.local_iterations):
                if self.update_mode == FULL_GRADIENT_UPDATE_MODE:
                    training_outputs += full_gradient_into_grad(
                        task=self.task,
                        model=model,
                        train_loader=train_loader,
                        client_id=self.client_id,
                    )
                    optimizer.step()
                    optimizer_steps += 1
                else:
                    for batch in train_loader:
                        output = self.task.train_step(model, batch, optimizer)
                        training_outputs.append(output)
                        optimizer_steps += 1
                        if (
                            self.max_local_steps is not None
                            and optimizer_steps >= self.max_local_steps
                        ):
                            break
                if self.max_local_steps is not None and optimizer_steps >= self.max_local_steps:
                    break
        finally:
            release_optimizer(optimizer)

        self._require_training_batches(optimizer_steps)

        metrics, evaluated_num_examples = self._post_fit_evaluation(model, train_data, request)
        return self._fit_result(
            request,
            metrics=metrics,
            evaluated_num_examples=evaluated_num_examples,
            training_outputs=training_outputs,
            optimizer_steps=optimizer_steps,
            model_state=self.task.get_federated_model_state(model),
            model_state_metadata=self.task.federated_model_state_metadata(model),
            trainable_parameters=trainable_parameter_count(model),
        )

    def _fit_result(
        self,
        request: FitRequest,
        *,
        metrics: dict[str, float],
        evaluated_num_examples: int,
        training_outputs: list[Mapping[str, float]],
        optimizer_steps: int,
        model_state: dict[str, Any],
        model_state_metadata: dict[str, Any],
        trainable_parameters: int,
        extra_metrics: Mapping[str, float] | None = None,
    ) -> FitResult:
        """The FitResult of a trained local update, whichever executor trained it.

        ``metrics`` are the post-fit pass's, which this extends in place;
        ``extra_metrics`` go last, after the update's own.
        """

        num_examples = self.task.federated_aggregation_weight(
            training_outputs,
            evaluated_num_examples,
        )
        if num_examples < 0:
            raise ValueError("task federated aggregation weight must be non-negative")
        model_state_scope = str(model_state_metadata["model_state_scope"])
        stack = getattr(model_state, "stack", None)
        if stack is None:
            communicated_parameters, communicated_bytes = model_state_size(model_state)
        else:
            # Every row of a stack has the same size, so it is measured once.
            if stack.row_size is None:
                stack.row_size = model_state_size(model_state)
            communicated_parameters, communicated_bytes = stack.row_size
        active_target_tokens = sum(float(output.get("total", 0.0)) for output in training_outputs)
        metrics.update(
            {
                "optimizer_steps": float(optimizer_steps),
                "active_target_tokens": float(active_target_tokens),
                "trainable_parameters": float(trainable_parameters),
                "communicated_parameters": float(communicated_parameters),
                "communicated_bytes": float(communicated_bytes),
            }
        )
        metrics.update(extra_metrics or {})
        return FitResult(
            round_id=request.round_id,
            client_id=self.client_id,
            num_examples=num_examples,
            payload={
                "model_state": model_state,
                "model_state_scope": model_state_scope,
                "model_state_metadata": model_state_metadata,
            },
            metrics=metrics,
        )

    # -- the batched executor (fedbrew/clients/batched_update.py) -----------

    #: The batched executor runs this rule's update: SGD with momentum,
    #: Nesterov and weight decay, a cosine rate and a step cap, under either
    #: of its modes.
    _batched_rule = "local_sgd"

    def batched_unsupported(self) -> str | None:
        """Why the batched executor cannot run this client's update, or None if it can.

        A rule runs batched only when its own class declares ``_batched_rule``:
        a subclass inherits the hooks below but not the declaration, so one
        whose update differs is never run as its parent's.
        """

        if type(self).__dict__.get("_batched_rule") is None:
            return f"update rule {type(self).__name__} declares no batched update"
        if not isinstance(self.task, BatchableTask):
            return (
                f"task {type(self.task).__name__} provides no functional_loss and "
                "functional_eval (fedbrew.tasks.base.BatchableTask)"
            )
        if not loss_averages_over_examples(self.task):
            return f"task {type(self.task).__name__}'s training loss is not a mean over examples"
        if self.base_seed is None:
            return "experiment.seed is unset, so the loaders draw from the process-wide stream"
        if getattr(self.task, "_scaler", None) is not None:
            return "runtime.use_amp is on, and GradScaler's loss scale is sequential state"
        return None

    def batched_program(self, request: FitRequest) -> LocalProgram:
        """What one step of this rule's own loop does this round (``_build_optimizer``)."""

        if self.momentum is None or self.weight_decay is None or self.nesterov is None:
            raise ValueError("local_sgd requires optimizer settings from configuration")
        return LocalProgram(
            optimizer=OptimizerSpec(
                "sgd",
                lr=self._round_learning_rate(request.round_id),
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=bool(self.nesterov),
            ),
            combine="full" if self.update_mode == FULL_GRADIENT_UPDATE_MODE else "batch",
        )

    def batched_plan(
        self,
        request: FitRequest,
        model: torch.nn.Module,
        start: Mapping[str, Any] | None = None,
    ) -> ClientBatchPlan:
        """This round's local update, as the batched executor runs it (``fit``'s loop).

        ``model`` is the architecture, for checking the broadcast against;
        its values are not read. ``start`` is the broadcast already checked
        by ``batched_start`` for another client of the same class and payload.
        """

        full = self.update_mode == FULL_GRADIENT_UPDATE_MODE
        return self._batch_plan(
            request,
            self.batched_program(request),
            self.batched_start(request, model) if start is None else start,
            self.batched_loop(),
            (lambda: no_training_batches(self.client_id))
            if full
            else self._no_training_batches_refusal,
            lambda loader: own_loop_updates(
                loader,
                local_iterations=self.local_iterations,
                update_mode=self.update_mode,
                max_local_steps=self.max_local_steps,
                client_id=self.client_id,
            ),
        )

    def batched_loop(self) -> LocalLoop:
        """The batches this rule's own loop takes of its loader's epochs, per update."""

        return LocalLoop(
            epochs=self.local_iterations,
            per_update="epoch" if self.update_mode == FULL_GRADIENT_UPDATE_MODE else "batch",
            max_updates=self.max_local_steps,
        )

    def batched_start(self, request: FitRequest, model: torch.nn.Module) -> Mapping[str, Any]:
        """The broadcast state the update starts from, checked as ``fit`` checks it."""

        return self._checked_federated_payload(model, request.payload, context="fit request")[0]

    def _batch_plan(
        self,
        request: FitRequest,
        program: LocalProgram,
        start: Mapping[str, Any],
        loop: LocalLoop,
        refuse: Any,
        draw: Any,
        **controls: Any,
    ) -> ClientBatchPlan:
        """The plan: the loop and what the loaders yield, or how to replay them.

        ``draw`` takes the training loader's batches as row indices and
        returns each update's, as the rule's own loop takes them; it is used
        only for a task that declares no ``loader_order``.
        """

        train_data = _get_train_data(self.client_data)
        round_id = request.round_id
        train_order = self._loader_order(
            train_data, self._train_loader_config(round_id, seeded=False)
        )
        eval_order = self._loader_order(
            train_data, self._eval_loader_config(round_id, seeded=False)
        )
        replay = None
        if train_order is None or eval_order is None:

            def replay() -> tuple[list[list[Any]], list[Any]]:
                updates = draw(
                    self.task.row_batches(train_data, self._train_loader_config(round_id))
                )
                evaluated = list(
                    self.task.row_batches(train_data, self._eval_loader_config(round_id))
                )
                return updates, evaluated

            train_order = eval_order = None
        return ClientBatchPlan(
            program=program,
            train_data=train_data,
            loop=loop,
            train_order=train_order,
            eval_order=eval_order,
            evaluate=request.post_fit_evaluation,
            start=start,
            client_id=self.client_id,
            seed=self.base_seed,
            refuse=refuse,
            replay=replay,
            **controls,
        )

    def _loader_order(self, data: Any, config: dict[str, Any]) -> Any:
        """The task's declaration of what its loader yields, or None to replay the loader.

        Unseeded, it is the same every round for the same unedited data and
        configuration, so it is kept per split.
        """

        declare = getattr(self.task, "loader_order", None)
        if not callable(declare):
            return None
        kept = getattr(self, "_kept_orders", None)
        if kept is None:
            kept = self._kept_orders = {}
        key = (id(data), tuple(config.items()))
        versions = data_versions(data)
        held = kept.get(key)
        if held is not None and held[0] is data and held[1] == versions:
            return held[2]
        order = declare(data, config)
        kept[key] = (data, versions, order)
        return order

    def batched_evaluation_supported(self) -> bool:
        """Whether the batched evaluator may measure this client: its evaluate is this class's."""

        cls = type(self)
        return (
            cls.__dict__.get("_batched_rule") is not None
            and cls.evaluate is TorchSGDClient.evaluate
            and cls._evaluate is TorchSGDClient._evaluate
            and self.batched_unsupported() is None
        )

    def batched_evaluation_plan(self, request: EvalRequest) -> ClientEvalPlan:
        """``_evaluate``'s passes over the requested splits, as the batched evaluator runs them."""

        splits = _eval_request_splits(request)
        if splits is None:
            raise ValueError("the batched evaluator measures requested splits")
        plan = ClientEvalPlan(splits=list(splits), client_id=self.client_id, seed=self.base_seed)
        for split in splits:
            split_data = _get_evaluation_split(self.client_data, split)
            if split_data is None:
                if split != "val":
                    plan.refusal = self._missing_split_refusal(split)
                    break
                plan.data.append(None)
                plan.orders.append(None)
                plan.batches.append(None)
                continue
            order = self._loader_order(
                split_data, self._eval_loader_config(request.round_id, seeded=False)
            )
            plan.data.append(split_data)
            plan.orders.append(order)
            plan.batches.append(
                None
                if order is not None
                else list(
                    self.task.row_batches(split_data, self._eval_loader_config(request.round_id))
                )
            )
        return plan

    def batched_evaluation_result(
        self,
        request: EvalRequest,
        plan: ClientEvalPlan,
        outputs: list[Any],
        model_state_metadata: dict[str, Any],
    ) -> EvalResult:
        """The EvalResult ``evaluate`` returns, from this client's share of the batched pass.

        ``outputs`` has, per requested split, its eval-step outputs, or
        ``compute_metrics`` of them and their example count as the task
        folded them for a whole chunk (``stacked_metrics``), or None for a
        split the client does not have.
        """

        if plan.refusal is not None:
            raise plan.refusal
        metrics: dict[str, float] = {}
        num_examples_by_split: dict[str, int] = {}
        requested_metrics = self._eval_request_metrics(request)
        for split, split_data, split_outputs in zip(plan.splits, plan.data, outputs, strict=True):
            if split_data is None:
                num_examples_by_split[split] = 0
                continue
            assert split_outputs is not None
            if isinstance(split_outputs, tuple):
                # compute_metrics of the split's outputs, and their examples,
                # folded by the task for the whole chunk (stacked_metrics).
                computed, num_examples = split_outputs
                split_metrics = filter_metrics(
                    computed, self.metrics if requested_metrics is None else requested_metrics
                )
            else:
                split_metrics, num_examples = self._evaluation_metrics(
                    split_outputs, split_data, metrics=requested_metrics
                )
            self._record_split(
                metrics, num_examples_by_split, split, split, split_metrics, num_examples
            )
        result = self._requested_splits_result(request, metrics, num_examples_by_split)
        result.payload.update(
            model_state_scope=str(model_state_metadata["model_state_scope"]),
            model_state_metadata=model_state_metadata,
        )
        return result

    def batched_result(
        self, request: FitRequest, plan: ClientBatchPlan, fit: ClientBatchFit
    ) -> FitResult:
        """The FitResult ``fit`` returns, from the batched executor's share for this client."""

        metrics, evaluated_num_examples = self._batched_post_fit(plan, fit)
        return self._fit_result(
            request,
            metrics=metrics,
            evaluated_num_examples=evaluated_num_examples,
            training_outputs=fit.training_outputs,
            optimizer_steps=fit.optimizer_steps,
            model_state=fit.model_state,
            model_state_metadata=fit.model_state_metadata,
            trainable_parameters=fit.trainable_parameters,
            extra_metrics=self._batched_extra_metrics(plan),
        )

    def _batched_extra_metrics(self, plan: ClientBatchPlan) -> dict[str, float]:
        """What this rule's result adds after the update's own metrics (``_fit_result``)."""

        del plan
        return {}

    def batched_stacked_supported(self) -> bool:
        """Whether this client's results can be built for a whole chunk at once.

        They can when they are ``_fit_result``'s, as this class builds them,
        and the task weighs a client by the examples its post-fit pass
        counted -- ``federated_aggregation_weight`` as ``TaskAdapter`` defines
        it -- so the per-step outputs are read for nothing but their
        ``total``. A rule whose result is its own (FedProx, SCAFFOLD) builds
        it client by client.
        """

        cls = type(self)
        return (
            cls.batched_result is TorchSGDClient.batched_result
            and cls._fit_result is TorchSGDClient._fit_result
            and cls._batched_post_fit is TorchSGDClient._batched_post_fit
            and type(self.task).federated_aggregation_weight
            is TaskAdapter.federated_aggregation_weight
        )

    def batched_stacked_results(
        self,
        requests: Sequence[FitRequest],
        plans: Sequence[ClientBatchPlan],
        chunk: Any,
        members: Sequence[TorchSGDClient[Any]],
    ) -> StackedFitResults:
        """``batched_result`` for every client of a chunk, as one stacked result.

        ``chunk`` is the chunk trained (``_train_chunk_stacked``), and
        ``members[p]`` is client ``p``'s rule, this one first. Each client's
        count, metrics and payload are the ones its ``batched_result`` would
        return -- the same numbers under the same names in the same order, and
        the same refusal -- computed a column at a time where the chunk holds
        them as columns: the post-fit metrics the task folded, and each
        client's own ``_fit_result`` arithmetic on them.
        """

        size = len(requests)
        columns = MetricColumns(size)
        evaluated = _stacked_post_fit(chunk, plans, members, columns)
        totals = _step_totals(chunk, size)
        num_examples = [int(count) for count in evaluated]
        if any(count < 0 for count in num_examples):
            raise ValueError("task federated aggregation weight must be non-negative")
        first = chunk.buckets[0].states
        if first.row_size is None:
            first.row_size = model_state_size(first.row(0))
        communicated_parameters, communicated_bytes = first.row_size
        everyone = range(size)
        columns.put("optimizer_steps", everyone, [float(len(plan.structure)) for plan in plans])
        columns.put("active_target_tokens", everyone, [float(total) for total in totals])
        for name, value in (
            ("trainable_parameters", float(chunk.trainable_parameters)),
            ("communicated_parameters", float(communicated_parameters)),
            ("communicated_bytes", float(communicated_bytes)),
        ):
            columns.put(name, everyone, [value] * size)
        for position, (member, plan) in enumerate(zip(members, plans, strict=True)):
            for name, value in member._batched_extra_metrics(plan).items():
                columns.put(name, [position], [value])
        metrics, reported = columns.tensors()
        metadata = chunk.model_state_metadata
        return StackedFitResults(
            round_id=requests[0].round_id,
            client_ids=[member.client_id for member in members],
            num_examples=torch.tensor(num_examples, dtype=torch.int64),
            states=[(bucket.states, bucket.positions) for bucket in chunk.buckets],
            metrics=metrics,
            reported=reported,
            payload={
                "model_state_scope": str(metadata["model_state_scope"]),
                "model_state_metadata": metadata,
            },
        )

    def _batched_post_fit(
        self,
        plan: ClientBatchPlan,
        fit: ClientBatchFit,
        metrics: list[str] | None = None,
    ) -> tuple[dict[str, float], int]:
        """``_post_fit_evaluation``'s result, from the executor's pass over the same batches."""

        if plan.evaluate and fit.eval_metrics is not None:
            computed, num_examples = fit.eval_metrics
            computed = {f"fit_{name}": value for name, value in computed.items()}
            return filter_metrics(
                computed, self.metrics if metrics is None else metrics
            ), num_examples
        if plan.evaluate:
            assert fit.eval_outputs is not None
            return self._evaluation_metrics(
                fit.eval_outputs, plan.train_data, metrics=metrics, prefix="fit_"
            )
        if plan.eval_rows:
            return {}, plan.eval_rows
        return {}, _infer_split_num_examples(plan.train_data)

    def _require_training_batches(self, optimizer_steps: int) -> None:
        """Refuse to report a fit that never ran a step.

        Five of the seven update rules already raise here; local_sgd and
        fedprox hand-roll their epoch loop and forgot to. Running the body zero
        times leaves every downstream quantity looking healthy -- model_state is
        the global state the client was sent, and num_examples is the client's
        full train count, because it comes from the separate post-fit evaluation
        pass whose loader never sets drop_last -- so the client is folded into
        the weighted mean at full weight while contributing nothing, diluting
        every other client's update, and the run exits "completed". Divergence
        detection cannot see it either: the watched metric is fit_loss, which is
        exactly what stops moving.
        """

        if optimizer_steps > 0:
            return
        raise self._no_training_batches_refusal()

    def _no_training_batches_refusal(self) -> ValueError:
        return ValueError(
            f"client {self.client_id!r} has no training batches "
            f"(train split {self._num_train_examples}, "
            f"batch_size {self.batch_size}, "
            f"drop_last {self.drop_last})"
        )

    def evaluate(self, request: EvalRequest) -> EvalResult:
        """Evaluate the provided model state on requested client splits."""

        return self._evaluate(request)

    def isolated_evaluation_rng(self, request: EvalRequest) -> Any:
        """Keep an evaluation's random draws out of the training stream.

        Evaluation is a measurement, and one that moves the generator the next
        round trains from turns evaluation.*.clients from a cost knob into a
        hyperparameter of the result.

        Plain evaluation does not need this. Its only draw is the base seed each
        DataLoader iterator takes, and _loader_seed already hands the loader its
        own generator whenever base_seed is set, which every shipped config
        does. Wrapping it anyway would add a fork and a restore to every client
        evaluation, to isolate draws that are already isolated.

        What does need it is a rule that TRAINS on the evaluation path:
        fedavg_ft fine-tunes in train() mode, so it draws dropout masks from the
        process-wide stream. That is the caller this exists for.

        Seeded inside the fork so the measurement stays reproducible rather than
        merely isolated.
        """

        fork = torch.random.fork_rng(devices=self._fork_devices(), enabled=True)
        seed = (
            None
            if self.base_seed is None
            else client_seed(int(self.base_seed), request.round_id, self.client_id)
        )
        return _SeededFork(fork, seed)

    def _fork_devices(self) -> list[torch.device]:
        """The CUDA device to fork, if the run is on one.

        Forking every visible device would save and restore a generator per GPU
        on a shared node; only the device this client computes on can advance.
        """

        device = getattr(self.task, "device", None)
        if isinstance(device, torch.device) and device.type == "cuda":
            return [device]
        return []

    def _evaluate(self, request: EvalRequest) -> EvalResult:
        model = self.task.build_model(self.model_config)
        model_state_metadata = self._load_federated_payload(
            model,
            request.payload,
            context="evaluation request",
            mutates=False,
        )
        model_state_scope = str(model_state_metadata["model_state_scope"])
        requested_splits = _eval_request_splits(request)
        if requested_splits is not None:
            result = self._evaluate_requested_splits(model, request, requested_splits)
            result.payload.update(
                model_state_scope=model_state_scope,
                model_state_metadata=model_state_metadata,
            )
            return result

        eval_data = _get_eval_data(self.client_data, self.client_id)
        metrics, num_examples = self._evaluate_model(
            model,
            eval_data,
            metrics=self._eval_request_metrics(request),
            round_id=request.round_id,
        )
        return EvalResult(
            round_id=request.round_id,
            client_id=self.client_id,
            num_examples=num_examples,
            metrics=metrics,
            payload={
                "model_state": self.task.get_federated_model_state(model),
                "model_state_scope": model_state_scope,
                "model_state_metadata": model_state_metadata,
            },
        )

    def _load_federated_payload(
        self,
        model: torch.nn.Module,
        payload: Mapping[str, Any],
        *,
        context: str,
        mutates: bool = True,
    ) -> dict[str, Any]:
        """Load a received model state, skipping copies that cannot matter.

        ``mutates`` describes what the caller does with the model afterwards.
        Fit trains it, so it must always take a fresh copy and must leave no
        claim that the received state is still resident. Evaluation only reads,
        so consecutive evaluations of the same state can share one load.

        That case is the whole cross-device evaluation round: with
        ``reuse_model=True`` every client evaluates through the same cached
        model instance, and ``client_scope: all`` hands all 3597 of them the
        same global state object, so the load after the first was copying a
        state dict onto itself. tools/profile_client_eval.py is how that cost
        was found.
        """

        model_state, expected_metadata = self._checked_federated_payload(
            model, payload, context=context
        )
        # Identity, not equality: the loop builds one payload per round and
        # shares it across every request, so `is` is exactly the question
        # "have these same weights already been loaded into this same model".
        if not mutates and getattr(model, RESIDENT_STATE_ATTR, None) is model_state:
            return expected_metadata
        # Dropped before the load, not after: if the load raises, the model is
        # left in an unknown state and must not claim to hold anything.
        forget_resident_state(model)
        # No clone here. Every load_federated_model_state implementation clones
        # the state before handing it to load_state_dict, which then copies
        # into the model's existing storage rather than aliasing the argument.
        self.task.load_federated_model_state(model, model_state)
        if not mutates:
            setattr(model, RESIDENT_STATE_ATTR, model_state)
        return expected_metadata

    def _checked_federated_payload(
        self,
        model: torch.nn.Module,
        payload: Mapping[str, Any],
        *,
        context: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """A received payload's model state, and the metadata it was checked against.

        Raises:
            ValueError: If the payload carries no model state, or one whose
                scope or metadata does not describe ``model``.
        """

        model_state = payload.get("model_state")
        if not isinstance(model_state, dict):
            raise ValueError(f"{context} payload must contain model_state")
        expected_metadata = self.task.federated_model_state_metadata(model)
        received_scope = payload_model_state_scope(payload, context=context)
        received_metadata = payload.get("model_state_metadata")
        validate_federated_state_metadata(
            expected_metadata,
            received_metadata if isinstance(received_metadata, Mapping) else None,
            received_scope=received_scope,
            context=context,
        )
        return model_state, expected_metadata

    def _evaluate_requested_splits(
        self,
        model: torch.nn.Module,
        request: EvalRequest,
        requested_splits: list[str],
    ) -> EvalResult:
        metrics, num_examples_by_split = self._split_metrics_and_counts(
            model,
            request,
            requested_splits,
        )
        return self._requested_splits_result(request, metrics, num_examples_by_split)

    def _requested_splits_result(
        self,
        request: EvalRequest,
        metrics: dict[str, float],
        num_examples_by_split: dict[str, int],
    ) -> EvalResult:
        """The EvalResult of a pass over the requested splits, whichever evaluator ran it."""

        return EvalResult(
            round_id=request.round_id,
            client_id=self.client_id,
            num_examples=sum(num_examples_by_split.values()),
            metrics=metrics,
            payload={
                "model_scope": str(request.payload.get("model_scope", "global")),
                "num_examples_by_split": num_examples_by_split,
            },
        )

    def _split_metrics_and_counts(
        self,
        model: torch.nn.Module,
        request: EvalRequest,
        requested_splits: list[str],
        split_prefix: str = "",
    ) -> tuple[dict[str, float], dict[str, int]]:
        """Measure ``model`` on each requested split, keyed by split name.

        ``split_prefix`` renames the reported splits without changing which
        data is read, so a personalized pass reports personal_val_accuracy over
        exactly the same val data the global pass used.
        """

        metrics: dict[str, float] = {}
        num_examples_by_split: dict[str, int] = {}
        requested_metrics = self._eval_request_metrics(request)
        for split in requested_splits:
            reported = f"{split_prefix}{split}"
            split_data = _get_evaluation_split(self.client_data, split)
            if split_data is None:
                # Validation is optional per client: a client with too few
                # samples to hold out a val split reports zero and is excluded
                # from the val aggregate rather than failing the whole round.
                if split == "val":
                    num_examples_by_split[reported] = 0
                    continue
                raise self._missing_split_refusal(split)
            split_metrics, num_examples = self._evaluate_model(
                model,
                split_data,
                metrics=requested_metrics,
                round_id=request.round_id,
            )
            self._record_split(
                metrics, num_examples_by_split, split, reported, split_metrics, num_examples
            )

        return metrics, num_examples_by_split

    def _missing_split_refusal(self, split: str) -> ValueError:
        return ValueError(
            f"client {self.client_id!r} has no non-empty {split} split; "
            "post-aggregation evaluation requires separate train and test data"
        )

    def _record_split(
        self,
        metrics: dict[str, float],
        num_examples_by_split: dict[str, int],
        split: str,
        reported: str,
        split_metrics: Mapping[str, float],
        num_examples: int,
    ) -> None:
        """One split's metrics under its reported name, and its count; an empty one refused."""

        if num_examples <= 0:
            raise ValueError(f"client {self.client_id!r} has an empty {split} split")
        num_examples_by_split[reported] = num_examples
        metrics.update({f"{reported}_{name}": value for name, value in split_metrics.items()})

    def get_state(self) -> dict[str, Any]:
        """Return serializable client metadata."""

        return {
            "client_id": self.client_id,
            "num_examples": self._num_examples,
            "local_iterations": self.local_iterations,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "nesterov": self.nesterov,
            "learning_rate_schedule": self.learning_rate_schedule,
            "min_learning_rate": self.min_learning_rate,
            "total_rounds": self.total_rounds,
            "max_local_steps": self.max_local_steps,
            "update_mode": self.update_mode,
            "metrics": list(self.metrics),
            "base_seed": self.base_seed,
            "train_shuffle": self.train_shuffle,
            "eval_shuffle": self.eval_shuffle,
            "drop_last": self.drop_last,
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore serializable client metadata.

        Raises:
            ValueError: If the checkpoint disagrees with this run's config
                about any configured setting, or carries `local_epochs` --
                written before the rename to `local_iterations`, and refused
                because the comparison skips a key the checkpoint lacks.

        Every key `get_state` writes is compared except two, and both
        exemptions are about what the value *is* rather than how much it
        matters. `num_examples` is measured from the client's shard, not
        configured. `metrics` is which columns the run writes, which the
        per-client CSV cursor already handles by invalidating on a changed
        header.

        The rest are compared whether or not `load_state` goes on to restore
        them. Only five of them are restored, and those were P10-F14 -- the
        checkpoint silently outranking the config. The other ten are the same
        defect facing the other way: `learning_rate`, `local_iterations` and the
        rest are checkpointed, read back by nothing, and so take silent effect
        from the resumed round on. See FINDINGS.csv POST-F04.

        `total_rounds` is compared only under a schedule that reads it. It is
        an input to `_round_learning_rate` and to nothing else, and that method
        returns before touching it when the schedule is ``constant``. So under
        ``constant`` a resume that extends the run changes no number a client
        computes, and under ``cosine`` it moves every remaining learning rate --
        the horizon of the curve *is* the schedule. Refusing it in both cases
        would refuse the one resume that is unambiguously fine; refusing it in
        neither would let `global_rounds: 500` -> `1000` re-anneal a cosine run
        from its midpoint without a word.
        """

        configured: dict[str, Any] = {
            "client_id": self.client_id,
            "local_iterations": self.local_iterations,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "nesterov": self.nesterov,
            "learning_rate_schedule": self.learning_rate_schedule,
            "min_learning_rate": self.min_learning_rate,
            "base_seed": self.base_seed,
            "train_shuffle": self.train_shuffle,
            "eval_shuffle": self.eval_shuffle,
            "drop_last": self.drop_last,
            "max_local_steps": self.max_local_steps,
            "update_mode": self.update_mode,
        }
        if self.learning_rate_schedule != "constant":
            configured["total_rounds"] = self.total_rounds
        # First: a key absent from `state` is not compared, so a checkpoint
        # carrying `local_epochs` instead of `local_iterations` would pass the
        # comparison below with that setting unchecked.
        refuse_a_pre_rename_client_state("local_sgd client", state)
        refuse_a_reconfigured_resume("local_sgd client", state, configured)
        self._num_examples = int(state.get("num_examples", self._num_examples))
        if "base_seed" in state:
            raw_seed = state["base_seed"]
            self.base_seed = None if raw_seed is None else int(raw_seed)
        if "train_shuffle" in state:
            self.train_shuffle = bool(state["train_shuffle"])
        if "eval_shuffle" in state:
            self.eval_shuffle = bool(state["eval_shuffle"])
        if "drop_last" in state:
            self.drop_last = bool(state["drop_last"])
        if "max_local_steps" in state:
            raw_max_steps = state["max_local_steps"]
            if raw_max_steps is None:
                self.max_local_steps = None
            elif (
                isinstance(raw_max_steps, bool)
                or not isinstance(raw_max_steps, int)
                or raw_max_steps <= 0
            ):
                raise ValueError("checkpoint max_local_steps must be positive or null")
            else:
                self.max_local_steps = raw_max_steps

    def _build_optimizer(
        self,
        model: torch.nn.Module,
        round_id: int,
    ) -> optim.Optimizer:
        if self.momentum is None or self.weight_decay is None or self.nesterov is None:
            raise ValueError("local_sgd requires optimizer settings from configuration")
        return reused_optimizer(
            optim.SGD,
            model.parameters(),
            lr=self._round_learning_rate(round_id),
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
        )

    def _round_learning_rate(self, round_id: int) -> float:
        if self.learning_rate_schedule is None or self.min_learning_rate is None:
            raise ValueError("local_sgd requires learning-rate schedule settings")
        if self.learning_rate_schedule == "constant":
            return self.learning_rate
        if self.total_rounds is None or self.total_rounds <= 1:
            return self.learning_rate
        progress = (round_id - 1) / (self.total_rounds - 1)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduled = self.min_learning_rate + (self.learning_rate - self.min_learning_rate) * cosine
        # Only the scheduled branch is clamped. `constant` returns
        # `self.learning_rate`, which __init__ already refuses at zero, so a
        # configured zero still fails loudly instead of being floored here.
        return max(scheduled, MIN_POSITIVE_LEARNING_RATE)

    def _evaluate_model(
        self,
        model: torch.nn.Module,
        data: Any,
        metrics: list[str] | None = None,
        round_id: int | None = None,
        prefix: str = "",
    ) -> tuple[dict[str, float], int]:
        """Evaluate ``model`` on ``data`` and return (metrics, example count).

        ``prefix`` is applied to the task's raw metric names *before* filtering,
        so a config asking for ``fit_loss`` selects the same number the task
        reports as ``loss``. Fit passes "fit_"; evaluation passes nothing and
        prefixes per split at the call site.
        """

        eval_loader = self.task.build_dataloader(
            data,
            self._eval_loader_config(round_id or 0),
        )
        outputs = [self.task.eval_step(model, batch) for batch in eval_loader]
        return self._evaluation_metrics(outputs, data, metrics=metrics, prefix=prefix)

    def _evaluation_metrics(
        self,
        outputs: list[Mapping[str, Any]],
        data: Any,
        metrics: list[str] | None = None,
        prefix: str = "",
    ) -> tuple[dict[str, float], int]:
        """``_evaluate_model``'s result from the eval-step outputs of its pass over ``data``."""

        metric_names = self.metrics if metrics is None else metrics
        computed = self.task.compute_metrics(outputs)
        if prefix:
            computed = {f"{prefix}{name}": value for name, value in computed.items()}
        filtered_metrics = filter_metrics(computed, metric_names)
        if outputs and all("total" in output for output in outputs):
            num_examples = int(sum(float(output["total"]) for output in outputs))
        else:
            num_examples = _infer_split_num_examples(data)
        return filtered_metrics, num_examples

    def _post_fit_evaluation(
        self,
        model: torch.nn.Module,
        train_data: Any,
        request: FitRequest,
        metrics: list[str] | None = None,
    ) -> tuple[dict[str, float], int]:
        """The trained model's fit_ metrics on its train split, and the split's example count.

        The count is the aggregation weight's input, so every fit needs it; the
        metrics only on the rounds evaluation.fit.every schedules. On the
        others the forward pass is skipped and the count is taken the way the
        pass would take it (_evaluated_example_count), so the weight, and every
        number that follows from it, is the same.
        """

        if request.post_fit_evaluation:
            return self._evaluate_model(
                model, train_data, metrics=metrics, round_id=request.round_id, prefix="fit_"
            )
        return {}, self._evaluated_example_count(model, train_data, request.round_id)

    def _evaluated_example_count(self, model: torch.nn.Module, data: Any, round_id: int) -> int:
        """The example count _evaluate_model returns, without its forward pass where it can.

        The same loader, built the same way and iterated in full, so the
        batches are the pass's and so is any draw a loader makes as it starts.
        Each batch's "total" comes from the task's evaluation_total, which
        reads it off the batch; a task that returns None there has its
        eval_step run for that batch instead, so the count is exact for any
        task and only the saving depends on the hook.
        """

        eval_loader = self.task.build_dataloader(data, self._eval_loader_config(round_id or 0))
        from_batch = getattr(self.task, "evaluation_total", lambda batch: None)
        totals: list[Any] = []
        for batch in eval_loader:
            total = from_batch(batch)
            if total is None:
                total = self.task.eval_step(model, batch).get("total")
            totals.append(total)
        if totals and all(total is not None for total in totals):
            return int(sum(float(total) for total in totals))
        return _infer_split_num_examples(data)

    def _eval_request_metrics(self, request: EvalRequest) -> list[str] | None:
        raw_metrics = request.payload.get("metrics")
        if not raw_metrics:
            return None
        if isinstance(raw_metrics, list) and all(isinstance(metric, str) for metric in raw_metrics):
            return list(raw_metrics)
        return None

    def _train_loader_config(
        self, round_id: int, phase: str = "fit", seeded: bool = True
    ) -> dict[str, Any]:
        """Loader options for a training pass over the client's train split.

        ``phase`` names which pass, and the name reaches `dataloader_seed`, so
        two training passes in the same round over the same data draw different
        orders. `fedavg_ft` is why it is a parameter: its fine-tuning pass ran
        under the default `"fit"` and therefore replayed the fit pass's batch
        order exactly, from the same starting weights. FINDINGS.csv P09-F09.
        """

        config: dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": self.train_shuffle,
            "drop_last": self.drop_last,
        }
        # Unseeded for the batched planner, which derives every client's seed
        # in one pass (fedbrew/clients/batch_orders.py).
        seed = self._loader_seed(round_id, phase) if seeded else None
        if seed is not None:
            config["seed"] = seed
        return config

    def _eval_loader_config(self, round_id: int, seeded: bool = True) -> dict[str, Any]:
        config: dict[str, Any] = {
            "batch_size": self.eval_batch_size,
            "shuffle": self.eval_shuffle,
        }
        seed = self._loader_seed(round_id, "eval") if seeded else None
        if seed is not None:
            config["seed"] = seed
        return config

    def _loader_seed(self, round_id: int, phase: str) -> int | None:
        if self.base_seed is None:
            return None
        return dataloader_seed(
            int(self.base_seed),
            int(round_id),
            self.client_id,
            phase,
        )


def _stacked_post_fit(
    chunk: Any,
    plans: Sequence[ClientBatchPlan],
    members: Sequence[TorchSGDClient[Any]],
    columns: MetricColumns,
) -> list[int]:
    """Each client's ``_batched_post_fit``: its metrics into ``columns``, its count returned.

    A bucket whose post-fit metrics the task folded is read a column at a
    time, each client's own metrics list filtering its values as it filters
    its dict (a run's clients share one list); any other bucket is read
    client by client, through ``_batched_post_fit`` itself.
    """

    evaluated = [0] * len(plans)
    for bucket in chunk.buckets:
        positions = bucket.positions
        if bucket.eval_metrics is None:
            for row, position in enumerate(positions):
                metrics, evaluated[position] = members[position]._batched_post_fit(
                    plans[position],
                    ClientBatchFit(
                        model_state={},
                        training_outputs=[],
                        eval_outputs=None
                        if bucket.eval_outputs is None
                        else bucket.eval_outputs[row],
                        optimizer_steps=0,
                        model_state_metadata={},
                        trainable_parameters=0,
                    ),
                )
                for name, value in metrics.items():
                    columns.put(name, [position], [value])
            continue
        computed = {f"fit_{name}": values for name, values in bucket.eval_metrics.items()}
        kept: dict[tuple[str, ...], list[int]] = {}
        for row, position in enumerate(positions):
            kept.setdefault(tuple(members[position].metrics), []).append(row)
            evaluated[position] = bucket.eval_examples[row]
        for requested, rows in kept.items():
            for name, values in filter_metrics(computed, list(requested)).items():
                columns.put(name, [positions[row] for row in rows], [values[row] for row in rows])
    return evaluated


def _step_totals(chunk: Any, size: int) -> list[float]:
    """Each client's ``active_target_tokens``: ``_fit_result``'s sum of its steps' ``total``."""

    totals = [0.0] * size
    for bucket in chunk.buckets:
        if bucket.step_totals is None:
            continue
        width = len(bucket.positions)
        for row, position in enumerate(bucket.positions):
            totals[position] = sum(float(total) for total in bucket.step_totals[row::width])
    return totals


def trainable_parameter_count(model: torch.nn.Module) -> int:
    """The elements of ``model``'s trainable parameters: the ``trainable_parameters`` metric."""

    return sum(
        int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
    )


def _get_train_data(client_data: Any) -> Any:
    if isinstance(client_data, Mapping):
        train_data = client_data.get("train")
        if isinstance(train_data, Mapping):
            return train_data
    return client_data


def _eval_request_splits(request: EvalRequest) -> list[str] | None:
    raw_splits = request.payload.get("splits")
    if raw_splits is None:
        return None
    if (
        not isinstance(raw_splits, list)
        or not raw_splits
        or not all(isinstance(split, str) for split in raw_splits)
    ):
        raise ValueError("eval request splits must be a non-empty list of strings")
    unknown = sorted(set(raw_splits) - {"train", "val", "test"})
    if unknown:
        raise ValueError(f"unsupported eval request splits: {', '.join(unknown)}")
    return list(dict.fromkeys(raw_splits))


def _get_evaluation_split(client_data: Any, split: str) -> Any | None:
    if not isinstance(client_data, Mapping):
        return client_data if split == "train" else None

    if split == "train":
        train_data = client_data.get("train")
        if isinstance(train_data, Mapping):
            return train_data if _infer_split_num_examples(train_data) > 0 else None
        return client_data if _infer_split_num_examples(client_data) > 0 else None

    # "val" is the config and metric name; "eval" is the on-disk shard key.
    # The mapping lives here so no other module has to know both names.
    shard_key = "eval" if split == "val" else split
    split_data = client_data.get(shard_key)
    if isinstance(split_data, Mapping) and _infer_split_num_examples(split_data) > 0:
        return split_data
    return None


def _get_eval_data(client_data: Any, client_id: str = "?") -> Any:
    """The val slice for the split-less `evaluate()` path, or a refusal.

    One resolver, not two: this delegates to `_get_evaluation_split(.., "val")`
    so the single-split path reads the val slice under exactly the rule the
    per-split path uses, rather than a second contract that happens to agree
    most of the time.

    It used to fall back to the train slice when val was missing or empty, and
    report those numbers under the caller's evaluation metric names. A client
    too small to hold out a val slice would score its own training data and
    nothing in the metric name, the count or the CSV would say so -- the
    reading-side half of exactly the leak chapter 05's disjoint slices are cut
    to prevent. `tests/test_split_reader_isolation.py` recorded the fallback as
    this resolver's contract, so the code and the chapter disagreed in writing.

    The per-split path treats a missing val slice as zero examples and drops
    the client from the val aggregate. That option is not available here: this
    path reports one unprefixed metric set with no per-split counts, so a
    caller cannot tell an excluded client from an evaluated one. It refuses
    instead. FINDINGS.csv P07-F08.
    """

    split_data = _get_evaluation_split(client_data, "val")
    if split_data is None:
        raise ValueError(
            f"client {client_id!r} has no non-empty val split, and a split-less "
            "evaluate() cannot report which split it read. Pass "
            'payload["splits"] to evaluate the splits this client does have.'
        )
    return split_data


def _infer_train_num_examples(client_data: Any) -> int:
    """The client's train rows, for messages about training."""

    if isinstance(client_data, Mapping):
        declared = client_data.get("num_train_examples")
        if isinstance(declared, int):
            return declared
        train_data = client_data.get("train")
        if isinstance(train_data, Mapping):
            return _infer_split_num_examples(train_data)
    return _infer_split_num_examples(client_data)


def _infer_num_examples(client_data: Any) -> int:
    if isinstance(client_data, Mapping):
        raw_num_examples = client_data.get("num_examples")
        if isinstance(raw_num_examples, int):
            return raw_num_examples
        train_data = client_data.get("train")
        eval_data = client_data.get("eval")
        test_data = client_data.get("test")
        splits = (train_data, eval_data, test_data)
        if any(isinstance(split, Mapping) for split in splits):
            # Every split, matching what clients.jsonl declares.
            return sum(_infer_split_num_examples(split) for split in splits)
    return _infer_split_num_examples(client_data)


def _infer_split_num_examples(data: Any) -> int:
    if isinstance(data, Mapping):
        raw_num_examples = data.get("num_examples")
        if isinstance(raw_num_examples, int):
            return raw_num_examples
        targets = data.get("y")
        if hasattr(targets, "__len__"):
            return len(cast(Any, targets))
        features = data.get("x", data.get("X"))
        if hasattr(features, "__len__"):
            return len(cast(Any, features))
    return 0
