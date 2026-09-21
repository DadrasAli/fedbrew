"""Experiment component factory built on top of registries."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from fedbrew.clients.base import ClientUpdate
from fedbrew.clients.lazy_pool import LazyClientPool
from fedbrew.clients.torch_delta_sgd_client import DEFAULT_DELTA, DEFAULT_GAMMA, DEFAULT_THETA_0
from fedbrew.clients.torch_fedlalr_client import DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_EPSILON

# The four rule sets below are config.py's, not this module's. They were
# defined in both, with different members -- see the comment beside them there
# and FINDINGS.csv P03-F04. Imported rather than re-exported deliberately:
# `tests/test_client_rule_sets_are_shared.py` fails on a second definition
# anywhere under fedbrew/.
from fedbrew.core.config import (
    CENTRALIZED_CLIENT_RULES,
    FEDAVG_ENGINE_CLIENT_RULES,
    FEDAVG_FT_CLIENT_RULES,
    FIXED_LR_SGD_CLIENT_RULES,
    FROZEN_WEIGHTING_CLIENT_RULES,
    UPDATE_MODE_CLIENT_RULES,
    UPDATE_MODE_OPTIONAL_CLIENT_RULES,
    FullConfig,
)
from fedbrew.core.federated_state import active_target_weighting_refusal
from fedbrew.core.paths import resolve_data_path
from fedbrew.core.refusal import RunRefused, yaml_number_cause
from fedbrew.core.registry import (
    BUILTIN,
    ModelFactory,
    Registry,
    client_updates,
    datasets,
    models,
    register_builtin_components,
    server_strategies,
    tasks,
)
from fedbrew.core.torch_utils import persistent_buffer_keys
from fedbrew.data.dataset import FederatedDataset
from fedbrew.servers.base import ServerStrategy
from fedbrew.servers.fedavg import SUPPORTED_AGGREGATION_WEIGHTING
from fedbrew.tasks.base import TaskAdapter

T = TypeVar("T")

FEDOPT_SERVER_STRATEGIES = {"fedavgm", "fedadam", "fedyogi", "fedadagrad", "fedopt"}
SCAFFOLD_SERVER_STRATEGIES = {"scaffold"}
CENTRALIZED_SERVER_STRATEGIES = {"centralized"}
SCAFFOLD_CLIENT_RULES = {"scaffold"}
#: Delta-SGD (arXiv:2306.11201). A client-only algorithm: the server is plain
#: FedAvg, and the step size is measured from the local smoothness rather than
#: configured, so no client.learning_rate is accepted.
DELTA_SGD_CLIENT_RULES = {"delta_sgd"}
#: FedLALR (arXiv:2309.09719). Server and client are a matched pair: the
#: server synchronizes the momentum and second moment that the client's local
#: AMSGrad reads, so neither half works with anything else.
FEDLALR_SERVER_STRATEGIES = {"fedlalr"}
FEDLALR_CLIENT_RULES = {"fedlalr"}
LOCAL_TRAINING_CLIENT_RULES = {
    "local_adamw",
    "fedprox",
    *FIXED_LR_SGD_CLIENT_RULES,
    *SCAFFOLD_CLIENT_RULES,
    *DELTA_SGD_CLIENT_RULES,
    *FEDLALR_CLIENT_RULES,
    *FEDAVG_FT_CLIENT_RULES,
}


def is_extension(component: Registry[Any], name: str) -> bool:
    """Whether a name was registered by an extension rather than the package.

    The dispatch tables above enumerate built-in names, and an out-of-tree
    component can appear in none of them. So the branch is taken on where the
    registration came from rather than on membership: everything the package
    ships keeps the dispatch it had, and everything else takes the documented
    extension contract (chapter 12).
    """

    return component.exists(name) and component.origin(name) != BUILTIN


def _declared(component: Registry[Any], name: str, extra: Mapping[str, Any]) -> dict[str, Any]:
    """The keys this component declared at registration, as the config set them.

    Declared and forwarded are one act: `Registry.register(config_keys=...)`
    is what makes a key loadable in the component's block, and this is what
    hands it to the factory. A declared key the config omits is absent rather
    than None, so the component's own default applies.
    """

    return {key: extra[key] for key in sorted(component.config_keys(name)) if key in extra}


def is_centralized(config: FullConfig) -> bool:
    """Return whether the run pools every client into one centralized client."""

    return (
        config.server.strategy in CENTRALIZED_SERVER_STRATEGIES
        or config.client.update_rule in CENTRALIZED_CLIENT_RULES
    )


@dataclass(slots=True)
class ExperimentComponents:
    """Concrete components needed to run one configured experiment."""

    server: ServerStrategy
    clients: Mapping[str, ClientUpdate]
    task: TaskAdapter
    dataset: FederatedDataset
    model_factory: ModelFactory
    config: FullConfig

    @property
    def client(self) -> Mapping[str, ClientUpdate]:
        """Compatibility alias for older call sites."""

        return self.clients


def build_components(config: FullConfig) -> ExperimentComponents:
    """Build all components required by the FL loop."""

    register_builtin_components()
    task_factory = _get_registered(tasks, config.task.name, "task")
    dataset_factory = _get_registered(datasets, config.data.name, "data backend")
    model_factory = _get_registered(models, config.model.name, "model")
    server_factory = _get_registered(
        server_strategies,
        config.server.strategy,
        "server strategy",
    )
    client_factory = _get_registered(
        client_updates,
        config.client.update_rule,
        "client update_rule",
    )

    dataset = _build_dataset(config, dataset_factory)
    model_config = _model_config(config, dataset)
    task = _build_task(config, task_factory, model_config, dataset)
    _refuse_federated_buffers(task, model_config)
    server = _build_server(config, task, server_factory, model_config)
    clients = _build_clients(config, task, dataset, client_factory, model_config)
    return ExperimentComponents(
        server=server,
        clients=clients,
        task=task,
        dataset=dataset,
        model_factory=model_factory,
        config=config,
    )


def _refuse_federated_buffers(task: TaskAdapter, model_config: Mapping[str, Any]) -> None:
    """Refuse a model that would federate registered buffers.

    WeightedStateAccumulator sees tensors and not a model, so it cannot tell a
    buffer from a parameter. Its handling of the two is one-sided by accident:
    a non-floating buffer that differs between clients raises -- which is the
    BatchNorm counter, and the loud case docs/06 4 documents -- while a
    *floating* buffer is added into the running sum like a weight. So a
    BatchNorm model whose clients happen to take the same number of steps,
    which equal shard sizes with drop_last or max_local_steps produce, averages
    running_mean and running_var across non-IID clients silently, and FedAdam
    or FedYogi then applies an adaptive update to running statistics.

    This is the one place with both the model and the federated state in hand,
    and it runs once per run rather than once per fit. Every model the
    repository ships returns nothing here -- the five classification builders
    have no buffers at all, and the two LLM models' buffers are non-persistent
    -- so this refuses only a model that does not exist yet.

    Scoped to the *federated* state, not the model: hf_causal_lm_lora
    federates adapter tensors only, so a buffer elsewhere in that model is not
    this function's business.

    Raises:
        ValueError: If any federated key is a persistent registered buffer.
    """

    model = task.build_model(model_config)
    federated = set(task.get_federated_model_state(model))
    offending = [name for name in persistent_buffer_keys(model) if name in federated]
    if not offending:
        return
    raise RunRefused(
        "these federated state keys are registered buffers, not parameters: "
        + ", ".join(offending)
        + ". Averaging them is not defined: a running statistic averaged over "
        "non-IID clients describes no client, and an integer counter has no "
        "mean at all. Use GroupNorm rather than BatchNorm (docs/06 section 4), "
        "or override get_federated_model_state in a task adapter to say which "
        "keys are federated (docs/12 section 6)."
    )


def _build_task(
    config: FullConfig,
    task_factory: Callable[..., TaskAdapter],
    model_config: Mapping[str, Any] | None = None,
    dataset: FederatedDataset | None = None,
) -> TaskAdapter:
    if is_extension(tasks, config.task.name):
        return task_factory(**_extension_task_kwargs(config, model_config, dataset))
    if config.task.name in {"classification", "causal_lm"}:
        kwargs: dict[str, Any] = {
            "model_config": dict(model_config or _model_config(config)),
            "batch_size": config.client.batch_size,
            "device": config.runtime.device,
            "dataloader_config": _dataloader_config(config),
        }
        performance = _performance_config(config)
        # Both tasks cache models under it. It reached only the classification
        # task, so every LLM config that set reuse_model: true got a fresh
        # from_pretrained per client per phase -- and, under LoRA, a fresh draw
        # from the process-wide RNG with it.
        kwargs["reuse_model"] = _performance_flag(performance, "reuse_model", True)
        _refuse_amp_a_task_cannot_honour(config)
        if config.task.name in AMP_AWARE_TASKS:
            kwargs["use_amp"] = bool(config.runtime.use_amp)
            kwargs["fast_batching"] = _performance_flag(performance, "fast_batching", True)
            kwargs["eval_batch_size"] = _eval_batch_size(config)
        return task_factory(**kwargs)
    return task_factory()


#: Tasks whose adapter takes ``use_amp`` and has an autocast/GradScaler path.
#: Read by `_build_task`, which used to spell the same membership as
#: ``config.task.name == "classification"``, and by
#: `_refuse_amp_a_task_cannot_honour` -- so which tasks understand AMP is one
#: fact with one home rather than a condition and an unstated assumption.
#:
#: An extension task is never in it, and that is correct: `EXTENSION_TASK_KEYS`
#: does not carry ``use_amp`` either, so an out-of-tree adapter cannot receive
#: the flag and must not be told a run honoured it.
AMP_AWARE_TASKS = {"classification"}


def _refuse_amp_a_task_cannot_honour(config: FullConfig) -> None:
    """Refuse ``runtime.use_amp: true`` on a task that has no AMP path.

    `TorchCausalLMTask.__init__` takes no ``use_amp`` and its `train_step` has
    no `autocast` or `GradScaler`, so the flag was accepted, echoed into
    `run.json`, and did nothing -- a run.json saying the run used mixed
    precision when it ran in fp32 throughout. The numbers stayed right, fp32
    being the reference, which is what made it quiet.

    It also made two guards read as protection they do not give: the
    ``getattr(task, "_scaler", None)`` checks in `delta_sgd` and `fedlalr` can
    never trip on a task that has no scaler to find.

    Refused rather than implemented: adding autocast to the causal-LM task is a
    feature and would move every LLM number, while every shipped LLM config
    already sets ``use_amp: false``. See FINDINGS.csv P03-F07.

    Raises:
        ValueError: If ``runtime.use_amp`` is true and the task is not in
            `AMP_AWARE_TASKS`.
    """

    if not config.runtime.use_amp or config.task.name in AMP_AWARE_TASKS:
        return
    raise RunRefused(
        f"runtime.use_amp is true and task {config.task.name!r} has no mixed-precision "
        f"path, so the run would execute in fp32 while run.json recorded AMP. Only "
        f"{', '.join(sorted(AMP_AWARE_TASKS))} implements it. Set runtime.use_amp to false."
    )


#: What an out-of-tree task adapter is built with. The shared half of what
#: the two built-in tasks receive -- `use_amp` and `fast_batching` are the
#: classification task's own -- plus the dataset's metadata, which is where a
#: generated problem states what a run on it is scored against. Documented in
#: chapter 12; tests/test_extension_components.py pins the names.
EXTENSION_TASK_KEYS: tuple[str, ...] = (
    "model_config",
    "batch_size",
    "eval_batch_size",
    "device",
    "dataloader_config",
    "reuse_model",
    "dataset_metadata",
)


def _extension_task_kwargs(
    config: FullConfig,
    model_config: Mapping[str, Any] | None,
    dataset: FederatedDataset | None,
) -> dict[str, Any]:
    """Build the keyword arguments an extension task adapter is handed."""

    performance = _performance_config(config)
    return {
        "model_config": dict(model_config or _model_config(config)),
        "batch_size": config.client.batch_size,
        "eval_batch_size": _eval_batch_size(config),
        "device": config.runtime.device,
        "dataloader_config": _dataloader_config(config),
        "reuse_model": _performance_flag(performance, "reuse_model", True),
        # The manifest, as the backend reports it -- including `reference`,
        # the reference optimum a generated problem is scored against. An
        # analytic task reads it here instead of closing over a spec, which
        # is what kept run.json from recording what a run was measured
        # against. Empty for a backend that reports nothing.
        "dataset_metadata": dict(dataset.get_metadata()) if dataset is not None else {},
    }


def _build_dataset(
    config: FullConfig,
    dataset_factory: Callable[..., FederatedDataset],
) -> FederatedDataset:
    dataset = _build_source_dataset(config, dataset_factory)
    if not is_centralized(config):
        return dataset

    from fedbrew.data.centralized_dataset import CentralizedFederatedDataset

    return CentralizedFederatedDataset(dataset)


def _build_source_dataset(
    config: FullConfig,
    dataset_factory: Callable[..., FederatedDataset],
) -> FederatedDataset:
    if is_extension(datasets, config.data.name):
        return dataset_factory(**_extension_dataset_kwargs(config))
    if config.data.name == "synthetic_classification":
        kwargs = _data_config(config)
        kwargs.setdefault("seed", config.experiment.seed)
        return dataset_factory(**kwargs)
    if config.data.name == "manifest_dataset":
        data_path = config.data.path
        if not data_path:
            raise RunRefused("data.path is required for manifest_dataset")
        return dataset_factory(
            resolve_data_path(data_path),
            shard_cache_bytes=_shard_cache_bytes(config),
        )
    return dataset_factory()


#: The named `data` fields an out-of-tree backend is handed, when the config
#: sets them, beside `seed` and whatever the backend declared at registration.
#: `path` is resolved first, the way manifest_dataset's is.
EXTENSION_DATASET_FIELDS: tuple[str, ...] = (
    "path",
    "num_clients",
    "samples_per_client",
    "input_dim",
    "num_classes",
)


def _extension_dataset_kwargs(config: FullConfig) -> dict[str, Any]:
    """Build the keyword arguments an extension dataset backend is handed."""

    kwargs: dict[str, Any] = {}
    for name in EXTENSION_DATASET_FIELDS:
        value = getattr(config.data, name)
        if value is None:
            continue
        kwargs[name] = resolve_data_path(value) if name == "path" else value
    kwargs.update(_declared(datasets, config.data.name, config.data.extra))
    # Last, and unconditional: a backend that generates its own data needs the
    # run's seed, and no config key carries it into this block.
    kwargs["seed"] = config.experiment.seed
    return kwargs


def _shard_cache_bytes(config: FullConfig) -> int:
    """Return the client-shard cache budget in bytes."""

    from fedbrew.data.manifest_dataset import DEFAULT_SHARD_CACHE_BYTES

    if is_centralized(config):
        # The pooled view reads every shard exactly once and then keeps the
        # concatenation resident, so a shard cache would only hold a second copy.
        return 0

    performance = _performance_config(config)
    configured = performance.get("shard_cache_bytes")
    if configured is None:
        return DEFAULT_SHARD_CACHE_BYTES
    if isinstance(configured, bool) or not isinstance(configured, int):
        raise RunRefused("runtime.performance.shard_cache_bytes must be an integer")
    if configured < 0:
        raise RunRefused("runtime.performance.shard_cache_bytes must be non-negative")
    return configured


def _build_server(
    config: FullConfig,
    task: TaskAdapter,
    server_factory: Callable[..., ServerStrategy],
    model_config: Mapping[str, Any] | None = None,
) -> ServerStrategy:
    resolved_model_config = dict(model_config or _model_config(config))
    common_kwargs = {
        "task": task,
        "model_config": resolved_model_config,
        "participation_rate": config.server.participation_rate,
        "participation_probability": config.server.participation_probability,
        "seed": config.experiment.seed,
        "metrics": config.server.metrics,
        "aggregation_weighting": _aggregation_weighting(config),
    }
    if is_extension(server_strategies, config.server.strategy):
        # Everything FedAvg is built with, plus the keys this strategy
        # declared. A strategy with server-side hyperparameters therefore
        # receives them named rather than digging in a config it was not
        # handed.
        return server_factory(
            **common_kwargs,
            **_declared(server_strategies, config.server.strategy, config.server.extra),
        )
    if config.server.strategy in {
        "fedavg",
        *SCAFFOLD_SERVER_STRATEGIES,
        *CENTRALIZED_SERVER_STRATEGIES,
    }:
        return server_factory(**common_kwargs)
    if config.server.strategy in FEDLALR_SERVER_STRATEGIES:
        # epsilon comes from the client block, not a server twin: the server
        # only needs it to seed v_hat_{-1} = epsilon^2, and it is the same
        # number the client's AMSGrad is floored by.
        return server_factory(
            **common_kwargs,
            epsilon=_client_extra_positive_float_with_default(config, "epsilon", DEFAULT_EPSILON),
        )
    if config.server.strategy in FEDOPT_SERVER_STRATEGIES:
        # `fedopt` is the only strategy whose optimizer the config chooses. For
        # the four named ones the builder supplies its own, which used to be
        # decided here as well -- `_server_optimizer` returned the strategy
        # name and the builders' setdefault could never fire. One fact, two
        # places, one of them dead. P10-F33.
        optimizer = (
            {"server_optimizer": _server_optimizer(config)}
            if config.server.strategy == "fedopt"
            else {}
        )
        return server_factory(
            **common_kwargs,
            **optimizer,
            server_learning_rate=_server_extra_float(config, "server_learning_rate"),
            beta1=_server_extra_float(config, "beta1"),
            # None, not a placeholder, for the hyperparameters this optimizer
            # never reads: validate_config has already refused a config that
            # sets one, and required every one that is read. P01-F07.
            beta2=_server_extra_optional_float(config, "beta2"),
            tau=_server_extra_optional_float(config, "tau"),
        )
    return server_factory()


def _build_clients(
    config: FullConfig,
    task: TaskAdapter,
    dataset: FederatedDataset,
    client_factory: Callable[..., ClientUpdate],
    model_config: Mapping[str, Any] | None = None,
) -> Mapping[str, ClientUpdate]:
    extension = is_extension(client_updates, config.client.update_rule)
    if extension or config.client.update_rule in LOCAL_TRAINING_CLIENT_RULES:
        client_ids = dataset.list_clients()

        def build_client(client_id: str) -> ClientUpdate:
            if extension:
                kwargs = _extension_client_kwargs(config, task, dataset, client_id, model_config)
            else:
                kwargs = _training_client_kwargs(config, task, dataset, client_id, model_config)
            return client_factory(**kwargs)

        if _use_lazy_clients(config):
            return LazyClientPool(client_ids, build_client)

        return {client_id: build_client(client_id) for client_id in client_ids}

    return {client_id: client_factory() for client_id in dataset.list_clients()}


#: What an out-of-tree client update rule is built with: the base every
#: locally-training built-in rule receives, before the per-rule gates that
#: enumerate built-in names. `learning_rate` may be None -- a rule that
#: derives its own step size is not made to state one, and preflight does not
#: require one for an extension rule.
EXTENSION_CLIENT_KEYS: tuple[str, ...] = (
    "client_id",
    "task",
    "model_config",
    "client_data",
    "local_iterations",
    "batch_size",
    "eval_batch_size",
    "learning_rate",
    "device",
    "metrics",
    "base_seed",
    "train_shuffle",
    "eval_shuffle",
    "drop_last",
    "total_rounds",
)


def _extension_client_kwargs(
    config: FullConfig,
    task: TaskAdapter,
    dataset: FederatedDataset,
    client_id: str,
    model_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the keyword arguments an extension client update rule is handed."""

    return {
        "client_id": client_id,
        "task": task,
        "model_config": dict(model_config or _model_config(config)),
        "client_data": dataset.get_client_data(client_id),
        "local_iterations": config.client.local_iterations,
        "batch_size": config.client.batch_size,
        "eval_batch_size": _eval_batch_size(config),
        "learning_rate": config.client.learning_rate,
        "device": config.runtime.device,
        "metrics": config.client.metrics,
        "base_seed": config.experiment.seed,
        "train_shuffle": _client_extra_bool(config, "train_shuffle", True),
        "eval_shuffle": _client_extra_bool(config, "eval_shuffle", False),
        "drop_last": _client_extra_bool(config, "drop_last", False),
        # A rule that schedules anything over the run needs the horizon, and
        # no other argument carries it.
        "total_rounds": config.server.global_rounds,
        **_declared(client_updates, config.client.update_rule, config.client.extra),
    }


def _training_client_kwargs(
    config: FullConfig,
    task: TaskAdapter,
    dataset: FederatedDataset,
    client_id: str,
    model_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if config.client.update_rule in DELTA_SGD_CLIENT_RULES:
        # TorchSGDClient wants a positive learning_rate, but Delta-SGD never
        # reads it: the rule starts at eta_0 and re-derives every step from
        # there. Passing eta_0 keeps the base class honest without inventing a
        # second knob that would look tunable and would not be.
        learning_rate = _client_extra_positive_float(config, "eta_0")
    else:
        learning_rate = _learning_rate(config)

    kwargs: dict[str, Any] = {
        "client_id": client_id,
        "task": task,
        "model_config": dict(model_config or _model_config(config)),
        "client_data": dataset.get_client_data(client_id),
        "local_iterations": config.client.local_iterations,
        "batch_size": config.client.batch_size,
        "eval_batch_size": _eval_batch_size(config),
        "learning_rate": learning_rate,
        "device": config.runtime.device,
        "metrics": config.client.metrics,
        "base_seed": config.experiment.seed,
        "train_shuffle": _client_extra_bool(config, "train_shuffle", True),
        "eval_shuffle": _client_extra_bool(config, "eval_shuffle", False),
        "drop_last": _client_extra_bool(config, "drop_last", False),
    }
    if config.client.update_rule in FIXED_LR_SGD_CLIENT_RULES:
        kwargs.update(
            momentum=_client_extra_float(config, "momentum"),
            weight_decay=_client_extra_float(config, "weight_decay"),
            nesterov=_client_extra_bool(config, "nesterov"),
            learning_rate_schedule=_client_extra_choice(config, "learning_rate_schedule"),
            min_learning_rate=_client_extra_float(config, "min_learning_rate"),
            total_rounds=config.server.global_rounds,
        )

    kwargs.update(_update_mode_kwargs(config))
    if config.client.update_rule in FEDAVG_ENGINE_CLIENT_RULES:
        kwargs.update(
            max_local_steps=_client_extra_optional_positive_int(
                config,
                "max_local_steps",
            ),
            max_grad_norm=_client_extra_optional_positive_float(
                config,
                "max_grad_norm",
            ),
        )
    if config.client.update_rule == "local_adamw":
        kwargs.update(
            weight_decay=_client_extra_float(config, "weight_decay"),
            beta1=_client_extra_float(config, "beta1"),
            beta2=_client_extra_float(config, "beta2"),
            epsilon=_client_extra_float(config, "epsilon"),
            learning_rate_schedule=_client_extra_choice(config, "learning_rate_schedule"),
            min_learning_rate=_client_extra_float(config, "min_learning_rate"),
            total_rounds=config.server.global_rounds,
            max_local_steps=_client_extra_optional_positive_int(config, "max_local_steps"),
        )
    if config.client.update_rule == "fedprox":
        kwargs["proximal_mu"] = _proximal_mu(config)

    if config.client.update_rule in FEDAVG_FT_CLIENT_RULES:
        kwargs.update(
            finetune_epochs=_client_extra_positive_int(config, "finetune_epochs"),
            # Optional: an unset fine-tuning rate inherits the training rate,
            # which is the conventional default for this baseline.
            finetune_learning_rate=_client_extra_optional_positive_float(
                config,
                "finetune_learning_rate",
            ),
        )

    if config.client.update_rule in FEDLALR_CLIENT_RULES:
        kwargs.update(
            beta1=_client_extra_positive_float_with_default(config, "beta1", DEFAULT_BETA1),
            beta2=_client_extra_positive_float_with_default(config, "beta2", DEFAULT_BETA2),
            epsilon=_client_extra_positive_float_with_default(config, "epsilon", DEFAULT_EPSILON),
        )

    if config.client.update_rule in DELTA_SGD_CLIENT_RULES:
        kwargs.update(
            update_mode=_client_extra_choice_with_default(
                config, "update_mode", "sequential_epoch"
            ),
            frozen_gradient_weighting=_client_extra_choice_with_default(
                config, "frozen_gradient_weighting", "examples"
            ),
            eta_0=_client_extra_positive_float(config, "eta_0"),
            theta_0=_client_extra_positive_float_with_default(config, "theta_0", DEFAULT_THETA_0),
            gamma=_client_extra_positive_float_with_default(config, "gamma", DEFAULT_GAMMA),
            delta=_client_extra_positive_float_with_default(config, "delta", DEFAULT_DELTA),
            eta_max=_client_extra_optional_positive_float(config, "eta_max"),
            max_grad_norm=_client_extra_optional_positive_float(
                config,
                "max_grad_norm",
            ),
        )

    return kwargs


def _update_mode_kwargs(config: FullConfig) -> dict[str, Any]:
    """The engine's mode for a rule that takes one, and the weighting for a rule that states it.

    Its own gate rather than a branch of the fixed-rate one, as the two sets are
    in `config.py`: which modes a rule runs does not follow from how it sets its
    learning rate.
    """

    kwargs: dict[str, Any] = {}
    if config.client.update_rule in UPDATE_MODE_OPTIONAL_CLIENT_RULES:
        # None is this rule's own state, not a missing key: see the set.
        kwargs["update_mode"] = config.client.extra.get("update_mode")
    elif config.client.update_rule in UPDATE_MODE_CLIENT_RULES:
        kwargs["update_mode"] = _client_extra_choice(config, "update_mode")
    if config.client.update_rule in FROZEN_WEIGHTING_CLIENT_RULES:
        kwargs["frozen_gradient_weighting"] = _client_extra_choice(
            config, "frozen_gradient_weighting"
        )
    return kwargs


def _get_registered(
    registry: Registry[T],
    name: str,
    label: str,
) -> T:
    if not registry.exists(name):
        raise RunRefused(f"Unknown {label}: {name}")
    return registry.get(name)


def _model_config(
    config: FullConfig,
    dataset: FederatedDataset | None = None,
) -> dict[str, Any]:
    values: dict[str, Any] = {"name": config.model.name}
    for name in ("input_dim", "hidden_dim", "num_classes"):
        value = getattr(config.model, name)
        if value is not None:
            values[name] = value
    values.update(config.model.extra)
    if dataset is not None:
        metadata = dataset.get_metadata()
        mismatch = model_data_shape_mismatch(values, metadata)
        if mismatch is not None:
            raise RunRefused(mismatch)
        if config.task.name == "causal_lm":
            _add_causal_manifest_metadata(values, metadata)
            refusal = active_target_weighting_refusal(
                config.client.update_rule, config.task.name, values, values.get("dataset_task")
            )
            if refusal is not None:
                raise RunRefused(refusal)
    return values


#: The two names `DataConfig` and `ModelConfig` both carry, and that a dataset
#: also reports in `get_metadata()`. They are the same word for two different
#: jobs -- `data.*` sizes the in-memory synthetic backend, `model.*` sizes the
#: input and output layers -- and nothing tied them together: `_model_config`
#: reads only `config.model.*` and `_data_config` only `config.data.*`, so an
#: edit to one block without the other produced either a shape-mismatch
#: traceback from inside a builder or, for `num_classes` set high, output units
#: that are never predicted and never mentioned. P11-F01.
SHARED_SHAPE_FIELDS: tuple[str, ...] = ("input_dim", "num_classes")


def model_data_shape_mismatch(
    model_values: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str | None:
    """Why this model is not shaped for this data, if it is not.

    Compared against the dataset's own `get_metadata()` rather than against
    `config.data.*`, which is what the finding suggested. `data.input_dim` and
    `data.num_classes` are set by exactly one shipped config -- every
    manifest-backed one leaves them unset, correctly, because the dimensions
    come from the manifest on disk -- so comparing the two config blocks would
    guard the one case that has never drifted and none of the cases that can.
    The metadata is the same number for the in-memory backend and is the
    manifest's for every other, so one comparison covers both.

    Args:
        model_values: The model keyword arguments, as `_model_config` collects
            them.
        metadata: The dataset's `get_metadata()`.

    Returns:
        A message naming the key and both values, or None when every key the
        two both declare agrees. A key only one of them declares is not a
        disagreement: `femnist_resnet18` takes no `input_dim`, and a manifest
        need not describe a shape its models take as given.
    """

    for name in SHARED_SHAPE_FIELDS:
        if name not in model_values or name not in metadata:
            continue
        configured = _as_shape(model_values[name])
        generated = _as_shape(metadata[name])
        if configured == generated:
            continue
        return (
            f"model.{name} does not match the dataset: {configured!r} != "
            f"{generated!r}. The model is sized for its data or it is sized "
            "for other data; change the config, or regenerate the dataset."
        )
    return None


def _as_shape(value: Any) -> Any:
    """A shape as it compares: a sequence of ints becomes a list of them.

    MNIST writes `input_dim: 784` when `flatten` is on and `[1, 28, 28]` when
    it is not, and a config's YAML gives a list where a manifest's JSON gives
    one too -- but a tuple reaches here from a builder's own defaults.
    """

    if isinstance(value, list | tuple):
        return [_as_shape(item) for item in value]
    return value


def _add_causal_manifest_metadata(
    values: dict[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    fields = (
        "task",
        "vocab_size",
        "tokenizer_vocabulary_size",
        "sequence_length",
        "stride",
        "padding_token_id",
        "eos_token_id",
        "tokenizer_identifier",
        "tokenizer_revision",
        "tokenizer_requested_revision",
        "tokenizer_resolved_revision",
        "tokenizer_asset_manifest",
        "asset_manifest",
        "corpus_hash",
        "source_sha256",
        "local_files_only",
        "trust_remote_code",
        "offline",
        "ignore_index",
    )
    for name in fields:
        if name in metadata:
            values[f"dataset_{name}"] = metadata[name]
    if "padding_token_id" in metadata:
        # Same rule as model.sequence_length and model.vocab_size, which the
        # builder refuses on mismatch (hf_causal_lm._expected_sequence_length,
        # _expected_vocab_size). This one used to overwrite in silence, and it
        # overwrote before the builder ran, so no later check could see that a
        # config had asked for a different token: the value reaching the
        # attention mask and the loss mask was the manifest's either way.
        generated = metadata["padding_token_id"]
        configured = values.get("pad_token_id")
        if configured is not None and configured != generated:
            raise RunRefused(
                "model.pad_token_id does not match the generated dataset: "
                f"{configured} != {generated}"
            )
        values["pad_token_id"] = generated


def _data_config(config: FullConfig) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in ("num_clients", "samples_per_client", "input_dim", "num_classes"):
        value = getattr(config.data, name)
        if value is not None:
            values[name] = value
    values.update(config.data.extra)
    return values


#: Evaluation runs under no_grad, so its batch size carries no scientific
#: meaning -- only throughput. Small training batches would otherwise leave the
#: GPU idle through every evaluation pass.
DEFAULT_MIN_EVAL_BATCH_SIZE = 256


def _eval_batch_size(config: FullConfig) -> int:
    """Return the evaluation batch size, defaulting well above the train size."""

    configured = config.client.extra.get("eval_batch_size")
    if configured is None:
        if config.task.name == "causal_lm":
            # A causal-LM forward produces batch x sequence_length x vocabulary
            # logits, which reaches tens of gigabytes at the classification
            # default. Evaluate at the training batch size instead.
            return int(config.client.batch_size)
        return max(int(config.client.batch_size), DEFAULT_MIN_EVAL_BATCH_SIZE)
    if isinstance(configured, bool) or not isinstance(configured, int):
        raise RunRefused("client.eval_batch_size must be an integer")
    if configured <= 0:
        raise RunRefused("client.eval_batch_size must be positive")
    return configured


def _performance_config(config: FullConfig) -> Mapping[str, Any]:
    performance = config.runtime.extra.get("performance", {})
    return performance if isinstance(performance, Mapping) else {}


def _performance_flag(
    performance: Mapping[str, Any],
    name: str,
    default: bool,
) -> bool:
    value = performance.get(name, default)
    if not isinstance(value, bool):
        raise RunRefused(f"runtime.performance.{name} must be a boolean")
    return value


def _dataloader_config(config: FullConfig) -> dict[str, Any]:
    performance = _performance_config(config)
    dataloader = performance.get("dataloader", {})
    if not isinstance(dataloader, Mapping):
        return {}
    return dict(dataloader)


def _learning_rate(config: FullConfig) -> float:
    if config.client.learning_rate is None:
        raise RunRefused("client.learning_rate must be configured")
    return config.client.learning_rate


def _proximal_mu(config: FullConfig) -> float:
    return _client_extra_float(config, "proximal_mu")


def _client_extra_bool(
    config: FullConfig,
    name: str,
    default: bool | None = None,
) -> bool:
    if name not in config.client.extra and default is None:
        raise RunRefused(f"client.{name} must be configured")
    value = config.client.extra.get(name, default)
    if not isinstance(value, bool):
        raise RunRefused(f"client.{name} must be a bool")
    return value


def _client_extra_float(config: FullConfig, name: str) -> float:
    if name not in config.client.extra:
        raise RunRefused(f"client.{name} must be configured")
    value = config.client.extra[name]
    if isinstance(value, bool) or not isinstance(value, float | int):
        raise RunRefused(f"client.{name} must be numeric" + yaml_number_cause(value))
    return float(value)


def _client_extra_choice(config: FullConfig, name: str) -> str:
    value = config.client.extra.get(name)
    if not isinstance(value, str) or not value:
        raise RunRefused(f"client.{name} must be a non-empty string")
    return value


def _client_extra_choice_with_default(
    config: FullConfig,
    name: str,
    default: str,
) -> str:
    """Return a string client extra, falling back to a documented default."""

    value = config.client.extra.get(name, default)
    if not isinstance(value, str) or not value:
        raise RunRefused(f"client.{name} must be a non-empty string")
    return value


def _client_extra_positive_float_with_default(
    config: FullConfig,
    name: str,
    default: float,
) -> float:
    """Return a positive float client extra, falling back to a paper default."""

    value = config.client.extra.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise RunRefused(f"client.{name} must be a positive number")
    return float(value)


def _client_extra_positive_int(config: FullConfig, name: str) -> int:
    """Return a required positive-integer client extra."""

    if name not in config.client.extra:
        raise RunRefused(f"client.{name} must be configured")
    value = config.client.extra[name]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RunRefused(f"client.{name} must be a positive integer")
    return value


def _client_extra_optional_positive_int(
    config: FullConfig,
    name: str,
) -> int | None:
    value = config.client.extra.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RunRefused(f"client.{name} must be a positive integer or null")
    return value


def _client_extra_optional_positive_float(
    config: FullConfig,
    name: str,
) -> float | None:
    value = config.client.extra.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise RunRefused(f"client.{name} must be a positive number or null")
    return float(value)


def _use_lazy_clients(config: FullConfig) -> bool:
    """Build clients on demand for a manifest dataset, eagerly otherwise.

    Derived, not configurable. A manifest dataset can hold thousands of
    clients whose shards are read from disk, so materialising every client up
    front is what the LazyClientPool exists to avoid; the in-process synthetic
    dataset holds a handful already in memory, where the pool would only add
    indirection. Neither case has a reason to want the other's behaviour, and
    the client.lazy_clients override that used to allow it was set by no config
    and no test.
    """

    return config.data.name == "manifest_dataset"


def _server_optimizer(config: FullConfig) -> str:
    """The optimizer a ``strategy: fedopt`` config names.

    Called only for `fedopt`. The four named strategies carry their own in the
    builder that registers them, so the branch that used to return
    ``config.server.strategy`` here is gone rather than merely unused.
    """

    raw_optimizer = config.server.extra.get("server_optimizer")
    if not isinstance(raw_optimizer, str) or not raw_optimizer.strip():
        raise RunRefused("server.strategy=fedopt requires server_optimizer")
    return raw_optimizer


def _aggregation_weighting(config: FullConfig) -> str:
    """Return how client model states are weighted in the aggregate.

    Absent from a config means "examples", so every baseline written before
    this knob existed keeps aggregating exactly as it did.
    """

    value = config.server.extra.get("aggregation_weighting", "examples")
    if not isinstance(value, str) or value not in SUPPORTED_AGGREGATION_WEIGHTING:
        raise RunRefused(
            "server.aggregation_weighting must be one of: "
            + ", ".join(sorted(SUPPORTED_AGGREGATION_WEIGHTING))
        )
    return value


def _server_extra_float(config: FullConfig, name: str) -> float:
    if name not in config.server.extra:
        raise RunRefused(f"server.{name} must be configured")
    value = config.server.extra[name]
    if isinstance(value, bool) or not isinstance(value, float | int):
        raise RunRefused(f"server.{name} must be numeric" + yaml_number_cause(value))
    return float(value)


def _server_extra_optional_float(config: FullConfig, name: str) -> float | None:
    """The value of a server setting, or None when the config omits it."""

    if name not in config.server.extra:
        return None
    return _server_extra_float(config, name)


def _client_extra_positive_float(config: FullConfig, name: str) -> float:
    value = _client_extra_float(config, name)
    if value <= 0.0:
        raise RunRefused(f"client.{name} must be positive")
    return value
