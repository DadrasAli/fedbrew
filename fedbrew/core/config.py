"""YAML configuration loading for benchmark experiments."""

from __future__ import annotations

import difflib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, cast

from fedbrew.core.refusal import RunRefused, yaml_number_cause
from fedbrew.servers.fedavg import SUPPORTED_AGGREGATION_WEIGHTING
from fedbrew.servers.fedopt import (
    FEDOPT_HYPERPARAMETERS,
    fedopt_bound_violation,
    unread_fedopt_hyperparameters,
)

try:
    import yaml as yaml_loader  # type: ignore[import-untyped]
except ModuleNotFoundError:  # pragma: no cover - used when PyYAML is unavailable.
    yaml_loader = None


@dataclass
class ExperimentConfig:
    """Top-level experiment metadata."""

    seed: int
    output_dir: str
    name: str = ""
    run_id: str | None = None
    use_run_subdir: bool = False
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    #: Components defined outside the package, loaded before any name in this
    #: config is checked: each entry a path ending in .py or a dotted module
    #: name, resolved like data.path. Part of the config rather than a flag,
    #: because the config is the whole description of the run and run.json
    #: is its record. fedbrew/core/extensions.py.
    extensions: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerConfig:
    """Server-side benchmark settings."""

    strategy: str
    global_rounds: int
    metrics: list[str]
    #: A fixed number of clients per round, ``ceil(rate x clients)``. Exactly one
    #: of this and ``participation_probability`` is set.
    participation_rate: float | None = None
    #: Each client joins each round independently with this probability, so the
    #: count varies by round and can be zero. One value for every client.
    participation_probability: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClientConfig:
    """Client-side benchmark settings."""

    update_rule: str
    #: Iterations of the local loop per selected client per round, from
    #: ``defaults.local_iterations``. What one iteration does is the rule's
    #: ``update_mode`` (K = this value, B = the client's training batches):
    #:
    #: - ``single_batch``: one optimizer step on the next mini-batch; the
    #:   loader starts over when it runs out. K updates.
    #: - ``sequential_epoch``: one pass over the client's train split, one
    #:   optimizer step per batch. K * B updates.
    #: - ``frozen_batch_gradients``: one pass whose batch gradients are all
    #:   evaluated at the iteration's starting point, then combined and applied
    #:   as one update. K updates.
    #: - ``full_gradient``: one optimizer step on the exact gradient of the
    #:   task's training loss over the whole train split, computed batch by
    #:   batch; ``batch_size`` only sets how much is in memory at once. K
    #:   updates.
    #:
    #: ``fedavg``, ``centralized``, ``fedavg_ft`` and ``delta_sgd`` take every
    #: ``update_mode``. ``fedprox``, ``scaffold``, ``fedlalr``, ``local_sgd``
    #: and ``local_adamw`` run their own loop, which is ``sequential_epoch``
    #: (also when the mode is unset), or ``full_gradient``. So arms differing
    #: in update shape are not comparable at equal ``local_iterations``
    #: (FINDINGS.csv POST-F15). ``max_local_steps`` caps the steps of
    #: ``local_adamw``, the one rule that honours it, in either of its modes.
    #: docs/04-configuration.md section 2.1.
    local_iterations: int
    batch_size: int
    metrics: list[str]
    learning_rate: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Dataset selection settings."""

    name: str = ""
    path: str | None = None
    num_clients: int | None = None
    samples_per_client: int | None = None
    input_dim: int | None = None
    num_classes: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskConfig:
    """Task adapter selection settings."""

    name: str


@dataclass
class ModelConfig:
    """Model selection settings."""

    name: str
    input_dim: int | None = None
    hidden_dim: int | None = None
    num_classes: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeConfig:
    """Runtime execution settings."""

    device: str
    use_amp: bool
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitEvaluationConfig:
    """When one data split is evaluated, and over which clients.

    Evaluation is the expensive part; aggregation is not. So these two knobs
    control cost only -- which metrics come out is fixed, and is identical for
    every split.
    """

    #: How often, in rounds.
    #:   <int>   rounds where round % every == 0, plus the final round
    #:   final   the final round only
    #:   never   not evaluated at all
    #: No default. The three splits do not share one -- EvaluationConfig
    #: supplies 10/participating, 5/all and 10/all -- so any value written
    #: here would be a fourth that nothing produces, and a reader of this
    #: dataclass would take it for the answer. It read ``= 1`` and ``= "all"``
    #: while no split used either.
    every: int | str
    #: Which clients. See parse_evaluation_client_scope for the four forms.
    clients: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CentralTestConfig:
    """The server's own pooled test shard, evaluated in one batched pass.

    A sibling of the three client splits rather than a flag on ``test``,
    because it is different data with a different cost, and either one can be
    wanted without the other. On datasets where the client test splits are a
    partition of this shard it is a cheaper second view of the same number; on
    datasets where the shard is separately held out (the LLM corpora split by
    conversation tree) it is the only measure of generalisation beyond the
    clients' own data.
    """

    #: No default, for the reason SplitEvaluationConfig has none: the
    #: EvaluationConfig factory supplies 10, and this read ``= 1``.
    every: int | str
    extra: dict[str, Any] = field(default_factory=dict)


#: Which model each evaluation pass measures.
#:   global    the aggregated server model, as every non-personalized arm does
#:   personal  each client's own model, which for a personalized update rule is
#:             what the algorithm actually produces
#:   both      one pass each, so a table can carry both columns
#: Personalized metrics are reported under a "personal_" split prefix --
#: personal_val_accuracy alongside val_accuracy -- which is what lets them
#: flow through the existing aggregation and dispersion statistics unchanged.
EVALUATION_MODEL_SCOPES = {"global", "personal", "both"}

#: Split-name prefix carrying the personalized pass's metrics and counts.
PERSONAL_SPLIT_PREFIX = "personal_"


@dataclass
class EvaluationConfig:
    """Post-aggregation evaluation, one block per data split.

    The roles are fixed and enforced elsewhere: train is an optimisation
    diagnostic, val is what model selection is allowed to look at, test is for
    reporting only.
    """

    train: SplitEvaluationConfig = field(
        default_factory=lambda: SplitEvaluationConfig(every=10, clients="participating")
    )
    val: SplitEvaluationConfig = field(
        default_factory=lambda: SplitEvaluationConfig(every=5, clients="all")
    )
    test: SplitEvaluationConfig = field(
        default_factory=lambda: SplitEvaluationConfig(every=10, clients="all")
    )
    central_test: CentralTestConfig = field(default_factory=lambda: CentralTestConfig(every=10))
    #: Which model the client passes measure. "global" keeps every existing
    #: config evaluating exactly what it evaluated before this field existed.
    model_scope: str = "global"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClientStatisticsConfig:
    """Statistics computed across clients from one evaluation pass.

    Each toggle is independent -- none of them implies or requires another.
    They are free: the expensive part is producing the per-client numbers, and
    every statistic below is then a pass over a list of floats. They are still
    optional because each one is a column in every round of every CSV.

    All of them apply to loss as well as accuracy, and to all three splits.
    """

    #: Write client_metrics.csv: one row per client per round. Off by default;
    #: at clients "all" that is 1.8M rows for a 500-round FEMNIST run, and the
    #: statistics below already summarise it.
    per_client_csv: bool = False
    std: bool = True
    variance: bool = False
    min: bool = True
    max: bool = True
    #: Mean over the worst N percent of clients -- lowest accuracies, highest
    #: losses. null or 0 turns it off. Emitted as e.g. test_accuracy_worst10.
    worst_percent: float | None = 10.0
    extra: dict[str, Any] = field(default_factory=dict)


#: The per-client metrics every evaluation pass reports. Every aggregate
#: column name is built from one of these, so the loop's aggregation and
#: client_metric_names below read the same tuple rather than each spelling the
#: pair out -- a metric added to one and not the other is the drift that makes
#: a name-based check worthless.
CLIENT_METRIC_BASES = ("loss", "accuracy")


def worst_percent_label(worst_percent: float) -> str:
    """Name fragment for the worst-N-percent column: 10 -> "10", 2.5 -> "2p5"."""

    return f"{float(worst_percent):g}".replace(".", "p")


def client_metric_names(
    split: str,
    statistics: ClientStatisticsConfig,
) -> set[str]:
    """Every aggregate name one evaluated split emits under this configuration.

    The two averages are unconditional -- they are what the split means -- and
    everything else is a client_statistics toggle. This mirrors the loop's
    aggregation so that a checkpoint metric can be checked against the columns
    a run will actually produce, before the run starts rather than after it
    finishes with best.pt missing.

    ``{split}_num_clients`` is unconditional too, and is not a
    ``{metric}_{suffix}`` name: it is how many clients the averages are over,
    which two configs sampling different client counts do not otherwise
    record. Selecting on it is refused a step earlier, by
    ``validate_selection_metric``, which wants a direction word the name does
    not carry. P07-F06.
    """

    suffixes = ["sample_weighted_avg", "avg"]
    if statistics.std:
        suffixes.append("std")
    if statistics.variance:
        suffixes.append("variance")
    if statistics.min:
        suffixes.append("min")
    if statistics.max:
        suffixes.append("max")
    if statistics.worst_percent:
        suffixes.append(f"worst{worst_percent_label(statistics.worst_percent)}")
    names = {f"{split}_{metric}_{suffix}" for metric in CLIENT_METRIC_BASES for suffix in suffixes}
    names.add(f"{split}_num_clients")
    return names


@dataclass
class DivergenceConfig:
    """Stop a run that is no longer learning, and record why.

    A hyperparameter sweep can spend much of its budget on arms that stopped
    learning in their first few rounds. Catching those early frees the GPU and
    leaves a machine-readable record of which hyperparameters blew up.

    Divergence and stagnation are reported as different statuses. A run whose
    loss went to NaN is a different claim from one that merely stopped
    improving, and conflating them would misreport the sweep.
    """

    #: The metric to watch. Must be produced every round: fit_loss is the local
    #: training loss and is free, whereas train_loss_sample_weighted_avg only
    #: exists on evaluation.train's schedule (default: every 10 rounds), so
    #: watching it would burn up to 10 rounds on an already-dead run.
    metric: str = "fit_loss"

    #: NaN or Inf. Nothing is recoverable from it, so this fires on the round
    #: it appears. Free and cannot false-positive.
    non_finite: bool = True

    #: Stop when the metric exceeds this multiple of its first observed value.
    #: Relative to round 1 rather than an absolute threshold so the same number
    #: works for MNIST, FEMNIST and the LLM tasks without retuning. null
    #: disables it.
    blowup_factor: float | None = 10.0

    #: Absolute ceiling on the metric. The relative check above cannot catch a
    #: run that was already pathological at its first observation, so this is
    #: the backstop. For a cross-entropy loss the natural reference is the
    #: random-guess value ln(num_classes) -- 2.30 for MNIST's 10 classes, 4.13
    #: for FEMNIST's 62 -- and a run an order of magnitude above that is worse
    #: than random by a wide margin. null disables it.
    blowup_absolute: float | None = None

    #: Rounds without improvement before the run is called stalled. Compares
    #: against the best value so far rather than counting consecutive
    #: increases: at participation rates of 1-2% the metric is measured on a
    #: different client subset each round, and for noise with no trend at all
    #: P(k consecutive increases) = 1/(k+1)!, so a 500-round run would see ~21
    #: false alarms at k=3. Off by default -- the right value is
    #: dataset-dependent and this is the only detector that can be wrong.
    patience: int | None = None

    #: Relative improvement required to reset the patience counter, so that
    #: noise-sized gains do not keep a stalled run alive.
    min_delta: float = 0.0

    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        """Whether any detector is on, which is what "enabled" now means.

        There used to be an ``enabled`` bool as well, so the same state had two
        spellings and validate_config had to reject the contradiction between
        them by name. Every detector off *is* off; ``divergence: null`` is the
        short way to write it.
        """

        return (
            self.non_finite
            or self.blowup_factor is not None
            or self.blowup_absolute is not None
            or self.patience is not None
        )


@dataclass
class FullConfig:
    """Resolved benchmark configuration."""

    experiment: ExperimentConfig
    server: ServerConfig
    client: ClientConfig
    task: TaskConfig
    data: DataConfig
    model: ModelConfig
    runtime: RuntimeConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    client_statistics: ClientStatisticsConfig = field(default_factory=ClientStatisticsConfig)
    divergence: DivergenceConfig = field(default_factory=DivergenceConfig)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if yaml_loader is not None:
        data = yaml_loader.safe_load(text)
    else:
        data = _load_simple_yaml(text)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise RunRefused(f"Expected YAML mapping in {config_path}")
    return cast(dict[str, Any], data)


def load_config(common_path: str | Path) -> FullConfig:
    """Load and validate a complete benchmark configuration."""

    common_config_path = Path(common_path)
    common = load_yaml(common_config_path)
    # First: a misspelled server or client block would otherwise be reported
    # as a missing one, which names the wrong problem.
    _refuse_unknown_root_keys(common)
    defaults = common.get("defaults", {})
    if not isinstance(defaults, dict):
        raise RunRefused("defaults must be a mapping")

    server = _load_component(common, common_config_path, "server", "server_config")
    client = _load_component(common, common_config_path, "client", "client_config")

    # Rejection first: _resolve_schedule_defaults writes global_rounds and
    # local_iterations into these same dicts, so running it first would hand
    # _reject_restated_keys the loader's own values and refuse every config.
    _reject_restated_keys(common, server, client)
    _refuse_unread_keys("defaults", defaults, DEFAULTS_KEYS)
    _resolve_schedule_defaults(defaults, server, client)

    # Imported here: registry pulls in the client/server packages, which import
    # this module for its config types.
    from fedbrew.core.extensions import load_extensions
    from fedbrew.core.registry import task_for_model

    experiment = _build_experiment_config(common["experiment"])
    if not experiment.name:
        # The config file already names the experiment; restating it invites drift.
        experiment.name = common_config_path.stem
    # Before anything is looked up by name: the extensions are where an
    # out-of-tree model, task, strategy, rule or backend gets its name.
    _validate_extensions(experiment.extensions)
    load_extensions(experiment.extensions)
    model = _build_model_config(common["model"])
    data = _build_data_config(common["data"])
    if not data.name:
        data.name = _infer_data_backend(data)

    config = FullConfig(
        experiment=experiment,
        server=_build_server_config(server),
        client=_build_client_config(client),
        task=TaskConfig(name=task_for_model(model.name)),
        data=data,
        model=model,
        runtime=_build_runtime_config(common["runtime"]),
        evaluation=_build_evaluation_config(common.get("evaluation", {})),
        client_statistics=_build_client_statistics_config(common.get("client_statistics", {})),
        divergence=_build_divergence_config(common.get("divergence", {})),
    )
    validate_config(config)
    return config


#: Local-update modes of the shared SGD engine (fedavg, centralized, fedavg_ft)
#: and of delta_sgd. Kept here so every validator names the same choices.
UPDATE_MODES = {"single_batch", "sequential_epoch", "frozen_batch_gradients", "full_gradient"}
#: Built-in tasks whose training loss is not a mean over the batch's examples:
#: their class overrides ``TaskAdapter.train_loss_denominator``. The frozen mode
#: weights batch gradients by example count, so on these it is not the gradient
#: of the pass and is refused at load (FINDINGS.csv POST-F19). An extension task
#: is not in it, because its class is not known until it is built; the engine
#: refuses it instead, before any update. tests/test_frozen_gradient_weighting.py
#: holds this set to the built-in task classes.
NON_EXAMPLE_MEAN_TASKS = frozenset({"causal_lm"})

#: delta_sgd takes every mode. Under ``full_gradient`` its step-size rule sees
#: the exact gradient of the client's objective.
FULL_GRADIENT_UPDATE_MODE = "full_gradient"
DELTA_SGD_UPDATE_MODES = set(UPDATE_MODES)
FROZEN_GRADIENT_WEIGHTINGS = {"examples", "uniform", "sum"}

#: The three rules whose client is ``FedAvgClient``: the shared engine of
#: ``fedbrew/clients/local_update_modes.py`` at a configured, scheduled learning
#: rate, in every ``update_mode``. ``centralized`` is the same engine over one
#: pooled client, and ``fedavg_ft`` trains exactly like ``fedavg`` -- its
#: personalization is in ``evaluate()``, behind ``evaluation.model_scope``.
#:
#: The sets below are defined here and imported by ``factory.py``, which used to
#: define the fixed-rate and update-mode sets with an extra member. The
#: factory's copy included ``fedavg_ft`` and this one did not, so the factory
#: built that rule with ``momentum``, ``weight_decay``, ``nesterov``,
#: ``learning_rate_schedule``, ``min_learning_rate``, ``update_mode`` and
#: ``frozen_gradient_weighting`` while `_validate_local_sgd_options` returned
#: early and checked none of them. Measured: dropping ``client.momentum`` from
#: ``configs/femnist/fedavg_ft.yaml`` was accepted by the validator and caught
#: only by ``TorchSGDClient.__init__``. See FINDINGS.csv P03-F04.
CENTRALIZED_CLIENT_RULES = {"centralized"}
FEDAVG_FT_CLIENT_RULES = {"fedavg_ft"}
FEDAVG_ENGINE_CLIENT_RULES = {"fedavg", *CENTRALIZED_CLIENT_RULES, *FEDAVG_FT_CLIENT_RULES}

#: Rules that step a torch SGD optimizer at ``client.learning_rate``, fixed or
#: scheduled, and so state ``momentum``, ``weight_decay``, ``nesterov``,
#: ``learning_rate_schedule`` and ``min_learning_rate``.
FIXED_LR_SGD_CLIENT_RULES = {"local_sgd", *FEDAVG_ENGINE_CLIENT_RULES}

#: Rules whose local iteration is one of the shared engine's ``update_mode``s,
#: and the modes each takes. Not derived from the fixed-rate set above, and it
#: used to be the set that one was derived from: taking a mode meant taking a
#: fixed rate's five settings, and `_validate_update_mode_options` ran only from
#: inside `_validate_local_sgd_options`, so a rule whose step size comes from
#: anywhere else could not take a mode at all. How a rule sets its rate and
#: which modes it runs are two questions, answered here and above separately.
#:
#: fedprox, scaffold and fedlalr run their own local loop, not the engine's,
#: and local_sgd and local_adamw run the base client's (`TorchSGDClient.fit`).
#: Each is ``sequential_epoch``: ``local_iterations`` chained passes, one step
#: of the rule's own update per batch. They also take ``full_gradient``: one
#: step of that same update per iteration, on the exact gradient of the whole
#: train split (`local_update_modes.full_gradient_into_grad`).
OWN_LOOP_CLIENT_RULES = {"fedprox", "scaffold", "fedlalr", "local_sgd", "local_adamw"}
OWN_LOOP_UPDATE_MODES = frozenset({"sequential_epoch", FULL_GRADIENT_UPDATE_MODE})
UPDATE_MODES_BY_CLIENT_RULE: dict[str, frozenset[str]] = {
    **{rule: frozenset(UPDATE_MODES) for rule in sorted(FEDAVG_ENGINE_CLIENT_RULES)},
    **dict.fromkeys(sorted(OWN_LOOP_CLIENT_RULES), OWN_LOOP_UPDATE_MODES),
}
UPDATE_MODE_CLIENT_RULES = set(UPDATE_MODES_BY_CLIENT_RULE)
#: Members that may leave ``update_mode`` unset, and what unset means for each:
#: the own-loop rules' loop, which is ``sequential_epoch``, and every config
#: they shipped with before they took a mode states none.
UPDATE_MODE_OPTIONAL_CLIENT_RULES = {*OWN_LOOP_CLIENT_RULES}
#: Rules that state ``frozen_gradient_weighting``: those that can run
#: ``frozen_batch_gradients``, the one mode that reads it. A rule without that
#: mode has nothing for the key to decide. Such a rule states it under every
#: mode, which is FINDINGS.csv POST-F18, open by decision.
FROZEN_WEIGHTING_CLIENT_RULES = {
    rule for rule, modes in UPDATE_MODES_BY_CLIENT_RULE.items() if "frozen_batch_gradients" in modes
}

#: Rules whose local step can run through `_GradientOnlyOptimizer` -- the
#: engine's gradient-only modes, and the own-loop rules' ``full_gradient`` --
#: and so hand `task.train_step` an optimizer-shaped wrapper rather than a
#: `torch.optim.Optimizer`. `delta_sgd` runs through the same engine and
#: refuses `use_amp` outright one rule up, so it needs no entry here.
SGD_ENGINE_CLIENT_RULES = set(UPDATE_MODE_CLIENT_RULES)

#: The `update_mode`s whose local step never constructs a real optimizer: each
#: evaluates every batch gradient at the iteration's starting model through
#: `_GradientOnlyOptimizer` and applies one combined update by hand.
GRADIENT_ONLY_UPDATE_MODES = frozenset({"frozen_batch_gradients", FULL_GRADIENT_UPDATE_MODE})

#: The nine client options _training_client_kwargs gates on the update rule
#: (factory.py:315-401). A rule outside a gate never receives the option, so
#: the key goes nowhere and the run -- and run.json -- claims momentum or a
#: cosine decay that never ran.
ENGINE_CLIENT_OPTIONS: tuple[str, ...] = (
    "momentum",
    "weight_decay",
    "nesterov",
    "learning_rate_schedule",
    "min_learning_rate",
    "update_mode",
    "frozen_gradient_weighting",
    "max_local_steps",
    "max_grad_norm",
)

#: Which of those nine each registered rule cannot honour, and why, as
#: rule -> (what the rule's local step is, options it never receives). A rule
#: that honours all nine has no row.
#:
#: This was a single tuple scoped to fedprox and scaffold, on the premise --
#: written into torch_sgd_client.py, torch_delta_sgd_client.py,
#: torch_fedlalr_client.py and this module -- that "every other rule already
#: fails on an option its engine cannot honour" in its constructor. It does
#: not. Those constructors test attributes the factory never sets for them:
#: TorchDeltaSGDClient checks self.momentum, and the factory hands momentum
#: only to FIXED_LR_SGD_CLIENT_RULES, so the attribute is always the base
#: class's None and the check can never fire. delta_sgd + momentum: 0.9,
#: fedlalr + max_grad_norm: 10.0 and three more pairs loaded, trained, and were
#: recorded in run.json as if they had applied -- the fedlalr one training
#: unclipped while its artifact said it was clipped.
#:
#: The refusal therefore belongs here, per rule, where the config is what is
#: being judged. tests/test_ignored_client_options.py derives the right-hand
#: column from _training_client_kwargs itself, so the table cannot drift from
#: the gates it describes.
UNHONOURED_CLIENT_OPTIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "fedprox": (
        "steps plain SGD with the proximal term written into the backward pass",
        tuple(name for name in ENGINE_CLIENT_OPTIONS if name != "update_mode"),
    ),
    "scaffold": (
        "steps plain SGD with the (c - c_i) correction written into the step",
        tuple(name for name in ENGINE_CLIENT_OPTIONS if name != "update_mode"),
    ),
    "fedlalr": (
        "steps with its own local AMSGrad, whose per-coordinate rate and "
        "synchronised moments are the algorithm",
        tuple(name for name in ENGINE_CLIENT_OPTIONS if name != "update_mode"),
    ),
    "delta_sgd": (
        "measures its own step size from the local smoothness at every step",
        (
            "momentum",
            "weight_decay",
            "nesterov",
            "learning_rate_schedule",
            "min_learning_rate",
            "max_local_steps",
        ),
    ),
    "local_adamw": (
        "steps with torch.optim.AdamW, which has no momentum term, no "
        "Nesterov variant and no clip, and does not run the shared "
        "local-update engine",
        (
            "momentum",
            "nesterov",
            "frozen_gradient_weighting",
            "max_grad_norm",
        ),
    ),
    # max_local_steps is the one entry here the rule's own engine implements
    # (TorchSGDClient.fit) and the factory simply does not forward.
    # Refusing it keeps the config honest about what will run; wiring it
    # through instead is a behaviour change and belongs in its own commit.
    "local_sgd": (
        "steps the base SGD client, which the factory hands no step cap or clip",
        (
            "frozen_gradient_weighting",
            "max_local_steps",
            "max_grad_norm",
        ),
    ),
}

#: Every key a section's ``extra`` dict may carry: the ones some reader
#: actually consumes. The dataclass split (_split_extra) files anything not a
#: named field into ``extra``, which is what lets a rule declare its own
#: options -- and, before this list existed, what let ``max_grad_nrom`` sit
#: there unread while the reader returned its default. A misspelling and a
#: deliberate extra were indistinguishable.
#:
#: The client and server lists are the union over all rules and strategies, so
#: they catch every typo but not an option belonging to a different rule. That
#: narrower check is per-rule and lives with the rule; see
#: _validate_unhonoured_client_options.
#: torch.set_float32_matmul_precision accepts exactly these. It does NOT
#: raise on anything else -- it emits a UserWarning and leaves the setting
#: untouched -- so a typo would run at "highest" while run.json recorded the
#: typo as if it had applied.
MATMUL_PRECISIONS = frozenset({"highest", "high", "medium"})

_KNOWN_EXTRA_KEYS: dict[str, frozenset[str]] = {
    # Nothing reads experiment.extra, evaluation.extra, the per-split extras,
    # client_statistics.extra or divergence.extra: every option those blocks
    # support is a named dataclass field. An empty set is the honest answer,
    # and it is what catches evaluation.val.client for evaluation.val.clients.
    "experiment": frozenset(),
    "evaluation": frozenset(),
    "evaluation.train": frozenset(),
    "evaluation.val": frozenset(),
    "evaluation.test": frozenset(),
    "evaluation.central_test": frozenset(),
    "client_statistics": frozenset(),
    "divergence": frozenset(),
    # data.extra is forwarded as keyword arguments only to
    # synthetic_classification, whose every parameter is already a named
    # DataConfig field; for manifest_dataset it is dropped entirely.
    "data": frozenset(),
    "server": frozenset(
        {
            "aggregation_weighting",
            "server_optimizer",
            "server_learning_rate",
            "beta1",
            "beta2",
            "tau",
        }
    ),
    "client": frozenset(
        {
            "beta1",
            "beta2",
            "delta",
            "drop_last",
            "epsilon",
            "eta_0",
            "eta_max",
            "eval_batch_size",
            "eval_shuffle",
            "finetune_epochs",
            "finetune_learning_rate",
            "frozen_gradient_weighting",
            "gamma",
            "learning_rate_schedule",
            "max_grad_norm",
            "max_local_steps",
            "min_learning_rate",
            "momentum",
            "nesterov",
            "proximal_mu",
            "theta_0",
            "train_shuffle",
            "update_mode",
            "weight_decay",
        }
    ),
    # resume_from, resume_latest, quiet, verbose, no_rich, print_every and
    # data_staging are written by apply_cli_overrides, which is followed by
    # another validate_config.
    "runtime": frozenset(
        {
            "checkpointing",
            "performance",
            "deterministic",
            "deterministic_warn_only",
            "resume_from",
            "resume_latest",
            "quiet",
            # Every terminal-output control is settable here. Some being config
            # keys and others only flags would be an asymmetry with nothing
            # behind it.
            "verbose",
            "no_rich",
            "print_every",
            "data_staging",
        }
    ),
    "runtime.checkpointing": frozenset(
        {
            "enabled",
            "interval",
            "save_last",
            "save_best",
            "save_every_round",
            "best_metric",
            "keep_last",
        }
    ),
    # data_staging had no list at all, so `mode: rsync` was accepted and then
    # printed a skip line at run time -- staging silently off, exit 0. Now the
    # only two keys it has are the two something reads.
    "runtime.data_staging": frozenset({"enabled", "local_root"}),
    "runtime.performance": frozenset(
        {
            "torch_num_threads",
            "cudnn_benchmark",
            "matmul_precision",
            "reuse_model",
            "fast_batching",
            "dataloader",
            "shard_cache_bytes",
        }
    ),
    # Only the four the loader takes from the config. batch_size, shuffle,
    # drop_last and seed are read by build_dataloader too, but the client
    # supplies all four per call and the per-call dict wins the merge -- except
    # drop_last on the evaluation path, which the client does not set, so a
    # config-level drop_last: true silently truncated every evaluated split.
    # generator and worker_init_fn are objects the seeding path builds; neither
    # is expressible in YAML.
    "runtime.performance.dataloader": frozenset(
        {"num_workers", "pin_memory", "persistent_workers", "prefetch_factor"}
    ),
}

#: The performance keys that change the numbers, and so make two runs
#: incomparable. Everything else under ``runtime.performance`` is throughput
#: only: it may change how long a round takes and must not change what the
#: round produces.
#:
#: The split lived in a table in docs/10 and nowhere else, which is how a
#: guard for it ended up unable to fail -- there was no authority to diff a
#: chapter against, so the check hardcoded a third copy of the list and never
#: opened the chapter at all. The two chapters that carry the split now diff
#: against this, the way SERVER_DIAGNOSTIC_METRICS puts the metric filter's
#: exemptions beside the filter.
#:
#: ``matmul_precision`` drops fp32 matmuls to TF32's 10 mantissa bits or a
#: bfloat16 pair's ~16; ``cudnn_benchmark`` picks kernels by timing them, so
#: the choice -- and the arithmetic -- depends on what else the machine was
#: doing. Adding a key here is a claim that it is safe to compare across;
#: chapter 10 owns the argument for each.
NUMERICS_PERFORMANCE_KEYS: frozenset[str] = frozenset({"matmul_precision", "cudnn_benchmark"})

#: The complement, derived rather than restated: no key can be in both, and a
#: new performance key is throughput-only until someone argues otherwise.
#: ``dataloader`` is excluded because it is a block, not a setting; its own
#: four keys are throughput-only and chapter 11 documents them.
THROUGHPUT_ONLY_PERFORMANCE_KEYS: frozenset[str] = (
    _KNOWN_EXTRA_KEYS["runtime.performance"] - NUMERICS_PERFORMANCE_KEYS - frozenset({"dataloader"})
) | _KNOWN_EXTRA_KEYS["runtime.performance.dataloader"]


def _validate_known_keys(
    section: str,
    keys: Iterable[str],
    declared: frozenset[str] = frozenset(),
) -> None:
    """Refuse a key in ``section`` that no reader consumes.

    ``declared`` is what the component in force for the section registered
    as its own keys (``Registry.register(config_keys=...)``): accepted here,
    and forwarded to that component's factory, only while it is the one the
    config selects.
    """

    _refuse_unread_keys(section, keys, _KNOWN_EXTRA_KEYS[section] | declared)


def _refuse_unread_keys(section: str, keys: Iterable[str], known: frozenset[str]) -> None:
    """Refuse any key in ``keys`` outside ``known``, naming what is accepted."""

    unknown = sorted(set(keys) - known)
    if not unknown:
        return
    # A key that was removed on purpose gets its own reason. _reject_restated_keys
    # says this at YAML load; repeating it here covers a config built in code.
    removed = [
        f"{section}.{name} has been removed ({_REMOVED_KEYS[section, name]})"
        for name in unknown
        if (section, name) in _REMOVED_KEYS
    ]
    if removed:
        raise RunRefused("; ".join(removed))
    named = ", ".join(f"{section}.{name}" for name in unknown)
    if known:
        detail = "This section accepts: " + ", ".join(sorted(known)) + "."
    else:
        detail = (
            f"This section takes no options beyond its named fields, so "
            f"nothing would read {'them' if len(unknown) > 1 else 'it'}."
        )
    raise RunRefused(
        f"unknown configuration {'keys' if len(unknown) > 1 else 'key'}: "
        f"{named}. {detail} A key nothing reads is silently dropped and the "
        "reader takes its default, so a misspelling changes the run without "
        "appearing anywhere in run.json."
    )


def _validate_performance_values(config: FullConfig) -> None:
    """Check the three performance values ``configure_runtime`` applies.

    matmul_precision is the one key in this block that changes the numbers:
    "high" and "medium" put fp32 matmuls on TensorFloat32 (10 stored mantissa
    bits) or a bfloat16 pair (~16), against 24 for "highest". A run at "high"
    and a run at "highest" are therefore not like-for-like, which is worth
    failing a typo over rather than silently reverting to "highest".

    The other two were unchecked, and each had its own silent path.
    `torch_num_threads` reaches `int()` and then `torch.set_num_threads`, both
    of which raise on values a config can hold -- `"not-an-int"`, `0`, `-4` --
    and `configure_runtime` caught every exception and returned early, so the
    matmul_precision line below never ran. `cudnn_benchmark` reaches
    `bool(...)`, which reads `"false"` as True: a config turning the
    autotuner off turned it on. Same rule `_validate_extra_bools` applies to
    the two determinism keys, which are the same kind of key one block up.
    P04-F07.
    """

    performance = config.runtime.extra.get("performance")
    if not isinstance(performance, Mapping):
        return

    threads = performance.get("torch_num_threads")
    if threads is not None and (isinstance(threads, bool) or not isinstance(threads, int)):
        raise RunRefused(f"runtime.performance.torch_num_threads must be an int, got {threads!r}")
    if isinstance(threads, int) and not isinstance(threads, bool) and threads < 1:
        raise RunRefused(
            "runtime.performance.torch_num_threads must be at least 1, got "
            f"{threads!r}. torch.set_num_threads rejects it, and a thread "
            "count is not a way to say 'let torch decide' -- omit the key."
        )

    benchmark = performance.get("cudnn_benchmark")
    if benchmark is not None and not isinstance(benchmark, bool):
        raise RunRefused(
            f"runtime.performance.cudnn_benchmark must be a bool, got {benchmark!r}. "
            "It is read through bool(), which takes any non-empty string -- "
            "including 'false' -- as true."
        )

    precision = performance.get("matmul_precision")
    if precision is None:
        return
    if not isinstance(precision, str) or precision not in MATMUL_PRECISIONS:
        raise RunRefused(
            "runtime.performance.matmul_precision must be one of "
            + ", ".join(sorted(MATMUL_PRECISIONS))
            + f", got {precision!r}. torch does not reject an unknown value; "
            "it warns and keeps the current setting, so the run would train at "
            "highest while run.json recorded this."
        )


def _validate_unknown_keys(config: FullConfig) -> None:
    """Check every ``extra`` dict, including the two nested runtime blocks."""

    declared = _declared_extra_keys(config)
    for section in ("experiment", "server", "client", "data", "runtime"):
        _validate_known_keys(
            section,
            getattr(config, section).extra,
            declared.get(section, frozenset()),
        )
    _validate_known_keys("evaluation", config.evaluation.extra)
    for split in ("train", "val", "test", "central_test"):
        _validate_known_keys(f"evaluation.{split}", getattr(config.evaluation, split).extra)
    _validate_known_keys("client_statistics", config.client_statistics.extra)
    _validate_known_keys("divergence", config.divergence.extra)

    for name in ("checkpointing", "performance", "data_staging"):
        block = config.runtime.extra.get(name)
        if isinstance(block, Mapping):
            _validate_known_keys(f"runtime.{name}", block)

    performance = config.runtime.extra.get("performance")
    if isinstance(performance, Mapping):
        dataloader = performance.get("dataloader")
        if isinstance(dataloader, Mapping):
            _validate_known_keys("runtime.performance.dataloader", dataloader)


def _declared_extra_keys(config: FullConfig) -> dict[str, frozenset[str]]:
    """The keys the selected strategy, rule and backend declared, per section.

    Read off the registries rather than off a table here, so the declaration
    and the forwarding live in one place -- the registration -- and a key
    declared for one strategy is not accepted while another is in force.
    """

    from fedbrew.core import registry

    selected = (
        ("server", registry.server_strategies, config.server.strategy),
        ("client", registry.client_updates, config.client.update_rule),
        ("data", registry.datasets, config.data.name),
    )
    return {
        section: component.config_keys(name)
        for section, component, name in selected
        if isinstance(name, str) and component.exists(name)
    }


def _validate_extensions(extensions: object) -> None:
    """Refuse an ``experiment.extensions`` that is not a list of distinct entries."""

    if not isinstance(extensions, list):
        raise RunRefused("experiment.extensions must be a list of paths or module names")
    for entry in extensions:
        if not isinstance(entry, str) or not entry.strip():
            raise RunRefused(
                "experiment.extensions entries must be non-empty strings: a path "
                f"ending in .py or a dotted module name, got {entry!r}"
            )
    if len(set(extensions)) != len(extensions):
        raise RunRefused("experiment.extensions lists an entry twice")


#: Keys that were removed because nothing read them, or because they restated a
#: value the code already knows. Mapped to the guidance shown when one appears.
_REMOVED_KEYS: dict[tuple[str, str], str] = {
    ("experiment", "task"): "the task is recorded by the model's registration "
    "(models.register(name, builder, task=...)) and read from model.name; a "
    "model cannot be registered without it, so there is nothing left to "
    "override, and an override that disagreed with it would run a model "
    "under a task it was not written for",
    ("server", "name"): "server.strategy already identifies the strategy",
    ("client", "name"): "client.update_rule already identifies the update rule",
    ("client", "type"): "client.update_rule is the only dispatch source; "
    "client.type was checked against the registry and then discarded, so a "
    "type that disagreed with update_rule validated clean and ran update_rule",
    ("model", "init"): "no model builder ever read it; models define their own initialization",
    ("data", "split"): "nothing read it",
    ("divergence", "enabled"): "every detector off is off -- write "
    "`divergence: null`, or turn off the detectors you do not want. The bool "
    "was a second spelling of a state the detectors already expressed, so a "
    "config could say enabled: true with nothing to detect",
    ("client", "lazy_clients"): "whether clients are built on demand follows "
    "from the dataset -- a manifest dataset reads thousands of shards from "
    "disk and gets the lazy pool, the in-process synthetic dataset is already "
    "in memory and does not",
    ("runtime.performance.dataloader", "batch_size"): "client.batch_size and "
    "client.eval_batch_size are the batch sizes; the client passes one of them "
    "on every build_dataloader call, so a value here was always overwritten",
    ("runtime.performance.dataloader", "shuffle"): "client.train_shuffle and "
    "client.eval_shuffle decide this per phase, and the client passes one of "
    "them on every call, so a value here was always overwritten",
    ("runtime.performance.dataloader", "drop_last"): "client.drop_last is the "
    "setting; the client passes it on the fit path only, so a value here was "
    "ignored while training and silently dropped the last partial batch of "
    "every evaluated split",
    ("runtime.performance.dataloader", "seed"): "loader seeding is derived per "
    "client, round and phase from experiment.seed; a fixed seed here would "
    "have given every client the same shuffle",
    ("runtime.data_staging", "mode"): "copy_tree was the only supported "
    "value; any other one printed a skip line and left staging off, so the "
    "key could only ever disable the feature it appeared to configure",
    ("runtime.data_staging", "fallback_local_root"): "local_root is the only "
    "staging destination; an unset or unexpanded one already means 'stage "
    "nothing' rather than 'try the next key'",
    ("runtime", "num_workers"): "nothing read it -- the DataLoader takes its "
    "worker count from runtime.performance.dataloader.num_workers, where the "
    "rest of the loader settings already live. Two shipped configs asked here "
    "for ten workers and ran at zero",
    # The two relocated keys. Every entry above was deleted because nothing
    # read it, so its dataclass field went with it and any field-based check
    # notices the key is gone. These two are different in kind: the value is
    # still read -- server.global_rounds drives the round loop -- and only the
    # spelling a config may use moved. The field therefore survives by design,
    # which is exactly why they have to be declared here rather than refused
    # in place: a check that asks "is this still a field?" cannot tell a
    # config that wrote the old name from the loader having filled the field
    # in. docs/04-configuration.md listed both as required keys of their
    # blocks for that reason, and a config copied from it did not load.
    ("server", "global_rounds"): "set defaults.global_rounds instead -- the "
    "round count is shared by the server and every client, and two spellings "
    "of one schedule is how they drift apart",
    ("client", "local_iterations"): "set defaults.local_iterations instead -- "
    "the iteration count is shared by every client, and two spellings of one "
    "schedule is how they drift apart",
    # The renamed key, under both of its old spellings. It counted epochs in
    # name only: update_mode decides what one iteration of the local loop is,
    # and under single_batch that is one optimizer step, not a pass. A config
    # written before the rename is refused rather than read under the new name,
    # so a value nobody has looked at since the rename is not taken on trust.
    ("defaults", "local_epochs"): "renamed to defaults.local_iterations, which "
    "counts iterations of the local loop -- what one iteration is depends on "
    "update_mode, docs/04-configuration.md section 2.1. This config predates "
    "the rename",
    ("client", "local_epochs"): "renamed to local_iterations and relocated: set "
    "defaults.local_iterations instead. This config predates the rename",
}


#: Top-level blocks a config may not write, mapped to the guidance shown when
#: one appears. Separate from _REMOVED_KEYS because a block is not a key, and
#: declared rather than checked inline for the same reason that list exists:
#: the configuration chapter diffs its block table against these, and a
#: refusal written as a bare ``if`` is one the chapter cannot see. ``task``
#: was refused that way while chapter 04 listed it as a required block.
_REMOVED_BLOCKS: dict[str, str] = {
    "task": "the task is inferred from model.name through the model's registration",
}

#: Blocks a config must write that are read at load and never stored on
#: ``FullConfig``. They are part of the surface a config author types and
#: absent from the surface the dataclass describes, so anything deriving the
#: documented blocks from ``FullConfig`` alone misses them -- which is how
#: ``defaults`` came to be required by the loader and documented nowhere.
_LOAD_TIME_BLOCKS: frozenset[str] = frozenset({"defaults"})

#: Every key the ``defaults`` block accepts. The block is closed like every
#: other one, and has to be declared rather than derived for the same reason
#: it is in _LOAD_TIME_BLOCKS: it is read at load and never stored, so the
#: unknown-key check, which walks ``FullConfig``'s ``extra`` dicts, never saw
#: it. ``defaults.bogus_key: 7`` loaded, and a stale or misspelled key beside
#: the two real ones was dropped without a word. FINDINGS.csv POST-F16.
DEFAULTS_KEYS: frozenset[str] = frozenset({"global_rounds", "local_iterations"})

#: The older spelling of the server and client blocks: a path to a YAML file
#: holding the block, in place of the block itself (`_load_component`).
_COMPONENT_PATH_KEYS: frozenset[str] = frozenset({"server_config", "client_config"})


def root_config_keys() -> frozenset[str]:
    """Every top-level key a run config may write.

    The blocks ``FullConfig`` stores, less the removed ones, plus the blocks
    read at load and never stored, plus the component-path spellings. Derived
    rather than listed, so a block added to ``FullConfig`` is accepted here
    without a second edit.
    """

    stored = {field.name for field in fields(FullConfig)} - set(_REMOVED_BLOCKS)
    return frozenset(stored | _LOAD_TIME_BLOCKS | _COMPONENT_PATH_KEYS)


def _refuse_unknown_root_keys(common: Mapping[str, Any]) -> None:
    """Refuse a top-level key that no block answers to.

    The blocks were each closed, but the root was not: ``load_config`` reads
    the blocks it knows by name and never looked at the rest, so
    ``evaluaton:`` loaded, the run took every evaluation default -- train
    metrics over the participating clients instead of all of them -- and
    ``divergance:`` ran with ``blowup_absolute`` unset. FINDINGS.csv POST-F26.

    A removed block is not unknown: `_reject_restated_keys` refuses it with
    the reason it was removed.
    """

    known = root_config_keys()
    unknown = sorted(str(key) for key in common if key not in known and key not in _REMOVED_BLOCKS)
    if not unknown:
        return
    problems = []
    for key in unknown:
        close = difflib.get_close_matches(key, sorted(known), n=1)
        hint = f" (did you mean {close[0]!r}?)" if close else ""
        problems.append(f"{key!r}{hint}")
    raise RunRefused(
        f"unknown top-level config key{'s' if len(unknown) > 1 else ''}: "
        f"{', '.join(problems)}. A config may write only these: "
        f"{', '.join(sorted(known))}"
    )


def _reject_restated_keys(
    common: Mapping[str, Any],
    server: Mapping[str, Any],
    client: Mapping[str, Any],
) -> None:
    """Fail on config keys that were removed, naming the replacement."""

    sections: dict[str, Mapping[str, Any]] = {
        "defaults": common.get("defaults") or {},
        "experiment": common.get("experiment") or {},
        "server": server,
        "client": client,
        "model": common.get("model") or {},
        "data": common.get("data") or {},
    }
    removed = [
        f"{section}.{key} has been removed ({reason})"
        for (section, key), reason in _REMOVED_KEYS.items()
        if key in sections.get(section, {})
    ]
    removed.extend(
        f"the top-level {block} block has been removed; {reason}"
        for block, reason in _REMOVED_BLOCKS.items()
        if block in common
    )
    if removed:
        raise RunRefused("; ".join(removed))


def _infer_data_backend(data: DataConfig) -> str:
    """Pick the dataset backend from whether a generated manifest was given."""

    return "manifest_dataset" if data.path else "synthetic_classification"


def _load_component(
    config: Mapping[str, Any],
    config_path: Path,
    section: str,
    legacy_path_key: str,
) -> dict[str, Any]:
    """Load an inline component, with legacy path support for external configs."""

    inline = config.get(section)
    if inline is not None:
        if not isinstance(inline, Mapping):
            raise RunRefused(f"{section} must be a mapping")
        return dict(inline)

    legacy_path = config.get(legacy_path_key)
    if legacy_path is None:
        raise RunRefused(f"config missing required mapping: {section}")
    return load_yaml(_resolve_config_path(config_path, str(legacy_path)))


def _resolve_schedule_defaults(
    defaults: Mapping[str, Any],
    server: dict[str, Any],
    client: dict[str, Any],
) -> None:
    """Set shared schedule values from their only supported configuration location.

    The spellings a config may not use -- ``server.global_rounds`` and
    ``client.local_iterations``, which name the resolved fields this writes,
    and ``defaults.local_epochs`` and ``client.local_epochs`` from before the
    rename -- are refused by ``_reject_restated_keys`` from ``_REMOVED_KEYS``,
    which runs first. They used to be refused here instead, in a raise this function
    kept for itself, and the guard that checks the configuration chapter lists
    every removed key reads ``_REMOVED_KEYS`` alone -- so the chapter went on
    calling both required keys of their blocks and nothing failed.
    """

    if "global_rounds" not in defaults:
        raise RunRefused("defaults.global_rounds is required")
    if "local_iterations" not in defaults:
        raise RunRefused("defaults.local_iterations is required")

    server["global_rounds"] = defaults["global_rounds"]
    client["local_iterations"] = defaults["local_iterations"]


def validate_config(config: FullConfig) -> None:
    """Validate resolved configuration values."""

    if config.experiment.seed < 0:
        raise RunRefused("seed must be non-negative")
    _validate_participation(config.server)
    if config.server.global_rounds <= 0:
        raise RunRefused("global_rounds must be positive")
    if config.client.local_iterations <= 0:
        raise RunRefused("local_iterations must be positive")
    if config.client.batch_size <= 0:
        raise RunRefused("batch_size must be positive")
    if config.runtime.device not in {"cpu", "cuda", "auto"}:
        raise RunRefused("device must be one of: cpu, cuda, auto")
    # Early: a half-paired config is wrong about which algorithm is running, so
    # every later message about that algorithm's options would be answering the
    # wrong question. "client.momentum must be configured" is not what someone
    # who wrote server.strategy=scaffold and forgot the client needs to read.
    _validate_extensions(config.experiment.extensions)
    _validate_registered_names(config)
    _validate_paired_strategies(config)
    _refuse_adapter_state_on_full_state_rules(config)
    _refuse_ignored_active_target_weighting(config)
    _refuse_retired_metric_names(config)
    _validate_evaluation(config)
    _validate_evaluation_model_scope(config)
    _validate_client_statistics(config.client_statistics)
    _validate_divergence(config.divergence)
    _validate_checkpoint_selection(config.runtime.extra.get("checkpointing"))
    _validate_checkpoint_metric_is_emitted(config)
    _require_derived_diagnostic_sources(config)
    _validate_divergence_metric_is_reachable(config)
    _validate_unhonoured_client_options(config)
    _refuse_frozen_off_examples(config)
    _validate_unknown_keys(config)
    _validate_performance_values(config)
    _validate_output_dir(config)
    if config.experiment.run_id is not None and not isinstance(
        config.experiment.run_id,
        str,
    ):
        raise RunRefused("experiment.run_id must be a string or null")
    if not isinstance(config.experiment.use_run_subdir, bool):
        raise RunRefused("experiment.use_run_subdir must be a bool")
    if not isinstance(config.experiment.notes, str):
        raise RunRefused("experiment.notes must be a string")
    _validate_metrics("experiment.tags", config.experiment.tags)
    _validate_metrics("server.metrics", config.server.metrics)
    _validate_metrics("client.metrics", config.client.metrics)
    _validate_extra_bools(
        "runtime",
        config.runtime.extra,
        ("deterministic", "deterministic_warn_only"),
    )
    _validate_print_every(config.runtime.extra.get("print_every"))
    _validate_extra_bools(
        "client",
        config.client.extra,
        ("train_shuffle", "eval_shuffle", "drop_last", "nesterov"),
    )
    _validate_local_sgd_options(config)
    _validate_update_mode_options(config)
    _validate_delta_sgd_options(config)
    _validate_fedlalr_options(config)
    _validate_fedavg_ft_options(config)
    _validate_local_adamw_options(config)
    _validate_sgd_engine_amp(config)
    _validate_server_algorithm_options(config)


def _validate_participation(server: ServerConfig) -> None:
    """Exactly one sampling scheme, its value in (0, 1]."""

    rate = server.participation_rate
    probability = server.participation_probability
    if (rate is None) == (probability is None):
        found = "both are set" if rate is not None else "neither is set"
        raise RunRefused(
            "server needs exactly one of participation_rate (a fixed number of clients "
            "per round, ceil(rate x clients)) and participation_probability (each client "
            f"independently, so the count varies by round and can be zero); {found}"
        )
    name = "participation_rate" if rate is not None else "participation_probability"
    value = rate if rate is not None else probability
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value <= 1:
        raise RunRefused(f"server.{name} must be in (0, 1], got {value!r}")


def _build_experiment_config(values: Mapping[str, Any]) -> ExperimentConfig:
    known, extra = _split_extra(values, ExperimentConfig)
    return ExperimentConfig(**known, extra=extra)


def _build_server_config(values: Mapping[str, Any]) -> ServerConfig:
    known, extra = _split_extra(values, ServerConfig)
    return ServerConfig(**known, extra=extra)


def _build_client_config(values: Mapping[str, Any]) -> ClientConfig:
    known, extra = _split_extra(values, ClientConfig)
    return ClientConfig(**known, extra=extra)


def _build_data_config(values: Mapping[str, Any]) -> DataConfig:
    known, extra = _split_extra(values, DataConfig)
    return DataConfig(**known, extra=extra)


def _build_model_config(values: Mapping[str, Any]) -> ModelConfig:
    known, extra = _split_extra(values, ModelConfig)
    return ModelConfig(**known, extra=extra)


def _build_runtime_config(values: Mapping[str, Any]) -> RuntimeConfig:
    known, extra = _split_extra(values, RuntimeConfig)
    return RuntimeConfig(**known, extra=extra)


def _build_evaluation_config(values: object) -> EvaluationConfig:
    if not isinstance(values, Mapping):
        raise RunRefused("evaluation must be a mapping")
    for retired, replacement in (
        ("test_set", "evaluation.test.every / evaluation.central_test.every"),
        ("test_sets", "evaluation.test.every and evaluation.central_test.every"),
        ("client_scope", "evaluation.<split>.clients, per split"),
        ("save_client_metrics", "client_statistics.per_client_csv"),
    ):
        if retired in values:
            raise RunRefused(f"evaluation.{retired} has been removed; use {replacement}")

    defaults = EvaluationConfig()
    splits = {
        split: _build_split_evaluation_config(
            values.get(split, {}), split, getattr(defaults, split)
        )
        for split in ("train", "val", "test")
    }
    central_values = values.get("central_test", {})
    if not isinstance(central_values, Mapping):
        raise RunRefused("evaluation.central_test must be a mapping")
    central_known, central_extra = _split_extra(central_values, CentralTestConfig)
    central_test = CentralTestConfig(
        every=central_known.get("every", defaults.central_test.every),
        extra=central_extra,
    )
    extra = {
        key: value
        for key, value in values.items()
        if key not in {"train", "val", "test", "central_test", "model_scope"}
    }
    return EvaluationConfig(
        **splits,
        central_test=central_test,
        model_scope=values.get("model_scope", defaults.model_scope),
        extra=extra,
    )


def _build_split_evaluation_config(
    values: object,
    split: str,
    defaults: SplitEvaluationConfig,
) -> SplitEvaluationConfig:
    if not isinstance(values, Mapping):
        raise RunRefused(f"evaluation.{split} must be a mapping")
    known, extra = _split_extra(values, SplitEvaluationConfig)
    return SplitEvaluationConfig(
        every=known.get("every", defaults.every),
        clients=known.get("clients", defaults.clients),
        extra=extra,
    )


def _build_client_statistics_config(values: object) -> ClientStatisticsConfig:
    if not isinstance(values, Mapping):
        raise RunRefused("client_statistics must be a mapping")
    known, extra = _split_extra(values, ClientStatisticsConfig)
    return ClientStatisticsConfig(**known, extra=extra)


#: Every detector off. `divergence: null` resolves to this -- the spelling that
#: replaced `enabled: false`.
_DIVERGENCE_OFF = {
    "non_finite": False,
    "blowup_factor": None,
    "blowup_absolute": None,
    "patience": None,
}


def _build_divergence_config(values: object) -> DivergenceConfig:
    if values is None:
        return DivergenceConfig(**_DIVERGENCE_OFF)
    if not isinstance(values, Mapping):
        raise RunRefused("divergence must be a mapping or null")
    known, extra = _split_extra(values, DivergenceConfig)
    return DivergenceConfig(**known, extra=extra)


def parse_evaluation_schedule(value: object, context: str) -> int | None:
    """Turn an ``every`` value into an interval in rounds.

    Returns the interval for an integer, 0 for "final" (the final round only),
    and None for "never". An interval pins round 1 and the final round as well
    as its own multiples, so a 500-round run with ``every: 10`` both starts
    from a measured baseline and ends on a measured round.
    """

    if isinstance(value, bool):
        raise RunRefused(f"{context}.every must be an integer, 'final' or 'never'")
    if isinstance(value, int):
        if value <= 0:
            raise RunRefused(
                f"{context}.every must be positive; use 'final' for the last "
                "round only, or 'never' to skip the split"
            )
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "final":
            return 0
        if text == "never":
            return None
        if text.isdigit() and int(text) > 0:
            return int(text)
    raise RunRefused(
        f"{context}.every must be a positive integer, 'final' or 'never'; got {value!r}"
    )


def evaluates_round(interval: int | None, round_id: int, final_round: int) -> bool:
    """Whether a split with this schedule is evaluated on this round.

    An interval always pins both endpoints: round 1 and the final round, plus
    every multiple in between -- 1, 10, 20, ..., final for ``every: 10``. Both
    are there so a curve is complete rather than merely periodic. Without round
    1 the first measurement of a ``every: 50`` split lands at round 50, so the
    plotted curve starts after most of the learning has already happened and
    there is no baseline to measure improvement from; without the final round a
    500-round run with ``every: 10`` would report a number 5 rounds stale.

    The cost is one extra pass per split per run, and it buys the point every
    other point is compared against.

    ``final`` (interval 0) is exempt: asking for the last round only means the
    last round only, and adding round 1 would quietly override that.
    """

    if interval is None:
        return False
    if interval == 0:
        return round_id == final_round
    return round_id == 1 or round_id % interval == 0 or round_id == final_round


def parse_evaluation_client_scope(scope: object) -> tuple[str, int | None]:
    """Split an ``evaluation.client_scope`` value into (mode, sample size).

    Returns the sample size only for the two sampling modes; it is None for
    "all" and "selected". Raises on anything else, so a typo fails at config
    load rather than silently evaluating the wrong client set for 500 rounds.
    """

    if not isinstance(scope, str):
        raise RunRefused("evaluation.client_scope must be a string")
    text = scope.strip()
    if text in {"all", "participating"}:
        return text, None
    for mode in ("sample", "resample"):
        prefix = f"{mode}:"
        if not text.startswith(prefix):
            continue
        raw = text[len(prefix) :].strip()
        if not raw.isdigit() or int(raw) <= 0:
            raise RunRefused(
                f"evaluation.client_scope {scope!r}: {mode} needs a positive "
                f"integer client count, for example {mode}:1000"
            )
        return mode, int(raw)
    raise RunRefused(
        "evaluation.client_scope must be one of: all, participating, "
        f"sample:<N>, resample:<N>; got {scope!r}"
    )


def _validate_evaluation(config: FullConfig) -> None:
    """Check every split's schedule and client scope, and the split roles."""

    parse_evaluation_schedule(config.evaluation.central_test.every, "evaluation.central_test")
    for split in ("train", "val", "test"):
        block = getattr(config.evaluation, split)
        parse_evaluation_schedule(block.every, f"evaluation.{split}")
        mode, _ = parse_evaluation_client_scope(block.clients)
        if split == "test" and mode == "participating":
            raise RunRefused(
                "evaluation.test.clients must not be 'participating': the "
                "reported test number would come from a biased subset of "
                "clients the model was just fitted on. Use all, sample:<N> or "
                "resample:<N>."
            )

    checkpointing = config.runtime.extra.get("checkpointing")
    selects_best = (
        isinstance(checkpointing, Mapping)
        and bool(checkpointing.get("enabled", True))
        and bool(checkpointing.get("save_best", True))
    )
    if (
        selects_best
        and parse_evaluation_schedule(config.evaluation.val.every, "evaluation.val") is None
    ):
        raise RunRefused(
            "runtime.checkpointing.save_best is on but evaluation.val.every is "
            "'never'; best.pt is selected on a validation metric, so the "
            "validation split has to be evaluated"
        )


def _validate_evaluation_model_scope(config: FullConfig) -> None:
    """Check which model the client evaluation passes measure.

    Also cross-checks checkpoint selection against it: selecting on a
    personal_ metric needs the personalized pass to run at all. Whether the
    scope then emits that exact name is the narrower question
    _validate_checkpoint_metric_is_emitted answers, once client_statistics has
    been validated.
    """

    scope = config.evaluation.model_scope
    if not isinstance(scope, str) or scope not in EVALUATION_MODEL_SCOPES:
        raise RunRefused(
            "evaluation.model_scope must be one of: " + ", ".join(sorted(EVALUATION_MODEL_SCOPES))
        )
    _validate_checkpoint_model_scope(config)


def _validate_checkpoint_model_scope(config: FullConfig) -> None:
    """Require the evaluation pass that produces the selection metric.

    Split out of `_validate_evaluation_model_scope`, which still calls it, so
    that `checkpoint_selection_problem` can compose the three checkpoint-
    selection checks without also re-raising that function's opinion of
    `model_scope` itself. P07-F09.
    """

    scope = config.evaluation.model_scope
    checkpointing = config.runtime.extra.get("checkpointing")
    if not isinstance(checkpointing, Mapping):
        return
    if not bool(checkpointing.get("save_best", True)):
        return

    from fedbrew.core.checkpointing import DEFAULT_SELECTION_METRIC

    best_metric = str(checkpointing.get("best_metric", DEFAULT_SELECTION_METRIC))
    wants_personal = best_metric.startswith(PERSONAL_SPLIT_PREFIX)
    emits_personal = scope in {"personal", "both"}
    emits_global = scope in {"global", "both"}
    if wants_personal and not emits_personal:
        raise RunRefused(
            f"runtime.checkpointing.best_metric={best_metric!r} needs the "
            f"personalized evaluation pass, but evaluation.model_scope is "
            f"{scope!r}; set it to personal or both"
        )
    if not wants_personal and not emits_global:
        raise RunRefused(
            f"runtime.checkpointing.best_metric={best_metric!r} needs the global "
            f"evaluation pass, but evaluation.model_scope is {scope!r}; select on "
            f"{PERSONAL_SPLIT_PREFIX}{best_metric} or set model_scope to both"
        )


def _validate_checkpoint_metric_is_emitted(config: FullConfig) -> None:
    """Require best_metric to be a column this run will actually produce.

    A name that passes validate_selection_metric (val_ prefix, one direction
    word) and the model_scope cross-check can still be a column nothing ever
    writes -- val_accuracy_bottom10 when the column is val_accuracy_worst10,
    val_accuracy_worst5 when worst_percent is 10, val_accuracy_variance when
    variance is off. The loop then finds None in the metrics dict every round
    and skips the update, so a 500-round run finishes with best_checkpoint
    null in run.json and no other sign of it.

    Runs last, after _validate_client_statistics and
    _validate_checkpoint_selection, because it reads worst_percent and assumes
    the val_ prefix both of those have already checked.
    """

    checkpointing = config.runtime.extra.get("checkpointing")
    if not isinstance(checkpointing, Mapping):
        return
    if not bool(checkpointing.get("save_best", True)):
        return

    from fedbrew.core.checkpointing import DEFAULT_SELECTION_METRIC

    best_metric = str(checkpointing.get("best_metric", DEFAULT_SELECTION_METRIC))
    scope = config.evaluation.model_scope
    splits = []
    if scope in {"global", "both"}:
        splits.append("val")
    if scope in {"personal", "both"}:
        splits.append(f"{PERSONAL_SPLIT_PREFIX}val")

    emitted: set[str] = set()
    for split in splits:
        emitted |= client_metric_names(split, config.client_statistics)
    if best_metric in emitted:
        return
    raise RunRefused(
        f"runtime.checkpointing.best_metric={best_metric!r} is never emitted, "
        "so best.pt would never be written. This configuration's validation "
        "metrics are: " + ", ".join(sorted(emitted)) + ". Metrics beyond the "
        "two averages are client_statistics toggles -- turn the one you want "
        "on, or set client_statistics.worst_percent to the percentage you are "
        "selecting on."
    )


def checkpoint_selection_problem(config: FullConfig) -> str | None:
    """The run path's first objection to this config's best.pt selection.

    `validate_config` raises this message; `--validate-only` reports it. The
    two must agree, and re-implementing the checks in `validation.py` is how
    they came apart: preflight accepted `best_mode` and checked it was max or
    min, a key `_validate_checkpoint_selection` has removed outright, and
    checked `best_metric` for non-emptiness alone while the run path checks the
    prefix, the direction, the model_scope pairing and whether the column is
    ever emitted. Measured on the pre-fix tree: `best_mode: min`,
    `best_metric: central_test_accuracy` and `best_metric:
    val_accuracy_bottom10` each passed preflight clean and were refused by
    `validate_config`. P07-F09.

    First objection, not all of them, for the reason
    `_validate_checkpoint_metric_is_emitted` gives in its own docstring: it
    assumes the prefix `_validate_checkpoint_selection` has already checked, so
    a metric that failed that one would draw a second, misleading complaint
    about a column that was never the problem.

    Args:
        config: A resolved config, whose `client_statistics` has already been
            validated -- the order `validate_config` runs these in.

    Returns:
        The message `validate_config` would raise, or None if it would not.
    """

    checks = (
        lambda: _validate_checkpoint_selection(config.runtime.extra.get("checkpointing")),
        lambda: _validate_checkpoint_model_scope(config),
        lambda: _validate_checkpoint_metric_is_emitted(config),
    )
    for check in checks:
        try:
            check()
        except ValueError as exc:
            return str(exc)
    return None


def _metrics_no_filter_can_remove(config: FullConfig) -> set[str]:
    """Round-record names that never pass through either metrics filter.

    The evaluation aggregates and the selected strategy's own diagnostics all
    reach the round record without going through filter_metrics, so a name
    among them survives whatever either list says. See docs/08-metrics.md
    section 4.3.

    The central-test metrics are the same case but are not enumerable here:
    `loop._evaluate_central_test_set` passes through any finite numeric key a
    task's `evaluate_global` reports, not just `central_test_loss` and
    `central_test_accuracy`, so `_require_metric_survives` checks that one by
    prefix instead of against this set.

    Splits are taken unconditionally rather than from each schedule: a
    schedule-gated metric is absent on most rounds by design and the monitor is
    built for that, so narrowing this by schedule would only turn correct
    configs into failures.
    """

    from fedbrew.core.metrics import server_diagnostic_metrics

    # Only the diagnostics this round will carry: one built from a client
    # metric that client.metrics filters out is absent, and was exempted here
    # all the same, so divergence.metric could watch a column no round had.
    # FINDINGS.csv POST-F12.
    unfiltered: set[str] = set(
        server_diagnostic_metrics(config.server.strategy, list(config.client.metrics or []))
    )
    prefixes = [""] if config.evaluation.model_scope in {"global", "both"} else []
    if config.evaluation.model_scope in {"personal", "both"}:
        prefixes.append(PERSONAL_SPLIT_PREFIX)
    for prefix in prefixes:
        for split in ("train", "val", "test"):
            unfiltered |= client_metric_names(f"{prefix}{split}", config.client_statistics)
    return unfiltered


def _validate_divergence_metric_is_reachable(config: FullConfig) -> None:
    """Require divergence.metric to survive both metrics filters.

    The sibling of _validate_checkpoint_metric_is_emitted, and the same defect:
    a name passes every syntactic check divergence has -- non-empty, not a
    test_ metric when patience is set -- and is still absent from every round,
    because a metrics list filtered it out on its way into the round record.

    What that costs is the whole monitor. Every detector reads the one metric
    name, so a name nothing emits silences all of them, non_finite included,
    for the entire run. Nothing fails. The loop prints a warning naming the
    metric -- after the last round, which for a 500-round FEMNIST arm is
    several GPU-hours after it would have been worth knowing.

    There are two filters on the path from the task to the round record, not
    one. This check covered server.metrics only, and the client-side filter is
    the earlier and stricter of the two: TorchSGDClient._evaluate_model applies
    client.metrics to the task metrics before the FitResult exists, so a
    client.metrics that omits fit_loss means no client ever reports it, the
    server aggregate has nothing to filter, and the default divergence.metric
    watches a name no round can contain. Nothing checked it.

    Here rather than in validation.py deliberately. validate_full_config runs
    only under --validate-only, so an issue raised there, even at severity
    "error", does not stop an ordinary run: the config that silences the
    monitor would still train. This function is on the run path, via
    load_config. Same reasoning as _validate_unhonoured_client_options.
    """

    from fedbrew.core.metrics import CLIENT_UNFILTERED_FIT_METRICS

    divergence = config.divergence
    if divergence is None or not divergence.active:
        return

    metric = divergence.metric
    unfiltered = _metrics_no_filter_can_remove(config)

    # The client filter first: it runs earlier, and a name it drops is gone
    # before server.metrics has anything to keep. Its exemptions are the
    # round-record names above plus the extras this rule adds after filtering.
    _require_metric_survives(
        metric=metric,
        requested=list(config.client.metrics or []),
        unfiltered=unfiltered
        | set(CLIENT_UNFILTERED_FIT_METRICS.get(config.client.update_rule, frozenset())),
        key="client.metrics",
        where=("so the client drops it before the FitResult is built and no round can contain it"),
    )
    _require_metric_survives(
        metric=metric,
        requested=list(config.server.metrics or []),
        unfiltered=unfiltered,
        key="server.metrics",
        where="so the filter drops it before it reaches the round record",
    )


def _refuse_retired_metric_names(config: FullConfig) -> None:
    """Refuse a metric name that was retired, naming what replaced it.

    FedLALR's ``client_effective_learning_rate_{mean,min,max}`` meant one
    thing on the client and another on the server; both were renamed
    (``RETIRED_METRIC_NAMES``, ``fedbrew/core/metrics.py``). A config still
    naming one would otherwise filter or watch a column that no longer
    exists -- and a monitor on it would stay silent for the run. POST-F14.
    """

    from fedbrew.core.metrics import RETIRED_METRIC_NAMES, retired_metric_message

    named: list[tuple[str, object]] = [
        *(("server.metrics", name) for name in config.server.metrics or []),
        *(("client.metrics", name) for name in config.client.metrics or []),
    ]
    if config.divergence is not None:
        named.append(("divergence.metric", config.divergence.metric))
    checkpointing = config.runtime.extra.get("checkpointing")
    if isinstance(checkpointing, Mapping) and "best_metric" in checkpointing:
        named.append(("runtime.checkpointing.best_metric", checkpointing["best_metric"]))
    for key, name in named:
        if isinstance(name, str) and name in RETIRED_METRIC_NAMES:
            raise RunRefused(f"{key} names {retired_metric_message(name)}")


def _require_derived_diagnostic_sources(config: FullConfig) -> None:
    """Refuse asking for a server diagnostic whose source the clients filter out.

    FedLALR's ``effective_learning_rate_across_clients_*`` are computed from
    each client's ``effective_learning_rate_coordinate_mean``. A non-empty
    ``client.metrics`` without it leaves the server nothing to spread, so the
    four columns are absent from every round. A ``divergence.metric`` naming
    one then silences every detector for the run, and a metrics list naming
    one asks for a column nothing writes. Both loaded; the reachability check
    exempted every strategy diagnostic whole. FINDINGS.csv POST-F12.
    """

    from fedbrew.core.metrics import SERVER_DIAGNOSTIC_SOURCES

    sources = SERVER_DIAGNOSTIC_SOURCES.get(config.server.strategy, {})
    client_metrics = list(config.client.metrics or [])
    if not sources or not client_metrics:
        return
    asked: list[tuple[str, str]] = [
        *(("server.metrics", name) for name in config.server.metrics or []),
        *(("client.metrics", name) for name in client_metrics),
    ]
    if config.divergence is not None and config.divergence.active:
        asked.insert(0, ("divergence.metric", config.divergence.metric))
    for key, name in asked:
        source = sources.get(name)
        if source is not None and source not in client_metrics:
            raise RunRefused(
                f"{key} names {name!r}, which the {config.server.strategy} server "
                f"computes from each client's {source!r}, and client.metrics is not "
                f"empty and does not name {source!r}: the clients drop it, and no "
                f"round can contain {name!r}. Add {source!r} to client.metrics."
            )


def _require_metric_survives(
    *,
    metric: str,
    requested: list[str],
    unfiltered: set[str],
    key: str,
    where: str,
) -> None:
    """Raise unless ``metric`` survives the non-empty metrics list ``key``."""

    if not requested:
        # The empty list keeps everything, which is the default and the case
        # that cannot go wrong.
        return
    # central_test_ is an open class, not a set _metrics_no_filter_can_remove
    # could enumerate: loop._evaluate_central_test_set writes any finite
    # numeric key evaluate_global reports under this prefix, not just loss and
    # accuracy, and none of it passes through filter_metrics either way.
    if metric in requested or metric in unfiltered or metric.startswith("central_test_"):
        return
    raise RunRefused(
        f"divergence.metric={metric!r} is not in {key}, and {key} is not "
        f"empty, {where}. Every detector reads that one name, so all of them "
        "-- non_finite included -- would stay silent for the whole run and the "
        f"warning would arrive after the last round. Add {metric!r} to {key}, "
        "or empty the list to keep every metric; evaluation columns and this "
        "strategy's own diagnostics are not filtered and need neither."
    )


def _validate_client_statistics(statistics: ClientStatisticsConfig) -> None:
    for name in ("per_client_csv", "std", "variance", "min", "max"):
        if not isinstance(getattr(statistics, name), bool):
            raise RunRefused(f"client_statistics.{name} must be a boolean")
    worst = statistics.worst_percent
    if worst is None:
        return
    if isinstance(worst, bool) or not isinstance(worst, int | float):
        raise RunRefused("client_statistics.worst_percent must be a number or null")
    if not 0 <= float(worst) <= 100:
        raise RunRefused("client_statistics.worst_percent must be between 0 and 100")


def _validate_divergence(divergence: DivergenceConfig) -> None:
    if not isinstance(divergence.non_finite, bool):
        raise RunRefused("divergence.non_finite must be a boolean")
    if not isinstance(divergence.metric, str) or not divergence.metric.strip():
        raise RunRefused("divergence.metric must be a non-empty metric name")
    # patience turns divergence into early stopping: it ends the run and writes
    # status "stalled" with stopped_round. Pointed at a test metric that is
    # model selection on the test set through a different door than the one
    # checkpointing.best_metric already refuses -- and it leaves no trace,
    # because run.json records the round it stopped at, not what it watched.
    # non_finite and blowup are safety stops rather than selection, so they may
    # still watch any metric.
    if divergence.patience is not None and divergence.metric.startswith(
        ("test_", "central_test_", "personal_test_")
    ):
        raise RunRefused(
            f"divergence.metric={divergence.metric!r} with patience set is "
            "early stopping on the test set, which biases the reported score "
            "exactly as checkpointing.best_metric does. Watch fit_loss, or a "
            "val_* metric."
        )

    ceiling = divergence.blowup_absolute
    if ceiling is not None:
        if isinstance(ceiling, bool) or not isinstance(ceiling, int | float):
            raise RunRefused("divergence.blowup_absolute must be a number or null")
        if not math.isfinite(float(ceiling)) or float(ceiling) <= 0.0:
            raise RunRefused("divergence.blowup_absolute must be a positive finite number")

    factor = divergence.blowup_factor
    if factor is not None:
        if isinstance(factor, bool) or not isinstance(factor, int | float):
            raise RunRefused("divergence.blowup_factor must be a number or null")
        if float(factor) <= 1.0:
            raise RunRefused(
                "divergence.blowup_factor must be greater than 1; it is a "
                "multiple of the metric's first observed value, so 1.0 or less "
                "would fire on the first round that is not an improvement"
            )

    patience = divergence.patience
    if patience is not None:
        if isinstance(patience, bool) or not isinstance(patience, int):
            raise RunRefused("divergence.patience must be an integer or null")
        if patience <= 0:
            raise RunRefused("divergence.patience must be positive; use null to disable it")

    delta = divergence.min_delta
    if isinstance(delta, bool) or not isinstance(delta, int | float):
        raise RunRefused("divergence.min_delta must be a number")
    if not 0.0 <= float(delta) < 1.0:
        raise RunRefused("divergence.min_delta is a relative improvement and must be in [0, 1)")


def _validate_checkpoint_selection(checkpointing: object) -> None:
    """Keep model selection off the test set, and off a configurable direction.

    best.pt is the checkpoint a paper reports, so the metric that picks it has
    to come from data the model was never selected on. Direction is derived
    from the metric name, so best_mode is no longer a setting.
    """

    if not isinstance(checkpointing, Mapping):
        return
    if "best_mode" in checkpointing:
        raise RunRefused(
            "runtime.checkpointing.best_mode has been removed; the direction is "
            "derived from best_metric (loss minimises, accuracy maximises)"
        )
    if not bool(checkpointing.get("save_best", True)):
        return

    from fedbrew.core.checkpointing import (
        DEFAULT_SELECTION_METRIC,
        validate_selection_metric,
    )

    validate_selection_metric(str(checkpointing.get("best_metric", DEFAULT_SELECTION_METRIC)))


def _split_extra(
    values: Mapping[str, Any],
    config_type: type[Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    known_names = {item.name for item in fields(config_type) if item.name != "extra"}
    known = {name: value for name, value in values.items() if name in known_names}
    extra = {name: value for name, value in values.items() if name not in known_names}
    return known, extra


def _resolve_config_path(common_path: Path, selected_path: str) -> Path:
    path = Path(selected_path)
    if path.is_absolute():
        return path

    candidates = [
        common_path.parent / path,
        common_path.parent.parent / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _validate_extra_bools(
    section: str,
    values: Mapping[str, Any],
    names: tuple[str, ...],
) -> None:
    for name in names:
        value = values.get(name)
        if value is not None and not isinstance(value, bool):
            raise RunRefused(f"{section}.{name} must be a bool")


def _validate_print_every(value: object) -> None:
    """runtime.print_every: a positive integer, or unset.

    Refused here rather than where the reporter reads it, so --validate-only
    catches it before a round runs. The flag's parser already refuses anything
    else; this is the config spelling's check.
    """

    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RunRefused(f"runtime.print_every must be a positive integer, not {value!r}")


def _validate_unhonoured_client_options(config: FullConfig) -> None:
    """Refuse client options the configured rule would silently ignore.

    The claim this replaces was that only fedprox and scaffold needed the
    check, because every other rule raises in its constructor on an option its
    engine cannot honour. Seven rules do carry such a constructor check, and
    for five of them it is unreachable from a config: it tests an attribute
    the factory only sets for the rules that honour the option, so for the
    rules that do not it is permanently None and the check never fires. What
    the constructor does still cover is direct construction, in tests and in
    library use, where the caller passes the argument itself.

    A soft preflight issue would not do: validate_full_config runs only under
    --validate-only, so an ordinary run would still train with a config
    claiming cosine decay it never applied -- and record the claim in
    run.json, which is the damage.
    """

    entry = UNHONOURED_CLIENT_OPTIONS.get(config.client.update_rule)
    if entry is None:
        return

    reason, unhonoured = entry
    ignored = [name for name in unhonoured if name in config.client.extra]
    if not ignored:
        return
    rule = config.client.update_rule
    raise RunRefused(
        f"client.update_rule={rule!r} {reason}, and cannot honour: "
        + ", ".join(f"client.{n}" for n in ignored)
        + ". Remove them, or compare against a fedavg arm that does implement "
        "them."
    )


def _validate_local_sgd_options(config: FullConfig) -> None:
    if config.client.update_rule not in FIXED_LR_SGD_CLIENT_RULES:
        return

    extra = config.client.extra
    momentum = _required_extra(extra, "client", "momentum")
    if (
        isinstance(momentum, bool)
        or not isinstance(momentum, int | float)
        or not 0.0 <= float(momentum) < 1.0
    ):
        raise RunRefused("client.momentum must be in [0, 1)")

    weight_decay = _required_extra(extra, "client", "weight_decay")
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, int | float)
        or float(weight_decay) < 0.0
    ):
        raise RunRefused("client.weight_decay must be non-negative")

    _validate_learning_rate_schedule(config)

    nesterov = _required_extra(extra, "client", "nesterov")
    if not isinstance(nesterov, bool):
        raise RunRefused("client.nesterov must be a bool")
    if nesterov and float(momentum) <= 0.0:
        raise RunRefused("client.nesterov requires positive client.momentum")


def _validate_update_mode_options(config: FullConfig) -> None:
    """Validate the local-update mode settings the shared SGD engine reads.

    Its own check, for every rule in `UPDATE_MODES_BY_CLIENT_RULE`, against the
    modes that rule takes. It used to run from inside
    `_validate_local_sgd_options`, so only a fixed-rate rule could reach it.
    """

    rule = config.client.update_rule
    if rule not in UPDATE_MODE_CLIENT_RULES:
        return
    if rule in UPDATE_MODE_OPTIONAL_CLIENT_RULES and config.client.extra.get("update_mode") is None:
        return

    checked = [("update_mode", UPDATE_MODES_BY_CLIENT_RULE[rule])]
    if rule in FROZEN_WEIGHTING_CLIENT_RULES:
        checked.append(("frozen_gradient_weighting", frozenset(FROZEN_GRADIENT_WEIGHTINGS)))
    for name, allowed in checked:
        value = _required_extra(config.client.extra, "client", name)
        if not isinstance(value, str) or value not in allowed:
            choices = ", ".join(sorted(allowed))
            raise RunRefused(f"client.{name} must be one of: {choices}")

    _refuse_drop_last_under_full_gradient(config)


def _refuse_drop_last_under_full_gradient(config: FullConfig) -> None:
    """full_gradient is the gradient over every training sample, and drop_last
    leaves the last partial batch of every pass out of it -- a different subset
    each round under train_shuffle, with nothing in run.json to say so."""

    extra = config.client.extra
    if extra.get("update_mode") == FULL_GRADIENT_UPDATE_MODE and extra.get("drop_last"):
        raise RunRefused(
            "client.drop_last: true leaves the last partial batch out of every pass, "
            "and update_mode: full_gradient is the gradient over every training "
            "sample. Set client.drop_last to false."
        )


def amp_unsupported_sgd_engine_setting(config: FullConfig) -> str | None:
    """The setting that makes the shared SGD engine unrunnable under `use_amp`.

    `GradScaler` needs an optimizer exposing `param_groups`, which it reads in
    `unscale_`. `update_mode: frozen_batch_gradients` and `full_gradient` pass
    `_GradientOnlyOptimizer`, which computes gradients and never steps so the
    mode can combine them by hand. It wraps no optimizer, so it has no groups
    to expose and `GradScaler.unscale_` raises `AttributeError`, as it does for
    the same wrapper under `fedlalr` and `delta_sgd`.

    Singular. This returned a list while `max_grad_norm` was refused too, on
    the reasoning that a config can hit both and a refusal naming one of two
    sends the reader to change the wrong one. That refusal is gone:
    `_ClippingOptimizer` does expose `param_groups`, and a `fedavg` +
    `max_grad_norm` run under AMP tracks its fp32 trajectory
    (`tests/test_amp_composes_with_wrapped_optimizers.py` records the
    measurement). With one setting left, the plural handling was a branch
    nothing could reach. FINDINGS.csv P03-F05.

    Returns the offending `client` key, or None when the config is runnable.
    """

    if not config.runtime.use_amp:
        return None
    if config.client.update_rule not in SGD_ENGINE_CLIENT_RULES:
        return None
    mode = config.client.extra.get("update_mode")
    if mode not in GRADIENT_ONLY_UPDATE_MODES:
        return None
    return f"update_mode: {mode}"


def _validate_sgd_engine_amp(config: FullConfig) -> None:
    """Refuse at load what would otherwise raise in round 1.

    Same dual-layer shape as delta_sgd and fedlalr: this is the config-load
    half, `torch_classification.train_step`'s TypeError is the backstop for a
    client constructed directly. Unlike those two the refusal is per setting
    rather than per rule -- `fedavg` under AMP is fine until it is asked to
    freeze its gradients.
    """

    offending = amp_unsupported_sgd_engine_setting(config)
    if offending is None:
        return

    raise RunRefused(
        f"client.{offending} cannot run under runtime.use_amp: true -- the "
        "local step passes GradScaler a gradient collector with no "
        "param_groups to unscale. Set runtime.use_amp to false, or choose "
        "another update_mode."
    )


#: Algorithms whose server strategy and client update rule are a matched pair.
#: Naming one half without the other is refused here, on the run path, rather
#: than discovered in round 1 when the server finds the wrong key in a fit
#: payload -- or, for centralized, not discovered at all.
#:
#: Each value says what the two halves exchange that nothing else supplies. It
#: goes into the error message, because "requires" without "because" is the
#: kind of refusal people work around.
PAIRED_STRATEGIES: dict[str, str] = {
    "centralized": (
        "the strategy pools the dataset into a single client and the update "
        "rule supplies the local update applied to it"
    ),
    "scaffold": (
        "the server broadcasts a control variate the client corrects its step "
        "with, and the client returns the control delta the server needs to "
        "update it"
    ),
    "fedlalr": (
        "the server synchronizes the momentum and second moment that the "
        "client's local AMSGrad reads"
    ),
}


def _validate_output_dir(config: FullConfig) -> None:
    """Refuse an output_dir that resolves to wherever the user is standing.

    Path("") is Path("."), so an empty experiment.output_dir writes
    run.json, round_metrics.csv, the per-client CSVs and every checkpoint into
    the working directory -- for an HPC job, whatever the submit script last
    cd'd to, and for an interactive run, the repository. Nothing fails and
    nothing warns; the artifacts are simply somewhere else.

    A whitespace-only value is worse in a quieter way: it is not empty, so it
    survives every emptiness check, and it creates a directory whose name is
    the spaces.

    "." itself is allowed. Someone who writes it has said where they want the
    artifacts; the value this refuses is the one nobody chose.

    Preflight has reported the empty case for a long time, under
    experiment.output_dir_empty. validate_full_config runs only under
    --validate-only, so that report did not stop a run. This is on the run
    path, via load_config.
    """

    output_dir = config.experiment.output_dir
    if not isinstance(output_dir, str):
        raise RunRefused("experiment.output_dir must be a string")
    if not output_dir.strip():
        raise RunRefused(
            "experiment.output_dir is empty, which resolves to the working "
            "directory: the run would write its artifacts wherever it happened "
            'to be started. Set a path, or "." if that is what you mean.'
        )


#: The five config values that name a registered component, as
#: (config key, the attribute path, the registry attribute on
#: fedbrew.core.registry, what the registry holds). One shape, five rows,
#: because five copies of the same lookup is how one of them ends up different.
REGISTERED_NAMES: tuple[tuple[str, str, str, str], ...] = (
    ("server.strategy", "server.strategy", "server_strategies", "server strategies"),
    ("client.update_rule", "client.update_rule", "client_updates", "client update rules"),
    ("task.name", "task.name", "tasks", "tasks"),
    ("data.name", "data.name", "datasets", "data backends"),
    ("model.name", "model.name", "models", "models"),
)

#: Registries no run config selects from, and the loader that checks each one
#: instead. The generator registry is selected by ``dataset.name`` in a
#: generator config, so ``generate_from_config`` is where an unknown name is
#: refused. tests/test_registered_names.py holds every registry to one of the
#: two tables, so a registry cannot be added without saying where its names
#: are checked.
REGISTRIES_CHECKED_ELSEWHERE: dict[str, str] = {
    "generators": "fedbrew.data.generate.generator_spec",
}


def _validate_registered_names(config: FullConfig) -> None:
    """Refuse a component name nothing is registered under.

    A typo in server.strategy used to load clean. The run then died inside
    build_components on a registry lookup, which reports the name it could not
    find and not the names it has -- and only after the config had been
    accepted, the output directory prepared and, for a staged dataset, the data
    copied. validate_full_config named the valid components, and lives in the
    module only --validate-only reaches, so it did not stop any of that.

    Registration is deliberately best-effort. register_builtin_components
    imports every builder, and one of them can fail on a missing optional
    extra; refusing the config in that case would turn "transformers is not
    installed" into "unknown model: hf_causal_lm", which is a worse message
    about a different problem. If registration raises, this check stands down
    and the run fails at import with the real reason.
    """

    from fedbrew.core import registry

    try:
        registry.register_builtin_components()
    except Exception:  # noqa: BLE001 - an optional extra is a build-time problem
        return

    for key, path, attribute, label in REGISTERED_NAMES:
        value = config
        for part in path.split("."):
            value = getattr(value, part)
        if not isinstance(value, str) or not value.strip():
            raise RunRefused(f"{key} must be set")
        component: Any = getattr(registry, attribute)
        if component.exists(value):
            continue
        raise RunRefused(
            f"unknown {key}={value!r}. Registered {label}: {component.listing()}. "
            "A component defined outside the package is loaded through "
            "experiment.extensions."
        )


def _validate_paired_strategies(config: FullConfig) -> None:
    """Refuse half of a matched server/client pair.

    Every one of these was already reported by validate_full_config, and two of
    the three -- centralized and fedlalr -- were also refused here, each with
    its own hand-written pair of raises. Scaffold was not, so naming one half
    loaded clean: a scaffold client on a FedAvg server ran until the server
    read a fit payload with no model_state, and a scaffold server on a FedAvg
    client until it found no control_delta. Round one, after the data was
    staged and the first local iterations were spent.

    Table-driven rather than hand-written pairs of raises, so another paired
    algorithm is one row and cannot be added with only one direction covered.
    tests/test_paired_strategies.py checks the table against the pairing
    constants in factory.py and probes all six directions.
    """

    for name, reason in PAIRED_STRATEGIES.items():
        server_is = config.server.strategy == name
        client_is = config.client.update_rule == name
        if server_is and not client_is:
            raise RunRefused(
                f"server.strategy={name} requires client.update_rule={name}, "
                f"got {config.client.update_rule!r}: {reason}"
            )
        if client_is and not server_is:
            raise RunRefused(
                f"client.update_rule={name} requires server.strategy={name}, "
                f"got {config.server.strategy!r}: {reason}"
            )


#: Built-in models whose federated state is an adapter, not the whole model:
#: their builder marks the model adapter-scoped, and the causal-LM task then
#: federates the adapter tensors alone. Named here so a rule that cannot train
#: adapter state is refused at load rather than in round 1.
ADAPTER_SCOPED_MODELS = frozenset({"hf_causal_lm_lora"})


def _refuse_adapter_state_on_full_state_rules(config: FullConfig) -> None:
    """Refuse fedprox, scaffold and fedlalr on an adapter-scoped model.

    Each loaded the broadcast into the whole model or keyed its state by the
    model's parameter names, so the pairing got past load and failed in round
    1, after the model was built and the data read -- while chapter 07 listed
    LoRA under "any rule". FINDINGS.csv POST-F29.
    """

    from fedbrew.core.federated_state import (
        FULL_STATE_ONLY_CLIENT_RULES,
        adapter_state_refusal,
    )

    rule = config.client.update_rule
    if config.model.name in ADAPTER_SCOPED_MODELS and rule in FULL_STATE_ONLY_CLIENT_RULES:
        raise RunRefused(
            f"model.name={config.model.name} is adapter-scoped: " + adapter_state_refusal(rule)
        )


def _refuse_ignored_active_target_weighting(config: FullConfig) -> None:
    """Refuse model.active_target_weighting: true under fedprox or scaffold.

    Both report their post-fit evaluation count and never call the task's
    aggregation-weight hook, which is the only reader of the key, so the run
    recorded a weighting it did not use. The dataset's default is judged where
    the manifest is read: `factory._model_config` and preflight's data check.
    FINDINGS.csv POST-F30.
    """

    from fedbrew.core.federated_state import active_target_weighting_refusal

    refusal = active_target_weighting_refusal(
        config.client.update_rule, config.task.name, config.model.extra
    )
    if refusal is not None:
        raise RunRefused(refusal)


def _validate_local_adamw_options(config: FullConfig) -> None:
    if config.client.update_rule != "local_adamw":
        return

    extra = config.client.extra
    weight_decay = _required_extra(extra, "client", "weight_decay")
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, int | float)
        or float(weight_decay) < 0.0
    ):
        raise RunRefused("client.weight_decay must be non-negative")

    for name in ("beta1", "beta2"):
        value = _required_extra(extra, "client", name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not 0.0 <= float(value) < 1.0
        ):
            raise RunRefused(f"client.{name} must be in [0, 1)")

    epsilon = _required_extra(extra, "client", "epsilon")
    if isinstance(epsilon, bool) or not isinstance(epsilon, int | float) or float(epsilon) <= 0.0:
        raise RunRefused("client.epsilon must be positive" + yaml_number_cause(epsilon))

    _validate_learning_rate_schedule(config)


def _validate_delta_sgd_options(config: FullConfig) -> None:
    """Validate the Delta-SGD client options (arXiv:2306.11201).

    Only eta_0 is required. theta_0, gamma and delta carry the paper's own
    defaults, which its authors used unchanged across every experiment, so
    restating them in each config would only invite drift.
    """

    if config.client.update_rule != "delta_sgd":
        return

    if config.client.learning_rate is not None:
        raise RunRefused(
            "delta_sgd measures its own step size from the local smoothness; "
            "remove client.learning_rate and set client.eta_0"
        )

    _require_positive_number(_required_extra(config.client.extra, "client", "eta_0"), "eta_0")
    for name in ("theta_0", "gamma", "delta"):
        if name in config.client.extra:
            _require_positive_number(config.client.extra[name], name)

    eta_max = config.client.extra.get("eta_max")
    if eta_max is not None:
        _require_positive_number(eta_max, "eta_max")

    _refuse_drop_last_under_full_gradient(config)
    for name, allowed in (
        ("update_mode", DELTA_SGD_UPDATE_MODES),
        ("frozen_gradient_weighting", FROZEN_GRADIENT_WEIGHTINGS),
    ):
        if name not in config.client.extra:
            continue
        value = config.client.extra[name]
        if not isinstance(value, str) or value not in allowed:
            choices = ", ".join(sorted(allowed))
            raise RunRefused(f"client.{name} must be one of: {choices}")

    # The rule reads the raw gradient off .grad, which the AMP path consumes
    # inside GradScaler.step instead of leaving there.
    if config.runtime.use_amp:
        raise RunRefused("delta_sgd is incompatible with runtime.use_amp: true")


def _refuse_frozen_off_examples(config: FullConfig) -> None:
    """Refuse `frozen_batch_gradients` on a task whose loss is not an example mean.

    Every rule that reads `update_mode` combines the frozen gradients by
    example count, equally or by sum, and none of the three is the gradient
    of the pass for a loss that averages over tokens. FINDINGS.csv POST-F19.
    """

    task = config.task.name
    if config.client.extra.get("update_mode") != "frozen_batch_gradients":
        return
    if task not in NON_EXAMPLE_MEAN_TASKS:
        return
    raise RunRefused(
        "update_mode: frozen_batch_gradients combines batch gradients by example "
        f"count, and task {task!r} averages its training loss over something else "
        "(active target tokens), so the combined update would not be the gradient "
        "of the pass (FINDINGS.csv POST-F19). Use update_mode: full_gradient, which "
        "weights each batch by what the task's loss averages over."
    )


def _validate_fedavg_ft_options(config: FullConfig) -> None:
    """Validate the FedAvg+FT client and the scope it needs to mean anything.

    fedavg_ft trains identically to fedavg; the only thing that distinguishes
    the arm is the personalized evaluation pass. Running it under
    model_scope: global would produce a plain fedavg run under a name implying
    otherwise, so that pairing is rejected rather than silently wasted.
    """

    if config.client.update_rule != "fedavg_ft":
        return

    epochs = _required_extra(config.client.extra, "client", "finetune_epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise RunRefused("client.finetune_epochs must be a positive integer")

    learning_rate = config.client.extra.get("finetune_learning_rate")
    if learning_rate is not None:
        _require_positive_number(learning_rate, "finetune_learning_rate")

    if config.evaluation.model_scope == "global":
        raise RunRefused(
            "client.update_rule=fedavg_ft needs evaluation.model_scope "
            "personal or both; under global it trains and evaluates exactly "
            "like plain fedavg"
        )


def _validate_fedlalr_options(config: FullConfig) -> None:
    """Validate the FedLALR pairing and client options (arXiv:2309.09719).

    The server synchronizes the momentum and second moment that the client's
    local AMSGrad reads, so either half alone would silently change what runs.
    beta1, beta2 and epsilon have defaults; clients/torch_fedlalr_client.py says
    which of them are the paper's.
    """

    # The pairing is enforced by _validate_paired_strategies, which runs first,
    # so reaching here with one half set is impossible: either both or neither.
    if config.client.update_rule != "fedlalr":
        return

    # alpha, the local learning rate, which the paper sets per task.
    if config.client.learning_rate is None or config.client.learning_rate <= 0.0:
        raise RunRefused("fedlalr requires client.learning_rate (alpha) > 0")

    for name in ("beta1", "beta2"):
        if name not in config.client.extra:
            continue
        value = config.client.extra[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not 0.0 <= float(value) < 1.0
        ):
            raise RunRefused(f"client.{name} must be in [0, 1)")

    if "epsilon" in config.client.extra:
        _require_positive_number(config.client.extra["epsilon"], "epsilon")

    # The AMSGrad update reads raw gradients off .grad, which GradScaler.step
    # consumes instead of leaving there.
    if config.runtime.use_amp:
        raise RunRefused("fedlalr is incompatible with runtime.use_amp: true")


# No _validate_scaffold_options / _validate_fedprox_options: both existed only
# to refuse runtime.use_amp, and that refusal was lifted once it was measured
# rather than predicted. Both rules correct .grad inside a wrapper's .step(),
# and GradScaler.step unscales .grad before delegating to a wrapped optimizer;
# tests/test_amp_composes_with_wrapped_optimizers.py records the measurement,
# and docs/07 sections 4.3-4.4 the reasoning.
#
# fedlalr and delta_sgd keep theirs: they step through _GradientOnlyOptimizer,
# which exposes no param_groups, and GradScaler needs those. That refusal is
# now a measurement too, not a prediction.


def _require_positive_number(value: object, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise RunRefused(
            f"client.{name} must be a finite positive number" + yaml_number_cause(value)
        )


def _validate_learning_rate_schedule(config: FullConfig) -> None:
    extra = config.client.extra
    schedule = _required_extra(extra, "client", "learning_rate_schedule")
    if not isinstance(schedule, str) or schedule not in {"constant", "cosine"}:
        raise RunRefused("client.learning_rate_schedule must be constant or cosine")
    min_learning_rate = _required_extra(extra, "client", "min_learning_rate")
    if (
        isinstance(min_learning_rate, bool)
        or not isinstance(min_learning_rate, int | float)
        or float(min_learning_rate) < 0.0
        or config.client.learning_rate is None
        or float(min_learning_rate) > config.client.learning_rate
    ):
        raise RunRefused("client.min_learning_rate must be in [0, learning_rate]")


def _required_extra(values: Mapping[str, Any], section: str, name: str) -> Any:
    if name not in values:
        raise RunRefused(f"{section}.{name} must be configured")
    return values[name]


#: The five spellings that build a FedOptServer: the four named aliases and
#: the bare `fedopt`, which names its optimizer in `server.server_optimizer`.
FEDOPT_STRATEGIES = frozenset({"fedavgm", "fedadam", "fedyogi", "fedadagrad", "fedopt"})


def fedopt_optimizer_name(config: FullConfig) -> str:
    """Which server optimizer a FedOpt config will actually run.

    The four named strategies *are* the optimizer; `fedopt` names one in
    `server.server_optimizer`. Normalised the way ``FedOptServer`` normalises
    it, so `FedAvgM ` and `fedavgm` are judged alike.

    Args:
        config: Any config; only meaningful for a `FEDOPT_STRATEGIES` one.

    Returns:
        The optimizer name, or the strategy itself when `fedopt` names none.
        That fallback has no row in ``UNREAD_FEDOPT_HYPERPARAMETERS``, so every
        hyperparameter stays required and the missing name is left to be the
        error it already is.
    """

    if config.server.strategy != "fedopt":
        return config.server.strategy
    named = config.server.extra.get("server_optimizer")
    if not isinstance(named, str) or not named.strip():
        return config.server.strategy
    return named.strip().lower()


def _validate_server_algorithm_options(config: FullConfig) -> None:
    extra = config.server.extra
    _validate_aggregation_weighting(config)
    if config.server.strategy in FEDOPT_STRATEGIES:
        optimizer = fedopt_optimizer_name(config)
        reason, unread = unread_fedopt_hyperparameters(optimizer)
        never_read = [name for name in unread if name in extra]
        if never_read:
            # Not "accepted and dropped": the schema used to require all four
            # of these of every FedOpt arm, so configs/femnist/fedadagrad.yaml
            # carried `beta2: 0.99   # unused by FedAdagrad; required by the
            # config schema` -- a placeholder the run then copied into
            # run.json's config block, where it reads like a setting that
            # shaped the result. Same rule as UNHONOURED_CLIENT_OPTIONS on the
            # client side. P01-F07.
            raise RunRefused(
                f"server_optimizer={optimizer!r} {reason}, and never reads: "
                + ", ".join(f"server.{name}" for name in never_read)
                + ". Remove them; no value of theirs would change the run."
            )
        for name in FEDOPT_HYPERPARAMETERS:
            if name in unread:
                continue
            value = _required_extra(extra, "server", name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise RunRefused(f"server.{name} must be numeric" + yaml_number_cause(value))
            # The bound, not only the type. This loop checked numeric-ness
            # alone and the bounds lived in validation.py, which only
            # --validate-only reaches. Three of the four were caught late by
            # FedOptServer.__init__; `tau: 0` was caught nowhere and turns
            # every exactly-zero delta coordinate into 0/0. P04-F05.
            violation = fedopt_bound_violation(name, float(value))
            if violation is not None:
                raise RunRefused(violation)


def _validate_aggregation_weighting(config: FullConfig) -> None:
    """Check the model-aggregation weighting mode, if one is configured."""

    if "aggregation_weighting" not in config.server.extra:
        return
    value = config.server.extra["aggregation_weighting"]
    if not isinstance(value, str) or value not in SUPPORTED_AGGREGATION_WEIGHTING:
        raise RunRefused(
            "server.aggregation_weighting must be one of: "
            + ", ".join(sorted(SUPPORTED_AGGREGATION_WEIGHTING))
        )


def _validate_metrics(name: str, metrics: object) -> None:
    if not isinstance(metrics, list) or not all(isinstance(metric, str) for metric in metrics):
        raise RunRefused(f"{name} must be a list of strings")


def _load_simple_yaml(text: str) -> dict[str, Any]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    data, index = _parse_mapping(lines, 0, 0)
    if index != len(lines):
        raise RunRefused("Could not parse YAML content")
    return data


def _parse_mapping(
    lines: list[str],
    index: int,
    indent: int,
) -> tuple[dict[str, Any], int]:
    data: dict[str, Any] = {}
    while index < len(lines):
        line = lines[index]
        current_indent = _indent_of(line)
        if current_indent < indent:
            break
        if current_indent != indent:
            raise RunRefused(f"Unexpected indentation: {line}")

        stripped = line.strip()
        key, separator, value = stripped.partition(":")
        if separator != ":":
            raise RunRefused(f"Expected mapping entry: {line}")

        index += 1
        value = value.strip()
        if value:
            data[key] = _parse_scalar(value)
        elif index < len(lines) and lines[index].strip().startswith("- "):
            data[key], index = _parse_list(lines, index, indent + 2)
        else:
            data[key], index = _parse_mapping(lines, index, indent + 2)
    return data, index


def _parse_list(lines: list[str], index: int, indent: int) -> tuple[list[Any], int]:
    values: list[Any] = []
    while index < len(lines):
        line = lines[index]
        current_indent = _indent_of(line)
        if current_indent < indent:
            break
        if current_indent != indent or not line.strip().startswith("- "):
            raise RunRefused(f"Expected list item: {line}")
        values.append(_parse_scalar(line.strip()[2:].strip()))
        index += 1
    return values, index


def _parse_scalar(value: str) -> Any:
    if value == "null":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))
