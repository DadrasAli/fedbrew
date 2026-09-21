"""Preflight validation helpers for experiment configurations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from fedbrew.core.config import (
    DELTA_SGD_UPDATE_MODES,
    FROZEN_GRADIENT_WEIGHTINGS,
    FullConfig,
    amp_unsupported_sgd_engine_setting,
    checkpoint_selection_problem,
    fedopt_optimizer_name,
)
from fedbrew.core.console import (
    AMBER,
    DIM,
    DONE,
    FAIL,
    FAINT,
    RED,
    WARN,
    Rail,
    Row,
    Surface,
    measure,
)
from fedbrew.core.data_staging import resolve_staging_root
from fedbrew.core.factory import (
    AMP_AWARE_TASKS,
    CENTRALIZED_CLIENT_RULES,
    CENTRALIZED_SERVER_STRATEGIES,
    DELTA_SGD_CLIENT_RULES,
    FEDAVG_FT_CLIENT_RULES,
    FEDLALR_CLIENT_RULES,
    FEDLALR_SERVER_STRATEGIES,
    FEDOPT_SERVER_STRATEGIES,
    SCAFFOLD_CLIENT_RULES,
    SCAFFOLD_SERVER_STRATEGIES,
    SUPPORTED_AGGREGATION_WEIGHTING,
    _model_config,
    is_extension,
    model_data_shape_mismatch,
)
from fedbrew.core.federated_state import active_target_weighting_refusal
from fedbrew.core.paths import (
    expand_path,
    has_unexpanded_env,
    resolve_data_path,
    resolve_output_dir,
)
from fedbrew.core.refusal import yaml_number_cause
from fedbrew.core.registry import (
    client_updates,
    datasets,
    models,
    register_builtin_components,
    server_strategies,
    tasks,
)
from fedbrew.servers.fedopt import (
    FEDOPT_HYPERPARAMETERS,
    fedopt_bound_violation,
    unread_fedopt_hyperparameters,
)

SEVERITIES = {"info", "warning", "error"}
FEDAVG_COMPATIBLE_CLIENTS = {
    "local_sgd",
    "fedavg",
    "local_adamw",
    "fedprox",
    "delta_sgd",
    "fedavg_ft",
}
FEDOPT_OPTIMIZERS = {"fedavgm", "fedadam", "fedyogi", "fedadagrad"}

#: Every registered server strategy, classified by whether a run weighting
#: clients by example count -- the default, and what every shipped arm does --
#: departs from what that strategy's source publishes. A strategy mapped to a
#: code emits that notice at ``info`` when ``server.aggregation_weighting`` is
#: not ``uniform``; a strategy mapped to ``None`` is exempt, and the reason is
#: beside it.
#:
#: This exists because the notice was written twice, for fedlalr and delta_sgd,
#: and the two arms that needed it most were not among them: SCAFFOLD's
#: pseudocode averages uniformly *and* its server scales the control variate by
#: 1/num_clients regardless, so an example-weighted run has x and c estimating
#: two different means. Nothing said so, while the same deviation on fedlalr
#: was reported. Two hand-written notices could not notice a third arm was
#: missing; this table can, because tests/test_aggregation_weighting_notice.py
#: requires every registered strategy to appear in it.
#:
#: delta_sgd is absent by construction: it is a client rule on a FedAvg server,
#: so it is not keyed by strategy. Its notice is in _validate_delta_sgd and its
#: code is checked beside these.
AGGREGATION_WEIGHTING_NOTICE: Mapping[str, str | None] = {
    "scaffold": "algorithm.scaffold_aggregation_weighting",
    "fedopt": "algorithm.fedopt_aggregation_weighting",
    "fedavgm": "algorithm.fedopt_aggregation_weighting",
    "fedadam": "algorithm.fedopt_aggregation_weighting",
    "fedyogi": "algorithm.fedopt_aggregation_weighting",
    "fedadagrad": "algorithm.fedopt_aggregation_weighting",
    "fedlalr": "algorithm.fedlalr_aggregation_weighting",
    # FedAvg is the source of example weighting: McMahan et al. Algorithm 1
    # averages by n_k, so the default is the paper and there is nothing to say.
    "fedavg": None,
    # The centralized baseline runs the FedAvg server over one pooled client.
    # Averaging one result is the identity, so no weighting is applied at all.
    "centralized": None,
}
FEDPROX_CLIENT_RULES = {"fedprox"}
#: Client rules that compute their own step size and so reject an explicit
#: client.learning_rate.
DERIVED_LEARNING_RATE_CLIENT_RULES = {*DELTA_SGD_CLIENT_RULES}


@dataclass(slots=True)
class ValidationIssue:
    """One preflight validation issue."""

    severity: str
    code: str
    message: str
    hint: str | None = None

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"Unknown validation severity: {self.severity}")


@dataclass(slots=True)
class ValidationReport:
    """Structured validation report for one config."""

    config_path: str | None
    issues: list[ValidationIssue]
    num_errors: int = field(init=False)
    num_warnings: int = field(init=False)
    is_valid: bool = field(init=False)

    def __post_init__(self) -> None:
        self.num_errors = sum(1 for issue in self.issues if issue.severity == "error")
        self.num_warnings = sum(1 for issue in self.issues if issue.severity == "warning")
        self.is_valid = self.num_errors == 0


def run_checks(
    config: FullConfig,
    on_check: Callable[[str, list[ValidationIssue]], None] | None = None,
) -> list[ValidationIssue]:
    """Run every preflight check, reporting each one's findings as it lands.

    `on_check` is called once per check with that check's own issues, which is
    what lets a rail settle a line at the moment the check finishes rather
    than after all ten have run.
    """

    issues: list[ValidationIssue] = []
    # CHECKS is defined at the bottom of this module, after the functions it
    # names; the lookup happens here, at call time.
    for name, check in CHECKS:
        already = len(issues)
        check(config, issues)
        if on_check is not None:
            on_check(name, issues[already:])
    return issues


def validate_full_config(
    config: FullConfig,
    config_path: str | None = None,
    on_check: Callable[[str, list[ValidationIssue]], None] | None = None,
) -> ValidationReport:
    """Validate a fully resolved experiment config without starting training."""

    return ValidationReport(config_path=config_path, issues=run_checks(config, on_check))


def report_from_exception(
    exc: Exception,
    config_path: str | None = None,
    code: str = "config.load_failed",
) -> ValidationReport:
    """Build a validation report for config loading or override failures."""

    return ValidationReport(
        config_path=config_path,
        issues=[
            ValidationIssue(
                severity="error",
                code=code,
                message=str(exc),
            )
        ],
    )


#: How one issue is marked and toned. Errors are red, warnings amber, and an
#: info issue is neither -- it reports something true about the config that the
#: reader may not know, and painting it would spend a warning colour on it.
_SEVERITY_STYLE = {
    "error": (FAIL, RED),
    "warning": (WARN, AMBER),
    "info": (DONE, None),
}


def stream_checks(
    config: FullConfig,
    config_path: str | None,
    rail: Rail,
) -> ValidationReport:
    """Run preflight with each check settling on the rail as it lands.

    One line per check. A clean check is its marker and its name and nothing
    else -- there is no result to report, and "ok" ten times is ten lines of
    noise around the two that matter. A check with findings settles under the
    worst severity it saw and lists them underneath, so a finding is attached
    to the check that raised it rather than arriving as a code the reader has
    to place.

    A failing check sets `rail.stopped`. Nothing after it stops running --
    preflight exists so every problem can be fixed in one pass -- but the rail
    has stopped in the sense that matters: it will not go on to settle into a
    plan, because there is no plan, only a config that cannot run.
    """

    issues: list[ValidationIssue] = []

    def settle(name: str, found: list[ValidationIssue]) -> None:
        issues.extend(found)
        with rail.stage(name) as stage:
            errors = sum(1 for issue in found if issue.severity == "error")
            warnings = sum(1 for issue in found if issue.severity == "warning")
            if errors:
                stage.fail(_count_summary(errors, warnings))
            elif warnings:
                stage.warn(_count_summary(errors, warnings))
            elif found:
                # info only: something the reader may not know, but nothing
                # they need to act on.
                stage.done(_plural(len(found), "note"))
            else:
                stage.done()
        if found:
            print_issues(rail.surface, found, prefix="  ")

    run_checks(config, on_check=lambda name, found: settle(name, found))
    return ValidationReport(config_path=config_path, issues=issues)


def _count_summary(errors: int, warnings: int) -> str:
    parts = []
    if errors:
        parts.append(_plural(errors, "error"))
    if warnings:
        parts.append(_plural(warnings, "warning"))
    return ", ".join(parts)


def print_validation_verdict(report: ValidationReport, surface: Surface) -> None:
    """The last word: what the reader has to do, and every error, together.

    The errors are deliberately printed twice -- once inline at the check that
    raised them, once here. A rail is read while it is running and a verdict is
    read when it stops, and a reader who watched a 10-line rail scroll past
    should not have to scroll back up to find out what to fix. What is *not*
    repeated is the warnings: they are advisory, they are attached to the check
    that knows why, and repeating them here would bury the errors among them.
    """

    errors = [issue for issue in report.issues if issue.severity == "error"]

    if errors:
        surface.rule("FIX ERRORS BEFORE RUNNING", tone=RED)
        for issue in errors:
            surface.line(
                f"{issue.code}  {issue.message}",
                tone=RED,
                marker=FAIL,
                marker_tone=RED,
                wrap=True,
            )
    elif report.num_warnings:
        surface.rule("READY — REVIEW WARNINGS", tone=AMBER)
    else:
        surface.rule("READY TO RUN")
    surface.line(
        f"{_plural(report.num_errors, 'error')}  {_plural(report.num_warnings, 'warning')}",
        tone=DIM,
    )


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def print_issues(
    surface: Surface,
    issues: list[ValidationIssue],
    *,
    prefix: str = "",
) -> None:
    """One row per issue, its hint indented under it.

    A row rather than a bordered table cell: a hint is a sentence, and the
    four-column table this replaced wrapped every one of them into a
    two-word-wide column on any terminal narrower than very wide.

    `prefix` indents the block, which is how findings sit under the rail stage
    that raised them rather than beside it.
    """

    width = measure([Row(issue.code, issue.message) for issue in issues])
    for issue in issues:
        marker, tone = _SEVERITY_STYLE.get(issue.severity, (DONE, None))
        surface.row(
            Row(
                issue.code,
                issue.message,
                tone=tone,
                marker=marker,
                marker_tone=tone or FAINT,
            ),
            width=width,
            prefix=prefix,
            wrap=True,
        )
        if issue.hint:
            surface.line(issue.hint, tone=FAINT, prefix=f"{prefix}    ", wrap=True)


def _register_components(issues: list[ValidationIssue]) -> None:
    try:
        register_builtin_components()
    except Exception as exc:  # pragma: no cover - defensive around optional imports.
        _add(
            issues,
            "error",
            "registry.load_failed",
            f"Could not register built-in components: {exc}",
        )


def _validate_experiment(config: FullConfig, issues: list[ValidationIssue]) -> None:
    experiment = config.experiment
    if not _non_empty(experiment.name):
        _add(issues, "error", "experiment.name_empty", "experiment.name is empty")

    if not _non_empty(experiment.output_dir):
        _add(
            issues,
            "error",
            "experiment.output_dir_empty",
            "experiment.output_dir is empty",
        )
    else:
        output_dir = resolve_output_dir(experiment.output_dir)
        if has_unexpanded_env(str(output_dir)):
            _add(
                issues,
                "warning",
                "experiment.output_dir_unresolved_env",
                f"Output directory contains unresolved environment variables: {output_dir}",
                "Set FL_OUTPUT_ROOT or use a concrete output path before launching.",
            )
        if output_dir.exists() and not experiment.use_run_subdir:
            try:
                has_files = any(output_dir.iterdir())
            except OSError as exc:
                _add(
                    issues,
                    "warning",
                    "experiment.output_dir_unreadable",
                    f"Could not inspect output directory {output_dir}: {exc}",
                )
            else:
                if has_files:
                    _add(
                        issues,
                        "warning",
                        "experiment.output_dir_not_empty",
                        f"Output directory already contains files: {output_dir}",
                        "Use --use-run-subdir or choose a fresh output directory for long runs.",
                    )

    if experiment.seed is not None and not _non_negative_int(experiment.seed):
        _add(
            issues,
            "error",
            "experiment.seed_invalid",
            "experiment.seed must be a non-negative integer",
        )


def _validate_server(config: FullConfig, issues: list[ValidationIssue]) -> None:
    server = config.server
    if not _positive_int(server.global_rounds):
        _add(
            issues,
            "error",
            "server.global_rounds_invalid",
            "server.global_rounds must be > 0",
        )
    rate, probability = server.participation_rate, server.participation_probability
    if (rate is None) == (probability is None):
        _add(
            issues,
            "error",
            "server.participation_scheme",
            "server needs exactly one of participation_rate and participation_probability",
        )
    if rate is not None and not _participation_rate(rate):
        _add(
            issues,
            "error",
            "server.participation_rate_invalid",
            "server.participation_rate must be in (0, 1]",
        )
    if probability is not None and not _participation_rate(probability):
        _add(
            issues,
            "error",
            "server.participation_probability_invalid",
            "server.participation_probability must be in (0, 1]",
        )
    if not _non_empty(server.strategy):
        _add(issues, "error", "server.strategy_empty", "server.strategy is empty")
    elif not server_strategies.exists(server.strategy):
        _add(
            issues,
            "error",
            "server.strategy_unknown",
            f"Unknown server strategy: {server.strategy}",
            f"Registered strategies: {server_strategies.listing()}",
        )


def _validate_client(config: FullConfig, issues: list[ValidationIssue]) -> None:
    client = config.client
    if not _positive_int(client.local_iterations):
        _add(
            issues,
            "error",
            "client.local_iterations_invalid",
            "client.local_iterations must be > 0",
        )
    if not _positive_int(client.batch_size):
        _add(
            issues,
            "error",
            "client.batch_size_invalid",
            "client.batch_size must be > 0",
        )
    if not _non_empty(client.update_rule):
        _add(issues, "error", "client.update_rule_empty", "client.update_rule is empty")
    elif not client_updates.exists(client.update_rule):
        _add(
            issues,
            "error",
            "client.update_rule_unknown",
            f"Unknown client update_rule: {client.update_rule}",
            f"Registered client rules: {client_updates.listing()}",
        )

    if is_extension(client_updates, client.update_rule):
        # Neither required nor refused: whether a rule needs a configured step
        # size is a fact about the rule, and nothing here knows it for a rule
        # the package did not write. The rule validates its own -- chapter 12.
        if client.learning_rate is not None and not _positive_number(client.learning_rate):
            _add(
                issues,
                "error",
                "client.learning_rate_invalid",
                "client.learning_rate must be > 0",
            )
    elif client.update_rule not in DERIVED_LEARNING_RATE_CLIENT_RULES:
        if client.learning_rate is None:
            _add(
                issues,
                "error",
                "client.learning_rate_missing",
                "client.learning_rate must be set",
            )
        elif not _positive_number(client.learning_rate):
            _add(
                issues,
                "error",
                "client.learning_rate_invalid",
                "client.learning_rate must be > 0",
            )
    elif client.learning_rate is not None:
        _add(
            issues,
            "error",
            "client.learning_rate_unexpected",
            f"{client.update_rule} derives its own learning rate",
            "delta_sgd measures it from the local smoothness starting at eta_0.",
        )

    for name in (
        "train_shuffle",
        "eval_shuffle",
        "drop_last",
        "nesterov",
    ):
        _validate_extra_bool(issues, client.extra, "client", name)


def _validate_task(config: FullConfig, issues: list[ValidationIssue]) -> None:
    task = config.task
    if not _non_empty(task.name):
        _add(issues, "error", "task.name_empty", "task.name is empty")
        return
    if not tasks.exists(task.name):
        _add(
            issues,
            "error",
            "task.name_unknown",
            f"Unknown task: {task.name}",
            f"Registered tasks: {tasks.listing()}",
        )
        return

    # The four AMP refusals below are per algorithm; this one is per task, and
    # it is the only one that was missing. `use_amp: true` on a causal-LM
    # config was accepted, echoed into run.json, and ignored -- the task takes
    # no such argument and its train_step has no autocast. `factory` raises on
    # it; this is the same fact said before anything is built. P03-F07.
    if config.runtime.use_amp and task.name not in AMP_AWARE_TASKS:
        _add(
            issues,
            "error",
            "task.amp_unsupported",
            f"task {task.name!r} has no mixed-precision path, so runtime.use_amp "
            "would be recorded in run.json and ignored",
            "Set runtime.use_amp to false.",
        )


def _validate_data(config: FullConfig, issues: list[ValidationIssue]) -> None:
    data = config.data
    if not _non_empty(data.name):
        _add(issues, "error", "data.name_empty", "data.name is empty")
        return
    if not datasets.exists(data.name):
        _add(
            issues,
            "error",
            "data.name_unknown",
            f"Unknown data backend: {data.name}",
            f"Registered data backends: {datasets.listing()}",
        )

    if data.name != "manifest_dataset":
        return

    data_path = data.path
    if not isinstance(data_path, str) or not data_path.strip():
        _add(issues, "error", "data.path_empty", "data.path is empty")
        return

    manifest_path = resolve_data_path(data_path)
    if has_unexpanded_env(str(manifest_path)):
        _add(
            issues,
            "error",
            "data.manifest_path_unresolved_env",
            f"Manifest path contains unresolved environment variables: {manifest_path}",
            "Set FL_DATA_ROOT or use a concrete generated manifest path.",
        )
        return

    if not manifest_path.exists():
        _add(
            issues,
            "error",
            "data.manifest_missing",
            f"Manifest path does not exist: {manifest_path}",
            "Run fedbrew generate with the matching data/configs config before training.",
        )
        return

    from fedbrew.data.manifest_validation import IDENTICAL_TO_TRAIN, validate_manifest

    issues.extend(
        validate_manifest(
            manifest_path,
            require_client_test=_evaluates_split(config, "test"),
            require_global_test=_evaluates_central(config),
        )
    )
    mismatch = model_data_shape_mismatch(_model_config(config), _manifest(manifest_path))
    if mismatch is not None:
        # The refusal build_components makes, reported before the job starts
        # rather than after the dataset has been read. P11-F01, and the same
        # rule as P07-F09: preflight reports the run path's answer.
        _add(issues, "error", "model.data_shape_mismatch", mismatch)
    ignored = active_target_weighting_refusal(
        config.client.update_rule,
        config.task.name,
        config.model.extra,
        _manifest(manifest_path).get("task"),
    )
    if ignored is not None:
        # The refusal _model_config makes once the manifest is read. POST-F30.
        _add(issues, "error", "client.active_target_weighting_ignored", ignored)

    if _client_test_source(manifest_path) == IDENTICAL_TO_TRAIN:
        _add(
            issues,
            "info",
            "data.test_is_training_data",
            "this dataset's client test split is its training data "
            f"(client_test_source={IDENTICAL_TO_TRAIN}): test_* and "
            "central_test_* are training numbers",
            "The generator declared that nothing is held out, so nothing "
            "measured on this data generalises. Report the numbers as what "
            "they are.",
        )


def _manifest(manifest_path: Any) -> Mapping[str, Any]:
    """The manifest as a mapping, or empty if it cannot be read.

    validate_manifest has already reported an unreadable manifest, so a
    second complaint here would be noise; an empty mapping declares no shape
    and so disagrees with nothing.
    """

    import json

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return manifest if isinstance(manifest, Mapping) else {}


def _client_test_source(manifest_path: Any) -> str | None:
    """The manifest's ``client_test_source``, or None if it cannot be read.

    validate_manifest has already reported an unreadable manifest as an
    error; this only decides whether the training-data note applies.
    """

    import json

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = manifest.get("client_test_source") if isinstance(manifest, Mapping) else None
    return value if isinstance(value, str) else None


#: The two builders that load weights from a prepared offline snapshot.
_ASSET_BACKED_MODELS = frozenset({"hf_causal_lm", "hf_causal_lm_lora"})


def _validate_model_assets(config: FullConfig, issues: list[ValidationIssue]) -> None:
    """Check a prepared model snapshot exists, before the run needs it.

    The builder already refuses to proceed without one -- it requires
    local_files_only, requires asset_manifest, and raises AssetManifestError
    naming the prepare-llm command when either asset directory is absent. So a
    missing snapshot has never caused a silent download or a hang.

    What it did cause is a late failure. The builder runs inside the round
    loop's setup, after config load, data staging and the dataset read, so a
    typo in asset_manifest surfaced minutes into a job that preflight had just
    called READY TO RUN. Preflight is where that belongs: it already checks the
    *data* manifest the same way, and this is the model's counterpart.
    """

    if config.model.name not in _ASSET_BACKED_MODELS:
        return

    raw_manifest = config.model.extra.get("asset_manifest")
    if not isinstance(raw_manifest, str) or not raw_manifest.strip():
        _add(
            issues,
            "error",
            "model.asset_manifest_missing",
            f"{config.model.name} requires model.asset_manifest",
            "Point it at the asset_manifest.json written by fedbrew prepare-llm.",
        )
        return

    manifest_path = expand_path(raw_manifest)
    if has_unexpanded_env(str(manifest_path)):
        # A warning, not an error: an $FL_CACHE_ROOT path is normal for a
        # config meant to run inside a job that exports it, and validating from
        # a login node cannot tell whether the job will.
        _add(
            issues,
            "warning",
            "model.asset_manifest_unresolved_env",
            f"Asset manifest path contains unresolved environment variables: {manifest_path}",
            "Export it before the job, or validate from inside one.",
        )
        return

    from fedbrew.data.llm_assets.manifest import AssetManifestError, load_asset_manifest

    preparation_config = config.model.extra.get("preparation_config")
    try:
        load_asset_manifest(
            manifest_path,
            preparation_config=preparation_config,
            require_assets=True,
        )
    except AssetManifestError as error:
        # The loader's message already names the prepare-llm command to run.
        _add(
            issues,
            "error",
            "model.asset_manifest_unusable",
            str(error),
            "Prepare the snapshot on a machine with network access first.",
        )


def _validate_model(config: FullConfig, issues: list[ValidationIssue]) -> None:
    model = config.model
    if not _non_empty(model.name):
        _add(issues, "error", "model.name_empty", "model.name is empty")
        return
    if not models.exists(model.name):
        _add(
            issues,
            "error",
            "model.name_unknown",
            f"Unknown model: {model.name}",
            f"Registered models: {models.listing()}",
        )

    _validate_model_assets(config, issues)

    if model.name == "mlp":
        for name in ("input_dim", "hidden_dim", "num_classes"):
            value = getattr(model, name)
            if not _positive_int(value):
                _add(
                    issues,
                    "error",
                    f"model.{name}_invalid",
                    f"model.{name} must be > 0 for mlp",
                )

    if model.name in {"cnn", "small_cnn"}:
        input_channels = model.extra.get("input_channels")
        if not _positive_int(input_channels):
            _add(
                issues,
                "error",
                "model.input_channels_invalid",
                f"model.input_channels must be > 0 for {model.name}",
            )
        for name in ("hidden_dim", "num_classes"):
            value = getattr(model, name)
            if not _positive_int(value):
                _add(
                    issues,
                    "error",
                    f"model.{name}_invalid",
                    f"model.{name} must be > 0 for {model.name}",
                )

    if model.name == "femnist_resnet18":
        required_dimensions = {
            "input_channels": model.extra.get("input_channels"),
            "base_channels": model.extra.get("base_channels"),
            "group_norm_groups": model.extra.get("group_norm_groups"),
            "num_classes": model.num_classes,
        }
        for name, value in required_dimensions.items():
            if not _positive_int(value):
                _add(
                    issues,
                    "error",
                    f"model.{name}_invalid",
                    f"model.{name} must be > 0 for femnist_resnet18",
                )
        dropout = model.extra.get("dropout", 0.1)
        if not _number_in_half_open_range(dropout, 0.0, 1.0):
            _add(
                issues,
                "error",
                "model.dropout_invalid",
                "model.dropout must be in [0, 1) for femnist_resnet18",
            )


def _evaluates_split(config: FullConfig, split: str) -> bool:
    from fedbrew.core.config import parse_evaluation_schedule

    try:
        return (
            parse_evaluation_schedule(
                getattr(config.evaluation, split).every, f"evaluation.{split}"
            )
            is not None
        )
    except ValueError:
        return False


def _evaluates_central(config: FullConfig) -> bool:
    from fedbrew.core.config import parse_evaluation_schedule

    try:
        return (
            parse_evaluation_schedule(
                config.evaluation.central_test.every, "evaluation.central_test"
            )
            is not None
        )
    except ValueError:
        return False


def _validate_evaluation(
    config: FullConfig,
    issues: list[ValidationIssue],
) -> None:
    from fedbrew.core.config import (
        parse_evaluation_client_scope,
        parse_evaluation_schedule,
    )

    for split in ("train", "val", "test"):
        block = getattr(config.evaluation, split)
        try:
            parse_evaluation_schedule(block.every, f"evaluation.{split}")
        except ValueError as error:
            _add(issues, "error", f"evaluation.{split}.every_invalid", str(error))
        try:
            mode, _ = parse_evaluation_client_scope(block.clients)
        except ValueError as error:
            _add(issues, "error", f"evaluation.{split}.clients_invalid", str(error))
            continue
        if split == "test" and mode == "participating":
            _add(
                issues,
                "error",
                "evaluation.test.clients_participating",
                "evaluation.test.clients must not be 'participating': the "
                "reported test number would come from a biased subset.",
            )

    try:
        parse_evaluation_schedule(config.evaluation.central_test.every, "evaluation.central_test")
    except ValueError as error:
        _add(issues, "error", "evaluation.central_test.every_invalid", str(error))


def _validate_runtime(config: FullConfig, issues: list[ValidationIssue]) -> None:
    runtime = config.runtime
    if runtime.device not in {"cpu", "cuda", "auto"}:
        _add(
            issues,
            "error",
            "runtime.device_invalid",
            "runtime.device must be one of: cpu, cuda, auto",
        )
    elif runtime.device == "cuda":
        _validate_cuda_available(issues)

    for name in ("deterministic", "deterministic_warn_only"):
        _validate_extra_bool(issues, runtime.extra, "runtime", name)

    _validate_checkpointing(config, issues)

    staging = runtime.extra.get("data_staging", {})
    if not isinstance(staging, Mapping) or not bool(staging.get("enabled", False)):
        return

    if config.data.name != "manifest_dataset":
        _add(
            issues,
            "warning",
            "runtime.staging_unused",
            "Data staging is enabled but data.name is not manifest_dataset",
            "Disable staging or switch to a manifest dataset.",
        )

    local_root = resolve_staging_root(staging)
    if local_root is None:
        _add(
            issues,
            "warning",
            "runtime.staging_root_unresolved",
            "Data staging is enabled but no local scratch root could be resolved",
            "Set FL_LOCAL_SCRATCH or configure runtime.data_staging.local_root.",
        )


def _pairing_is_out_of_scope(
    strategy: str,
    update_rule: str,
    issues: list[ValidationIssue],
) -> bool:
    """Whether this pair includes a component the shipped rules cannot judge.

    Every pairing rule below enumerates names the package ships, and an
    out-of-tree component can appear in none of them -- so refusing a pair
    for not being on a list it cannot be on would close the hook at the last
    step. The extension is left to refuse a partner it cannot work with,
    which is the deliberate cost: an incompatible pair then fails when it is
    built or in the first round rather than at preflight. Reported as an info
    issue so the gap is stated rather than silent; chapter 12 §1.5 says the
    same thing to the person writing the component.
    """

    if not (is_extension(client_updates, update_rule) or is_extension(server_strategies, strategy)):
        return False
    _add(
        issues,
        "info",
        "algorithm.extension_pairing_unchecked",
        f"server.strategy={strategy!r} with client.update_rule={update_rule!r} "
        "includes a component from outside the package, so the shipped pairing "
        "rules do not apply",
        "The extension is responsible for refusing a partner it cannot work with.",
    )
    return True


def _validate_algorithm_compatibility(
    config: FullConfig,
    issues: list[ValidationIssue],
) -> None:
    """Check the server/client pair, and each half's own options.

    Two parts, split because only the second is about the shipped set:
    ``aggregation_weighting`` is a config value every run has, while every
    rule below names components the package ships and so cannot judge a pair
    that includes an extension.
    """

    _validate_aggregation_weighting(config, issues)
    if _pairing_is_out_of_scope(config.server.strategy, config.client.update_rule, issues):
        return
    _validate_shipped_algorithm_compatibility(config, issues)


def _validate_shipped_algorithm_compatibility(
    config: FullConfig,
    issues: list[ValidationIssue],
) -> None:
    """Every pairing and per-algorithm rule for the components the package ships."""

    strategy = config.server.strategy
    update_rule = config.client.update_rule

    if strategy == "fedavg" and update_rule not in FEDAVG_COMPATIBLE_CLIENTS:
        _add(
            issues,
            "error",
            "algorithm.fedavg_client_incompatible",
            f"FedAvg is not compatible with client update_rule={update_rule}",
            "Use fedavg, local_sgd, local_adamw, fedprox or delta_sgd clients with FedAvg.",
        )
    if update_rule in DELTA_SGD_CLIENT_RULES:
        _validate_delta_sgd(config, issues)
    if update_rule in FEDAVG_FT_CLIENT_RULES:
        _validate_fedavg_ft(config, issues)
    if strategy in FEDLALR_SERVER_STRATEGIES or update_rule in FEDLALR_CLIENT_RULES:
        _validate_fedlalr(config, issues)
    if update_rule in FEDPROX_CLIENT_RULES:
        proximal_mu = config.client.extra.get("proximal_mu")
        if proximal_mu is None:
            _add(
                issues,
                "error",
                "algorithm.fedprox_mu_missing",
                "FedProx client config must contain proximal_mu",
            )
        elif not _non_negative_number(proximal_mu):
            _add(
                issues,
                "error",
                "algorithm.fedprox_mu_invalid",
                "FedProx proximal_mu must be >= 0",
            )

    if strategy in FEDOPT_SERVER_STRATEGIES:
        _validate_fedopt(config, issues)

    server_is_centralized = strategy in CENTRALIZED_SERVER_STRATEGIES
    client_is_centralized = update_rule in CENTRALIZED_CLIENT_RULES
    if server_is_centralized and not client_is_centralized:
        _add(
            issues,
            "error",
            "algorithm.centralized_client_incompatible",
            "The centralized server strategy requires client.update_rule=centralized",
            "Set client.update_rule to centralized in the experiment config.",
        )
    if client_is_centralized and not server_is_centralized:
        _add(
            issues,
            "error",
            "algorithm.centralized_server_incompatible",
            "The centralized client rule requires server.strategy=centralized",
            "Set server.strategy to centralized in the experiment config.",
        )
    if server_is_centralized and client_is_centralized:
        _add(
            issues,
            "info",
            "algorithm.centralized_pooled_clients",
            "Centralized training pools every client into one client, so "
            "participation and client_test dispersion metrics are degenerate",
            "Compare against federated runs on central_test_accuracy.",
        )

    server_is_scaffold = strategy in SCAFFOLD_SERVER_STRATEGIES
    client_is_scaffold = update_rule in SCAFFOLD_CLIENT_RULES
    if server_is_scaffold and not client_is_scaffold:
        _add(
            issues,
            "error",
            "algorithm.scaffold_client_incompatible",
            "SCAFFOLD server requires a scaffold client update rule",
            "Set client.update_rule to scaffold in the experiment config.",
        )
    if client_is_scaffold and not server_is_scaffold:
        _add(
            issues,
            "error",
            "algorithm.scaffold_server_incompatible",
            "SCAFFOLD client requires the scaffold server strategy",
            "Set server.strategy to scaffold in the experiment config.",
        )
    if server_is_scaffold and not _positive_number(config.client.learning_rate):
        _add(
            issues,
            "error",
            "algorithm.scaffold_learning_rate_invalid",
            "SCAFFOLD requires client.learning_rate > 0",
        )
    partial_participation = [
        name
        for name, value in (
            ("participation_rate", config.server.participation_rate),
            ("participation_probability", config.server.participation_probability),
        )
        if value is not None and value < 1.0
    ]
    if server_is_scaffold and partial_participation:
        # Exactly one of the two is set -- _validate_participation refuses the
        # config otherwise -- so the notice can name the one this run used. The
        # figure carries over to Bernoulli sampling exactly: 1/p is the expected
        # gap between a client's draws, where under a fixed rate it is only the
        # mean age.
        setting = partial_participation[0]
        _add(
            issues,
            "warning",
            "algorithm.scaffold_partial_participation",
            "SCAFFOLD refreshes a client's control variate only when that client is "
            f"sampled, so at {setting} < 1.0 a variate is on average about "
            f"1/{setting} rounds old when it is used",
            "participation_rate=1.0 or participation_probability=1.0 keeps every "
            "control variate current.",
        )
    # Per setting rather than per rule: fedavg, centralized and fedavg_ft all
    # run under AMP happily until they are asked to freeze
    # their gradients, at which point the local step passes GradScaler a
    # collector with no param_groups. max_grad_norm was refused here too, on
    # the same code, until it was measured and found to compose.
    # P03-F05.
    sgd_engine_setting = amp_unsupported_sgd_engine_setting(config)
    if sgd_engine_setting is not None:
        _add(
            issues,
            "error",
            "algorithm.sgd_engine_amp_unsupported",
            f"client.{sgd_engine_setting} cannot run under runtime.use_amp: "
            "GradScaler has no param_groups to unscale",
            "Set runtime.use_amp to false, or choose another update_mode.",
        )
    if server_is_scaffold:
        _add_scaffold_notices(config, issues)


def _add_scaffold_notices(config: FullConfig, issues: list[ValidationIssue]) -> None:
    """The two things a SCAFFOLD run should be told about itself.

    Split out of _validate_shipped_algorithm_compatibility rather than added to
    it: that function was already at the C901 ceiling the tree is pinned to, so
    a second notice inline would have raised the ceiling for every function in
    the package. Both are `info` -- neither describes a misconfiguration.
    """

    # The FedLALR counterpart of this notice has existed since that arm
    # landed; SCAFFOLD has the same multi-state payload and had none, so
    # nothing said its rounds cost double.
    _add(
        issues,
        "info",
        "algorithm.scaffold_communication_cost",
        "SCAFFOLD sends the control variate beside the model every round "
        "in both directions: 2x the per-round volume of a FedAvg arm",
        "Compare arms on communicated_bytes, not on round count alone.",
    )
    if config.server.extra.get("aggregation_weighting") == "uniform":
        return
    # Sharper than the FedLALR and Delta-SGD notices this copies, because
    # SCAFFOLD is the one arm where the deviation is internally inconsistent
    # rather than merely off-paper. scaffold.py's two folds do not agree: the
    # model goes through WeightedStateAccumulator at _result_weight, while the
    # summed control delta is always scaled by 1/num_clients. So x moves along
    # the example-weighted mean of the local models and c estimates their
    # uniform mean -- and the uniform one is where c = (1/N) sum(c_i) lives,
    # the invariant the -c_i + c correction rests on.
    _add(
        issues,
        "info",
        "algorithm.scaffold_aggregation_weighting",
        "SCAFFOLD as published averages clients uniformly, but this run "
        "weights the model by example count while the control variate "
        "stays uniform, so the drift correction is not the one the paper "
        "analyses",
        "Set server.aggregation_weighting: uniform to match the paper, or "
        "keep examples to stay comparable with the other arms.",
    )


def _validate_fedavg_ft(
    config: FullConfig,
    issues: list[ValidationIssue],
) -> None:
    """Preflight FedAvg + local fine-tuning."""

    extra = config.client.extra
    epochs = extra.get("finetune_epochs")
    if epochs is None:
        _add(
            issues,
            "error",
            "algorithm.fedavg_ft_epochs_missing",
            "fedavg_ft requires client.finetune_epochs",
            "One or a few epochs of local SGD on the client's own train split.",
        )
    elif isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        _add(
            issues,
            "error",
            "algorithm.fedavg_ft_epochs_invalid",
            "fedavg_ft requires client.finetune_epochs to be a positive integer",
        )

    learning_rate = extra.get("finetune_learning_rate")
    if learning_rate is not None and not _positive_number(learning_rate):
        _add(
            issues,
            "error",
            "algorithm.fedavg_ft_learning_rate_invalid",
            "fedavg_ft requires client.finetune_learning_rate > 0 when set",
            "Leave it unset to inherit client.learning_rate.",
        )

    if config.evaluation.model_scope == "global":
        _add(
            issues,
            "error",
            "algorithm.fedavg_ft_scope_missing",
            "fedavg_ft under evaluation.model_scope=global is just fedavg",
            "Set evaluation.model_scope to both (global and personalized "
            "columns side by side) or personal.",
        )

    for split in ("train", "val", "test"):
        split_config = getattr(config.evaluation, split)
        # A split that is never scheduled costs nothing, however it is scoped.
        if str(split_config.every) == "never":
            continue
        if split_config.clients == "all":
            _add(
                issues,
                "info",
                f"algorithm.fedavg_ft_{split}_scope_cost",
                f"evaluation.{split}.clients=all fine-tunes every client on "
                f"every evaluated round, not just the round's participants",
                "A fixed sample:<N> caps that cost and tracks the same clients across rounds.",
            )


def _validate_fedlalr(
    config: FullConfig,
    issues: list[ValidationIssue],
) -> None:
    """Preflight the FedLALR pair (arXiv:2309.09719)."""

    server_is_fedlalr = config.server.strategy in FEDLALR_SERVER_STRATEGIES
    client_is_fedlalr = config.client.update_rule in FEDLALR_CLIENT_RULES
    if server_is_fedlalr and not client_is_fedlalr:
        _add(
            issues,
            "error",
            "algorithm.fedlalr_client_incompatible",
            "FedLALR server requires client.update_rule=fedlalr",
            "The server synchronizes the momentum and second moment that only "
            "the FedLALR client produces.",
        )
    if client_is_fedlalr and not server_is_fedlalr:
        _add(
            issues,
            "error",
            "algorithm.fedlalr_server_incompatible",
            "FedLALR client requires server.strategy=fedlalr",
            "Only that server broadcasts the momentum and second moment the "
            "client's local AMSGrad starts from.",
        )
    if not client_is_fedlalr:
        return

    if not _positive_number(config.client.learning_rate):
        _add(
            issues,
            "error",
            "algorithm.fedlalr_learning_rate_invalid",
            "FedLALR requires client.learning_rate (alpha) > 0",
        )

    extra = config.client.extra
    for name in ("beta1", "beta2"):
        if name in extra and not _unit_interval(extra[name]):
            _add(
                issues,
                "error",
                f"algorithm.fedlalr_{name}_invalid",
                f"FedLALR requires client.{name} in [0, 1)",
            )
    if "epsilon" in extra and not _positive_number(extra["epsilon"]):
        _add(
            issues,
            "error",
            "algorithm.fedlalr_epsilon_invalid",
            "FedLALR requires client.epsilon > 0",
            "It is the floor on v_hat, so 1/sqrt(v_hat) stays bounded by 1/epsilon.",
        )

    if config.runtime.use_amp:
        _add(
            issues,
            "error",
            "algorithm.fedlalr_amp_unsupported",
            "fedlalr reads raw gradients and cannot run under runtime.use_amp",
            "Set runtime.use_amp to false.",
        )

    if config.server.extra.get("aggregation_weighting") != "uniform":
        _add(
            issues,
            "info",
            "algorithm.fedlalr_aggregation_weighting",
            "FedLALR as published averages clients uniformly, but this run "
            "weights them by example count",
            "Set server.aggregation_weighting: uniform to match the paper, or "
            "keep examples to stay comparable with the other arms.",
        )

    _add(
        issues,
        "info",
        "algorithm.fedlalr_communication_cost",
        "FedLALR sends x, m and v_hat every round: 3x the per-round volume of "
        "a FedAvg arm in both directions",
        "Compare arms on communicated_bytes, not on round count alone.",
    )


def _validate_delta_sgd(
    config: FullConfig,
    issues: list[ValidationIssue],
) -> None:
    """Preflight the Delta-SGD client (arXiv:2306.11201)."""

    extra = config.client.extra
    if "eta_0" not in extra:
        _add(
            issues,
            "error",
            "algorithm.delta_sgd_eta_0_missing",
            "delta_sgd requires client.eta_0",
            "The paper uses eta_0: 0.2 unchanged across every experiment.",
        )
    elif not _positive_number(extra["eta_0"]):
        _add(
            issues,
            "error",
            "algorithm.delta_sgd_eta_0_invalid",
            "delta_sgd requires client.eta_0 > 0",
        )

    for name in ("theta_0", "gamma", "delta", "eta_max"):
        if name in extra and extra[name] is not None and not _positive_number(extra[name]):
            _add(
                issues,
                "error",
                f"algorithm.delta_sgd_{name}_invalid",
                f"delta_sgd requires client.{name} > 0 when set",
            )

    for name, allowed in (
        ("update_mode", DELTA_SGD_UPDATE_MODES),
        ("frozen_gradient_weighting", FROZEN_GRADIENT_WEIGHTINGS),
    ):
        if name in extra and extra[name] not in allowed:
            _add(
                issues,
                "error",
                f"algorithm.delta_sgd_{name}_invalid",
                f"Unknown client.{name}: {extra[name]}",
                f"Choose one of: {', '.join(sorted(allowed))}.",
            )

    if config.runtime.use_amp:
        _add(
            issues,
            "error",
            "algorithm.delta_sgd_amp_unsupported",
            "delta_sgd reads raw gradients and cannot run under runtime.use_amp",
            "Set runtime.use_amp to false.",
        )

    if config.server.extra.get("aggregation_weighting") != "uniform":
        _add(
            issues,
            "info",
            "algorithm.delta_sgd_aggregation_weighting",
            "Delta-SGD as published averages clients uniformly, but this run "
            "weights them by example count",
            "Set server.aggregation_weighting: uniform to match the paper, or "
            "keep examples to stay comparable with the other arms.",
        )


def _validate_aggregation_weighting(
    config: FullConfig,
    issues: list[ValidationIssue],
) -> None:
    """Check server.aggregation_weighting, which defaults to example weighting."""

    if "aggregation_weighting" not in config.server.extra:
        return
    value = config.server.extra["aggregation_weighting"]
    if not isinstance(value, str) or value not in SUPPORTED_AGGREGATION_WEIGHTING:
        _add(
            issues,
            "error",
            "server.aggregation_weighting_invalid",
            f"Unknown server.aggregation_weighting: {value}",
            "Use examples (default, weight by client example count) or uniform "
            "(weight every participating client equally).",
        )


def _validate_checkpointing(
    config: FullConfig,
    issues: list[ValidationIssue],
) -> None:
    checkpointing = config.runtime.extra.get("checkpointing")
    if checkpointing is None:
        return
    if not isinstance(checkpointing, Mapping):
        _add(
            issues,
            "error",
            "runtime.checkpointing_invalid",
            "runtime.checkpointing must be a mapping",
        )
        return

    interval = checkpointing.get("interval", 1)
    if not _positive_int(interval):
        _add(
            issues,
            "error",
            "runtime.checkpointing_interval_invalid",
            "runtime.checkpointing.interval must be > 0",
        )

    keep_last = checkpointing.get("keep_last")
    if keep_last is not None and not _non_negative_int(keep_last):
        _add(
            issues,
            "error",
            "runtime.checkpointing_keep_last_invalid",
            "runtime.checkpointing.keep_last must be >= 0 when provided",
        )

    # The run path's own answer, not a second opinion. This block used to
    # accept `best_mode` and check it was max or min -- a key config.py has
    # removed -- and to check `best_metric` for non-emptiness while the run
    # path checks the prefix, the direction, the model_scope pairing and
    # whether the column is emitted at all. Both directions were wrong at
    # once: preflight refused what the run accepts and accepted what the run
    # refuses. P07-F09.
    problem = checkpoint_selection_problem(config)
    if problem is not None:
        _add(
            issues,
            "error",
            "runtime.checkpointing_best_metric_invalid",
            problem,
        )

    if (
        bool(checkpointing.get("save_every_round", False))
        and (config.server.global_rounds or 0) > 10
    ):
        _add(
            issues,
            "warning",
            "runtime.checkpointing_save_every_round_long_run",
            "save_every_round=true may use substantial storage for long runs",
            "Use interval and keep_last for storage-efficient HPC runs.",
        )


def _validate_fedopt(config: FullConfig, issues: list[ValidationIssue]) -> None:
    strategy = config.server.strategy
    extra = config.server.extra
    if strategy == "fedopt":
        optimizer = extra.get("server_optimizer")
        if optimizer not in FEDOPT_OPTIMIZERS:
            _add(
                issues,
                "error",
                "algorithm.fedopt_optimizer_invalid",
                "server.strategy=fedopt requires server_optimizer in fedavgm/fedadam/fedyogi",
            )
    elif extra.get("server_optimizer") is not None:
        # Under a named strategy the builder supplies its own optimizer and
        # nothing reads this key, so the config states a fact the run does not
        # use. It was a warning, and only when the value *differed* -- a
        # warning about a value that has no effect, silent when it agreed. The
        # rule for a key its component never receives is the one
        # UNHONOURED_CLIENT_OPTIONS already applies on the client side: refuse
        # it. P10-F33.
        _add(
            issues,
            "error",
            "algorithm.fedopt_optimizer_unread",
            f"server.strategy={strategy} already names its optimizer, so "
            f"server_optimizer={extra.get('server_optimizer')!r} is never read",
            "Remove server_optimizer, or use server.strategy=fedopt to choose it.",
        )

    reason, unread = unread_fedopt_hyperparameters(fedopt_optimizer_name(config))
    never_read = [name for name in unread if name in extra]
    if never_read:
        # The same refusal validate_config makes on the run path, so
        # --validate-only does not pass a config a run would reject. P01-F07.
        _add(
            issues,
            "error",
            "algorithm.fedopt_hyperparameter_unread",
            f"server.strategy={strategy} {reason}, and never reads: "
            + ", ".join(f"server.{name}" for name in never_read),
            "Remove them; no value of theirs would change the run.",
        )

    # Only what this optimizer reads, and against the run path's own bounds.
    # Four unconditional checks used to run here, so after P01-F07 removed
    # `beta2` from every fedavgm and fedadagrad config, preflight reported
    # them as missing -- a key it is now an error to set. And the bounds were
    # spelled out here rather than read from FEDOPT_BOUNDS, which is how
    # `tau: 0` came to be refused by preflight alone. P04-F05.
    for name in FEDOPT_HYPERPARAMETERS:
        if name in unread:
            continue
        _validate_fedopt_hyperparameter(issues, extra, name)

    if extra.get("aggregation_weighting") != "uniform":
        # Milder than the SCAFFOLD notice, deliberately. Algorithm 2 of Reddi
        # et al. writes the client average as 1/|S| like every other
        # pseudocode here, but that paper's own experiments weight by example
        # count, so example weighting is a departure from the pseudocode and
        # not from the method as evaluated. The notice exists so the reader
        # is told which of the two they are running, not to argue for either.
        _add(
            issues,
            "info",
            "algorithm.fedopt_aggregation_weighting",
            "FedOpt's pseudocode averages clients uniformly, but this run "
            "weights them by example count -- which is what the paper's own "
            "experiments do",
            "Set server.aggregation_weighting: uniform to match the "
            "pseudocode, or keep examples to stay comparable with the other "
            "arms.",
        )


def _validate_extra_bool(
    issues: list[ValidationIssue],
    values: Mapping[str, Any],
    section: str,
    name: str,
) -> None:
    value = values.get(name)
    if value is not None and not isinstance(value, bool):
        _add(
            issues,
            "error",
            f"{section}.{name}_invalid",
            f"{section}.{name} must be a bool",
        )


def _validate_fedopt_hyperparameter(
    issues: list[ValidationIssue],
    values: Mapping[str, Any],
    name: str,
) -> None:
    """Report what `validate_config` would raise about one FedOpt setting.

    The bound and its wording come from `FEDOPT_BOUNDS`, so this cannot drift
    from the run path the way the four hand-written predicates here did.
    """

    if name not in values:
        _add(
            issues,
            "error",
            f"algorithm.fedopt_{name}_missing",
            f"FedOpt requires server.{name}",
        )
        return
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, int | float):
        _add(
            issues,
            "error",
            f"algorithm.fedopt_{name}_invalid",
            f"server.{name} must be numeric" + yaml_number_cause(value),
        )
        return
    violation = fedopt_bound_violation(name, float(value))
    if violation is not None:
        _add(issues, "error", f"algorithm.fedopt_{name}_invalid", violation)


def _validate_cuda_available(issues: list[ValidationIssue]) -> None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local environment.
        _add(
            issues,
            "warning",
            "runtime.cuda_torch_unavailable",
            f"runtime.device=cuda but torch could not be imported: {exc}",
            "Validate inside the same environment that will run training.",
        )
        return

    if not torch.cuda.is_available():
        _add(
            issues,
            "warning",
            "runtime.cuda_unavailable",
            "runtime.device=cuda but torch.cuda.is_available() is false",
            "Use --device cpu/auto locally or validate inside a GPU allocation.",
        )


def _add(
    issues: list[ValidationIssue],
    severity: str,
    code: str,
    message: str,
    hint: str | None = None,
) -> None:
    issues.append(ValidationIssue(severity=severity, code=code, message=message, hint=hint))


def _non_empty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value > 0


def _unit_interval(value: object) -> bool:
    """Return whether value is a number in [0, 1) -- an exponential decay rate."""

    return (
        isinstance(value, int | float) and not isinstance(value, bool) and 0.0 <= float(value) < 1.0
    )


def _non_negative_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value >= 0


def _participation_rate(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and 0 < float(value) <= 1


def _number_in_half_open_range(value: object, lower: float, upper: float) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and lower <= float(value) < upper
    )


def _check_components(config: FullConfig, issues: list[ValidationIssue]) -> None:
    """Adapter: the registry check needs no config, and the sequence needs one
    signature."""

    _register_components(issues)


#: The preflight checks, in the order they run.
#:
#: A named sequence rather than ten calls in a row, because `--validate-only`
#: now streams them: a reader watches each one settle and sees a finding
#: attributed to the check that raised it, instead of a flat list of issue
#: codes they have to map back onto a mental model of what preflight does.
#:
#: Every check runs, including after one fails. That is the property this
#: sequence must not lose: the run path's `validate_config` stops at the first
#: problem, and preflight exists precisely so a reader can fix everything in
#: one pass rather than discovering the next error after the next launch.
CHECKS: tuple[tuple[str, Callable[[FullConfig, list[ValidationIssue]], None]], ...] = (
    ("components", _check_components),
    ("experiment", _validate_experiment),
    ("server", _validate_server),
    ("client", _validate_client),
    ("task", _validate_task),
    ("data", _validate_data),
    ("model", _validate_model),
    ("evaluation", _validate_evaluation),
    ("runtime", _validate_runtime),
    ("algorithm compatibility", _validate_algorithm_compatibility),
)

#: Every check name, for a rail that wants to measure its label column before
#: the first stage runs.
CHECK_NAMES: tuple[str, ...] = tuple(name for name, _ in CHECKS)
