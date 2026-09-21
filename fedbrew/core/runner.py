"""Entrypoint for running benchmark experiments."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from fedbrew.core.artifacts import (
    append_run_index,
    prepare_output_dir,
    save_client_metrics_csv,
    save_client_update_metrics_csv,
    save_round_metrics_csv,
    save_run_json,
)
from fedbrew.core.config import (
    FullConfig,
    evaluates_round,
    load_config,
    parse_evaluation_schedule,
    validate_config,
)
from fedbrew.core.console import (
    FAIL,
    RED,
    add_output_arguments,
    build_surface,
    surface_from_args,
)
from fedbrew.core.data_staging import maybe_stage_manifest_dataset
from fedbrew.core.divergence import (
    STATUS_COMPLETED,
    STATUS_DIVERGED,
    STATUS_STALLED,
    DivergenceVerdict,
)
from fedbrew.core.factory import ExperimentComponents, build_components
from fedbrew.core.logging import (
    RoundProgress,
    client_progress_reporter,
    print_experiment_end,
    print_plan_header,
    print_round_metrics,
    print_round_table_header,
    print_termination_notice,
)
from fedbrew.core.loop import run_fl_loop
from fedbrew.core.paths import resolve_named_config, resolve_output_dir
from fedbrew.core.refusal import RunRefused
from fedbrew.core.run_metadata import (
    build_dataset_provenance,
    build_extension_provenance,
    build_hf_causal_lm_trace,
    capture_code_state,
    generate_run_id,
)
from fedbrew.core.runtime_setup import (
    configure_deterministic_environment,
    configure_runtime,
    seed_everything,
)
from fedbrew.core.state import ExperimentState, MetricRecord
from fedbrew.core.validation import (
    CHECK_NAMES,
    print_validation_verdict,
    report_from_exception,
    stream_checks,
)

DEFAULT_CONFIG_PATH = "configs/dev/smoke.yaml"

#: Where a short name (``synthetic/fedavg``) resolves to a file
#: (``configs/synthetic/fedavg.yaml``).
_CONFIG_BASE_DIR = "configs"


def run(
    common_path: str | Path = DEFAULT_CONFIG_PATH,
    args: argparse.Namespace | None = None,
) -> ExperimentState:
    """Load config, build components, run the loop, and save artifacts."""

    config = load_config(common_path)
    if args is not None:
        config = apply_cli_overrides(config, args)
    deterministic = _runtime_extra_bool(config, "deterministic", False)
    # Strict by default: a config that says deterministic: true and does not
    # say otherwise gets determinism, not a warning about the lack of it.
    # An arm that cannot afford the deterministic kernel opts out in its own
    # config, where the cost is visible.
    deterministic_warn_only = _runtime_extra_bool(
        config,
        "deterministic_warn_only",
        False,
    )
    configure_deterministic_environment(deterministic)
    config, run_metadata, index_root = resolve_run_metadata(config)
    # Both of these return what they actually did -- the seeds they set, the
    # torch/CUDA versions, the flags torch ended up holding, and any setup
    # error that was swallowed. Both returns used to be discarded apart from
    # resolved_device, so a run that silently failed to apply a performance
    # setting left nothing on disk saying so.
    run_metadata["seeding"] = seed_everything(
        config.experiment.seed,
        deterministic=deterministic,
        warn_only=deterministic_warn_only,
    )
    runtime_info = configure_runtime(config, deterministic)
    run_metadata["runtime"] = dict(runtime_info)
    if runtime_info.get("resolved_device") in {"cpu", "cuda"}:
        config = replace(
            config,
            runtime=replace(config.runtime, device=str(runtime_info["resolved_device"])),
        )
    config = maybe_stage_manifest_dataset(config)
    run_metadata["output_dir"] = config.experiment.output_dir

    output_dir = prepare_output_dir(config.experiment.output_dir)
    config, run_metadata = _resolve_resume_latest(config, run_metadata, output_dir)
    _refuse_a_foreign_seed(config, output_dir)
    _refuse_to_replace_a_finished_run(config, output_dir)
    _carry_previous_attempt(config, run_metadata, output_dir)
    components = build_components(config)

    # After build_components, so the client roster is a counted fact rather
    # than whatever the config guessed, and after _resolve_resume_latest, so
    # the plan names the checkpoint this run will actually load.
    roster = _client_roster_size(components)
    print_plan_header(
        config,
        deterministic=deterministic,
        deterministic_warn_only=deterministic_warn_only,
        client_count=roster,
        resume_from=config.runtime.extra.get("resume_from"),
    )
    # One object, both halves of the run surface. The live footer and the
    # settled block report the same rate and the same estimate because they
    # read the same durations, rather than each keeping its own count.
    progress = RoundProgress(total_rounds=config.server.global_rounds or 0, roster=roster)
    print_round_table_header(config)
    run_metadata["started_at"] = _utc_timestamp()
    run_started = time.perf_counter()
    state = run_fl_loop(
        server=components.server,
        client=components.clients,
        dataset=components.dataset,
        global_rounds=config.server.global_rounds or 0,
        output_dir=output_dir,
        resume_from=config.runtime.extra.get("resume_from"),
        checkpointing=config.runtime.extra.get("checkpointing"),
        evaluation=config.evaluation,
        client_statistics=config.client_statistics,
        evaluation_seed=config.experiment.seed,
        divergence=config.divergence,
        on_round_end=_round_progress_reporter(config, progress),
        on_round_flush=_run_json_writer(config, run_metadata, output_dir, run_started),
        on_client_progress=client_progress_reporter(config, progress),
        on_termination=_termination_reporter(config),
    )
    run_metadata["finished_at"] = _utc_timestamp()
    _record_durations(run_metadata, run_started)

    run_metadata["checkpointing"] = dict(state.checkpointing)
    # Carried through run_metadata so run.json and runs_index.jsonl report the
    # same outcome from one source.
    run_metadata["status"] = state.status
    run_metadata["termination"] = dict(state.termination) if state.termination else None
    # What the loop actually did, overwriting what _carry_previous_attempt
    # assumed from the config. POST-F05.
    run_metadata["resumed"] = state.resumed
    run_metadata["stopped_round"] = (
        state.metrics_history[-1].round_id
        if state.status != "completed" and state.metrics_history
        else None
    )
    _add_federated_model_state_metadata(run_metadata, components)

    # The loop already flushed these every round; this final pass is what
    # records the last round when the run ends cleanly, and what brings the
    # per-client CSVs (flushed on checkpoint rounds only) fully up to date.
    save_round_metrics_csv(state.metrics_history, output_dir)
    # Per-client detail is opt-in: it is one row per client per round, which
    # dwarfs every other artifact, and round_metrics.csv already carries the
    # aggregates (macro/std/min/bottom10) that the round-level analysis uses.
    if config.client_statistics.per_client_csv:
        save_client_metrics_csv(state.client_metrics_history, output_dir)
        save_client_update_metrics_csv(state.client_update_metrics_history, output_dir)

    artifact_files = _artifact_file_names(config)
    save_run_json(
        state.metrics_history,
        output_dir,
        config,
        state.client_metrics_history,
        artifact_files=artifact_files,
        run_metadata=run_metadata,
        client_update_history=state.client_update_metrics_history,
    )
    final_record = state.metrics_history[-1] if state.metrics_history else None
    append_run_index(
        {
            **run_metadata,
            "final_metrics": final_record.metrics if final_record else {},
        },
        index_root,
    )
    print_experiment_end(
        state.metrics_history,
        output_dir,
        config,
        {"status": state.status, **state.termination} if state.termination else None,
    )
    return state


def _client_roster_size(components: ExperimentComponents) -> int | None:
    """How many clients this run actually has, or None if asking would cost.

    The dataset knows; the config often does not (a manifest dataset carries
    its own roster). Guarded because list_clients is a dataset method with no
    contract about being cheap, and a header must not be the thing that fails
    a run.
    """

    try:
        return len(list(components.dataset.list_clients()))
    except Exception:  # pragma: no cover - a header must never break a run.
        return None


def _round_progress_reporter(
    config: FullConfig,
    progress: RoundProgress | None = None,
) -> Callable[[MetricRecord], None]:
    """Return an on_round_end callback that prints each round's block.

    The estimate itself lives on `RoundProgress`, which the live footer reads
    too. What is decided here is only which rounds print: every Nth under
    `runtime.print_every` (`--print-every N`), else every round under
    `verbose`, else the evaluation rounds. Printing is all it decides -- the
    loop has already evaluated the round on its schedules and flushed it to
    round_metrics.csv by the time this is called.
    """

    total_rounds = config.server.global_rounds or 0
    tracker = progress if progress is not None else RoundProgress(total_rounds=total_rounds)
    verbose = _runtime_extra_bool(config, "verbose", False)
    print_every = config.runtime.extra.get("print_every")
    intervals = _evaluation_intervals(config)

    def report(record: MetricRecord) -> None:
        duration = record.timings.total if record.timings is not None else None
        # Recorded for every round, reported for some: a round that prints
        # nothing still took time, and dropping it from the median would make
        # the estimate describe only the expensive rounds.
        tracker.record(duration)
        # print_every replaces the evaluation schedule rather than thinning it:
        # an intersection would print nothing between evaluations that N does
        # not happen to land on. evaluates_round pins round 1 and the final
        # round, as it does for an `every: N` split.
        if print_every is not None:
            if not evaluates_round(print_every, record.round_id, total_rounds):
                return
        elif not verbose and not _reports_round(intervals, record.round_id, total_rounds):
            return
        print_round_metrics(
            record.round_id,
            record.metrics,
            record.num_clients,
            config,
            num_examples=record.num_examples,
            duration_sec=duration,
            eta_sec=tracker.eta_seconds(record.round_id),
            progress=tracker,
        )

    return report


def _evaluation_intervals(config: FullConfig) -> list[int | None]:
    """The parsed `every` of each evaluated split, including the central pass.

    Read from the schedules rather than from what a round happened to produce.
    Deciding "this was an evaluation round" by looking for evaluation columns
    in the output would infer intent from output, which is how a config that
    silently stopped evaluating a split goes unnoticed for a whole sweep.
    """

    intervals = []
    for split in ("train", "val", "test"):
        try:
            intervals.append(
                parse_evaluation_schedule(getattr(config.evaluation, split).every, split)
            )
        except ValueError:  # pragma: no cover - load_config rejects these first.
            continue
    try:
        intervals.append(
            parse_evaluation_schedule(config.evaluation.central_test.every, "central_test")
        )
    except ValueError:  # pragma: no cover - as above.
        pass
    return intervals


def _reports_round(intervals: list[int | None], round_id: int, total_rounds: int) -> bool:
    """Whether this round is one the terminal reports by default.

    Evaluation rounds only. A 500-round run reports 50 times at the default
    `every: 10` instead of 500, and the rounds it skips are the ones whose
    output would repeat the previous round's evaluation numbers unchanged --
    the fit metrics move, but nothing that was measured does.

    `evaluates_round` is the loop's own predicate, not a copy of it, so the
    rounds reported are exactly the rounds evaluated -- including the two it
    pins at both ends, round 1 and the final round.

    A config that evaluates nothing at all reports every round: there is no
    evaluation round to wait for, and silence for the whole run is worse than
    a line per round.
    """

    active = [interval for interval in intervals if interval is not None]
    if not active:
        return True
    return any(evaluates_round(interval, round_id, total_rounds) for interval in active)


def _termination_reporter(config: FullConfig) -> Callable[[DivergenceVerdict], None]:
    """Return an on_termination callback that announces a stop as it happens.

    Wired the same way _round_progress_reporter wires print_round_metrics --
    quiet/plain/rich all go through print_termination_notice, which shares
    print_round_metrics' own suppression rules.
    """

    def report(verdict: DivergenceVerdict) -> None:
        print_termination_notice(verdict, config)

    return report


def _refuse_a_foreign_seed(config: FullConfig, output_dir: Path) -> None:
    """Refuse to write a second seed's run on top of a first seed's.

    Replicate seeds are the only way to put a dispersion on a comparison, and
    nothing in the output path distinguishes them unless the caller puts it
    there. Two seeds of the same arm otherwise resolve to one output_dir, where
    the second overwrites the first's run.json, appends to its
    round_metrics.csv, and -- if the checkpoint is picked up -- continues the
    first seed's model while reporting the second seed's config. The result is a
    directory whose contents cannot be attributed to either run, and the
    "3-seed spread" computed from it is not one.

    An error rather than a silent path rewrite: appending seed_N ourselves would
    move every existing run's output location.
    """

    path = Path(output_dir) / "run.json"
    if not path.is_file():
        return
    try:
        with path.open(encoding="utf-8") as handle:
            previous = json.load(handle)
    except (OSError, ValueError):
        return

    experiment = previous.get("config", {})
    if not isinstance(experiment, dict):
        return
    previous_seed = (experiment.get("experiment") or {}).get("seed")
    seed = config.experiment.seed
    if previous_seed is None or previous_seed == seed:
        return

    raise RunRefused(
        f"refusing to run seed {seed} in {output_dir}: it already holds a run "
        f"with seed {previous_seed}.\n"
        "  Replicates need one directory each, or their metrics interleave in "
        "the same CSV and neither run can be recovered.\n"
        f"  Put the seed in the path, e.g. --output-dir {output_dir}/seed_{seed} "
        "(the example sweep scripts in SLURMs/ show the pattern)."
    )


#: The statuses a run.json keeps once its run has ended. The other one,
#: `running`, is what a crash leaves as well as what a live run writes, and
#: restarting a crashed run in place is ordinary, so only these are protected.
_FINISHED_STATUSES = frozenset({STATUS_COMPLETED, STATUS_DIVERGED, STATUS_STALLED})

#: Keys that name a run or say how it was launched, not what it computes. Two
#: runs that differ only here are the same experiment. output_dir is here
#: because the comparison is only ever made inside that directory: a relative
#: and an absolute spelling of it are the same place.
_NOT_CONFIGURATION = frozenset(
    {
        "experiment.run_id",
        "experiment.name",
        "experiment.tags",
        "experiment.notes",
        "experiment.output_dir",
        "runtime.extra.quiet",
        "runtime.extra.verbose",
        "runtime.extra.no_rich",
        "runtime.extra.print_every",
        "runtime.extra.resume_from",
        "runtime.extra.resume_latest",
    }
)


def config_differences(
    recorded: Mapping[str, Any],
    config: FullConfig,
) -> list[tuple[str, Any, Any]]:
    """Every setting a run.json's recorded config and this config disagree on.

    Each as (dotted key, recorded value, this config's value), compared the way
    run.json writes a config (`_as_written`), so a tuple equals the list it was
    written as. A key present on one side only is not compared,
    the rule `refuse_a_reconfigured_resume` applies to a checkpoint: a run
    recorded before a setting existed is not different for lacking it. Keys in
    `_NOT_CONFIGURATION` are skipped, and so is `data.path` under data staging,
    where it names a per-job copy of the same manifest.
    """

    current = _as_written(asdict(config))
    before = _as_written(recorded)
    skipped = set(_NOT_CONFIGURATION)
    staging = config.runtime.extra.get("data_staging")
    if isinstance(staging, Mapping) and staging.get("enabled"):
        skipped.add("data.path")

    differences: list[tuple[str, Any, Any]] = []

    def walk(old: Mapping[str, Any], new: Mapping[str, Any], prefix: str) -> None:
        for key in sorted(set(old) & set(new)):
            path = f"{prefix}{key}"
            if path in skipped:
                continue
            if isinstance(old[key], dict) and isinstance(new[key], dict):
                walk(old[key], new[key], f"{path}.")
            elif old[key] != new[key]:
                differences.append((path, old[key], new[key]))

    walk(before, current, "")
    return differences


def _as_written(value: Any) -> Any:
    """A config value as run.json holds it.

    What `save_run_json` makes of it: `json_safe` turns a non-finite float into
    null, and `json.dumps(default=str)` a tuple into a list and anything else
    that is not JSON into its string.
    """

    if isinstance(value, Mapping):
        return {str(key): _as_written(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_written(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _refuse_to_replace_a_finished_run(config: FullConfig, output_dir: Path) -> None:
    """Refuse a fresh start over a finished run whose config differs. POST-F22.

    `_refuse_a_foreign_seed` guards the seed and nothing else. At the same seed
    a fresh start replaced whatever finished run output_dir held -- run.json,
    round_metrics.csv, the checkpoints -- with a run of different settings, and
    nothing on disk said it had: a directory named for one learning rate held
    another's curve. The same config may still rerun in place, which replaces a
    run with the same experiment. A resume is not a fresh start and has its own
    check (`refuse_a_reconfigured_resume`), and a run.json still saying
    `running`, a crash or a live run, may restart in place as before.
    """

    if config.runtime.extra.get("resume_from"):
        return
    path = Path(output_dir) / "run.json"
    if not path.is_file():
        return
    try:
        with path.open(encoding="utf-8") as handle:
            previous = json.load(handle)
    except (OSError, ValueError):
        return
    if not isinstance(previous, dict) or previous.get("status") not in _FINISHED_STATUSES:
        return
    recorded = previous.get("config")
    if not isinstance(recorded, dict):
        return
    differences = config_differences(recorded, config)
    if not differences:
        return

    lines = "\n".join(
        f"  {key}: that run {old!r}, this run {new!r}" for key, old, new in differences
    )
    count = len(differences)
    raise RunRefused(
        f"refusing to start a fresh run in {output_dir}: it holds a finished run "
        f"({previous.get('status')}, {previous.get('run_id')}) whose config differs in "
        f"{count} {'setting' if count == 1 else 'settings'}:\n{lines}\n"
        "Starting here would replace that run's results with this one's, and nothing "
        "in the directory would say the settings had changed.\n"
        "  Run the new settings in their own directory: name the setting in "
        "output_dir, or pass --output-dir.\n"
        "  Or move or delete the old run first, if it is no longer wanted."
    )


def _carry_previous_attempt(
    config: FullConfig,
    run_metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    """Pick up what the attempt we are resuming already spent.

    A requeue starts a fresh process, so time.perf_counter() measures this
    attempt and nothing before it. Without carrying the earlier total forward,
    duration_sec silently reports the tail of a run as the whole of it, and a
    time budget built from it under-counts by however much the killed attempt
    had already burned. The previous total is in the run.json this same
    output_dir already holds -- the loop rewrites it every round, so it is
    present even when the earlier attempt was killed mid-round.
    """

    resume_from = config.runtime.extra.get("resume_from")
    run_metadata["resume_from"] = str(resume_from) if resume_from else None
    # A provisional answer to a question only the loop can settle: whether the
    # checkpoint can be resumed from at all. The loop refuses one that cannot,
    # before anything is written (POST-F25), and every write takes the loop's
    # answer from ExperimentState, so this value never reaches disk on its own.
    run_metadata["resumed"] = bool(resume_from)
    run_metadata["previous_duration_sec"] = 0.0
    run_metadata["attempts"] = 1
    if not resume_from:
        return

    try:
        with (Path(output_dir) / "run.json").open(encoding="utf-8") as handle:
            previous = json.load(handle)
    except (OSError, ValueError):
        # First resume of a run written before these fields existed, or an
        # unreadable file. The attempt count is then a lower bound; say so by
        # leaving the defaults rather than guessing.
        return

    duration = previous.get("duration_sec")
    if isinstance(duration, (int, float)) and math.isfinite(duration):
        run_metadata["previous_duration_sec"] = float(duration)
    attempts = previous.get("attempts")
    run_metadata["attempts"] = (attempts if isinstance(attempts, int) else 1) + 1


def _record_durations(metadata: dict[str, Any], run_started: float) -> None:
    """Wall time for this attempt, and for the run across every attempt."""

    attempt = round(time.perf_counter() - run_started, 3)
    metadata["attempt_duration_sec"] = attempt
    metadata["duration_sec"] = round(
        float(metadata.get("previous_duration_sec") or 0.0) + attempt, 3
    )


def _run_json_writer(
    config: FullConfig,
    run_metadata: dict[str, Any],
    output_dir: Path,
    run_started: float,
) -> Callable[[ExperimentState], None]:
    """Return a per-round writer that keeps run.json current mid-run.

    The CSVs the loop flushes carry the numbers; run.json carries the config,
    the environment and the summary that make them interpretable. Writing it
    only at the end left a cancelled run's artifacts unreadable by anything
    that keys off run.json, so it is now rewritten alongside them. status is
    "running" until the loop returns and run() overwrites this with the final
    record, so a run.json saying "running" is exactly the signal that the job
    did not finish.
    """

    def write(state: ExperimentState) -> None:
        metadata = {
            **run_metadata,
            "status": "running",
            "stopped_round": state.metrics_history[-1].round_id if state.metrics_history else None,
            # Decided before round 1, so every mid-run write carries the
            # loop's answer rather than the config's. POST-F05.
            "resumed": state.resumed,
        }
        _record_durations(metadata, run_started)
        save_run_json(
            state.metrics_history,
            output_dir,
            config,
            state.client_metrics_history,
            artifact_files=_artifact_file_names(config),
            run_metadata=metadata,
            client_update_history=state.client_update_metrics_history,
        )

    return write


def _artifact_file_names(config: FullConfig) -> list[str]:
    """Names of the artifacts this run writes, per its statistics config."""

    names = ["round_metrics.csv"]
    if config.client_statistics.per_client_csv:
        names.extend(["client_metrics.csv", "client_update_metrics.csv"])
    names.append("run.json")
    return names


def _resolve_resume_latest(
    config: FullConfig,
    run_metadata: dict[str, Any],
    output_dir: Path,
) -> tuple[FullConfig, dict[str, Any]]:
    """Resolve --resume-latest after the final output directory is known."""

    runtime_extra = dict(config.runtime.extra)
    if not bool(runtime_extra.get("resume_latest", False)):
        return config, run_metadata

    from fedbrew.core.checkpointing import find_latest_checkpoint

    latest_checkpoint = find_latest_checkpoint(output_dir)
    if latest_checkpoint is None:
        raise RunRefused(
            f"No checkpoint found under {output_dir / 'checkpoints'} for --resume-latest"
        )

    runtime_extra["resume_from"] = str(latest_checkpoint)
    resolved_config = replace(
        config,
        runtime=replace(config.runtime, extra=runtime_extra),
    )
    metadata = dict(run_metadata)
    metadata["resume_from"] = str(latest_checkpoint)
    return resolved_config, metadata


def _add_federated_model_state_metadata(
    run_metadata: dict[str, Any],
    components: ExperimentComponents,
) -> None:
    """Lift the server's validated state contract into reproducibility metadata."""

    state = components.server.save_state()
    scope = state.get("model_state_scope")
    metadata = state.get("model_state_metadata")
    if isinstance(scope, str):
        run_metadata["model_state_scope"] = scope
    if isinstance(metadata, dict):
        copied = dict(metadata)
        run_metadata["federated_model_state"] = copied
        llm = run_metadata.get("llm")
        if isinstance(llm, dict):
            llm.update(copied)


def resolve_run_metadata(config: FullConfig) -> tuple[FullConfig, dict[str, Any], Path]:
    """Resolve final output directory and collect run metadata."""

    experiment = config.experiment
    root_output_dir = resolve_output_dir(experiment.output_dir)
    run_id = experiment.run_id or generate_run_id(
        experiment.name,
        experiment.seed,
    )
    final_output_dir = root_output_dir / run_id if experiment.use_run_subdir else root_output_dir
    experiment = replace(
        experiment,
        output_dir=str(final_output_dir),
        run_id=run_id,
        tags=list(experiment.tags),
        notes=experiment.notes,
    )
    resolved_config = replace(config, experiment=experiment)
    validate_config(resolved_config)

    llm_trace = build_hf_causal_lm_trace(resolved_config)
    dataset_provenance = build_dataset_provenance(resolved_config)
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "experiment_name": experiment.name,
        "seed": experiment.seed,
        "output_dir": str(final_output_dir),
        "created_at": _utc_timestamp(),
        "tags": list(experiment.tags),
        "notes": experiment.notes,
        # Once per run, not once per round: run.json is rewritten on every
        # flush and this shells out to git.
        "code_state": capture_code_state(),
    }
    if llm_trace is not None:
        metadata["llm"] = llm_trace
    if dataset_provenance is not None:
        metadata["dataset"] = dataset_provenance
    extension_provenance = build_extension_provenance(resolved_config)
    if extension_provenance is not None:
        metadata["extensions"] = extension_provenance
    return resolved_config, metadata, _runs_index_root(root_output_dir)


def _runtime_extra_bool(config: FullConfig, name: str, default: bool) -> bool:
    value = config.runtime.extra.get(name, default)
    if not isinstance(value, bool):
        raise RunRefused(f"runtime.{name} must be a bool")
    return value


def apply_cli_overrides(
    config: FullConfig,
    args: argparse.Namespace,
) -> FullConfig:
    """Return an effective config with supported CLI overrides applied."""

    experiment = config.experiment
    server = config.server
    client = config.client
    runtime = config.runtime
    evaluation = config.evaluation
    runtime_extra = dict(runtime.extra)
    experiment_tags = list(experiment.tags)

    output_dir = getattr(args, "output_dir", None)
    if output_dir is not None:
        _validate_non_empty("output_dir", output_dir)
        experiment = replace(experiment, output_dir=str(output_dir))

    seed = getattr(args, "seed", None)
    if seed is not None:
        _validate_non_negative_int("seed", seed)
        experiment = replace(experiment, seed=seed)

    rounds = getattr(args, "rounds", None)
    if rounds is not None:
        _validate_positive_int("rounds", rounds)
        server = replace(server, global_rounds=rounds)

    local_iterations = getattr(args, "local_iterations", None)
    if local_iterations is not None:
        _validate_positive_int("local_iterations", local_iterations)
        client = replace(client, local_iterations=local_iterations)

    learning_rate = getattr(args, "lr", None)
    if learning_rate is not None:
        _validate_positive_float("lr", learning_rate)
        client = replace(client, learning_rate=learning_rate)

    batch_size = getattr(args, "batch_size", None)
    if batch_size is not None:
        _validate_positive_int("batch_size", batch_size)
        client = replace(client, batch_size=batch_size)

    participation_rate = getattr(args, "participation_rate", None)
    if participation_rate is not None:
        _validate_participation_rate(participation_rate)
        server = replace(server, participation_rate=participation_rate)

    device = getattr(args, "device", None)
    if device is not None:
        _validate_device(device)
        runtime = replace(runtime, device=device)

    num_workers = getattr(args, "num_workers", None)
    if num_workers is not None:
        _validate_non_negative_int("num_workers", num_workers)
        runtime_extra = _set_dataloader_num_workers(runtime_extra, num_workers)

    if bool(getattr(args, "no_staging", False)):
        runtime_extra = _set_data_staging_enabled(runtime_extra, False)
    if bool(getattr(args, "staging", False)):
        runtime_extra = _set_data_staging_enabled(runtime_extra, True)

    resume_from = getattr(args, "resume_from", None)
    resume_latest = bool(getattr(args, "resume_latest", False))
    if resume_from is not None and resume_latest:
        raise RunRefused("--resume-from and --resume-latest cannot be used together")
    if resume_from is not None:
        _validate_non_empty("resume_from", resume_from)
        runtime_extra["resume_from"] = str(resume_from)
    if resume_latest:
        runtime_extra["resume_latest"] = True

    run_id = getattr(args, "run_id", None)
    if run_id is not None:
        _validate_non_empty("run_id", run_id)
        experiment = replace(experiment, run_id=str(run_id))

    if bool(getattr(args, "use_run_subdir", False)):
        experiment = replace(experiment, use_run_subdir=True)

    for tag in getattr(args, "tag", []) or []:
        _validate_non_empty("tag", tag)
        experiment_tags.append(str(tag))

    notes = getattr(args, "notes", None)
    if notes is not None:
        experiment = replace(experiment, notes=str(notes))

    experiment = replace(experiment, tags=experiment_tags)

    if bool(getattr(args, "quiet", False)):
        runtime_extra["quiet"] = True
    if bool(getattr(args, "verbose", False)):
        runtime_extra["verbose"] = True
    if bool(getattr(args, "no_rich", False)):
        runtime_extra["no_rich"] = True
    runtime_extra = _set_print_every(runtime_extra, getattr(args, "print_every", None))

    effective_config = replace(
        config,
        experiment=experiment,
        server=server,
        client=client,
        runtime=replace(runtime, extra=runtime_extra),
        evaluation=evaluation,
    )
    validate_config(effective_config)
    return effective_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse runner command-line arguments."""

    parser = argparse.ArgumentParser(description="Run a fedbrew experiment.")
    parser.add_argument(
        "config_name",
        nargs="?",
        metavar="name",
        help=(
            f"Short name resolved against {_CONFIG_BASE_DIR}/, .yaml implied "
            "(e.g. 'synthetic/fedavg'). Mutually exclusive with --config."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"Path to a common experiment config YAML file. Defaults to {DEFAULT_CONFIG_PATH}.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run preflight validation and exit without training.",
    )
    parser.add_argument(
        "--output-dir",
        type=_parse_non_empty,
        help="Override config.experiment.output_dir.",
    )
    parser.add_argument(
        "--seed",
        type=_parse_non_negative_int,
        help="Override config.experiment.seed.",
    )
    parser.add_argument(
        "--rounds",
        type=_parse_positive_int,
        help="Override config.server.global_rounds.",
    )
    parser.add_argument(
        "--local-iterations",
        type=_parse_positive_int,
        help="Override config.client.local_iterations.",
    )
    parser.add_argument(
        "--lr",
        type=_parse_positive_float,
        help="Override config.client.learning_rate.",
    )
    parser.add_argument(
        "--batch-size",
        type=_parse_positive_int,
        help="Override config.client.batch_size.",
    )
    parser.add_argument(
        "--participation-rate",
        type=_parse_participation_rate,
        help="Override config.server.participation_rate in (0, 1].",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        help="Override config.runtime.device.",
    )
    parser.add_argument(
        "--num-workers",
        type=_parse_non_negative_int,
        help=(
            "Override runtime.performance.dataloader.num_workers, creating "
            "the block if the config has none."
        ),
    )
    staging_group = parser.add_mutually_exclusive_group()
    staging_group.add_argument(
        "--staging",
        action="store_true",
        help='Enable config.runtime.extra["data_staging"].',
    )
    staging_group.add_argument(
        "--no-staging",
        action="store_true",
        help='Disable config.runtime.extra["data_staging"].',
    )
    parser.add_argument(
        "--resume-from",
        type=_parse_non_empty,
        help='Override config.runtime.extra["resume_from"].',
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume from checkpoints/latest.pt or the highest numbered checkpoint.",
    )
    parser.add_argument(
        "--run-id",
        type=_parse_non_empty,
        help="Override experiment.run_id.",
    )
    parser.add_argument(
        "--use-run-subdir",
        action="store_true",
        help="Write artifacts under config.experiment.output_dir / run_id.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        type=_parse_non_empty,
        help="Append a run tag. Can be repeated.",
    )
    parser.add_argument(
        "--notes",
        help="Override experiment notes.",
    )
    # Defined once in fedbrew.core.console and added to every command that
    # prints, so --quiet means the same thing everywhere it is accepted.
    add_output_arguments(parser)
    # Here rather than beside --quiet: only a run has rounds to report.
    parser.add_argument(
        "--print-every",
        type=_parse_positive_int,
        metavar="N",
        help=(
            "Report rounds 1, N, 2N, ... and the final round, in place of the "
            "evaluation rounds. Every round is still evaluated on its schedule "
            "and written to round_metrics.csv."
        ),
    )
    args = parser.parse_args(argv)

    # Refused as --quiet --verbose is, by the parser: both on one command line
    # contradict each other. A config that sets print_every run with --quiet
    # is not refused -- quiet wins there, as it does over verbose, so a driver
    # that always passes --quiet keeps working.
    if args.quiet and args.print_every is not None:
        parser.error("--print-every reports rounds and --quiet reports none; give one")
    if args.config_name and args.config:
        parser.error(
            f"give a config name or --config, not both ({args.config_name!r} and {args.config!r})"
        )
    if args.config_name:
        try:
            args.config = str(resolve_named_config(_CONFIG_BASE_DIR, args.config_name))
        except FileNotFoundError as exc:
            parser.error(str(exc))
    elif not args.config:
        args.config = DEFAULT_CONFIG_PATH
    return args


def default_override_namespace(**overrides: Any) -> argparse.Namespace:
    """The namespace :func:`apply_cli_overrides` reads when no flag was passed.

    For a caller driving :func:`run` from Python rather than from the command
    line -- an out-of-tree component's entry script, a sweep driver -- that
    wants one or two settings and not the job of supplying the rest.

    ``parse_args([])`` is what produces it, so a flag added to the parser
    appears here in the same commit. The alternative is a second list of flag
    names kept by hand somewhere else, which is correct exactly until the next
    flag lands and silently thereafter, since a missing attribute is not an
    error downstream: ``apply_cli_overrides`` reads every one of them through
    ``getattr(args, name, default)``.

    The empty list is load-bearing. ``parse_args(None)`` reads ``sys.argv``,
    which for a caller with its own flags would parse those, and for a caller
    with a flag this parser also defines would silently adopt its value.

    Args:
        **overrides: Values to set, named as the parsed attribute rather than
            as the flag -- ``local_iterations``, not ``--local-iterations``. A name the
            parser does not define raises rather than being set, for the same
            reason ``_KNOWN_EXTRA_KEYS`` rejects an unread config key: nothing
            downstream distinguishes a misspelled override from an absent one,
            so the run would proceed with the setting quietly not applied.

    Raises:
        ValueError: If an override names an attribute ``parse_args`` does not
            define.
    """

    namespace = parse_args([])
    for name, value in overrides.items():
        if not hasattr(namespace, name):
            known = ", ".join(sorted(vars(namespace)))
            raise ValueError(f"unknown override: {name}. parse_args defines: {known}")
        setattr(namespace, name, value)
    return namespace


#: The status a refused run exits with. argparse already exits 2 when `fedbrew run`
#: refuses its own arguments, and an uncaught exception exits 1, so a script can
#: tell a run that declined its input from one that crashed.
EXIT_REFUSED = 2


def main(argv: Sequence[str] | None = None) -> None:
    """Run the configured benchmark."""

    args = parse_args(argv)
    if args.validate_only:
        if _run_preflight(args):
            raise SystemExit(1)
        return
    try:
        run(args.config, args)
    except RunRefused as refusal:
        _print_refusal(refusal, args)
        raise SystemExit(EXIT_REFUSED) from None


def _print_refusal(refusal: RunRefused, args: argparse.Namespace) -> None:
    """How every refusal looks: a rule and the reason, on stderr, whatever --quiet says."""

    surface = build_surface(no_rich=bool(getattr(args, "no_rich", False)), file=sys.stderr)
    surface.rule("RUN REFUSED", tone=RED)
    surface.line(str(refusal), tone=RED, marker=FAIL, marker_tone=RED, wrap=True)


def _run_preflight(args: argparse.Namespace) -> bool:
    """Stream `--validate-only` as a rail, then its result. True if it failed.

    Three surfaces in sequence, all on one console: the checks as they settle,
    the plan the config describes, and the verdict. The plan is the rail's
    *result* -- it is what a clean preflight has proved -- so a failing check
    withholds it. Printing "here is what will happen" under an error that
    guarantees nothing will happen is worse than printing nothing.
    """

    surface = surface_from_args(args)
    surface.rule("PREFLIGHT")
    rail = surface.rail(["load config", *CHECK_NAMES])

    config: FullConfig | None = None
    with rail.stage("load config") as stage:
        try:
            config = load_config(args.config)
            config = apply_cli_overrides(config, args)
        except Exception as exc:
            # Caught rather than raised: a config that will not load is a
            # preflight finding like any other, and the traceback preflight
            # exists to prevent is not an improvement on the message.
            stage.fail(str(exc))
            report = report_from_exception(exc, config_path=str(args.config))
        else:
            stage.done(str(args.config))

    if config is None:
        print_validation_verdict(report, surface)
        return True

    report = stream_checks(config, str(args.config), rail)
    if not report.num_errors:
        # Resolved the same way run() resolves them, through the same helper,
        # so --validate-only prints the determinism the run would apply
        # rather than its own reading of the same two keys.
        print_plan_header(
            config,
            deterministic=_runtime_extra_bool(config, "deterministic", False),
            deterministic_warn_only=_runtime_extra_bool(config, "deterministic_warn_only", False),
            surface=surface,
        )
    print_validation_verdict(report, surface)
    return bool(report.num_errors)


def _utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _runs_index_root(root_output_dir: Path) -> Path:
    if root_output_dir.name == "outputs":
        return root_output_dir
    return root_output_dir.parent


def _validate_non_empty(name: str, value: Any) -> None:
    if not str(value).strip():
        raise RunRefused(f"{name} must not be empty")


def _validate_non_negative_int(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RunRefused(f"{name} must be a non-negative integer")


def _validate_positive_int(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RunRefused(f"{name} must be a positive integer")


def _validate_positive_float(name: str, value: Any) -> None:
    if not isinstance(value, float | int) or isinstance(value, bool) or float(value) <= 0:
        raise RunRefused(f"{name} must be a positive float")


def _validate_participation_rate(value: Any) -> None:
    if not isinstance(value, float | int) or isinstance(value, bool) or not 0 < float(value) <= 1:
        raise RunRefused("participation_rate must be in (0, 1]")


def _validate_device(value: str) -> None:
    if value not in {"cpu", "cuda", "auto"}:
        raise RunRefused("device must be one of: cpu, cuda, auto")


def _set_dataloader_num_workers(runtime_extra: dict[str, Any], num_workers: int) -> dict[str, Any]:
    """Write the flag's worker count into the only place the loader reads.

    Both blocks are created when absent. They used to be required to exist
    already, so --num-workers was silently a no-op on the ten shipped configs
    that carry no runtime.performance block -- which is most of the configs a
    throughput experiment would start from.
    """

    runtime_extra = dict(runtime_extra)
    performance = runtime_extra.get("performance")
    performance = dict(performance) if isinstance(performance, dict) else {}
    dataloader = performance.get("dataloader")
    dataloader = dict(dataloader) if isinstance(dataloader, dict) else {}
    dataloader["num_workers"] = num_workers
    performance["dataloader"] = dataloader
    runtime_extra["performance"] = performance
    return runtime_extra


def _set_data_staging_enabled(runtime_extra: dict[str, Any], enabled: bool) -> dict[str, Any]:
    runtime_extra = dict(runtime_extra)
    data_staging = runtime_extra.get("data_staging")
    if not isinstance(data_staging, dict):
        data_staging = {}
    else:
        data_staging = dict(data_staging)
    data_staging["enabled"] = enabled
    runtime_extra["data_staging"] = data_staging
    return runtime_extra


def _set_print_every(runtime_extra: dict[str, Any], print_every: Any) -> dict[str, Any]:
    """runtime.print_every from --print-every, or runtime_extra unchanged without it."""

    if print_every is None:
        return runtime_extra
    _validate_positive_int("print_every", print_every)
    return {**runtime_extra, "print_every": print_every}


def _parse_non_empty(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("value must not be empty")
    return value


def _parse_positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a float") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _parse_participation_rate(value: str) -> float:
    parsed = _parse_positive_float(value)
    if parsed > 1:
        raise argparse.ArgumentTypeError("value must be in (0, 1]")
    return parsed


def _parse_non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ExperimentComponents",
    "apply_cli_overrides",
    "build_components",
    "default_override_namespace",
    "main",
    "parse_args",
    "resolve_run_metadata",
    "run",
]


if __name__ == "__main__":
    main()
