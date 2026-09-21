"""Training loop coordination for federated learning experiments."""

from __future__ import annotations

import csv
import math
import random
import statistics
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fedbrew.clients.base import ClientUpdate
from fedbrew.core.artifacts import (
    clear_stale_temp_files,
    flush_round_artifacts,
    retired_metric_columns,
    round_metrics_gap,
)
from fedbrew.core.checkpointing import (
    DEFAULT_SELECTION_METRIC,
    StagedCheckpoints,
    selection_mode_for_metric,
)
from fedbrew.core.config import (
    CLIENT_METRIC_BASES,
    PERSONAL_SPLIT_PREFIX,
    ClientStatisticsConfig,
    DivergenceConfig,
    EvaluationConfig,
    evaluates_round,
    parse_evaluation_client_scope,
    parse_evaluation_schedule,
    worst_percent_label,
)
from fedbrew.core.divergence import (
    STATUS_DIVERGED,
    DivergenceMonitor,
    DivergenceVerdict,
)
from fedbrew.core.protocol import (
    ClientInfo,
    EvalRequest,
    EvalResult,
    FitRequest,
    FitResult,
    RoundInfo,
)
from fedbrew.core.refusal import RunRefused
from fedbrew.core.runtime_setup import capture_rng_state, restore_rng_state
from fedbrew.core.state import (
    ClientEvaluationRecord,
    ClientMetricRecord,
    ExperimentState,
    MetricRecord,
    RoundState,
    RoundTimings,
)
from fedbrew.core.torch_utils import NonFiniteStateError, model_state_is_all_zeros
from fedbrew.data.dataset import FederatedDataset
from fedbrew.servers.base import ServerStrategy

ClientPool = ClientUpdate | Mapping[str, ClientUpdate]

#: The metrics every evaluated client must report, and the split each is
#: measured on. accuracy is deliberately not among them: a task with no
#: notion of correct/incorrect (examples/pl-1d's scalar objective) has none
#: to report, and _aggregate_client_split_metrics and
#: _build_client_evaluation_record both already treat a metric a task never
#: reports as absent for that column rather than crashing on it.
_REQUIRED_CLIENT_METRIC_SPLITS = {
    "train_loss": "train",
    "test_loss": "test",
}

# Splits every in-scope client must be able to report. "val" is deliberately
# absent: a client too small to hold out a validation split reports zero val
# examples and is dropped from the val aggregate instead of failing the round.
_REQUIRED_EVALUATION_SPLITS = {"train", "test"}


def _scope_split_names(model_scope: str, splits: Iterable[str]) -> list[str]:
    """Return the metric split names one evaluation pass produces.

    The personalized pass reports under a "personal_" split prefix --
    personal_val_accuracy beside val_accuracy -- rather than a metric suffix.
    That is deliberate: everything downstream keys off ``f"{split}_{metric}"``,
    so a prefixed split flows through the aggregation, the dispersion
    statistics and the worst-percent summary with no changes at all.
    """

    names: list[str] = []
    for split in splits:
        if model_scope in {"global", "both"}:
            names.append(split)
        if model_scope in {"personal", "both"}:
            names.append(f"{PERSONAL_SPLIT_PREFIX}{split}")
    return names


def run_fl_loop(
    server: ServerStrategy,
    client: ClientPool,
    dataset: FederatedDataset,
    global_rounds: int,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    checkpointing: Mapping[str, Any] | None = None,
    on_round_end: Callable[[MetricRecord], None] | None = None,
    on_round_flush: Callable[[ExperimentState], None] | None = None,
    # Fires as each client finishes fit or evaluation within a round --
    # (round_id, done, total, phase), done 1-indexed, phase one of
    # "fit"/"client_eval". Both are the only stages in the round body that
    # visit a known, bounded client sequence one at a time; nothing surfaced
    # that before this, and `print_round_metrics` only reports once the whole
    # round is done. A no-op default: this is the hook, not a renderer.
    #
    # round_id is passed even though the caller could count rounds itself,
    # because it cannot count them *correctly*: the only round boundary a
    # caller sees is on_round_end, which fires after a round rather than
    # before, so the first round of a resumed process -- which starts at the
    # checkpoint's round, not at 1 -- would be reported under the wrong
    # number. The loop knows which round it is executing; a live progress line
    # without that number is missing the thing a watcher most wants.
    on_client_progress: Callable[[int, int, int, str], None] | None = None,
    # Fires once, the moment a divergence/stall stop is decided -- before the
    # round loop's own `break`. Without this the stop is silent at the moment
    # it happens: the reason string already exists (DivergenceVerdict.reason)
    # but previously reached the caller only in the final summary, after the
    # process had already finished.
    on_termination: Callable[[DivergenceVerdict], None] | None = None,
    evaluation: EvaluationConfig | None = None,
    client_statistics: ClientStatisticsConfig | None = None,
    evaluation_seed: int | None = None,
    divergence: DivergenceConfig | None = None,
) -> ExperimentState:
    """Run a minimal task-agnostic federated loop."""

    _require_positive_global_rounds(global_rounds)
    evaluation = evaluation or EvaluationConfig()
    statistics = client_statistics or ClientStatisticsConfig()
    monitor = DivergenceMonitor(divergence or DivergenceConfig())
    # Every split declares its own schedule and its own client set, so the
    # three passes are priced independently: train is a cheap diagnostic over
    # this round's trainers, val is what selection reads, test is the reported
    # number. Parsed up front so a bad value fails before any work is done.
    schedules = {
        split: parse_evaluation_schedule(getattr(evaluation, split).every, f"evaluation.{split}")
        for split in ("train", "val", "test")
    }
    central_schedule = parse_evaluation_schedule(
        evaluation.central_test.every, "evaluation.central_test"
    )

    # Decided before anything is written: a resume that cannot be taken is
    # refused with the directory exactly as it was. POST-F25.
    server_payload, start_round, checkpoint = _initialize_or_resume(server, resume_from, output_dir)
    # A run that was killed inside an artifact write can leave one temp file
    # behind. Swept here, before anything opens one for this run.
    clear_stale_temp_files(output_dir)
    checkpoint_policy = _checkpoint_policy(checkpointing)
    state = ExperimentState(
        final_payload=server_payload,
        # A resume that is not taken is refused (POST-F25), so a run that got
        # this far with a checkpoint did continue it.
        resumed=resume_from is not None,
    )
    if start_round > 1 and output_dir is not None:
        _load_existing_metric_history(state, output_dir, start_round)
        # The monitor is built fresh above, so without this a resumed run's
        # blow-up ceiling anchors on the resumed round instead of round 1.
        # Primed from the replayed history rather than from checkpointed
        # monitor state, so it also works for a checkpoint written before this
        # existed and stays correct if the history was reloaded from the CSV.
        monitor.prime((record.round_id, record.metrics) for record in state.metrics_history)
    checkpoint_tracker = _initialize_checkpoint_tracker(
        output_dir,
        checkpoint_policy,
        state.metrics_history,
    )
    # How much of each per-client history is already on disk, so the round
    # flush appends its new rows instead of rewriting the whole run. Starts
    # empty even on a resume: the first flush then rewrites both files once,
    # which is what puts the replayed history back under a correct header.
    csv_cursor: dict[str, Any] = {}

    client_infos = _build_client_infos(dataset)
    evaluation_clients = {
        split: _evaluation_client_selector(
            client_infos,
            getattr(evaluation, split).clients,
            evaluation_seed,
            split,
        )
        for split in ("train", "val", "test")
    }
    _setup_clients(client, client_infos)
    _restore_client_states(client, checkpoint)
    if start_round > global_rounds:
        state.checkpointing = _checkpointing_summary(
            output_dir, checkpoint_policy, checkpoint_tracker
        )
        return state

    for round_id in range(start_round, global_rounds + 1):
        round_started = time.perf_counter()
        round_info = RoundInfo(round_id=round_id, total_rounds=global_rounds)
        requests = list(server.configure_round(round_info, client_infos))
        selected_clients = [request.client_id for request in requests]

        # Announced before the first client trains: under participation_probability
        # the selection varies by round and can be empty, and a footer waiting for
        # a finished client would show the previous round's count until one did.
        if on_client_progress is not None:
            on_client_progress(round_id, 0, len(requests), "fit")
        fit_totals = _FitPhaseTotals()
        fit_phase_started = time.perf_counter()
        try:
            # Skipped when no client was selected: the model and every server
            # state carry over unchanged, rather than each strategy defining an
            # update over no results.
            if requests:
                server_payload = server.aggregate_stream(
                    round_info,
                    _stream_fit_results(
                        client, requests, state, fit_totals, round_id, on_client_progress
                    ),
                )
        except NonFiniteStateError as error:
            # Same contract as the monitor below: a model that went non-finite
            # is a recorded outcome, not a crash, so the job exits zero and the
            # sweep index keeps the row. Recorded here rather than one round
            # later because aggregation is where the damage would become
            # permanent -- the server's second-moment/control state would carry
            # the NaN forever, and _update_checkpoints would write it to disk.
            # The round is abandoned before either, so the last checkpoint is
            # the last healthy one.
            state.status = STATUS_DIVERGED
            verdict = DivergenceVerdict(
                status=STATUS_DIVERGED,
                detector="non_finite_client_state",
                round_id=round_id,
                metric="client_model_state",
                value=float("nan"),
                threshold=None,
                reason=(
                    f"round {round_id} aggregation refused a non-finite client "
                    f"state ({error}); the model cannot recover from it"
                ),
            )
            state.termination = verdict.as_dict()
            if on_termination is not None:
                on_termination(verdict)
            break
        fit_phase_seconds = time.perf_counter() - fit_phase_started
        # The clients' own post-fit loss and accuracy used to be dropped here:
        # named "loss" and "accuracy" they were indistinguishable from the
        # global model's numbers on the same round. Prefixed "fit_" they are
        # unambiguous and worth keeping -- they are the only view of what the
        # local models did before averaging.

        # Each split has its own schedule and its own client set, so a client
        # may be due for one split and not another. Grouping by client keeps
        # the one-load-per-client property: a client evaluated for two splits
        # is still visited once.
        splits_by_client, infos_by_client = _round_evaluation_plan(
            evaluation_clients,
            schedules,
            round_id,
            global_rounds,
            selected_clients,
        )
        client_eval_started = time.perf_counter()
        evaluated = _evaluate_models_on_clients(
            client,
            round_id,
            [
                (infos_by_client[client_id], splits)
                for client_id, splits in splits_by_client.items()
            ],
            server_payload,
            evaluation.model_scope,
            on_client_progress,
        )
        client_eval_seconds = time.perf_counter() - client_eval_started
        selected_client_set = set(selected_clients)
        state.client_metrics_history.extend(
            _build_client_evaluation_record(
                result,
                participated=result.client_id in selected_client_set,
                evaluated_splits=splits,
                model_scope=evaluation.model_scope,
            )
            for result, splits in evaluated
        )
        for split in ("train", "val", "test"):
            subset = [result for result, splits in evaluated if split in splits]
            if not subset:
                continue
            # One aggregation per scope. The personal pass is just another
            # split name, so this is the same call with a different key.
            for metric_split in _scope_split_names(evaluation.model_scope, [split]):
                round_info.metrics.update(
                    _aggregate_client_split_metrics(subset, metric_split, statistics)
                )
        global_eval_started = time.perf_counter()
        if evaluates_round(central_schedule, round_id, global_rounds):
            round_info.metrics.update(_evaluate_central_test_set(server, dataset))
        global_eval_seconds = time.perf_counter() - global_eval_started

        num_examples = fit_totals.num_examples
        metrics = dict(round_info.metrics)
        if isinstance(server_payload, dict):
            server_payload["metrics"] = metrics
        checkpoint_started = time.perf_counter()
        checkpoint_payload = _build_checkpoint_payload(
            server,
            client,
            server_payload,
            metrics,
            round_id,
        )
        # Written now, so checkpoint_sec times the write; visible only at the
        # commit below, after this round's CSV rows and run.json. POST-F24.
        staged = _update_checkpoints(
            checkpoint_payload,
            metrics,
            output_dir,
            round_id,
            checkpoint_policy,
            checkpoint_tracker,
        )
        checkpoint_seconds = time.perf_counter() - checkpoint_started

        # server_payload is deliberately not stored here: nothing reads it back,
        # and retaining one global model state per round grew without bound over
        # a long run. state.final_payload carries the model the run ends with.
        timings = RoundTimings(
            fit=fit_totals.fit_seconds,
            # The server consumes fit results as they stream in, so the wall
            # time of aggregate_stream contains the client fits. Subtracting
            # them leaves the server's own aggregation cost.
            aggregate=max(fit_phase_seconds - fit_totals.fit_seconds, 0.0),
            client_eval=client_eval_seconds,
            global_eval=global_eval_seconds,
            checkpoint=checkpoint_seconds,
            total=time.perf_counter() - round_started,
        )
        state.rounds.append(
            RoundState(
                round_id=round_id,
                metrics=metrics,
                num_clients=len(selected_clients),
                num_examples=num_examples,
                timings=timings,
            )
        )
        metric_record = MetricRecord(
            round_id=round_id,
            metrics=metrics,
            num_clients=len(selected_clients),
            num_examples=num_examples,
            timings=timings,
        )
        state.metrics_history.append(metric_record)
        # Written every round rather than once at the end: a run that is
        # cancelled, preempted or hits its wall clock never reaches the save in
        # runner.run(), and used to leave a checkpoint at round N beside no
        # metrics at all -- so --resume-latest had nothing to replay and the
        # resumed run's CSV started mid-experiment.
        flush_round_artifacts(
            state.metrics_history,
            state.client_metrics_history,
            state.client_update_metrics_history,
            output_dir,
            statistics.per_client_csv,
            csv_cursor,
        )
        # run.json is assembled from config and run metadata the loop does not
        # hold, so the runner supplies its own writer rather than the loop
        # reaching for them.
        if on_round_flush is not None:
            on_round_flush(state)
        # Last of the round's writes: a checkpoint must never be visible for a
        # round the history does not hold yet, or a kill in between leaves a
        # run no resume can continue. POST-F24.
        _commit_checkpoints(staged, output_dir, checkpoint_policy)
        if on_round_end is not None:
            on_round_end(metric_record)

        # Checked last, so the round that triggers the stop is still fully
        # recorded: its metrics, timings and checkpoint are the evidence of
        # what went wrong. Breaking rather than raising keeps the exit code
        # zero, which is what stops a packed SLURM job from reporting a
        # diverged arm as a failed one.
        verdict = monitor.update(round_id, metrics)
        if verdict is not None:
            state.status = verdict.status
            state.termination = verdict.as_dict()
            if on_termination is not None:
                on_termination(verdict)
            break

    if divergence is not None and divergence.active and not monitor.observed:
        # Every detector reads one metric name; a name nothing emits silences
        # all of them -- including non_finite -- for the whole run, with no
        # error, because "absent" is indistinguishable from "not evaluated this
        # round".
        print(
            f"Warning: divergence.metric={monitor.metric!r} was never present in "
            "any round's metrics, so no divergence detector ran. Check the name "
            "against the metrics this config emits.",
            flush=True,
        )

    state.final_payload = server_payload
    state.checkpointing = _checkpointing_summary(output_dir, checkpoint_policy, checkpoint_tracker)
    return state


def _initialize_or_resume(
    server: ServerStrategy,
    resume_from: str | Path | None,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, Any], int, Mapping[str, Any] | None]:
    """Start the run, continuing an earlier attempt, or refuse to.

    Returns:
        The broadcast payload, the round to start at, and the checkpoint the
        run was resumed from (None when none was asked for). A resume that
        cannot be taken raises RunRefused and changes nothing on disk.
    """

    if resume_from is None:
        return server.initialize(), 1, None

    from fedbrew.core.checkpointing import (
        get_checkpoint_round_id,
        load_checkpoint,
        refuse_a_pre_rename_checkpoint,
    )

    if not Path(resume_from).is_file():
        raise RunRefused(
            f"cannot resume from {resume_from}: no checkpoint file exists at that "
            "path. --resume-latest picks the newest checkpoint in the output directory."
        )
    checkpoint = load_checkpoint(resume_from)
    round_id = get_checkpoint_round_id(checkpoint)

    # A resume is only worth taking when the rounds before the checkpoint's are
    # all on record: continuing from round N+1 onto a CSV that stops at round 10
    # produces a file with a silent hole in it. It used to answer by deleting
    # the attempt -- checkpoints, CSVs, run.json -- and starting again from
    # round 1, which turned a stopped 10000-round run into a fresh one and left
    # nothing to decide from: refused now, with everything where it was.
    # POST-F25.
    # Before the gap check, which reads the same CSV: a file whose header is
    # in the retired FedLALR names cannot be continued under the new ones
    # without holding both. POST-F14.
    retired = retired_metric_columns(output_dir)
    if retired:
        from fedbrew.core.metrics import RETIRED_METRIC_NAMES

        listed = "; ".join(f"{name}: {', '.join(columns)}" for name, columns in retired.items())
        renames = "; ".join(
            f"{column} -> {RETIRED_METRIC_NAMES[column]}"
            for column in sorted({c for columns in retired.values() for c in columns})
        )
        raise RunRefused(
            f"cannot resume from {resume_from}: the run's CSVs were written under "
            f"retired FedLALR metric names ({listed}), and a resumed round writes the "
            "new ones, so one file would hold both spellings of one quantity. Nothing "
            f"was run and nothing in {output_dir} was changed. The names were renamed "
            f"by what they compute ({renames}). Finish the run with the release that "
            "wrote it, or move the directory aside and run again from round 1."
        )
    gap = round_metrics_gap(output_dir, round_id)
    if gap is not None:
        raise RunRefused(
            f"cannot resume from {resume_from} (round {round_id}): {gap}. Continuing would "
            "leave those rounds out of every artifact, so nothing was run and nothing in "
            f"{output_dir} was changed.\n"
            "  Resume from an earlier checkpoint whose round the CSV reaches, with "
            "--resume-from <that checkpoint>, if the run kept one.\n"
            "  Or move the directory aside and run again without a resume flag, which "
            "starts from round 1."
        )

    # Before anything is restored, and for every client at once: the lazy pool
    # calls load_state only when a client is first built, so the per-client
    # check there could fire rounds into the resumed run.
    refuse_a_pre_rename_checkpoint(checkpoint)
    _refuse_a_half_restored_resume(server, checkpoint)
    _restore_server_state(server, checkpoint)
    # initialize() before the RNG is put back, not after. It builds the global
    # model through the task, and model construction draws from the
    # process-wide RNG. The uninterrupted run makes that draw once, before
    # round 1 -- so before the checkpoint was written. Restoring first and
    # initializing after left the resumed run exactly one model construction
    # ahead in the stream, and every dropout mask from the resume round on was
    # drawn from the wrong position. Measured on the four-round CPU fixture in
    # tests/test_reproducibility.py: the resumed run's round 3 consumed 18
    # RNG-moving calls against the uninterrupted run's 17.
    server_payload = server.initialize()
    _restore_rng_state(checkpoint)
    if isinstance(checkpoint.get("metrics"), dict):
        server_payload["metrics"] = checkpoint["metrics"]
    return server_payload, round_id + 1, checkpoint


def _refuse_a_half_restored_resume(
    server: ServerStrategy,
    checkpoint: Mapping[str, Any],
) -> None:
    """Refuse a checkpoint that carries coupled server state and no client state.

    SCAFFOLD defines its server control variate as ``c = (1/N) sum_i c_i`` and
    corrects every local step by ``c - c_i``. `_restore_server_state` puts
    ``c`` back; `_restore_client_states` returns early when the checkpoint has
    no ``client_states``, so every ``c_i`` stays at its initial zeros. The
    correction becomes ``+c`` for every client, the server keeps updating from
    the restored ``c``, and the gap never closes -- the two sides move by the
    same increments from then on, so the error is preserved exactly rather than
    decaying. Nothing in the run notices: it completes and reports success.

    `best.pt` is such a checkpoint by construction (`_without_client_states`
    strips per-client state from it, which is worth 40 GB a file on FEMNIST
    SCAFFOLD), and ``--resume-from .../best.pt`` is one thing to type. Measured
    on a four-client SCAFFOLD run checkpointed at round 3 and resumed for three
    more: from `latest.pt` the invariant held to 4.8e-08, from `best.pt` the
    residual was 0.4668 -- the whole of ``||c||`` -- and the final models
    differed by 7.4% of the model norm.

    The refusal is narrow on purpose. It fires only when the checkpoint really
    does carry a non-zero coupled value, so a round-1 checkpoint whose ``c`` is
    still zeros resumes as before, and it never fires for a strategy whose
    clients carry nothing across rounds -- FedAvg resuming from `best.pt` is
    fine and stays fine.

    Raises:
        RunRefused: If the strategy declares `coupled_client_state` and the
            checkpoint holds a non-zero value under one of those keys while
            carrying no `client_states`.
    """

    coupled = getattr(type(server), "coupled_client_state", {})
    if not coupled or checkpoint.get("client_states"):
        return

    server_state = checkpoint.get("server_state")
    if not isinstance(server_state, Mapping):
        return

    stranded = sorted(
        f"{key} (the clients' {client_key})"
        for key, client_key in coupled.items()
        if _carries_a_non_zero_state(server_state, key)
    )
    if not stranded:
        return

    raise RunRefused(
        f"cannot resume {type(server).__name__} from this checkpoint: it carries "
        f"{', '.join(stranded)} but no client_states, so the server half would be "
        "restored while every client's half resets to zero. The two are defined "
        "in terms of each other and neither side would notice; the run would "
        "complete and report success. best.pt is written without client state on "
        "purpose -- resume from checkpoints/latest.pt, or --resume-latest."
    )


def _carries_a_non_zero_state(server_state: Mapping[str, Any], key: str) -> bool:
    value = server_state.get(key)
    return isinstance(value, Mapping) and bool(value) and not model_state_is_all_zeros(value)


def _restore_rng_state(checkpoint: Mapping[str, Any]) -> None:
    """Put the RNG back where the checkpointed round left it."""

    rng_state = checkpoint.get("rng_state")
    if not isinstance(rng_state, Mapping) or not rng_state:
        # A checkpoint written before rng_state existed. Continuing on a fresh
        # stream is what this fix exists to stop being silent, so say it.
        print(
            "Warning: this checkpoint carries no rng_state, so the resumed run "
            "will not reproduce the uninterrupted run at this seed. Any model "
            "with dropout or another global-RNG consumer will diverge from it.",
            flush=True,
        )
        return

    restored = restore_rng_state(rng_state)
    missing = sorted(set(rng_state) - set(restored))
    if missing:
        print(
            "Warning: resumed without restoring the "
            f"{', '.join(missing)} RNG stream(s) recorded in the checkpoint "
            "(a different device count or a missing optional dependency).",
            flush=True,
        )


def _restore_server_state(
    server: ServerStrategy,
    checkpoint: Mapping[str, Any],
) -> None:
    raw_state = checkpoint.get("server_state")
    if raw_state is not None and not isinstance(raw_state, Mapping):
        raise RunRefused("checkpoint server_state must be a mapping")

    server_state = dict(raw_state or {})
    # The model lives at the top level and is no longer duplicated inside
    # server_state. Re-inject it for load_state, which reads it from there.
    # A checkpoint written before that change still carries its own copy, so
    # take the nested one when it is present: the two were byte-identical, and
    # preferring it keeps an old checkpoint restoring exactly as it always did.
    if "model_state" not in server_state:
        for field in ("model_state", "model_state_scope", "model_state_metadata"):
            if field in checkpoint:
                server_state[field] = checkpoint[field]

    # load_state skips the model restore when model_state is absent rather
    # than complaining, so without this a checkpoint that carried neither copy
    # would resume from freshly initialised weights and report it as a
    # successful resume -- the whole run silently starting over at round R.
    if not isinstance(server_state.get("model_state"), dict):
        raise RunRefused(
            "checkpoint must contain a model_state, at the top level or inside "
            "server_state; resuming without one would restart from the "
            "initial weights while reporting a successful resume"
        )
    server.load_state(server_state)


def _restore_client_states(
    client: ClientPool,
    checkpoint: Mapping[str, Any] | None,
) -> None:
    if checkpoint is None or not isinstance(client, Mapping):
        return
    client_states = checkpoint.get("client_states")
    if client_states is None:
        return
    if not isinstance(client_states, Mapping):
        raise RunRefused("checkpoint client_states must be a mapping")
    load_state_snapshot = getattr(client, "load_state_snapshot", None)
    if callable(load_state_snapshot):
        load_state_snapshot(client_states)
        return

    for client_id, client_state in client_states.items():
        if client_id not in client:
            continue
        if not isinstance(client_state, Mapping):
            raise RunRefused("checkpoint client state must be a mapping")
        client[str(client_id)].load_state(client_state)


def _setup_clients(client: ClientPool, client_infos: list[ClientInfo]) -> None:
    setup_client_infos = getattr(client, "setup_client_infos", None)
    if callable(setup_client_infos):
        setup_client_infos(client_infos)
        return

    if isinstance(client, Mapping):
        for client_info in client_infos:
            if client_info.client_id in client:
                client[client_info.client_id].setup(client_info)
        return

    for client_info in client_infos:
        client.setup(client_info)


@dataclass(slots=True)
class _FitPhaseTotals:
    """Totals collected while fit results stream past on their way to the server."""

    num_examples: int = 0
    fit_seconds: float = 0.0


def _stream_fit_results(
    client: ClientPool,
    requests: Sequence[FitRequest],
    state: ExperimentState,
    totals: _FitPhaseTotals,
    round_id: int,
    on_progress: Callable[[int, int, int, str], None] | None = None,
) -> Iterator[FitResult]:
    """Yield fit results one at a time, recording per-client update metrics.

    Yielding instead of returning a list means a client's model state becomes
    unreachable as soon as the server has folded it into its aggregate, so peak
    memory stays at one client model rather than one per participating client.

    Clients are deliberately not released here even though the evaluation phase
    releases the ones it touches. The evaluation phase runs immediately after
    this one over an overlapping client set, so releasing here just forces every
    client to be rebuilt moments later. The cost is that a round holds every
    participating client's data at once, which is what makes peak RSS scale with
    participation_rate x client count. See docs/11-performance-and-cost.md.
    """

    total = len(requests)
    for done, request in enumerate(requests, start=1):
        fit_started = time.perf_counter()
        result = _fit_client(client, request)
        totals.fit_seconds += time.perf_counter() - fit_started
        state.client_update_metrics_history.append(_build_client_metric_record(result))
        totals.num_examples += result.num_examples
        if on_progress is not None:
            on_progress(round_id, done, total, "fit")
        yield result


def _fit_client(client: ClientPool, request: FitRequest) -> FitResult:
    if isinstance(client, Mapping):
        return client[request.client_id].fit(request)
    return client.fit(request)


def _evaluate_client(client: ClientPool, request: EvalRequest) -> EvalResult:
    if isinstance(client, Mapping):
        return client[request.client_id].evaluate(request)
    return client.evaluate(request)


def _evaluation_client_selector(
    all_client_infos: list[ClientInfo],
    scope: str,
    seed: int | None,
    split: str,
) -> Callable[[int, list[str]], list[ClientInfo]]:
    """Return a function mapping (round_id, selected_clients) -> clients to evaluate.

    The four modes differ in what changes between rounds:

    ``all``       every client -- exact, and the most expensive.
    ``participating`` this round's training clients -- free, but the set changes
                  every round and is biased toward clients the model has just
                  been fitted on.
    ``sample:N``  one draw for the whole run. The client set is constant, so a
                  round-over-round change in the metric can only come from the
                  model. This is the mode to select or early-stop on.
    ``resample:N`` a fresh draw every round. Converges to the whole-population
                  mean, but every round-over-round comparison carries two
                  rounds' worth of sampling noise, which is worst for the tail
                  statistics (_std, _min, _bottom10).

    Both sampling modes are deterministic functions of the run seed, the client
    roster and the split, so a resumed run keeps measuring exactly the same
    clients without any of it being persisted in a checkpoint.

    The split is in that list because it used to be missing from it. The draw
    key carried seed, sample size and round and nothing else, so two splits
    configured at the same `N` drew the *identical* clients: the clients whose
    validation data selects `best.pt` were exactly the clients whose test data
    is then reported. Two shipped configs are in that shape --
    `configs/openimage/fedavg.yaml` at `sample:2000` and
    `configs/reference_evaluation.yaml` at `sample:40` -- and both overlapped
    100% where independent draws would overlap 14.5% and 4%.
    """

    mode, sample_size = parse_evaluation_client_scope(scope)

    if mode == "all":
        return lambda round_id, selected_clients: all_client_infos

    if mode == "participating":
        return lambda round_id, selected_clients: _participating_client_infos(
            all_client_infos, selected_clients
        )

    if mode == "sample":
        fixed = _sampled_client_infos(all_client_infos, sample_size, seed, None, split)
        return lambda round_id, selected_clients: fixed

    return lambda round_id, selected_clients: _sampled_client_infos(
        all_client_infos, sample_size, seed, round_id, split
    )


def _participating_client_infos(
    all_client_infos: list[ClientInfo],
    selected_clients: list[str],
) -> list[ClientInfo]:
    client_infos_by_id = {client_info.client_id: client_info for client_info in all_client_infos}
    unique_selected_clients = list(dict.fromkeys(selected_clients))
    return [
        client_infos_by_id[client_id]
        for client_id in unique_selected_clients
        if client_id in client_infos_by_id
    ]


def _sampled_client_infos(
    all_client_infos: list[ClientInfo],
    sample_size: int | None,
    seed: int | None,
    round_id: int | None,
    split: str,
) -> list[ClientInfo]:
    """Draw a reproducible subset of clients for one split.

    ``round_id`` is None for a fixed draw and set for a per-round redraw; it is
    the only thing that distinguishes the two, so the fixed draw is literally
    the same call with the round left out of the key.

    ``split`` is in the key so that two splits sampled at the same ``N`` draw
    different clients. Without it they drew the same ones, which made the
    reported test score a measurement of the very clients `best.pt` was
    selected on.

    The draw runs over ids sorted by name so it does not depend on the order
    the dataset happens to list clients in, and the result is returned in
    roster order so downstream artifacts keep their usual ordering.
    """

    if sample_size is None or sample_size >= len(all_client_infos):
        return all_client_infos
    key = f"evaluation_clients:{split}:{seed}:{sample_size}:{round_id}"
    chosen = set(
        random.Random(key).sample(
            sorted(client_info.client_id for client_info in all_client_infos),
            sample_size,
        )
    )
    return [client_info for client_info in all_client_infos if client_info.client_id in chosen]


def _round_evaluation_plan(
    evaluation_clients: Mapping[str, Callable[[int, list[str]], list[ClientInfo]]],
    schedules: Mapping[str, int | None],
    round_id: int,
    final_round: int,
    selected_clients: list[str],
) -> tuple[dict[str, list[str]], dict[str, ClientInfo]]:
    """Return which splits each client is evaluated on this round."""

    splits_by_client: dict[str, list[str]] = {}
    infos_by_client: dict[str, ClientInfo] = {}
    for split in ("train", "val", "test"):
        if not evaluates_round(schedules[split], round_id, final_round):
            continue
        for client_info in evaluation_clients[split](round_id, selected_clients):
            splits_by_client.setdefault(client_info.client_id, []).append(split)
            infos_by_client[client_info.client_id] = client_info
    return splits_by_client, infos_by_client


def _evaluate_models_on_clients(
    client: ClientPool,
    round_id: int,
    work: list[tuple[ClientInfo, list[str]]],
    server_payload: Mapping[str, Any],
    model_scope: str = "global",
    on_progress: Callable[[int, int, int, str], None] | None = None,
) -> list[tuple[EvalResult, list[str]]]:
    """Evaluate the requested model(s) on each client's own splits.

    ``model_scope`` selects which model is measured. A client rule that has no
    personal model must reject a non-global scope rather than silently
    evaluating the global one under a personalized name.
    """

    base_payload = dict(server_payload)
    base_payload.update({"metrics": ["loss", "accuracy"], "model_scope": model_scope})

    total = len(work)
    results: list[tuple[EvalResult, list[str]]] = []
    for done, (client_info, splits) in enumerate(work, start=1):
        request = EvalRequest(
            round_id=round_id,
            client_id=client_info.client_id,
            payload={**base_payload, "splits": list(splits)},
        )
        try:
            result = _evaluate_client(client, request)
            _validate_client_evaluation(result, splits, model_scope)
            results.append((result, splits))
            if on_progress is not None:
                on_progress(round_id, done, total, "client_eval")
        finally:
            _release_client(client, request.client_id)
    return results


#: Longest first, so `central_test_` is stripped whole rather than leaving a
#: `test_`-shaped remainder. Tried, in order, against every key
#: `evaluate_global` reports -- both to resolve `loss`/`accuracy` under
#: whichever spelling the server used, and to name everything else it
#: reported the same way.
_CENTRAL_METRIC_PREFIXES = ("central_test_", "global_test_", "global_", "test_")

#: `loss` and `accuracy` are resolved by the multi-spelling loop below, under
#: whichever prefix the server used; nothing else may claim either bare name,
#: even if a task's `evaluate_global` also reports one under a spelling that
#: loop did not try.
_RESOLVED_CENTRAL_METRICS = frozenset({"loss", "accuracy"})

#: The subset of `_RESOLVED_CENTRAL_METRICS` a central pass must report.
#: `accuracy` is not among them: a task with no notion of correct/incorrect
#: (examples/pl-1d's scalar objective) has none to report, and that is not
#: the failure `loss` missing or non-numeric is.
_REQUIRED_CENTRAL_METRICS = frozenset({"loss"})

#: What each refusal of the central test pass tells the reader to do. The pass
#: is on by default, every 10 rounds, so a run can meet one without its config
#: having asked for central evaluation at all.
_CENTRAL_TEST_OFF = "Set evaluation.central_test.every to never to run without it."


def _require_positive_global_rounds(global_rounds: int) -> None:
    """Refuse a round count below one from Python code calling `run_fl_loop`.

    A ValueError, not a RunRefused: `fedbrew run` cannot get here with one,
    because argparse refuses `--rounds` below one and `validate_config` refuses
    `global_rounds` below one, so reaching this is a bug in the caller.
    """

    if global_rounds <= 0:
        raise ValueError("global_rounds must be positive")


def _evaluate_central_test_set(
    server: ServerStrategy,
    dataset: FederatedDataset,
) -> dict[str, float]:
    evaluator = getattr(server, "evaluate_global", None)
    if not callable(evaluator):
        raise RunRefused(
            "evaluation.central_test is on, but the server strategy has no "
            f"evaluate_global to run it. {_CENTRAL_TEST_OFF}"
        )

    try:
        global_data = dataset.get_global_data()
    except (FileNotFoundError, KeyError) as error:
        raise RunRefused(
            "evaluation.central_test is on, but the dataset does not provide global_test "
            f"data. {_CENTRAL_TEST_OFF}"
        ) from error
    if global_data is None:
        raise RunRefused(
            "evaluation.central_test is on, but the dataset does not provide global_test "
            f"data. {_CENTRAL_TEST_OFF}"
        )
    return _central_test_metrics(evaluator(global_data))


def _central_test_metrics(raw_metrics: Any) -> dict[str, float]:
    """The round metrics a strategy's ``evaluate_global`` result stands for.

    Split from `_evaluate_central_test_set` because the two refuse different
    things. That function refuses a run the central pass cannot serve -- a
    strategy with no ``evaluate_global``, a dataset with no global test data --
    which is input. This one holds what ``evaluate_global`` returned to its
    contract, a mapping with a numeric loss, and a strategy that breaks it has
    a defect, which keeps its traceback.
    """

    if not isinstance(raw_metrics, Mapping):
        raise ValueError("central test evaluation must return a metric mapping")

    metrics: dict[str, float] = {}
    for name in ("loss", "accuracy"):
        value = next(
            (
                raw_metrics[key]
                for key in (
                    f"central_test_{name}",
                    f"global_test_{name}",
                    f"global_{name}",
                    f"test_{name}",
                    name,
                )
                if key in raw_metrics
            ),
            None,
        )
        if isinstance(value, bool) or not isinstance(value, int | float):
            if name in _REQUIRED_CENTRAL_METRICS:
                raise ValueError(f"central test evaluation must report a numeric {name} metric")
            continue
        metrics[f"central_test_{name}"] = float(value)

    # Everything else evaluate_global reported, under whichever bare name is
    # left after stripping one known prefix -- so a task's own central-pass
    # diagnostic (optimality_gap) reaches round_metrics.csv as
    # central_test_optimality_gap the same way loss (and accuracy, when a
    # task has one) do, instead of being discarded for being neither.
    # Silently dropped rather than raising: unlike loss nothing downstream
    # requires these, and a non-finite one leaves the round's cell blank
    # rather than failing the round -- the divergence monitor already owns
    # what a diverged run means, and a stray extra diagnostic should not get
    # a second, competing say.
    for key, value in raw_metrics.items():
        bare = _strip_central_metric_prefix(key)
        if bare in _RESOLVED_CENTRAL_METRICS or isinstance(value, bool):
            continue
        if not isinstance(value, int | float) or not math.isfinite(value):
            continue
        metrics.setdefault(f"central_test_{bare}", float(value))
    return metrics


def _strip_central_metric_prefix(key: str) -> str:
    for prefix in _CENTRAL_METRIC_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _release_client(client: ClientPool, client_id: str) -> None:
    release_client = getattr(client, "release_client", None)
    if callable(release_client):
        release_client(client_id)


def _validate_client_evaluation(
    result: EvalResult,
    evaluated_splits: list[str],
    model_scope: str = "global",
) -> None:
    """Check that a client reported every metric and count the scope implies.

    A personalized pass has to be held to the same standard as the global one:
    a client rule that quietly skipped it would otherwise leave the personal
    columns absent from the CSV rather than failing the round.
    """

    prefixes = _scope_split_names(model_scope, [""])
    missing_metrics = [
        f"{prefix}{name}"
        for prefix in prefixes
        for name, split in _REQUIRED_CLIENT_METRIC_SPLITS.items()
        if split in evaluated_splits and f"{prefix}{name}" not in result.metrics
    ]
    if missing_metrics:
        raise ValueError(
            f"client {result.client_id!r} evaluation is missing metrics: "
            + ", ".join(missing_metrics)
        )

    counts = result.payload.get("num_examples_by_split")
    if not isinstance(counts, Mapping):
        raise ValueError(f"client {result.client_id!r} evaluation must report split counts")
    for split in _scope_split_names(model_scope, evaluated_splits):
        if split.removeprefix(PERSONAL_SPLIT_PREFIX) not in _REQUIRED_EVALUATION_SPLITS:
            continue
        count = counts.get(split)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(
                f"client {result.client_id!r} evaluation must report a positive "
                f"{split} example count"
            )


def _aggregate_client_split_metrics(
    results: list[EvalResult],
    split: str,
    statistics_config: ClientStatisticsConfig,
) -> dict[str, float]:
    """Summarise one evaluated split across clients.

    The two averages are always emitted for a metric that is actually
    present: they are what the split means. The dispersion statistics are
    configurable because each is a column in every row of every CSV, not
    because they cost anything -- the expensive part is producing the
    per-client numbers, and everything below is a pass over a list of floats.

    Clients reporting zero examples for this split are excluded; see
    ``_REQUIRED_EVALUATION_SPLITS``. A base metric -- accuracy, in practice,
    since loss stays required at ``_validate_client_evaluation`` -- that is
    absent from any client counted into this split is skipped entirely
    rather than aggregated from whichever clients happened to report it: a
    partial average would silently mix "this task has no accuracy" with
    "this task has accuracy and one client's rule forgot to report it," and
    those are not the same defect.
    """

    counts: list[int] = []
    qualifying: list[EvalResult] = []
    for result in results:
        count = _optional_evaluation_split_count(result, split)
        if count <= 0:
            continue
        counts.append(count)
        qualifying.append(result)

    if not counts:
        raise RunRefused(
            f"no client reported a non-empty {split} split; the dataset has no "
            f"{split} data and cannot be used for this evaluation"
        )

    total_examples = sum(counts)
    # How many clients every average below is over. Emitted beside them
    # because the aggregate's name does not say: `val_accuracy_sample_
    # weighted_avg` means one estimator at `val.clients: all` and another at
    # `sample:200`, and configs/femnist/fedavg_ft.yaml is the one FEMNIST arm
    # that samples -- 200 writers against the other ten arms' 3,597, a
    # standard error sqrt(3597/200) = 4.2x larger on a column tabled beside
    # theirs. The count is the one fact that makes the difference visible in
    # the data rather than only in the configs. P07-F06.
    #
    # len(qualifying), not len(results): a client reporting zero examples for
    # this split is excluded from the averages, so counting it here would
    # describe a population the numbers are not over.
    aggregated: dict[str, float] = {f"{split}_num_clients": float(len(qualifying))}
    for metric in CLIENT_METRIC_BASES:
        if not all(f"{split}_{metric}" in result.metrics for result in qualifying):
            continue
        values = [float(result.metrics[f"{split}_{metric}"]) for result in qualifying]
        prefix = f"{split}_{metric}"
        # The whole system's number: total over total, identical to pooling
        # every client's data into one set.
        aggregated[f"{prefix}_sample_weighted_avg"] = (
            sum(value * count for value, count in zip(values, counts, strict=True)) / total_examples
        )
        # The uniform mean over clients: every client counts once.
        aggregated[f"{prefix}_avg"] = _overflow_safe(statistics.fmean, values)
        aggregated.update(
            _client_distribution_statistics(prefix, metric, values, statistics_config)
        )
    return aggregated


def _overflow_safe(statistic: Callable[[list[float]], float], values: list[float]) -> float:
    """Run one statistic over finite values, answering NaN if it cannot fit.

    Three of the helpers on this path raise OverflowError on inputs that are
    every one of them a finite float. ``pstdev`` and ``pvariance`` sum the squared
    deviations as exact rationals and convert at the end, so they raise once the
    values pass about 1e154 -- their square exceeds float range even though they
    do not. ``fmean`` raises from fsum's intermediate accumulator near 1e308.
    A diverging run walks through both thresholds in order.

    The condition is the statistic's, not the data's, so the guard belongs on
    the call and not on the inputs: the values are finite, the caller's
    ``finite`` check is right to pass them, and NaN is the answer that check
    already gives for a distribution nothing can summarise. It keeps the column
    set identical to a healthy round and leaves the divergence monitor to say
    what a run that got here means.

    Only OverflowError is caught. fsum also raises ValueError on a mix of
    infinities, but that needs a non-finite input, which never reaches here.

    The guard is per statistic, and that leaves an asymmetry on purpose. From
    CPython 3.11 ``pstdev`` takes the square root before converting, so on a
    round where ``pvariance`` overflows ``pstdev`` can still return a float: a
    row then carries a finite ``_std`` beside a NaN ``_variance``. Both are
    true -- the variance exceeds float range and its square root does not --
    and forcing the pair to NaN together would hide a fact to make the row
    look tidy, the same trade as recomputing in floats and rejected for the
    same reason. On 3.10 both overflow and both read NaN. A reader comparing
    the two columns across interpreters is seeing ``statistics`` change, not
    the run.

    See FINDINGS.csv POST-F02.
    """

    try:
        return statistic(values)
    except OverflowError:
        return math.nan


def _client_distribution_statistics(
    prefix: str,
    metric: str,
    values: list[float],
    config: ClientStatisticsConfig,
) -> dict[str, float]:
    """Statistics over the per-client values, each independently switchable.

    Every helper here is NaN-hostile, in two different ways. ``pstdev`` and
    ``pvariance`` use exact rational arithmetic and raise outright on a
    non-finite value (ValueError for NaN, OverflowError for an infinity, on
    CPython 3.10), which would kill the process mid-round, before the
    divergence monitor gets to record the blow-up as an outcome. ``min``, ``max`` and
    ``sorted`` are worse: NaN comparisons are all False, so they return
    whichever element the NaN happened to sit next to and give no sign of it.

    So a non-finite input short-circuits the whole block to NaN. That is the
    answer the monitor is built to read (divergence.non_finite), it keeps the
    emitted column set identical to a finite round, and it leaves the decision
    about what a diverged run means in the one place that owns it.

    Finiteness is necessary and not sufficient: ``pstdev`` and ``pvariance``
    also raise on finite values too large to square. ``_overflow_safe`` catches
    that and gives the same NaN, so the two hazards get one answer.
    """

    # Same key set either way: a round that dropped columns instead of
    # reporting NaN would change the CSV schema partway through a run.
    finite = all(math.isfinite(value) for value in values)

    computed: dict[str, float] = {}
    if config.std:
        computed[f"{prefix}_std"] = (
            _overflow_safe(statistics.pstdev, values) if finite else math.nan
        )
    if config.variance:
        computed[f"{prefix}_variance"] = (
            _overflow_safe(statistics.pvariance, values) if finite else math.nan
        )
    if config.min:
        computed[f"{prefix}_min"] = min(values) if finite else math.nan
    if config.max:
        computed[f"{prefix}_max"] = max(values) if finite else math.nan

    worst_percent = config.worst_percent
    if worst_percent:
        # "Worst" is the direction that is bad for the metric: the lowest
        # accuracies, but the highest losses. Calling it "bottom" would have
        # meant opposite things for the two.
        count = max(1, math.ceil(len(values) * float(worst_percent) / 100.0))
        label = worst_percent_label(worst_percent)
        if finite:
            ordered = sorted(values, reverse=metric != "accuracy")
            computed[f"{prefix}_worst{label}"] = _overflow_safe(statistics.fmean, ordered[:count])
        else:
            computed[f"{prefix}_worst{label}"] = math.nan
    return computed


def _optional_evaluation_split_count(result: EvalResult, split: str) -> int:
    """Return the split's example count, or 0 when the client has no such data."""

    counts = result.payload.get("num_examples_by_split")
    if not isinstance(counts, Mapping):
        return 0
    count = counts.get(split)
    if isinstance(count, bool) or not isinstance(count, int):
        return 0
    return max(count, 0)


def _build_client_evaluation_record(
    result: EvalResult,
    *,
    participated: bool,
    evaluated_splits: list[str],
    model_scope: str = "global",
) -> ClientEvaluationRecord:
    """One per-client row, from whichever pass this scope actually measured.

    The prefix is load-bearing. A personalized pass reports its splits and its
    counts under "personal_" (see _scope_split_names), so reading the
    unprefixed names under model_scope: personal found nothing and wrote a row
    whose three counts were 0 and whose six metric columns were empty -- for
    every client, every round, in a file the user had switched on deliberately.

    Under "both" the global pass is the one recorded, because that is what
    these columns have always held and what an existing reader expects; the
    personalized numbers reach round_metrics.csv through the personal_
    aggregates either way.
    """

    prefix = PERSONAL_SPLIT_PREFIX if model_scope == "personal" else ""

    # A client can now be due for one split and not another, so every split is
    # optional in the record and reports zero examples when it was not run.
    counts = {
        split: (
            _optional_evaluation_split_count(result, f"{prefix}{split}")
            if split in evaluated_splits
            else 0
        )
        for split in ("train", "val", "test")
    }

    def measured(split: str, metric: str) -> float | None:
        if not counts[split]:
            return None
        # accuracy is optional (_REQUIRED_CLIENT_METRIC_SPLITS): a task
        # without one leaves this key absent rather than reporting it as
        # zero, so an absent key is a second reason for None here, same as
        # a zero-count split -- both mean this column has nothing to say for
        # this client. A present-but-malformed value still raises: only the
        # metric's absence is the part that is now expected.
        key = f"{prefix}{split}_{metric}"
        if key not in result.metrics:
            return None
        return float(result.metrics[key])

    return ClientEvaluationRecord(
        round_id=result.round_id,
        client_id=result.client_id,
        participated=participated,
        train_num_examples=counts["train"],
        test_num_examples=counts["test"],
        val_num_examples=counts["val"],
        global_model_train_loss=measured("train", "loss"),
        global_model_train_accuracy=measured("train", "accuracy"),
        global_model_test_loss=measured("test", "loss"),
        global_model_test_accuracy=measured("test", "accuracy"),
        global_model_val_loss=measured("val", "loss"),
        global_model_val_accuracy=measured("val", "accuracy"),
        model_scope="personal" if prefix else "global",
    )


def _build_client_metric_record(result: FitResult) -> ClientMetricRecord:
    return ClientMetricRecord(
        round_id=result.round_id,
        client_id=result.client_id,
        phase="fit",
        num_examples=result.num_examples,
        metrics=dict(result.metrics),
    )


def _build_checkpoint_payload(
    server: ServerStrategy,
    client: ClientPool,
    server_payload: dict[str, Any],
    metrics: dict[str, float],
    round_id: int,
) -> dict[str, Any] | None:
    if "model_state" not in server_payload:
        return None

    server_state = server.save_state()
    checkpoint_state = {
        "round_id": round_id,
        "model_state": server_payload["model_state"],
        "metrics": metrics,
        "server_state": server_state,
        # The position in the process-wide RNG stream is part of the run's
        # state: nn.Dropout draws from it on every forward pass, so a run
        # resumed at round R with a freshly seeded stream is not the run that
        # reached round R uninterrupted. On FEMNIST a resume without it moved
        # fit_accuracy on CPU and on GPU, and stayed bit-identical with
        # dropout: 0.0, which isolates the cause.
        "rng_state": capture_rng_state(),
    }
    for field in ("model_state_scope", "model_state_metadata"):
        if field in server_payload:
            checkpoint_state[field] = server_payload[field]

    # save_state() returns a complete snapshot, and every strategy override
    # begins with super().save_state(), so it re-clones the same
    # self._model_state the top level already holds -- a second full copy of
    # the model, independently cloned and independently pickled into the same
    # file. On femnist_resnet18 that was 11.24 MB of a 22.5 MB checkpoint;
    # with save_last: true, which every shipped training config sets, it was
    # written every round of every run of every strategy. Nothing read both:
    # _restore_server_state takes one or the other and eval_medmcqa_choice
    # reads the top-level key. Stripping here rather than in save_state()
    # leaves that method's contract intact for its three other callers.
    for field in ("model_state", "model_state_scope", "model_state_metadata"):
        if field in checkpoint_state:
            server_state.pop(field, None)

    client_states = _collect_client_states(client)
    if client_states is not None:
        checkpoint_state["client_states"] = client_states
    return checkpoint_state


def _without_client_states(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the payload minus per-client state, for best.pt.

    best.pt answers "which model scored highest", so it needs the model and
    nothing else. latest.pt and round_*.pt answer "how do I continue", and
    find_latest_checkpoint only ever reads those two, so dropping client state
    from best.pt cannot break a resume.

    The distinction is worth 40 GB per SCAFFOLD checkpoint: SCAFFOLD holds one
    model-sized control variate per client, so on FEMNIST's 3597 writers the
    client states are ~3600x the model itself.
    """

    if "client_states" not in payload:
        return payload
    return {key: value for key, value in payload.items() if key != "client_states"}


def _update_checkpoints(
    checkpoint_payload: dict[str, Any] | None,
    metrics: dict[str, float],
    output_dir: str | Path | None,
    round_id: int,
    checkpoint_policy: Mapping[str, Any],
    checkpoint_tracker: dict[str, Any],
) -> StagedCheckpoints | None:
    """Stage the round's numbered, latest and best checkpoints; return them.

    Written in full here and made visible by _commit_checkpoints once the
    round's CSV rows and run.json are on disk (POST-F24). This used to report
    whether any file was written, and the per-client CSV flush keyed off that
    to decide whether to rewrite itself. The signal was always true --
    save_last writes latest.pt every round, independently of interval -- and
    the flush no longer needs it either way: it appends its new rows rather
    than rewriting the run.
    """

    if output_dir is None or checkpoint_payload is None:
        return None
    if not bool(checkpoint_policy.get("enabled", True)):
        return None

    from fedbrew.core.checkpointing import (
        save_best_checkpoint,
        save_checkpoint,
        save_latest_checkpoint,
        should_save_checkpoint,
    )

    staged = StagedCheckpoints()
    if should_save_checkpoint(round_id, checkpoint_policy):
        save_checkpoint(checkpoint_payload, output_dir, round_id, staged)

    if bool(checkpoint_policy.get("save_last", False)):
        latest_path = save_latest_checkpoint(checkpoint_payload, output_dir, staged)
        checkpoint_tracker["latest_checkpoint"] = latest_path

    if bool(checkpoint_policy.get("save_best", False)):
        best_metric = str(checkpoint_policy.get("best_metric", DEFAULT_SELECTION_METRIC))
        metric_value = _checkpoint_metric_value(metrics, best_metric)
        _warn_once_on_a_non_finite_selection_metric(
            checkpoint_tracker, best_metric, metric_value, round_id
        )
        if metric_value is not None and _is_better_checkpoint_metric(
            metric_value,
            checkpoint_tracker.get("best_metric_value"),
            str(checkpoint_policy.get("best_mode", "max")),
        ):
            best_path = save_best_checkpoint(
                _without_client_states(checkpoint_payload),
                output_dir,
                best_metric,
                metric_value,
                staged,
            )
            checkpoint_tracker["best_metric_value"] = metric_value
            checkpoint_tracker["best_round_id"] = round_id
            checkpoint_tracker["best_checkpoint"] = best_path
    return staged


def _commit_checkpoints(
    staged: StagedCheckpoints | None,
    output_dir: str | Path | None,
    checkpoint_policy: Mapping[str, Any],
) -> None:
    """Make the round's staged checkpoints visible, then prune to keep_last.

    Pruned after the commit, not before, so keep_last counts this round's
    numbered checkpoint among the ones it keeps.
    """

    if staged is None or output_dir is None:
        return
    staged.commit()

    from fedbrew.core.checkpointing import prune_old_checkpoints

    keep_last = checkpoint_policy.get("keep_last")
    if keep_last is not None:
        prune_old_checkpoints(output_dir, int(keep_last))


def _warn_once_on_a_non_finite_selection_metric(
    checkpoint_tracker: dict[str, Any],
    best_metric: str,
    metric_value: float | None,
    round_id: int,
) -> None:
    """Say the first time a round cannot be considered for `best.pt`.

    `_is_better_checkpoint_metric` refuses a non-finite candidate, which is the
    fix for the freeze it used to cause. Refusing silently would leave the
    other half of the same problem: `best.pt` stops tracking the run and
    nothing says why. Said once rather than every round -- a diverging run
    produces NaN in every round after the first, and 500 identical lines is a
    log nobody reads.
    """

    if metric_value is None or math.isfinite(metric_value):
        return
    if checkpoint_tracker.get("warned_non_finite_selection_metric"):
        return
    checkpoint_tracker["warned_non_finite_selection_metric"] = True
    print(
        f"Warning: round {round_id} reported {best_metric}={metric_value}, which "
        "cannot be compared, so this round is not a candidate for best.pt. Later "
        "rounds still are. Said once; see the divergence monitor for what a run "
        "that got here means.",
        flush=True,
    )


def _checkpoint_policy(checkpointing: Mapping[str, Any] | None) -> dict[str, Any]:
    from fedbrew.core.checkpointing import checkpoint_config_with_defaults

    return checkpoint_config_with_defaults(checkpointing)


def _load_existing_metric_history(
    state: ExperimentState,
    output_dir: str | Path,
    start_round: int,
) -> None:
    """Replay the rounds before the checkpoint out of the run's metric CSVs.

    Refused, not skipped, when a file cannot be read. The first flush after a
    resume rewrites the per-client files from this history, so continuing with
    an empty one deleted every earlier round's rows -- and because the files
    were loaded under one handler, a corrupt client_metrics.csv emptied an
    intact client_update_metrics.csv as well (POST-F10). A torn final row is
    not refused: the readers drop it with a warning, since an interrupted
    append leaves exactly that.
    """

    from fedbrew.core.artifacts import (
        load_client_metrics_csv,
        load_client_update_metrics_csv,
        load_round_metrics_csv,
    )

    try:
        rounds = load_round_metrics_csv(output_dir)
        evaluations = load_client_metrics_csv(output_dir)
        updates = load_client_update_metrics_csv(output_dir)
    except (OSError, ValueError, csv.Error) as error:
        raise RunRefused(
            f"cannot resume at round {start_round}: {str(error).rstrip('.')}. A resume "
            "rewrites the per-client metric files from the rounds it replays, so "
            "continuing would delete every earlier round's rows. Repair or remove that "
            "row, or move the file aside to resume without its history."
        ) from error
    state.metrics_history.extend(record for record in rounds if record.round_id < start_round)
    state.client_metrics_history.extend(
        record for record in evaluations if record.round_id < start_round
    )
    state.client_update_metrics_history.extend(
        record for record in updates if record.round_id < start_round
    )


def _initialize_checkpoint_tracker(
    output_dir: str | Path | None,
    checkpoint_policy: Mapping[str, Any],
    history: list[MetricRecord],
) -> dict[str, Any]:
    tracker: dict[str, Any] = {
        "latest_checkpoint": None,
        "best_checkpoint": None,
        "best_metric_value": None,
        "best_round_id": None,
    }
    if output_dir is None:
        return tracker

    from fedbrew.core.checkpointing import find_latest_checkpoint

    latest_path = find_latest_checkpoint(output_dir)
    if latest_path is not None:
        tracker["latest_checkpoint"] = latest_path

    best_path = Path(output_dir) / "checkpoints" / "best.pt"
    if best_path.is_file():
        tracker["best_checkpoint"] = best_path

    if bool(checkpoint_policy.get("save_best", False)):
        best_metric = str(checkpoint_policy.get("best_metric", DEFAULT_SELECTION_METRIC))
        best_mode = selection_mode_for_metric(best_metric)
        for record in history:
            metric_value = _checkpoint_metric_value(record.metrics, best_metric)
            if metric_value is not None and _is_better_checkpoint_metric(
                metric_value,
                tracker.get("best_metric_value"),
                best_mode,
            ):
                tracker["best_metric_value"] = metric_value
                tracker["best_round_id"] = record.round_id
                if best_path.is_file():
                    tracker["best_checkpoint"] = best_path
    return tracker


def _checkpoint_metric_value(
    metrics: Mapping[str, float],
    metric_name: str,
) -> float | None:
    value = metrics.get(metric_name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _is_better_checkpoint_metric(
    metric_value: float,
    best_metric_value: Any,
    mode: str,
) -> bool:
    """Whether this round's selection metric beats the best seen so far.

    A non-finite candidate is never better, including when nothing has been
    seen yet. Without that clause a NaN first observation was accepted -- the
    `best_metric_value is None` branch below takes any first value -- and then
    froze `best.pt` for the whole run: every later comparison is `x < nan` or
    `x > nan`, both False, in `min` mode and `max` mode alike. The run wrote
    `best.pt` once, at the round whose metric was NaN, kept it to the end, and
    reported `best_metric_value: nan` with `best_round_id: 1`. A NaN arriving
    after a real value was always rejected correctly, so the defect was the
    first observation alone.

    That first observation is not hypothetical. `_overflow_safe` answers NaN by
    design for a statistic that cannot fit -- see POST-F02 -- and
    `val_loss_avg` goes through it and is a legal `checkpointing.best_metric`.
    A run whose first evaluated round is already diverging therefore produces
    exactly this, and the divergence monitor's job is to say what the run
    means, not to keep `best.pt` moving.

    Infinities are refused with NaN rather than compared. `+inf` would be
    "better" than every later accuracy under `max`, and `-inf` better than
    every later loss under `min`, so accepting them freezes `best.pt` the same
    way while looking like a comparison that worked.
    """

    if not math.isfinite(metric_value):
        return False
    if best_metric_value is None:
        return True
    if isinstance(best_metric_value, bool) or not isinstance(best_metric_value, int | float):
        return True
    if mode == "min":
        return metric_value < float(best_metric_value)
    return metric_value > float(best_metric_value)


def _checkpointing_summary(
    output_dir: str | Path | None,
    checkpoint_policy: Mapping[str, Any],
    checkpoint_tracker: Mapping[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "enabled": bool(checkpoint_policy.get("enabled", True)),
        "latest_checkpoint": _relative_checkpoint_path(
            output_dir, checkpoint_tracker.get("latest_checkpoint")
        ),
        "best_checkpoint": _relative_checkpoint_path(
            output_dir, checkpoint_tracker.get("best_checkpoint")
        ),
        "best_metric": (
            checkpoint_policy.get("best_metric")
            if bool(checkpoint_policy.get("save_best", False))
            else None
        ),
        "best_metric_value": checkpoint_tracker.get("best_metric_value"),
        "best_round_id": checkpoint_tracker.get("best_round_id"),
        "kept_checkpoints": _kept_numbered_checkpoints(output_dir),
    }
    return summary


def _kept_numbered_checkpoints(output_dir: str | Path | None) -> list[str]:
    if output_dir is None:
        return []
    checkpoint_dir = Path(output_dir) / "checkpoints"
    if not checkpoint_dir.is_dir():
        return []
    return [
        _relative_checkpoint_path(output_dir, path) or str(path)
        for path in sorted(checkpoint_dir.glob("round_*.pt"))
    ]


def _relative_checkpoint_path(output_dir: str | Path | None, path: Any) -> str | None:
    if output_dir is None or path is None:
        return None
    checkpoint_path = Path(path)
    try:
        return str(checkpoint_path.relative_to(Path(output_dir)))
    except ValueError:
        return str(checkpoint_path)


def _collect_client_states(client: ClientPool) -> dict[str, Any] | None:
    get_state_snapshot = getattr(client, "get_state_snapshot", None)
    if callable(get_state_snapshot):
        return dict(get_state_snapshot())

    if not isinstance(client, Mapping):
        return None
    return {client_id: client_update.get_state() for client_id, client_update in client.items()}


def _build_client_infos(dataset: FederatedDataset) -> list[ClientInfo]:
    client_infos: list[ClientInfo] = []
    for client_id in dataset.list_clients():
        client_metadata = dataset.get_client_metadata(client_id)
        payload: dict[str, Any]
        if isinstance(client_metadata, Mapping):
            payload = dict(client_metadata)
        else:
            payload = {"metadata": client_metadata}

        client_infos.append(
            ClientInfo(
                client_id=client_id,
                num_examples=_infer_num_examples(payload),
                payload=payload,
            )
        )
    return client_infos


def _infer_num_examples(payload: dict[str, Any]) -> int:
    raw_num_examples = payload.get("num_examples")
    if isinstance(raw_num_examples, int):
        return raw_num_examples

    for key in ("y", "X", "x", "data"):
        value = payload.get(key)
        if isinstance(value, Sized):
            return len(value)
    return 0
