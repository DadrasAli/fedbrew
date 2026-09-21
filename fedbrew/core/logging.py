"""Clear, live terminal output helpers for benchmark runs.

What each line *says* is decided here; how it looks is decided in
fedbrew.core.console, which owns the palette, the markers and the TTY gate.
This module composes rows and tones and hands them over -- it holds no colour
of its own, and tests/test_console_is_the_only_renderer.py keeps it that way.

There is no longer a rich path and a parallel plain path per function. The one
composition runs against a Surface that has already decided whether its
destination is a terminal, so the two can no longer drift -- which they had,
in both directions: the plain footer named the detector and the rich one did
not spell the status the same way.
"""

from __future__ import annotations

import math
import re
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fedbrew.core.config import (
    CLIENT_METRIC_BASES,
    PERSONAL_SPLIT_PREFIX,
    ClientStatisticsConfig,
    FullConfig,
    client_metric_names,
    worst_percent_label,
)
from fedbrew.core.console import (
    AMBER,
    DIM,
    FAINT,
    GOLD,
    GROUP_ACCURACY,
    GROUP_ALGORITHM,
    GROUP_LOSS,
    GROUP_SPREAD,
    IVORY,
    PULSE,
    RED,
    SPLIT_CENTRAL,
    SPLIT_TEST,
    SPLIT_VAL,
    UNCLASSIFIED,
    MetricRow,
    Row,
    Surface,
    Verbosity,
    build_surface,
    measure,
)
from fedbrew.core.divergence import DivergenceVerdict
from fedbrew.core.metrics import (
    FIXED_METRIC_GLOSSES,
    METRIC_SUFFIX_GLOSSES,
    SPLIT_GLOSSES,
    metric_gloss,
    server_diagnostic_metrics,
    surviving_client_fit_extras,
)
from fedbrew.core.paths import resolve_output_dir
from fedbrew.core.state import MetricRecord

_CONTEXT_DEFINITIONS = (
    ("round", "Completed federated round / configured total."),
    ("clients", "Number of clients selected and trained in the round."),
    ("examples", "Sum of post-fit train examples reported by those clients."),
    ("time", "Wall-clock seconds for the round, end to end."),
    ("eta", "Projected time left, from the median of the last 10 rounds."),
)

_DATA_SPLIT_DEFINITIONS = (
    (
        "eval",
        "Validation data held out from each client's training partition, used "
        "for validation and model monitoring.",
    ),
    (
        "test",
        "Saved local client test data, used for client_test evaluation and never replaced by eval.",
    ),
    (
        "global_test",
        "Complete centralized/global test data, used for global_test evaluation.",
    ),
)

#: The client-test entries of the run legend, in display order. A curated
#: subset: the loop emits twelve columns per split and listing all of them
#: would drown the legend it is meant to explain.
#:
#: Derived rather than written out. As a fixed tuple this claimed
#: "test_accuracy_bottom10" -- a spelling the loop stopped emitting when the
#: suffix became worst{P} -- so the legend named a column no run produced and
#: omitted the one every run did. It also promised _std and _min
#: unconditionally, which is wrong for any config that turns those toggles off.
#: Asking client_metric_names makes both classes of mistake impossible: a name
#: survives only if the run will actually write it.
_CURATED_CLIENT_TEST_METRICS = (
    "test_loss_sample_weighted_avg",
    "test_loss_avg",
    "test_accuracy_sample_weighted_avg",
    "test_accuracy_avg",
    "test_accuracy_std",
    "test_accuracy_min",
)


def _client_test_metric_names(
    statistics: ClientStatisticsConfig | None = None,
) -> tuple[str, ...]:
    """Legend entries for the client test split under `statistics`."""

    statistics = ClientStatisticsConfig() if statistics is None else statistics
    curated = list(_CURATED_CLIENT_TEST_METRICS)
    if statistics.worst_percent:
        label = worst_percent_label(statistics.worst_percent)
        curated.append(f"test_accuracy_worst{label}")
    emitted = client_metric_names("test", statistics)
    return tuple(name for name in curated if name in emitted)


#: The validation entries, which are a shorter list than the test ones on
#: purpose. The test split is the reported result and gets its dispersion
#: alongside; val exists to select a checkpoint, so what a reader needs from it
#: is the two averages and -- added below -- whichever column selection
#: actually reads.
_CURATED_CLIENT_VAL_METRICS = (
    "val_loss_sample_weighted_avg",
    "val_loss_avg",
    "val_accuracy_sample_weighted_avg",
    "val_accuracy_avg",
)


def _client_val_metric_names(config: FullConfig | None) -> tuple[str, ...]:
    """Legend entries for the validation split, including what selects on it.

    Without this the validation split had no entries at all, which was
    survivable while the terminal reported every round and became a defect the
    moment it reported evaluation rounds only: `evaluation.val` defaults to
    `every: 5` against test's `every: 10`, so the most frequent evaluation
    round in a default run is a val round -- and a val round that printed only
    the fit metrics would have shown everything except the thing it measured.
    """

    statistics = ClientStatisticsConfig() if config is None else config.client_statistics
    curated = list(_CURATED_CLIENT_VAL_METRICS)
    selected = _checkpoint_selection_metric(config)
    # The column model selection reads is the reason this round happened. It is
    # usually one of the four above; when a config selects on val_accuracy_min
    # or a worst-percent column, that is the one to show.
    if selected is not None and selected.startswith("val_") and selected not in curated:
        curated.append(selected)
    emitted = client_metric_names("val", statistics)
    return tuple(name for name in curated if name in emitted)


def _checkpoint_selection_metric(config: FullConfig | None) -> str | None:
    if config is None:
        return None
    checkpointing = config.runtime.extra.get("checkpointing")
    if not isinstance(checkpointing, Mapping):
        return None
    metric = checkpointing.get("best_metric")
    return str(metric) if metric else None


_FIT_METRIC_DEFAULTS = {
    "local_sgd": ("fit_loss", "fit_accuracy"),
    "fedprox": ("fit_loss", "fit_accuracy", "fit_proximal_loss", "fit_total_loss"),
    "scaffold": (
        "fit_loss",
        "fit_accuracy",
        "control_delta_norm",
        "client_control_norm",
        "local_steps",
    ),
}


def print_plan_header(
    config: FullConfig,
    *,
    deterministic: bool,
    deterministic_warn_only: bool,
    client_count: int | None = None,
    resume_from: str | Path | None = None,
    surface: Surface | None = None,
) -> None:
    """Print what this run is about to do, before it does any of it.

    Four blocks -- data, federation, algorithm, metrics -- above an identity
    block naming the run itself. Every value is the *resolved* one: after the
    config's defaults, after the CLI overrides, and after the runtime resolved
    `device: auto` to something concrete. A header that echoed the YAML would
    be answering a question nobody asks -- the file is right there -- while
    hiding the two facts a reader actually needs, which are what the defaults
    filled in and what the flags changed.

    The two determinism flags are passed in rather than read off ``config``
    here, and have no defaults: this function prints what the run resolved,
    and a default of its own would be a second answer to "what did the run
    do" that is right only for as long as it happens to match. See
    ``runner.run``, which resolves them once for the run and for this.

    Amber appears in exactly five places, and nowhere else in this header: a
    matmul precision that changes the numbers, determinism downgraded to
    warn-only, an output directory that already holds files, a resumed run,
    and components loaded from outside the package. Each is something a
    reader would otherwise assume was not the case.
    """

    if _is_quiet(config):
        return
    surface = _surface_for(config) if surface is None else surface
    verbose = surface.verbosity is Verbosity.VERBOSE

    blocks: list[tuple[str | None, list[Row]]] = [
        (
            None,
            _identity_rows(config, resume_from, deterministic, deterministic_warn_only),
        ),
        ("data", _data_rows(config, client_count)),
        ("federation", _federation_rows(config, client_count)),
        ("algorithm", _algorithm_rows(config)),
        ("metrics", _metrics_rows(config, verbose=verbose)),
    ]
    # One label column across all four blocks, measured over every row in the
    # header at once. The column names in the metrics block are the widest
    # labels in it, so measuring per block would step the values in and out
    # four times down a single screen.
    width = measure([row for _, rows in blocks for row in rows])

    surface.rule("EXPERIMENT PLAN")
    for name, rows in blocks:
        if name is not None:
            surface.blank()
            surface.line(name, tone=IVORY)
        for row in rows:
            surface.row(row, width=width, wrap=True)
    surface.blank()


def _identity_rows(
    config: FullConfig,
    resume_from: str | Path | None,
    deterministic: bool,
    deterministic_warn_only: bool,
) -> list[Row]:
    experiment = config.experiment
    runtime_extra = config.runtime.extra
    rows = [Row("Experiment", experiment.name or "(unnamed)")]
    if experiment.run_id:
        rows.append(Row("Run", str(experiment.run_id)))
    rows.append(Row("Output", str(experiment.output_dir)))
    note = _output_dir_note(config, resume_from)
    if note is not None:
        # Its own row rather than a trailing note on the one above: an output
        # path is routinely 90 characters of scratch directory, and a warning
        # appended after one is a warning past the right-hand edge.
        rows.append(Row("", note, tone=AMBER))
    if experiment.extensions:
        # Amber: some of what this run is built from is not the package. A
        # reader comparing two runs would otherwise take the strategy, task
        # and model names as naming the shipped components.
        rows.append(Row("Extensions", ", ".join(experiment.extensions), tone=AMBER))
    rows.append(Row("Device", _resolved_device(config.runtime.device)))
    rows.append(Row("Seed", str(experiment.seed)))

    resume = resume_from or runtime_extra.get("resume_from")
    if resume:
        # Amber: the model this run reports did not start from the seed above,
        # and every number in the output is a continuation of an earlier
        # attempt rather than a fresh draw.
        rows.append(Row("Resumed from", str(resume), tone=AMBER))
    elif runtime_extra.get("resume_latest"):
        rows.append(Row("Resumed from", "latest checkpoint in the output directory", tone=AMBER))

    if deterministic:
        rows.append(
            Row(
                "Determinism",
                "on, warn_only" if deterministic_warn_only else "on",
                # warn_only lets a non-deterministic kernel run anyway, so the
                # run is reproducible only where torch happened to have a
                # deterministic implementation. "on" would be a false claim.
                tone=AMBER if deterministic_warn_only else GOLD,
            )
        )
    precision = _matmul_precision(config)
    if precision is not None:
        # The one performance key that changes the numbers -- but only when it
        # is not "highest", which is what torch does anyway. Amber on every
        # shipped config (they all set it explicitly, by policy) would spend
        # the colour on the case where nothing changed and leave nothing to
        # say it with when something did.
        changes_numerics = precision != "highest"
        rows.append(Row("Matmul precision", precision, tone=AMBER if changes_numerics else GOLD))
    return rows


def _data_rows(config: FullConfig, client_count: int | None) -> list[Row]:
    data = config.data
    rows = [Row("Dataset", data.name or "(unset)")]
    if data.path:
        rows.append(Row("Manifest", str(data.path)))
    resolved_clients = client_count if client_count is not None else data.num_clients
    if resolved_clients is not None:
        rows.append(Row("Clients", str(resolved_clients)))
    for label, value in (
        ("Samples per client", data.samples_per_client),
        ("Input dimension", data.input_dim),
        ("Classes", data.num_classes),
    ):
        if value is not None:
            rows.append(Row(label, str(value)))
    return rows


def _federation_rows(config: FullConfig, client_count: int | None) -> list[Row]:
    server = config.server
    rows = [
        Row("Strategy", server.strategy),
        Row("Rounds", str(server.global_rounds)),
        Row("Participation probability", f"{server.participation_probability:g}")
        if server.participation_probability is not None
        else Row("Participation rate", f"{server.participation_rate:g}"),
    ]
    per_round = _clients_per_round(config, client_count)
    if per_round is not None:
        rows.append(Row("Clients per round", per_round))
    weighting = server.extra.get("aggregation_weighting")
    if weighting is not None:
        rows.append(Row("Aggregation weighting", str(weighting)))
    return rows


def _algorithm_rows(config: FullConfig) -> list[Row]:
    client = config.client
    rows = [
        Row("Update rule", client.update_rule),
        Row("Task", config.task.name),
        Row("Model", config.model.name),
        Row("Local iterations", str(client.local_iterations)),
        Row("Batch size", str(client.batch_size)),
    ]
    if client.learning_rate is not None:
        rows.append(Row("Learning rate", f"{client.learning_rate:g}"))
    else:
        # delta_sgd derives its own step size and rejects an explicit one, so
        # "unset" here is the algorithm working, not a gap.
        rows.append(Row("Learning rate", "derived per client by the update rule", tone=DIM))
    rows.extend(
        Row(_extra_label(key), _format_extra(value))
        for key, value in sorted(client.extra.items())
        if key in _REPORTED_CLIENT_EXTRAS and value is not None
    )
    return rows


def _metrics_rows(config: FullConfig, *, verbose: bool) -> list[Row]:
    rows: list[Row] = []
    for split in ("train", "val", "test"):
        split_config = getattr(config.evaluation, split)
        schedule = _schedule_text(split_config.every, f"evaluation.{split}")
        if schedule is None:
            continue
        rows.append(Row(f"evaluation.{split}", f"{schedule}, {split_config.clients} clients"))
    central = _schedule_text(config.evaluation.central_test.every, "evaluation.central_test")
    if central is not None:
        rows.append(Row("evaluation.central_test", central))
    rows.append(Row("evaluation.model_scope", config.evaluation.model_scope))

    checkpointing = config.runtime.extra.get("checkpointing")
    if isinstance(checkpointing, Mapping) and checkpointing.get("best_metric"):
        rows.append(Row("checkpoint selects on", str(checkpointing["best_metric"])))
    rows.append(Row("divergence watches", config.divergence.metric))

    planned = _planned_metric_names(config)
    names = planned if verbose else _progress_metric_names(config)
    rows.append(
        Row(
            "Columns",
            f"{len(planned)}, all listed"
            if verbose
            else f"{len(names)} of {len(planned)} listed; --verbose lists them all",
        )
    )
    asked_and_dropped = _client_metrics_the_server_filter_removes(config)
    if asked_and_dropped:
        # Where a config's request and the run's output disagree without either
        # list being wrong. client.metrics naming a metric reads as asking for
        # that column; the client does emit it, and then the server filters the
        # whole aggregated dict against server.metrics, which does not list it.
        # Nothing raises, and the column is simply absent from a CSV whose
        # config appears to have asked for it.
        rows.append(
            Row(
                "not written",
                f"{', '.join(asked_and_dropped)} -- named in client.metrics "
                "and emitted, then removed by server.metrics. Add them there "
                "to keep them.",
                tone=AMBER,
            )
        )
    rows.extend(Row(name, _metric_definition(name, config), tone=None) for name in names)
    return rows


def _client_metrics_the_server_filter_removes(config: FullConfig) -> list[str]:
    """What this config asks for under client.metrics and round_metrics.csv will not carry.

    Every rule, not only the ones with exempt extras. The server filters the
    whole aggregated dict against a non-empty server.metrics, so any name that
    client.metrics lists and server.metrics does not is emitted and then
    dropped. This used to consult only the extras the FedAvg family exempts
    from its own filter, so the four rules that filter everything -- fedprox,
    scaffold, delta_sgd, fedlalr -- printed no row while their shipped configs
    asked for communicated_bytes and got no column.

    Defined as the complement of the planned column list rather than as a
    second rule, so this row and the column list above it cannot disagree.
    """

    planned = set(_planned_metric_names(config))
    return sorted(set(config.client.metrics or ()) - planned)


#: Client knobs worth a line in the header: each one changes what the local
#: update does, and none is visible in any other output. The rest of
#: `client.extra` is plumbing.
_REPORTED_CLIENT_EXTRAS = {
    "momentum",
    "weight_decay",
    "max_grad_norm",
    "proximal_mu",
    "update_mode",
    "learning_rate_schedule",
    "max_local_steps",
    "aggregation_weighting",
}


def _extra_label(key: str) -> str:
    return key.replace("_", " ").capitalize()


def _format_extra(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, Mapping):
        return ", ".join(f"{name}={_format_extra(item)}" for name, item in sorted(value.items()))
    return str(value)


def _schedule_text(every: object, context: str) -> str | None:
    """ "every 10 rounds", "the final round only", or None when never run."""

    from fedbrew.core.config import parse_evaluation_schedule

    try:
        interval = parse_evaluation_schedule(every, context)
    except ValueError:
        return None
    if interval is None:
        return None
    if interval == 0:
        return "the final round only"
    if interval == 1:
        return "every round"
    return f"every {interval} rounds"


def _clients_per_round(config: FullConfig, client_count: int | None) -> str | None:
    """How many clients a round will actually fit, when the roster is known.

    The same rule the server samples with (`FedAvgServer.sample_clients`):
    a rate at or above 1.0 takes every client, and anything below rounds up,
    never to fewer than one. Recomputed rather than reported, because a header
    printed before the first round has nothing measured to report yet.

    Under participation_probability the count is a draw, so this states its
    mean and, unless negligible, how often a round selects no client -- a
    round the loop does not aggregate, which should not first be seen mid-run.
    """

    total = client_count if client_count is not None else config.data.num_clients
    if total is None or total <= 0:
        return None
    probability = config.server.participation_probability
    if probability is not None:
        if probability >= 1.0:
            return str(total)
        text = f"{total * probability:.3g} on average, varying by round"
        empty = (1.0 - probability) ** total
        if empty >= 0.0005:
            text += f"; none in {empty:.1%} of rounds, which leave the model unchanged"
        return text
    rate = config.server.participation_rate
    if rate >= 1.0:
        return str(total)
    return str(max(1, math.ceil(total * rate)))


def _resolved_device(device: str) -> str:
    """The device this run will use, not the one the config asked for.

    `run()` has already replaced "auto" with what torch resolved by the time
    the header prints, so this only fires for `--validate-only`, where nothing
    has configured the runtime yet. Worth resolving there too: "auto" on a
    node that hands out no CUDA is the single most expensive thing to discover
    after a job has been queued rather than before.
    """

    if device != "auto":
        return device
    try:
        import torch

        return f"auto → {'cuda' if torch.cuda.is_available() else 'cpu'}"
    except Exception:  # pragma: no cover - a header must never break a run.
        return device


def _matmul_precision(config: FullConfig) -> str | None:
    performance = config.runtime.extra.get("performance")
    if not isinstance(performance, Mapping):
        return None
    precision = performance.get("matmul_precision")
    return None if precision is None else str(precision)


def _output_dir_note(config: FullConfig, resume_from: str | Path | None) -> str | None:
    """ "already holds N files", or None.

    Preflight raises this as a warning, but preflight is a separate command:
    a real run has never said it, and writing a second experiment's artifacts
    on top of a first one's is exactly the mistake that is invisible until the
    CSVs are read months later.

    Silent while resuming. A resumed run's output directory holds the earlier
    attempt by definition, and the "Resumed from" row above already says so;
    two ambers for one fact is how amber stops meaning anything.
    """

    if resume_from or config.runtime.extra.get("resume_from"):
        return None
    if config.runtime.extra.get("resume_latest"):
        return None
    try:
        directory = resolve_output_dir(config.experiment.output_dir)
        if not directory.is_dir():
            return None
        count = sum(1 for _ in directory.iterdir())
    except OSError:
        return None
    if not count:
        return None
    return f"already holds {count} file{'s' if count != 1 else ''}"


def _planned_metric_names(config: FullConfig) -> list[str]:
    """Every column round_metrics.csv will carry, as far as it is knowable.

    Assembled from the sources that already answer this for preflight rather
    than from a second list: `client_metric_names` for the evaluation
    aggregates, and the two dictionaries in fedbrew.core.metrics naming what a
    `client.metrics`/`server.metrics` list is unable to filter out. A column
    listed here that the run does not write, or written and not listed, would
    make the header worse than no header.
    """

    names = list(_fit_metric_names(config))
    names.extend(surviving_client_fit_extras(config.client.update_rule, config.server.metrics))
    for split in ("train", "val", "test"):
        if not _split_is_evaluated(config, split):
            continue
        for metric_split in _model_scope_splits(config.evaluation.model_scope, split):
            names.extend(sorted(client_metric_names(metric_split, config.client_statistics)))
    if _central_is_evaluated(config):
        names.extend(("central_test_loss", "central_test_accuracy"))
    names.extend(
        sorted(server_diagnostic_metrics(config.server.strategy, config.client.metrics or ()))
    )
    return _ordered_metric_names(_deduplicate(names))


def _model_scope_splits(model_scope: str, split: str) -> list[str]:
    """The split names one evaluation pass reports under.

    Mirrors loop._scope_split_names: a personalized pass reports the same
    splits under a "personal_" prefix, which is what lets it flow through the
    aggregation unchanged -- and what puts a second set of columns in the CSV.
    """

    names = []
    if model_scope in {"global", "both"}:
        names.append(split)
    if model_scope in {"personal", "both"}:
        names.append(f"{PERSONAL_SPLIT_PREFIX}{split}")
    return names


@dataclass
class RoundProgress:
    """Where a run is, shared by the settled block and the live footer.

    Both halves of the run surface need the same three facts -- how many
    rounds there are, how many clients the dataset holds, and how long the
    completed rounds took -- and only the loop is in a position to know them.
    Computing the estimate twice is how the number under the bar and the
    number in the block come to disagree by the end of a long run.
    """

    total_rounds: int
    #: Every client the dataset has, or None when asking would have cost.
    roster: int | None = None
    #: One entry per completed round, including the rounds that printed
    #: nothing: a round that reports no metrics still took time, and dropping
    #: it would make the estimate describe only the expensive rounds.
    durations: list[float] = field(default_factory=list)
    #: Clients the fit phase actually selected, from the phase itself rather
    #: than recomputed from the config -- what happened, not what should have.
    sampled: int | None = None

    def record(self, duration: float | None) -> None:
        if duration is not None:
            self.durations.append(float(duration))

    @property
    def median_seconds(self) -> float | None:
        """Median of the last few rounds, not the mean and not all of them.

        Round one carries one-off setup -- CUDA context, shard cache fill,
        lazily built clients -- and checkpoint-writing rounds spike. A mean
        would carry both for the rest of the job.
        """

        if not self.durations:
            return None
        return statistics.median(self.durations[-10:])

    def eta_seconds(self, round_id: int) -> float | None:
        typical = self.median_seconds
        remaining = self.total_rounds - round_id
        if typical is None or remaining <= 0:
            return None
        return typical * remaining

    def fraction(self, round_id: int) -> float:
        if self.total_rounds <= 0:
            return 0.0
        return round_id / self.total_rounds

    def clients_text(self, sampled: int | None = None) -> str:
        """`10/1000 clients` -- this round's selection against the roster."""

        selected = self.sampled if sampled is None else sampled
        if selected is None:
            return "" if self.roster is None else f"{self.roster} clients"
        if self.roster is None:
            return f"{selected} clients"
        return f"{selected}/{self.roster} clients"


def print_round_metrics(
    round_id: int,
    metrics: dict[str, float],
    num_clients: int,
    config: FullConfig | None = None,
    *,
    num_examples: int | None = None,
    duration_sec: float | None = None,
    eta_sec: float | None = None,
    progress: RoundProgress | None = None,
) -> None:
    """Print one completed round: a bar, then its numbers grouped by kind."""

    if _is_quiet(config):
        return
    surface = _surface_for(config)
    _print_round_header(
        surface,
        round_id,
        _total_rounds(config),
        num_clients,
        progress,
        num_examples=num_examples,
        duration_sec=duration_sec,
        eta_sec=eta_sec,
    )
    surface.blank()

    shown = _round_display_metrics(metrics, config, verbose=surface.verbosity is Verbosity.VERBOSE)
    if shown:
        _print_metric_groups(surface, shown, config)
        hidden = len(metrics) - len(shown)
        if hidden > 0:
            surface.line(f"{hidden} more columns written; --verbose shows them", tone=FAINT)
            surface.blank()
    else:
        surface.line("no metrics this round", tone=FAINT)
        surface.blank()


def _print_round_header(
    surface: Surface,
    round_id: int,
    total_rounds: int | None,
    num_clients: int,
    progress: RoundProgress | None,
    *,
    num_examples: int | None,
    duration_sec: float | None,
    eta_sec: float | None,
) -> None:
    """Open a round block.

    On a terminal the bar is frozen where this round landed, so scrolling back
    through a long run shows how far in each block was written.

    Off a terminal there is no bar -- without colour it is a row of dashes
    carrying nothing a reader can use -- and no live footer either, so the
    facts the footer would have carried are appended here instead. A sweep log
    that stopped recording how long a round took would be a regression.
    """

    progress_text = f"round {round_id}/{total_rounds}" if total_rounds else f"round {round_id}"
    clients = (
        progress.clients_text(num_clients) if progress is not None else f"{num_clients} clients"
    )
    tail = "   ".join(part for part in (progress_text, clients) if part)

    if not surface.is_tty:
        extras = (
            f"{num_examples} examples" if num_examples is not None else "",
            _format_duration(duration_sec) if duration_sec is not None else "",
            f"ETA {_format_duration(eta_sec)}" if eta_sec is not None else "",
        )
        surface.line("   ".join(part for part in (tail, *extras) if part))
        return

    tracker = progress if progress is not None else RoundProgress(total_rounds or 0)
    # Same verbosity the footer was sized under, or the settled bar and the
    # live one below it would be different lengths on the same screen.
    verbose = surface.verbosity is Verbosity.VERBOSE
    surface.spans(
        [
            (_BLOCK_GUTTER, None),
            *surface.bar(
                tracker.fraction(round_id),
                _run_bar_width(surface, tracker, verbose=verbose),
            ),
            ("   ", None),
            # Where the footer's pulse goes. A settled header is the live
            # footer with the glyph removed and the clock dropped, so its text
            # starts in the same column -- which is what lets a reader compare
            # the frozen bar above with the moving one below without counting.
            (" " * (len(PULSE[0]) + 1), None),
            (tail, DIM),
        ]
    )


#: Two spaces before the bar on every line the run surface draws, so the bar
#: in a header and the bar in the footer start at the same column.
_BLOCK_GUTTER = "  "


def _run_bar_width(surface: Surface, progress: RoundProgress, *, verbose: bool = False) -> int:
    """One bar width for the whole run surface, headers and footer alike.

    Measured from the widest the tail can ever become rather than from the
    tail in hand, so the bar is the same length in every block of a run and in
    the footer between them. Sized from the current text instead, it would be
    shorter at `round 1/500` than at `round 500/500` and shorter again once an
    estimate appeared -- and a progress bar that changes length is one nobody
    can read the fill of.

    Floors at zero rather than at a legible minimum: on a terminal too narrow
    for both, the bar is the half that can be dropped. Holding a minimum would
    push the line past the right edge, and the footer is redrawn in place --
    a footer that wrapped would leave its own first line behind on every
    redraw.
    """

    tail = _footer_tail_width(progress, verbose=verbose)
    return max(0, surface.width - len(_BLOCK_GUTTER) - 3 - len(PULSE[0]) - 1 - tail - 1)


#: Splits in the order a group prints them: the training data first, then the
#: split model selection may look at, then the reported one, then the server's
#: own. Anything with no split sorts last.
_SPLIT_DISPLAY_ORDER = ("train", "validation", "test", "central")


def _print_metric_groups(
    surface: Surface,
    shown: list[tuple[str, float]],
    config: FullConfig | None,
) -> None:
    """The round's numbers, grouped by kind and coloured by split.

    A list rather than a table, so --verbose grows it in place instead of
    reshaping it: the same headings in the same order, with more rows under
    each.
    """

    selected = _checkpoint_selection_metric(config)
    classified = [
        (index, name, value, classify_metric(name)) for index, (name, value) in enumerate(shown)
    ]
    for key, heading, tone in _METRIC_GROUPS:
        members = [entry for entry in classified if entry[3][0] == key]
        if not members:
            continue
        surface.heading(heading, tone=tone)
        surface.metric_rows(
            [
                MetricRow(
                    split=label,
                    qualifier=qualifier,
                    value=_format_metric(name, value, key),
                    tone=split_tone,
                    selected=name == selected,
                )
                for _, name, value, (_, label, split_tone, qualifier) in sorted(
                    members, key=_group_sort_key
                )
            ]
        )
        surface.blank()


def _group_sort_key(
    entry: tuple[int, str, float, tuple[str, str, str, str]],
) -> tuple[int, int, int]:
    """Within a group: by split, then the plainest form of it, then as given.

    The plainest form first means `fit_loss` opens the loss group's train rows
    and the aggregates follow it, rather than the em-dash row landing wherever
    the preferred-order list happened to leave it.
    """

    index, _, _, (_, label, _, qualifier) = entry
    try:
        split_rank = _SPLIT_DISPLAY_ORDER.index(label)
    except ValueError:
        split_rank = len(_SPLIT_DISPLAY_ORDER)
    return split_rank, 0 if qualifier == "—" else 1, index


def _round_display_metrics(
    metrics: dict[str, float],
    config: FullConfig | None,
    *,
    verbose: bool,
) -> list[tuple[str, float]]:
    """Which of a round's columns to print, in display order.

    A three-split run writes 40+ columns per round. Printed in full that is 40
    lines every evaluation round, which at 500 rounds is 20,000 lines of log
    that nobody reads and that buries the four numbers anybody watches. The
    default is the same curated set the plan header lists -- so a reader meets
    each name with its gloss once, then sees those names again every round --
    plus the fit metrics, which are what the divergence monitor watches and so
    the numbers that decide whether the run continues.

    Every column still reaches round_metrics.csv. This governs the terminal
    only, and --verbose prints all of them.
    """

    ordered = _ordered_metrics(metrics)
    if verbose or config is None:
        return ordered
    curated = set(_progress_metric_names(config)) | {"fit_loss", "fit_accuracy"}
    return [(name, value) for name, value in ordered if name in curated]


def print_round_table_header(config: FullConfig | None = None) -> None:
    """Close the plan and open the round stream.

    The legend of metric definitions this used to print now lives in the plan
    header's metrics block, beside the columns it defines. It was printed
    twice-removed from them before -- a list of names above the header, a
    second list of names in every round -- and a reader matching one to the
    other was doing the join by hand.
    """

    if _is_quiet(config):
        return
    _surface_for(config).rule("TRAINING")


#: What the loop calls each phase, and what a reader should see instead.
#: "client_eval" is the loop's own name for the per-client evaluation pass; on
#: screen it sits beside "fit" and wants to be the same length as a word.
_PHASE_LABELS = {"fit": "fit", "client_eval": "eval"}

#: Redraws per second while a phase is running. The natural upper bound is one
#: callback per client, and `client_scope: all` on FEMNIST is 3,500 of them per
#: evaluation pass -- a write per client costs more than it tells anyone.
_PROGRESS_REDRAWS_PER_SECOND = 10.0

#: Space reserved for the rate and the remaining time, so the tail is the same
#: width on every redraw and the bar in front of it does not step left and
#: right as "9m 58s left" becomes "2h 29m left". Wide enough for an LLM config
#: whose rounds are minutes rather than fractions of a second.
_RATE_BUDGET = len("1234.56s/round")
_ETA_BUDGET = len("99h 59m left")


def client_progress_reporter(
    config: FullConfig | None = None,
    progress: RoundProgress | None = None,
) -> Callable[[int, int, int, str], None] | None:
    """Build the on_client_progress callback, or None if nothing would see it.

    The fit and evaluation phases are the only places in a round that visit a
    known, bounded sequence of clients one at a time, and until this they were
    silent: `print_round_metrics` reports once the whole round is over, so a
    round whose fit phase runs for twenty minutes had nothing to say for
    twenty minutes.

    One line, redrawn in place. Not a rail: a rail is for a bounded list whose
    entries are each worth keeping on screen, and 500 rounds of five clients is
    neither bounded on screen nor worth 2,500 settled lines.

    The client count does not animate by default. Participation is a property
    of the round -- ten of a thousand, decided when the round was selected --
    not a thing in motion, and a number that counts up to a total it reaches
    every round says less each time it is read. Liveness comes from the pulse
    and the clock instead, which keep moving on a config where a round takes
    minutes and the bar advances once. --verbose adds the within-phase count
    back, for `client_scope: all` on FEMNIST: 3,500 clients is a pass long
    enough that a reader wants to know where in it they are.

    Returns None -- so the loop skips the callback entirely rather than calling
    into a no-op -- when the destination is not a terminal. A redraw is the one
    thing a redirected stream must not receive, and the round block that
    follows carries the same facts settled: which clients, how many, how long.
    """

    if _is_quiet(config):
        return None
    # Built once and captured: this fires per client per round, and a rich
    # Console per call would be the most expensive thing in a cheap round.
    surface = _surface_for(config)
    if not surface.is_tty:
        return None

    verbose = surface.verbosity is Verbosity.VERBOSE
    tracker = progress if progress is not None else RoundProgress(_total_rounds(config) or 0)
    tail_width = _footer_tail_width(tracker, verbose=verbose)
    bar_width = _run_bar_width(surface, tracker, verbose=verbose)
    state: dict[str, object] = {"drawn": 0.0, "pulse": 0}

    def report(round_id: int, done: int, total: int, phase: str) -> None:
        # The selection is the fit phase's size, which the loop announces with
        # done=0 before the first client trains. Taken from the phase rather
        # than recomputed from the config so the footer states the clients this
        # round actually had, none included.
        if phase == "fit":
            tracker.sampled = total
        now = time.perf_counter()
        final = done >= total
        if not final and now - state["drawn"] < 1.0 / _PROGRESS_REDRAWS_PER_SECOND:
            return
        state["drawn"] = now
        state["pulse"] = int(state["pulse"]) + 1
        # The round comes from the loop rather than from counting completed
        # rounds here: on_round_end fires after a round, so a resumed run --
        # which starts at the checkpoint's round, not at 1 -- would have its
        # first round counted wrong, and a progress line is read precisely
        # when nothing else on screen says where the run is.
        tail = _footer_tail(tracker, round_id, done, total, phase, verbose=verbose)
        surface.redraw_spans(
            [
                (_BLOCK_GUTTER, None),
                *surface.bar(tracker.fraction(round_id), bar_width),
                ("   ", None),
                (PULSE[int(state["pulse"]) % len(PULSE)], GOLD),
                (" ", None),
                (tail.ljust(tail_width), FAINT),
            ]
        )

    return report


def _footer_tail(
    progress: RoundProgress,
    round_id: int,
    done: int,
    total: int,
    phase: str,
    *,
    verbose: bool,
) -> str:
    """The footer's right-hand side: where, over how many, how fast, how long."""

    parts = [
        f"round {round_id}/{progress.total_rounds}"
        if progress.total_rounds
        else f"round {round_id}",
        progress.clients_text(),
    ]
    if verbose:
        parts.append(f"{_PHASE_LABELS.get(phase, phase)} {done}/{total}")
    rate = progress.median_seconds
    if rate is not None:
        parts.append(f"{rate:.2f}s/round")
    eta = progress.eta_seconds(round_id)
    if eta is not None:
        parts.append(f"{_format_duration(eta)} left")
    return "   ".join(part for part in parts if part)


def _footer_tail_width(progress: RoundProgress, *, verbose: bool) -> int:
    """The tail's fixed width, computed once from the widest it can become.

    Padding to whatever the current tail happens to need would move the bar
    every time the estimate crossed a unit boundary -- the one element on the
    line that is supposed to be still.
    """

    rounds = f"round {progress.total_rounds}/{progress.total_rounds}"
    clients = progress.clients_text(progress.roster)
    width = len(rounds) + 3 + len(clients) + 3 + _RATE_BUDGET + 3 + _ETA_BUDGET
    if verbose:
        # "eval 3500/3500" at its widest, plus its separator.
        width += 3 + len("eval ") + 2 * len(str(progress.roster or 0)) + 1
    return width


def print_termination_notice(verdict: DivergenceVerdict, config: FullConfig | None = None) -> None:
    """Announce a divergence/stall stop the moment it is decided.

    Without this, a run that stops early gives no live signal at all: the
    round loop's per-round body has no other print (confirmed by grep --
    print_round_metrics is the only thing that fires there, once per
    completed round), and `verdict.reason` otherwise first reaches the
    terminal in `print_experiment_end`'s footer, after the process has
    already finished. A packed sweep watching stdout could not tell "still
    training" from "quietly diverged 40 rounds ago" until the job exited.
    Uses the same quiet/plain/rich fallback as every other function here --
    this is the same reporting path as print_round_metrics, not a second one.
    """

    if _is_quiet(config):
        return
    surface = _surface_for(config)
    # A diverged run is an error; a stalled one is a warning. The palette
    # already says which is which, so the two no longer need different words.
    tone = RED if verdict.status == "diverged" else AMBER
    surface.rule(f"STOPPING EARLY: {verdict.status.upper()}", tone=tone)
    surface.line(verdict.reason, tone=tone, wrap=True)
    surface.spans(
        [
            ("Detector: ", DIM),
            (f"{verdict.detector} on {verdict.metric} (round {verdict.round_id})", GOLD),
        ]
    )
    surface.blank()


def print_experiment_end(
    history: list[MetricRecord],
    output_dir: str | Path,
    config: FullConfig | None = None,
    termination: Mapping[str, Any] | None = None,
) -> None:
    """Print a styled experiment footer and flush it immediately.

    A run stopped by the divergence monitor exits successfully, so the footer
    is the only place that says so on screen. It is coloured differently and
    names the detector, because "finished" and "stopped at round 12 because the
    loss went to NaN" must not read the same in a log.
    """

    output_path = Path(output_dir)
    surface = _surface_for(config)
    status = str((termination or {}).get("status") or "completed")
    if _is_quiet(config):
        # Quiet is "tell me how it went", not "tell me nothing". One line, and
        # it has to carry the outcome: a sweep grepping its logs must be able
        # to tell a finished run from one that diverged at round 12, and the
        # exit code cannot -- a divergence stop exits zero on purpose.
        surface.final(_final_line(status, history, output_path, termination))
        return
    rows: list[Row] = []
    if status == "completed":
        surface.rule("EXPERIMENT COMPLETE")
        surface.line("Finished successfully.", tone=IVORY)
    else:
        tone = RED if status == "diverged" else AMBER
        surface.rule(f"EXPERIMENT STOPPED EARLY: {status.upper()}", tone=tone)
        surface.line(str((termination or {}).get("reason", "")), tone=tone, wrap=True)
        rows.append(
            Row(
                "Detector",
                f"{(termination or {}).get('detector')} on {(termination or {}).get('metric')}",
            )
        )
    rows.append(Row("Rounds completed", str(len(history))))
    wall_clock = _wall_clock_summary(history)
    if wall_clock is not None:
        rows.append(Row("Wall clock", wall_clock))
    rows.append(Row("Metrics", str(output_path / "round_metrics.csv")))
    rows.append(Row("Artifacts", str(output_path)))
    surface.rows(rows)


def print_download_progress(
    name: str,
    done: int,
    total: int | None,
    elapsed: float,
    *,
    finished: bool,
) -> None:
    """Render one redirected third-party download as a single styled line.

    An interactive terminal gets a single line, redrawn in place as bytes
    arrive -- one line for the thing being fetched, no nesting, no per-shard
    sub-bars. Anything else -- a redirected/piped stream, a SLURM log file --
    gets one line, printed once the download finishes: a carriage-return-
    redrawn line in a log file is noise, one line per byte written rather
    than one per file.

    Deliberately independent of print_experiment_start/print_round_metrics'
    --quiet/--no-rich config gating: generate/prepare-llm/prepare-oasst1 have
    no FullConfig to read those from. Whether the destination is an actual
    terminal is the only signal available here, and it is also the right one
    -- a log file should get what the library's own bars would have logged
    there too: one summary line, not a redraw.
    """

    surface = build_surface()
    line = _format_download_line(name, done, total, elapsed)
    if finished:
        surface.line(line, tone=GOLD)
        return
    surface.redraw(line, tone=DIM)


def _format_download_line(name: str, done: int, total: int | None, elapsed: float) -> str:
    size = _format_bytes(done) if total is None else f"{_format_bytes(done)}/{_format_bytes(total)}"
    percent = f"{min(100.0, 100 * done / total):3.0f}%" if total else "  ?%"
    rate = f"{_format_bytes(done / elapsed)}/s" if elapsed >= 0.05 else "--B/s"
    return f"{name}: {percent} {size} {rate}"


def _format_bytes(num_bytes: float) -> str:
    """Human-readable size, matching the 1000-based units these libraries'
    own tqdm bars already use (`unit_scale=True` defaults to a 1000 divisor,
    not 1024)."""

    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1000
    return f"{value:.1f}TB"


def _progress_definitions(
    config: FullConfig | None,
) -> list[tuple[str, str]]:
    definitions = list(_CONTEXT_DEFINITIONS)
    definitions.extend(_DATA_SPLIT_DEFINITIONS)
    definitions.extend(
        (name, _metric_definition(name, config)) for name in _progress_metric_names(config)
    )
    return definitions


def _split_is_evaluated(config: FullConfig | None, split: str) -> bool:
    """Whether this run will evaluate `split` at all.

    Recomputed from `evaluation.<split>.every` through the same parser the
    loop schedules with, rather than inferred from which columns a round
    happened to produce: reading intent back out of output is how a config
    that silently stopped evaluating a split goes unnoticed.
    """

    from fedbrew.core.config import parse_evaluation_schedule

    if config is None:
        return True
    try:
        return (
            parse_evaluation_schedule(
                getattr(config.evaluation, split).every, f"evaluation.{split}"
            )
            is not None
        )
    except ValueError:
        return False


def _central_is_evaluated(config: FullConfig | None) -> bool:
    from fedbrew.core.config import parse_evaluation_schedule

    if config is None:
        return True
    try:
        return (
            parse_evaluation_schedule(
                config.evaluation.central_test.every, "evaluation.central_test"
            )
            is not None
        )
    except ValueError:
        return False


def _progress_metric_names(config: FullConfig | None) -> list[str]:
    def evaluated(split: str) -> bool:
        return _split_is_evaluated(config, split)

    def central_evaluated() -> bool:
        return _central_is_evaluated(config)

    names: list[str] = []
    if evaluated("train"):
        names.extend(("train_loss_sample_weighted_avg", "train_accuracy_sample_weighted_avg"))
    if central_evaluated():
        names.extend(("central_test_loss", "central_test_accuracy"))
    if evaluated("val"):
        names.extend(_client_val_metric_names(config))
    if evaluated("test"):
        names.extend(
            _client_test_metric_names(None if config is None else config.client_statistics)
        )
    if config is None:
        return _ordered_metric_names(names)

    names.extend(
        name for name in _fit_metric_names(config) if name not in {"fit_loss", "fit_accuracy"}
    )
    if config.server.strategy == "scaffold":
        names.extend(("server_control_norm", "mean_client_control_delta_norm"))
    return _ordered_metric_names(_deduplicate(names))


def _fit_metric_names(config: FullConfig) -> list[str]:
    server_metrics = list(config.server.metrics)
    client_metrics = list(config.client.metrics)
    if not client_metrics and config.task.name == "classification":
        client_metrics = list(_FIT_METRIC_DEFAULTS.get(config.client.update_rule, ()))
    elif not client_metrics:
        client_metrics = list(server_metrics)

    if not server_metrics:
        return client_metrics
    if not client_metrics:
        return server_metrics
    return [name for name in server_metrics if name in client_metrics]


# --------------------------------------------------------------------------
# Which group, which split, which statistic -- derived, not tabulated
# --------------------------------------------------------------------------
# This replaced sixteen hand-written display labels. A table of labels goes
# stale silently: a column with no entry fell through to `name.title()` and
# printed as if it had been thought about. The derivation below reads the two
# authorities that actually define the column vocabulary --
# `client_metric_names` for the split aggregates and `FIXED_METRIC_GLOSSES`
# for everything else -- so a column that has a gloss has a group, and a
# column that has neither is loud rather than plausible.

#: The four groups, in the order a round block prints them, with the heading
#: text and the hue each is drawn in.
_METRIC_GROUPS: tuple[tuple[str, str, str], ...] = (
    ("loss", "loss", GROUP_LOSS),
    ("accuracy", "accuracy", GROUP_ACCURACY),
    ("spread", "client spread", GROUP_SPREAD),
    ("algorithm", "algorithm", GROUP_ALGORITHM),
    # Not a group anyone designed. It exists so that a column the derivation
    # cannot place is visible on screen instead of being filed under whichever
    # group happens to be the fallback.
    ("unclassified", "unclassified", UNCLASSIFIED),
)

#: The suffixes that describe how clients differ rather than how they did.
#: `worst{P}` carries its percentage in the name, so it is matched rather than
#: listed; the rest are `METRIC_SUFFIX_GLOSSES` keys.
_SPREAD_SUFFIXES = frozenset({"std", "variance", "min", "max"})
_WORST_SUFFIX = re.compile(r"^worst[0-9p]+$")

#: Every statistic suffix a split aggregate can carry, longest first so that
#: `sample_weighted_avg` is matched before `avg`.
_METRIC_SUFFIXES = tuple(
    sorted(
        (name for name in METRIC_SUFFIX_GLOSSES if name != "worst{P}"),
        key=len,
        reverse=True,
    )
)

#: Split token -> (label, tone). Longest token first: `central_test` must be
#: matched before `test`, or every central column reads as a client one.
_SPLIT_DISPLAY: tuple[tuple[str, str, str], ...] = (
    ("central_test", "central", SPLIT_CENTRAL),
    ("train", "train", GOLD),
    ("val", "validation", SPLIT_VAL),
    ("test", "test", SPLIT_TEST),
    # The fit pass measures the same data the train split does, one phase
    # earlier, so it shares the split's colour and its word. What separates
    # them on screen is the qualifier: a fit column has none.
    ("fit", "train", GOLD),
)

#: A row whose column name carries no split at all -- a per-round diagnostic
#: like `optimizer_steps`. An em dash rather than a blank, so the column reads
#: as deliberately empty instead of as a rendering fault.
_NO_SPLIT = ("—", FAINT)


def classify_metric(name: str) -> tuple[str, str, str, str]:
    """Return (group, split label, split tone, qualifier) for one column.

    The qualifier is what the column name says that its group heading does
    not. Under "loss", `val_loss_avg` is "avg"; under "client spread", where
    the heading names neither metric nor statistic, `test_accuracy_std` keeps
    both. A column with nothing left to say -- `fit_loss` under "loss" -- gets
    an em dash rather than an empty cell.
    """

    personal = name.startswith(PERSONAL_SPLIT_PREFIX)
    bare = name.removeprefix(PERSONAL_SPLIT_PREFIX)

    aggregate = _split_aggregate(bare)
    if aggregate is not None:
        split, base, suffix = aggregate
        spread = suffix in _SPREAD_SUFFIXES or bool(_WORST_SUFFIX.match(suffix))
        group = "spread" if spread else base
        qualifier = f"{base}_{suffix}" if spread else suffix
        label, tone = _split_display(split)
        return group, label, tone, _personalized(qualifier, personal)

    # central_test_ is an open class, not an enumeration: `evaluate_global`
    # passes through any finite numeric key a task reports, so a name this
    # module has never seen is still real -- unlike the split/base/suffix
    # vocabulary above, config typos cannot reach this prefix, only
    # `loop._evaluate_central_test_set`'s own output can.
    if bare in FIXED_METRIC_GLOSSES or bare.startswith("central_test_"):
        group = _fixed_metric_group(bare)
        label, tone = _split_display(_leading_split(bare))
        return group, label, tone, _personalized(_fixed_metric_qualifier(bare, group), personal)

    # Neither authority knows this name. Printing it under a plausible heading
    # would be the same defect the label table had, so it prints under its own.
    return "unclassified", "?", UNCLASSIFIED, name


def _split_aggregate(name: str) -> tuple[str, str, str] | None:
    """Decompose `{split}_{base}_{suffix}`, or None if it is not one."""

    for split, _, _ in _SPLIT_DISPLAY:
        if split == "fit" or not name.startswith(f"{split}_"):
            continue
        rest = name[len(split) + 1 :]
        for base in CLIENT_METRIC_BASES:
            if not rest.startswith(f"{base}_"):
                continue
            suffix = rest[len(base) + 1 :]
            if suffix in _METRIC_SUFFIXES or _WORST_SUFFIX.match(suffix):
                return split, base, suffix
    return None


def _leading_split(name: str) -> str | None:
    for split, _, _ in _SPLIT_DISPLAY:
        if name == split or name.startswith(f"{split}_"):
            return split
    return None


def _split_display(split: str | None) -> tuple[str, str]:
    for candidate, label, tone in _SPLIT_DISPLAY:
        if candidate == split:
            return label, tone
    return _NO_SPLIT


#: Fixed columns whose group cannot be read off the end of their name.
#: `{split}_num_clients` is a fact about the client population an aggregate is
#: over, which is what the "client spread" block already describes; the
#: fallthrough below would file it under "algorithm", beside the SCAFFOLD
#: control norms, where it says nothing about the algorithm at all. P07-F06.
_FIXED_METRIC_GROUPS: Mapping[str, str] = {
    f"{split}_num_clients": "spread" for split in ("train", "val", "test")
}


def _fixed_metric_group(name: str) -> str:
    """Loss, accuracy, spread or algorithm, from the column name.

    `fit_proximal_loss` and `fit_total_loss` are losses and belong beside the
    loss the run is actually minimising; everything that is neither, and that
    the table above does not place, is a diagnostic about how the algorithm
    behaved.
    """

    placed = _FIXED_METRIC_GROUPS.get(name)
    if placed is not None:
        return placed
    for base in CLIENT_METRIC_BASES:
        if name.endswith(f"_{base}"):
            return base
    return "algorithm"


def _fixed_metric_qualifier(name: str, group: str) -> str:
    """What is left of a fixed metric's name after its group and split.

    An algorithm diagnostic keeps its whole name: the heading says only
    "algorithm", so nothing in the name is redundant. Some of those names are
    longer than the qualifier column, which pushes that row's value right
    rather than truncating it -- a metric name cut in half cannot be matched
    against the CSV column it names, and these appear under --verbose, where a
    reader is looking at names rather than comparing numbers.
    """

    if group == "algorithm":
        return name
    split = _leading_split(name)
    trimmed = name if split is None else name.removeprefix(f"{split}_")
    trimmed = trimmed.removesuffix(f"_{group}").removesuffix(group)
    return trimmed.strip("_") or "—"


def _personalized(qualifier: str, personal: bool) -> str:
    """Personal-model columns say so in the qualifier, not the split column.

    `personal_val_loss_avg` measures the validation split -- the personal part
    is which model was evaluated on it, not which data. It also does not fit
    in an eleven-column split label.
    """

    if not personal:
        return qualifier
    return "personal" if qualifier == "—" else f"personal {qualifier}"


def _metric_definition(name: str, config: FullConfig | None) -> str:
    """The gloss for one column, under this config.

    The text itself lives in fedbrew.core.metrics, composed from the base
    metric and the suffix rather than written out per column -- 72 columns at
    `model_scope: both`, of which 71 would repeat the same six sentences. All
    that is decided here is the one thing the column name cannot carry: which
    clients a split was measured on.
    """

    split_glosses = dict(SPLIT_GLOSSES)
    if config is not None and config.evaluation.train.clients == "participating":
        # Not every client: this round's trainers. A gloss saying otherwise
        # would describe a measurement the run does not perform.
        split_glosses["train"] = "the selected clients' train data"
    return metric_gloss(name, split_glosses=split_glosses)


def _deduplicate(names: list[str]) -> list[str]:
    return list(dict.fromkeys(names))


def _ordered_metrics(metrics: dict[str, float]) -> list[tuple[str, float]]:
    return [(name, metrics[name]) for name in _ordered_metric_names(list(metrics))]


#: Display order for the metrics a run is likely to have. Everything else is
#: appended in sorted order after these.
_PREFERRED_METRIC_ORDER = (
    "train_loss_sample_weighted_avg",
    "central_test_loss",
    "val_loss_sample_weighted_avg",
    "val_loss_avg",
    "test_loss_sample_weighted_avg",
    "test_loss_avg",
    "train_accuracy_sample_weighted_avg",
    "central_test_accuracy",
    "val_accuracy_sample_weighted_avg",
    "val_accuracy_avg",
    "test_accuracy_sample_weighted_avg",
    "test_accuracy_avg",
    "test_accuracy_std",
    "test_accuracy_min",
)

#: The worst-percent column carries its percentage in its name, so it cannot be
#: a literal in the tuple above. It sorts last among the preferred names, which
#: is where the fixed "test_accuracy_bottom10" entry used to sit.
_WORST_PERCENT_METRIC = re.compile(r"^test_accuracy_worst[0-9p]+$")


def _ordered_metric_names(names: list[str]) -> list[str]:
    worst = sorted(name for name in names if _WORST_PERCENT_METRIC.match(name))
    preferred = (*_PREFERRED_METRIC_ORDER, *worst)
    ordered_names = [name for name in preferred if name in names]
    ordered_names.extend(sorted(name for name in names if name not in preferred))
    return ordered_names


def _format_metric(name: str, value: float | None, group: str | None = None) -> str:
    """Four decimal places, percentages for accuracies, integers for counts.

    The integer case is confined to the algorithm group, where every column
    that holds a whole number holds a count of something -- optimizer steps,
    parameters, bytes, tokens. `optimizer_steps 40.0000` invites a reader to
    wonder what the fractional part of a step would be, and doubt spreads to
    the losses beside it. It is confined to that group because a loss that
    lands exactly on 2.0 is still a loss measured to four places, and printing
    it as `2` would claim a precision it does not have in the other direction.
    """

    if value is None:
        return "-"
    if "accuracy" in name.lower():
        return f"{value:.2%}"
    if group == "algorithm" and float(value).is_integer():
        return f"{int(value)}"
    return f"{value:.4f}"


#: The metric keys are deliberately explicit, which makes a few of them long.
#: These are the shorter forms used for terminal headers only; the keys written
#: to CSV and JSON are always the full ones.
_METRIC_LABELS = {
    "fit_loss": "Fit Loss",
    "fit_accuracy": "Fit Accuracy",
    "train_loss_sample_weighted_avg": "Train Loss (sample-weighted)",
    "train_accuracy_sample_weighted_avg": "Train Accuracy (sample-weighted)",
    "train_loss_avg": "Train Loss (per-client)",
    "train_accuracy_avg": "Train Accuracy (per-client)",
    "val_loss_sample_weighted_avg": "Val Loss (sample-weighted)",
    "val_accuracy_sample_weighted_avg": "Val Accuracy (sample-weighted)",
    "val_loss_avg": "Val Loss (per-client)",
    "val_accuracy_avg": "Val Accuracy (per-client)",
    "test_loss_sample_weighted_avg": "Test Loss (sample-weighted)",
    "test_accuracy_sample_weighted_avg": "Test Accuracy (sample-weighted)",
    "test_loss_avg": "Test Loss (per-client)",
    "test_accuracy_avg": "Test Accuracy (per-client)",
    "central_test_loss": "Central Test Loss",
    "central_test_accuracy": "Central Test Accuracy",
}


def _metric_label(name: str) -> str:
    known = _METRIC_LABELS.get(name)
    if known is not None:
        return known
    return name.replace("_", " ").title()


def _final_line(
    status: str,
    history: list[MetricRecord],
    output_path: Path,
    termination: Mapping[str, Any] | None,
) -> str:
    """The whole run in one greppable line, for --quiet."""

    rounds = len(history)
    elapsed = _total_wall_clock(history)
    spent = "" if elapsed is None else f" in {_format_duration(elapsed)}"
    if status == "completed":
        return f"completed {rounds} rounds{spent} -> {output_path}"
    reason = str((termination or {}).get("reason") or "").strip()
    stopped_at = f" at round {history[-1].round_id}" if history else ""
    return f"{status}{stopped_at}{spent}: {reason} -> {output_path}"


def _total_wall_clock(history: list[MetricRecord]) -> float | None:
    durations = [record.timings.total for record in history if record.timings is not None]
    return sum(durations) if durations else None


def _format_duration(seconds: float) -> str:
    """Render seconds the way a job-time budget is read, not as raw floats."""

    if seconds < 10:
        return f"{seconds:.2f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {remaining_seconds:02d}s"


def _wall_clock_summary(history: list[MetricRecord]) -> str | None:
    """Summarise measured round time, or None when no round was timed."""

    durations = [record.timings.total for record in history if record.timings is not None]
    if not durations:
        return None
    total = sum(durations)
    mean = total / len(durations)
    return (
        f"{_format_duration(total)} over {len(durations)} timed rounds "
        f"({_format_duration(mean)}/round)"
    )


def _total_rounds(config: FullConfig | None) -> int | None:
    if config is None:
        return None
    return config.server.global_rounds


def _is_quiet(config: FullConfig | None) -> bool:
    return config is not None and bool(config.runtime.extra.get("quiet"))


def _use_plain(config: FullConfig | None) -> bool:
    return config is not None and bool(config.runtime.extra.get("no_rich"))


def _is_verbose(config: FullConfig | None) -> bool:
    return config is not None and bool(config.runtime.extra.get("verbose"))


def _surface_for(config: FullConfig | None) -> Surface:
    """The surface this run's config asks for.

    Built per call rather than once, matching what the removed _build_console
    did: these functions are called from four sites in runner.run() with no
    object between them to hold state, and a surface is a few attributes and a
    rich Console. The one patch point the tests need is this function.
    """

    return build_surface(
        quiet=_is_quiet(config),
        verbose=_is_verbose(config),
        no_rich=_use_plain(config),
    )
