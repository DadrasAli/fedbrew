"""Experiment state containers and lifecycle helpers."""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Any

from fedbrew.core.divergence import STATUS_COMPLETED


@dataclass(slots=True)
class RoundTimings:
    """Wall-clock seconds spent in each phase of one completed round.

    Deliberately kept out of the ``metrics`` dict: timings are run diagnostics,
    not learning signal, and everything downstream that iterates over metric
    names (progress table, filter_metrics, the CSV plotter) would otherwise
    treat seconds as a metric to plot and select on.

    ``fit`` and ``aggregate`` are measured separately even though the server
    consumes fit results as a stream: ``fit`` is the time spent inside client
    updates, ``aggregate`` is the streaming server's own time around them.
    """

    total: float = 0.0
    fit: float = 0.0
    aggregate: float = 0.0
    client_eval: float = 0.0
    global_eval: float = 0.0
    checkpoint: float = 0.0


@dataclass(slots=True)
class MetricRecord:
    """Metrics collected for one completed round."""

    round_id: int
    metrics: dict[str, float]
    #: How many clients trained this round. The identities used to be stored
    #: here and written to every artifact; nothing read them back except a
    #: count, and at participation_rate 1 on FEMNIST it held 3597 ids per round.
    #: client_update_metrics.csv still carries which client did what.
    num_clients: int
    num_examples: int
    timings: RoundTimings | None = None


@dataclass(slots=True)
class ClientMetricRecord:
    """Selected-client update diagnostics for one phase of one round."""

    round_id: int
    client_id: str
    phase: str
    num_examples: int
    metrics: dict[str, float]


@dataclass(slots=True)
class ClientEvaluationRecord:
    """Post-aggregation train/test evaluation for one client and round."""

    round_id: int
    client_id: str
    participated: bool
    train_num_examples: int
    test_num_examples: int
    global_model_train_loss: float | None
    global_model_train_accuracy: float | None
    global_model_test_loss: float | None
    global_model_test_accuracy: float | None
    val_num_examples: int = 0
    global_model_val_loss: float | None = None
    global_model_val_accuracy: float | None = None
    #: Which model the numbers in this row came from: "global" or "personal".
    #: The column names say global_model_* for every row because they were
    #: named before evaluation.model_scope existed and a rename would break
    #: every existing reader; this field is what disambiguates them.
    model_scope: str = "global"


@dataclass(slots=True)
class ClientHistorySummary:
    """Running totals over one client history, maintained as records arrive.

    run.json's scale block needs six aggregates over these histories, and
    on_round_flush rewrites run.json every round so a killed job's artifacts
    stay readable. Recomputing the aggregates meant re-scanning every record
    since round 1 on every round: O(records so far) per round, quadratic in
    round count, and with the square of the client count too, so it is worst
    exactly where cross-device runs are largest.

    Every field here is O(1) per record instead.
    """

    client_ids: set[str] = field(default_factory=set)
    phase_counts: dict[str, int] = field(default_factory=dict)
    train_examples: int = 0
    test_examples: int = 0
    num_examples: int = 0
    #: Every metric name seen, which is client_update_metrics.csv's column set.
    #: Deriving it was another full scan of the history on every flush.
    metric_names: set[str] = field(default_factory=set)
    #: How many records arrived, whether the history keeps them or not
    #: (``_AppendOnlyHistory.keeps``): run.json's counts of them.
    records: int = 0

    def snapshot(self) -> ClientHistorySummary:
        """This summary as it is now, in containers of its own: what a flush writes.

        Its elements are strings and numbers, which nothing changes, so new
        containers of the same elements are a deep copy's equal at a fraction
        of its cost (a 1000-client summary: 10 against 376 us).
        """

        return replace(
            self,
            client_ids=set(self.client_ids),
            phase_counts=dict(self.phase_counts),
            metric_names=set(self.metric_names),
        )


@dataclass(slots=True)
class RoundTimingSummary:
    """Running aggregates of the rounds' wall clock, for run.json's timing block.

    run.json is rewritten every round, and its timing block summed, averaged,
    sorted for the median and took the extremes of every round's timings since
    round 1, each time: O(rounds so far) per round, which took the per-round
    write from 1.0 to 2.3 ms between 200 and 2000 rounds. Each aggregate is
    kept here as records arrive instead, in O(log rounds), and each is the
    value the full computation gives, bit for bit: the totals add left to
    right in round order (sum()'s order before Python 3.12, whose sum()
    compensates as it goes), ``partials`` holds the exact sum math.fsum -- and so
    statistics.fmean -- rounds, and the two heaps hold the halves of the sorted
    durations whose middle statistics.median reads.
    """

    timed_rounds: int = 0
    total_sec: float = 0.0
    minimum_sec: float = math.inf
    maximum_sec: float = -math.inf
    #: Non-overlapping partial sums of every duration (Shewchuk), whose exact
    #: total math.fsum rounds once, as it does over the durations themselves.
    partials: list[float] = field(default_factory=list)
    #: The smaller half of the durations, negated so heapq keeps the largest
    #: on top, and the larger half; the smaller half is never the shorter.
    lower_half: list[float] = field(default_factory=list)
    upper_half: list[float] = field(default_factory=list)
    #: Seconds per RoundTimings field other than "total", summed in order.
    phase_sec: dict[str, float] = field(default_factory=dict)

    def add(self, timings: RoundTimings) -> None:
        duration = timings.total
        self.timed_rounds += 1
        self.total_sec += duration
        self.minimum_sec = min(self.minimum_sec, duration)
        self.maximum_sec = max(self.maximum_sec, duration)
        _add_exactly(self.partials, duration)
        if not self.lower_half or duration <= -self.lower_half[0]:
            heapq.heappush(self.lower_half, -duration)
        else:
            heapq.heappush(self.upper_half, duration)
        if len(self.lower_half) > len(self.upper_half) + 1:
            heapq.heappush(self.upper_half, -heapq.heappop(self.lower_half))
        elif len(self.upper_half) > len(self.lower_half):
            heapq.heappush(self.lower_half, -heapq.heappop(self.upper_half))
        for timing in fields(RoundTimings):
            if timing.name != "total":
                self.phase_sec[timing.name] = self.phase_sec.get(timing.name, 0.0) + getattr(
                    timings, timing.name
                )

    def snapshot(self) -> RoundTimingSummary:
        """This summary as it is now, in containers of its own, as ``ClientHistorySummary``'s."""

        return replace(
            self,
            partials=list(self.partials),
            lower_half=list(self.lower_half),
            upper_half=list(self.upper_half),
            phase_sec=dict(self.phase_sec),
        )

    def mean_sec(self) -> float:
        return math.fsum(self.partials) / self.timed_rounds

    def median_sec(self) -> float:
        if len(self.lower_half) > len(self.upper_half):
            return -self.lower_half[0]
        return (-self.lower_half[0] + self.upper_half[0]) / 2


def _add_exactly(partials: list[float], value: float) -> None:
    """Fold ``value`` into ``partials`` so they still sum exactly to every value so far."""

    kept = 0
    for partial in partials:
        if abs(value) < abs(partial):
            value, partial = partial, value
        high = value + partial
        low = partial - (high - value)
        if low:
            partials[kept] = low
            kept += 1
        value = high
    partials[kept:] = [value]


class _AppendOnlyHistory(list):  # type: ignore[type-arg]
    """A list that keeps a summary of itself current, and refuses to forget.

    Subclassing list keeps every reader unchanged -- the CSV writers, the
    round-table code and the tests all iterate, index and len() these exactly
    as before. Only the two ways records actually arrive are overridden.

    The mutators that would invalidate the summary raise instead of silently
    desynchronising it. Nothing in the codebase uses them: these histories are
    appended to and read, never edited, and a summary that can drift from the
    list it describes is worse than the full scan it replaces.

    ``keeps`` false keeps the summary and not the records: a run whose
    per-client records nobody reads -- no per-client CSV, the only reader
    (``run_fl_loop``'s ``client_records``) -- summarises them as they arrive
    and holds none, and the stacked path builds none (``extend_stacked``).
    """

    __slots__ = ("summary", "keeps")

    def __init__(self) -> None:
        super().__init__()
        self.summary = ClientHistorySummary()
        self.keeps = True

    def _accumulate(self, record: Any) -> None:
        raise NotImplementedError

    def append(self, record: Any) -> None:
        self._accumulate(record)
        if self.keeps:
            super().append(record)

    def extend(self, records: Any) -> None:
        records = list(records)
        self._accumulate_all(records)
        if self.keeps:
            super().extend(records)

    def _accumulate_all(self, records: list[Any]) -> None:
        """``_accumulate`` of each record in turn: a round's records arrive together."""

        for record in records:
            self._accumulate(record)

    def _refuse(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(
            f"{type(self).__name__} is append-only: it maintains a running "
            "summary that any other mutation would silently invalidate"
        )

    insert = _refuse
    pop = _refuse
    remove = _refuse
    clear = _refuse
    sort = _refuse
    reverse = _refuse
    __setitem__ = _refuse
    __delitem__ = _refuse
    __iadd__ = _refuse


class RoundHistory(_AppendOnlyHistory):
    """The completed rounds' records, with running timing aggregates."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__()
        self.summary = RoundTimingSummary()

    def _accumulate(self, record: MetricRecord) -> None:
        if record.timings is not None:
            self.summary.add(record.timings)


class ClientEvaluationHistory(_AppendOnlyHistory):
    """Post-aggregation per-client evaluations, with running totals."""

    __slots__ = ()

    def _accumulate(self, record: ClientEvaluationRecord) -> None:
        summary = self.summary
        summary.client_ids.add(record.client_id)
        summary.train_examples += record.train_num_examples
        summary.test_examples += record.test_num_examples
        summary.records += 1


class ClientUpdateHistory(_AppendOnlyHistory):
    """Per-client update diagnostics, with running totals."""

    __slots__ = ()

    def _accumulate(self, record: ClientMetricRecord) -> None:
        self._accumulate_all([record])

    def _accumulate_all(self, records: list[ClientMetricRecord]) -> None:
        # The same additions in the same order, the summary's fields looked up once.
        summary = self.summary
        client_ids, counts, names = summary.client_ids, summary.phase_counts, summary.metric_names
        examples = summary.num_examples
        for record in records:
            client_ids.add(record.client_id)
            counts[record.phase] = counts.get(record.phase, 0) + 1
            examples += record.num_examples
            names.update(record.metrics)
        summary.num_examples = examples
        summary.records += len(records)

    def extend_stacked(
        self,
        records: Callable[[], list[ClientMetricRecord]],
        phase: str,
        client_ids: Sequence[str],
        counts: Sequence[int],
        names: Iterable[str],
    ) -> None:
        """``extend`` with one stack's records, its summary taken a column at a time.

        ``records`` makes the stack's clients' records of ``phase``, client
        ``p`` of ``client_ids`` with ``counts[p]`` examples, and is called only
        where the history keeps them; ``names`` every metric name some record
        holds. The totals ``_accumulate_all`` keeps -- a set of ids, a count
        per phase, a sum of whole numbers, a set of names, how many -- are the
        same taken this way.
        """

        if client_ids:
            summary = self.summary
            summary.client_ids.update(client_ids)
            summary.phase_counts[phase] = summary.phase_counts.get(phase, 0) + len(client_ids)
            summary.num_examples += sum(counts)
            summary.metric_names.update(names)
            summary.records += len(client_ids)
        if self.keeps:
            list.extend(self, records())


@dataclass(slots=True)
class RoundState:
    """Serializable state for one completed round."""

    round_id: int
    metrics: dict[str, float]
    num_clients: int
    num_examples: int
    timings: RoundTimings | None = None


@dataclass(slots=True)
class ExperimentState:
    """State returned by a completed benchmark run."""

    rounds: list[RoundState] = field(default_factory=list)
    # Append-only lists that summarise themselves: run.json is rewritten every
    # round and its timing and scale blocks would otherwise re-scan all three,
    # every round, for the whole run.
    metrics_history: list[MetricRecord] = field(default_factory=RoundHistory)
    client_metrics_history: list[ClientEvaluationRecord] = field(
        default_factory=ClientEvaluationHistory
    )
    client_update_metrics_history: list[ClientMetricRecord] = field(
        default_factory=ClientUpdateHistory
    )
    #: The model state the run ended with. run_fl_loop is this library's public
    #: entry point, and this is how a caller gets the final model back --
    #: nothing inside the package reads it, and that is expected, not dead.
    final_payload: dict[str, Any] = field(default_factory=dict)
    checkpointing: dict[str, Any] = field(default_factory=dict)
    #: "completed", "diverged" or "stalled". A run that stops early is not a
    #: failed run -- it exits normally and records why here, so that a packed
    #: SLURM job cannot mistake divergence for a crash.
    status: str = STATUS_COMPLETED
    #: The divergence verdict that ended the run, empty when it completed.
    termination: dict[str, Any] = field(default_factory=dict)
    #: Whether this run *continued* an earlier attempt. A resume that cannot
    #: be taken is refused before the run starts (POST-F25), so this is also
    #: whether one was requested; it used to restart from round 1 and still
    #: be called a resume (POST-F05).
    resumed: bool = False
