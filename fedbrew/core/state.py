"""Experiment state containers and lifecycle helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
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


class _AppendOnlyHistory(list):  # type: ignore[type-arg]
    """A list that keeps a summary of itself current, and refuses to forget.

    Subclassing list keeps every reader unchanged -- the CSV writers, the
    round-table code and the tests all iterate, index and len() these exactly
    as before. Only the two ways records actually arrive are overridden.

    The mutators that would invalidate the summary raise instead of silently
    desynchronising it. Nothing in the codebase uses them: these histories are
    appended to and read, never edited, and a summary that can drift from the
    list it describes is worse than the full scan it replaces.
    """

    __slots__ = ("summary",)

    def __init__(self) -> None:
        super().__init__()
        self.summary = ClientHistorySummary()

    def _accumulate(self, record: Any) -> None:
        raise NotImplementedError

    def append(self, record: Any) -> None:
        self._accumulate(record)
        super().append(record)

    def extend(self, records: Any) -> None:
        for record in records:
            self.append(record)

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


class ClientEvaluationHistory(_AppendOnlyHistory):
    """Post-aggregation per-client evaluations, with running totals."""

    __slots__ = ()

    def _accumulate(self, record: ClientEvaluationRecord) -> None:
        summary = self.summary
        summary.client_ids.add(record.client_id)
        summary.train_examples += record.train_num_examples
        summary.test_examples += record.test_num_examples


class ClientUpdateHistory(_AppendOnlyHistory):
    """Per-client update diagnostics, with running totals."""

    __slots__ = ()

    def _accumulate(self, record: ClientMetricRecord) -> None:
        summary = self.summary
        summary.client_ids.add(record.client_id)
        summary.phase_counts[record.phase] = summary.phase_counts.get(record.phase, 0) + 1
        summary.num_examples += record.num_examples
        summary.metric_names.update(record.metrics)


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
    metrics_history: list[MetricRecord] = field(default_factory=list)
    # Append-only lists that summarise themselves: run.json is rewritten every
    # round and its scale block would otherwise re-scan all of both, every
    # round, for the whole run.
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
