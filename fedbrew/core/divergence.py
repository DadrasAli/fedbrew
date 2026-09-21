"""Stop a run that is no longer learning, and record why.

The monitor is fed one metric value per round and answers a single question:
should this run stop, and under what name. It holds no model state and does no
I/O, so it is cheap to call every round and easy to test directly.

Three detectors, deliberately reported as two different statuses:

``non_finite`` and ``blowup`` produce ``diverged``
    The run's loss went to NaN/Inf or exploded. Both are unambiguous and fire
    within a round or two, which is what makes them worth running by default.

``blowup_absolute`` also produces ``diverged``
    A fixed ceiling, needed because the relative check has nothing to anchor on
    when the very first observation is already pathological.

``patience`` produces ``stalled``
    The run is still finite but has stopped improving. That is a weaker and
    different claim -- reporting a plateau as divergence would misstate a
    sweep -- so it gets its own status and is off unless asked for.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from fedbrew.core.config import DivergenceConfig

#: Terminal statuses a run can end with. "completed" is the normal case; the
#: other two are set by this module.
STATUS_COMPLETED = "completed"
STATUS_DIVERGED = "diverged"
STATUS_STALLED = "stalled"


@dataclass(frozen=True)
class DivergenceVerdict:
    """Why a run stopped early, in the form written to run.json."""

    status: str
    detector: str
    round_id: int
    metric: str
    value: float
    #: The number the value was compared against: the blow-up ceiling for
    #: ``blowup``, the best value so far for ``patience``. None for
    #: ``non_finite``, which has nothing to compare to.
    threshold: float | None
    reason: str

    def as_dict(self) -> dict[str, object]:
        """The termination block as written to run.json.

        ``status`` is deliberately absent: it is a top-level key there, and
        run.json's contract is that nothing appears twice in it.
        """

        return {
            "detector": self.detector,
            "round_id": self.round_id,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "reason": self.reason,
        }


class DivergenceMonitor:
    """Watches one metric across rounds and decides when to stop.

    ``update`` is called once per round with that round's metrics and returns
    a verdict the first time a detector fires, then None forever after -- the
    caller is expected to break out of the loop on the first verdict.
    """

    def __init__(self, config: DivergenceConfig) -> None:
        """Arm the detectors described by ``config``.

        Args:
            config: The resolved ``divergence`` block. Nothing is copied, so
                the monitor reflects later mutation of the config object.

        No state is derived from the first round until :meth:`update` sees the
        watched metric: the relative blow-up threshold is anchored on the first
        *observed* value, not on round 1, so a schedule-gated metric anchors on
        the first round it is actually produced.
        """

        self._config = config
        self._first_value: float | None = None
        self._best_value: float | None = None
        self._best_round: int | None = None
        #: Observations since the last real improvement -- not rounds. A
        #: schedule-gated metric is absent on most rounds, and counting rounds
        #: would make patience measure the evaluation schedule instead of the
        #: run's progress.
        self._since_best = 0
        self._verdict: DivergenceVerdict | None = None
        #: Whether the watched metric was ever present in a round's metrics. A
        #: name that is never emitted -- a typo, or a metric the configured
        #: evaluation schedule does not produce -- makes _metric_value return
        #: None every round, which update() reads as "not evaluated yet". That
        #: silently disables every detector, including non_finite, for the whole
        #: run. Tracked so the caller can say so instead.
        self._observed = False

    @property
    def metric(self) -> str:
        """Name of the metric being watched, from the config."""

        return self._config.metric

    @property
    def verdict(self) -> DivergenceVerdict | None:
        """The verdict that fired, or None while the run is still healthy.

        Latched: once a detector fires this keeps returning that first verdict,
        and :meth:`update` returns None thereafter. The caller is expected to
        stop the run on the first verdict, so a second one would describe a run
        that should already have ended.
        """

        return self._verdict

    @property
    def observed(self) -> bool:
        """True once the watched metric has actually been seen in a round."""

        return self._observed

    def update(self, round_id: int, metrics: dict[str, float] | None) -> DivergenceVerdict | None:
        """Feed one round's metrics; return a verdict the first time one fires."""

        if not self._config.active or self._verdict is not None:
            return None

        value = _metric_value(metrics, self._config.metric)
        if value is not None:
            self._observed = True
        if value is None:
            # The watched metric is not in this round -- a schedule-gated metric
            # on an unevaluated round, say. Skipping keeps the patience counter
            # measured in rounds that actually carry an observation.
            return None

        verdict = self._evaluate(round_id, value)
        if verdict is not None:
            self._verdict = verdict
        return verdict

    def _evaluate(self, round_id: int, value: float) -> DivergenceVerdict | None:
        config = self._config

        if config.non_finite and not math.isfinite(value):
            return DivergenceVerdict(
                status=STATUS_DIVERGED,
                detector="non_finite",
                round_id=round_id,
                metric=config.metric,
                value=value,
                threshold=None,
                reason=(
                    f"{config.metric} is {value} at round {round_id}; the model "
                    "cannot recover from a non-finite loss"
                ),
            )

        if not math.isfinite(value):
            # non_finite detection is off, so a NaN cannot be compared against
            # anything below without poisoning the baseline. Ignore the round.
            return None

        if self._first_value is None and value > 0.0:
            # Anchored on the first strictly positive observation, not simply
            # the first. A metric can legitimately start at 0 -- on a
            # one-label-per-client split every client fits its single label
            # exactly, so fit_loss is 0.0 with accuracy 1.0 in round 1 -- and
            # anchoring there would give a ceiling of 0 and either fire on
            # everything or, with a non-positive guard, never fire at all.
            self._first_value = value
        absolute = self._check_absolute(round_id, value)
        if absolute is not None:
            return absolute
        blowup = self._check_blowup(round_id, value)
        if blowup is not None:
            return blowup
        return self._check_patience(round_id, value)

    def prime(self, observations: Iterable[tuple[int, dict[str, float] | None]]) -> None:
        """Re-anchor from rounds that already happened, without judging them.

        A resumed run builds a fresh monitor, so without this the blow-up
        ceiling re-anchors on the resumed round's loss instead of round 1's. On
        FEMNIST fit_loss starts near ln 62 = 4.1 (ceiling 41 at blowup_factor
        10); a requeue after it had fallen to 0.3 would re-anchor the ceiling
        at 3 -- and a noisy round on a 1% participation draw could then
        be recorded as diverged for an arm that would have completed had it
        never been preempted. runs_index.jsonl would carry that verdict with
        nothing beside it saying the run had resumed.

        Deliberately does not return or store a verdict: these rounds already
        happened and the run continued past them. Priming answers "where were
        the anchors", not "should this have stopped".
        """

        if not self._config.active:
            return
        for round_id, metrics in observations:
            value = _metric_value(metrics, self._config.metric)
            if value is None or not math.isfinite(value):
                continue
            self._observed = True
            if self._first_value is None and value > 0.0:
                self._first_value = value
            self._track_best(round_id, value)

    def _track_best(self, round_id: int, value: float) -> None:
        """The patience bookkeeping, without the verdict."""

        if self._best_value is None or self._best_round is None:
            self._best_value = value
            self._best_round = round_id
            self._since_best = 0
            return
        required = self._best_value - abs(self._best_value) * float(self._config.min_delta)
        if value <= required:
            self._best_value = value
            self._best_round = round_id
            self._since_best = 0
        else:
            self._since_best += 1

    def _check_absolute(self, round_id: int, value: float) -> DivergenceVerdict | None:
        ceiling = self._config.blowup_absolute
        if ceiling is None or value <= float(ceiling):
            return None
        # The backstop for a run that was already pathological at its first
        # observation, where no relative anchor exists to measure against.
        return DivergenceVerdict(
            status=STATUS_DIVERGED,
            detector="blowup_absolute",
            round_id=round_id,
            metric=self._config.metric,
            value=value,
            threshold=float(ceiling),
            reason=(
                f"{self._config.metric} reached {value:.6g} at round {round_id}, "
                f"above the absolute ceiling of {float(ceiling):.6g}"
            ),
        )

    def _check_blowup(self, round_id: int, value: float) -> DivergenceVerdict | None:
        factor = self._config.blowup_factor
        if factor is None or self._first_value is None:
            return None
        # Relative to the first positive observation, so the same factor
        # transfers across tasks with different loss scales without needing a
        # per-task number. blowup_absolute covers what this cannot: a run that
        # was already astronomically bad the first time it was measured.
        ceiling = self._first_value * float(factor)
        if value <= ceiling:
            return None
        return DivergenceVerdict(
            status=STATUS_DIVERGED,
            detector="blowup",
            round_id=round_id,
            metric=self._config.metric,
            value=value,
            threshold=ceiling,
            reason=(
                f"{self._config.metric} reached {value:.6g} at round {round_id}, "
                f"above {factor:g}x its round-1 value of "
                f"{self._first_value:.6g} ({ceiling:.6g})"
            ),
        )

    def _check_patience(self, round_id: int, value: float) -> DivergenceVerdict | None:
        patience = self._config.patience
        if patience is None:
            return None
        if self._best_value is None or self._best_round is None:
            self._best_value = value
            self._best_round = round_id
            self._since_best = 0
            return None

        # min_delta gates what counts as an improvement at all, rather than
        # only the comparison: otherwise a noise-sized gain would reset the
        # counter every observation and a stalled run would never be caught.
        # Written against the magnitude so it behaves the same for a metric
        # that can go negative.
        required = self._best_value - abs(self._best_value) * float(self._config.min_delta)
        if value <= required:
            self._best_value = value
            self._best_round = round_id
            self._since_best = 0
            return None

        self._since_best += 1
        if self._since_best < patience:
            return None
        return DivergenceVerdict(
            status=STATUS_STALLED,
            detector="patience",
            round_id=round_id,
            metric=self._config.metric,
            value=value,
            threshold=self._best_value,
            reason=(
                f"{self._config.metric} has not improved on {self._best_value:.6g} "
                f"(round {self._best_round}) in {self._since_best} evaluations"
            ),
        )


def _metric_value(metrics: dict[str, float] | None, name: str) -> float | None:
    if not metrics:
        return None
    value = metrics.get(name)
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, int | float):
        return None
    return float(value)
