"""The run's randomized-output means: the ``convergence`` config section.

Convergence rates for stochastic non-convex methods are stated for a
randomized output (Ghadimi and Lan, "Stochastic first- and zeroth-order
methods for nonconvex stochastic programming", SIAM J. Optim. 23(4), 2013):
the method returns ``x_R`` with ``R`` drawn uniformly from ``{1, .., T}``, and
the bound is on ``E[||grad F(x_R)||^2] = (1/T) sum_t ||grad F(x_t)||^2``. The
same mean of an optimality gap ``F(x_t) - F*`` is what the convex rates bound
for an averaged output. Neither can be read off a curve evaluated every tenth
round, so a run that is asked for it (``convergence.metrics``) evaluates the
metric at every round and writes the mean.

Which iterates. The iterates are the global model after each round's
aggregation, ``x_1, .., x_t`` -- the model every ``central_test_*`` column and
``grad_norm_sq`` is measured at. The mean written on round ``t`` is over
those ``t`` values, ``(1/t) sum_{s<=t} m(x_s)``; the initial model ``x_0`` is
not an iterate, and a round that is not evaluated does not exist: every round
is. The column is ``<metric>_running_mean``; on the final round it is the
expected metric at the randomized output of the whole run.

Exact. The sum of the values is kept as Shewchuk's exact partials (the
algorithm behind ``math.fsum``), so the mean is the correctly rounded sum
divided by ``t``: ``math.fsum(values) / t`` bit for bit, whatever the order or
the number of rounds, and the same on a resumed run, whose partials are in the
checkpoint (``state``).

Off, which is the default, the loop never builds this object: no schedule is
changed, no value read, nothing written.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from fedbrew.core.config import evaluates_round
from fedbrew.core.metrics import GRAD_NORM_COLUMN, RUNNING_MEAN_SUFFIX
from fedbrew.core.refusal import RunRefused

CENTRAL_PREFIX = "central_test_"

#: The checkpoint key the partial sums are written under.
CHECKPOINT_KEY = "convergence"


def running_mean_column(metric: str) -> str:
    """The column that holds the running mean of ``metric``."""

    return f"{metric}{RUNNING_MEAN_SUFFIX}"


def resolve_convergence_metrics(
    names: Sequence[str],
    *,
    central: Sequence[str],
    grad_norm: bool,
    task: str,
) -> list[str]:
    """``convergence.metrics`` as round columns, refusing a name nothing evaluates.

    ``central`` is what the task's central pass reports, ``grad_norm`` whether
    the task declares ``grad_norm_sq``. A bare name of a central metric
    (``optimality_gap``) is that metric's ``central_test_`` column.
    """

    columns: list[str] = []
    for name in names:
        column = _column(name, central, grad_norm, task)
        if column in columns:
            raise RunRefused(f"convergence.metrics names {column!r} twice ({name!r})")
        columns.append(column)
    return columns


def _column(name: str, central: Sequence[str], grad_norm: bool, task: str) -> str:
    if name == GRAD_NORM_COLUMN:
        if not grad_norm:
            raise RunRefused(
                f"convergence.metrics names {name!r}, but task {task!r} declares no gradient "
                "of its objective (TaskAdapter.GRAD_NORM_GLOSS), so it has no grad_norm_sq"
            )
        return name
    bare = name.removeprefix(CENTRAL_PREFIX)
    if bare in central:
        return f"{CENTRAL_PREFIX}{bare}"
    valid = [f"{CENTRAL_PREFIX}{metric}" for metric in central]
    if grad_norm:
        valid.append(GRAD_NORM_COLUMN)
    raise RunRefused(
        f"convergence.metrics names {name!r}, which task {task!r} does not evaluate at the "
        f"global model every round. It can be asked for: {', '.join(valid)} (a central "
        "metric may be written without its central_test_ prefix)"
    )


class ExactSum:
    """A sum of floats kept exactly, as Shewchuk's partials: ``value()`` is ``math.fsum``'s.

    Adding costs a few operations (the partials of a sum of ordinary
    magnitudes number about two); reading is one ``fsum`` over them. The
    values must be finite: infinities and NaN do not survive the partials, and
    the caller treats them apart.
    """

    __slots__ = ("partials",)

    def __init__(self, partials: Sequence[float] = ()) -> None:
        self.partials = [float(p) for p in partials]

    def add(self, x: float) -> None:
        partials = self.partials
        i = 0
        for y in partials:
            if abs(x) < abs(y):
                x, y = y, x
            high = x + y
            low = y - (high - x)
            if low:
                partials[i] = low
                i += 1
            x = high
        partials[i:] = [x]

    def value(self) -> float:
        return math.fsum(self.partials)


class RunningMeans:
    """Each chosen metric's exact mean over the iterates seen so far; what the loop asks of it.

    The loop calls ``observe`` once per round, in order, after the round's
    evaluations are in ``metrics``. Rounds are evaluated every round where
    this run needs a pass (``schedules``); ``observe`` then removes the
    columns of a pass the config's own schedule did not ask for on that round,
    so the CSV holds the evaluations ``evaluation`` schedules and the means
    ``convergence`` adds, and no others.
    """

    def __init__(
        self,
        metrics: Sequence[str],
        *,
        central_schedule: int | None,
        grad_norm_schedule: int | None,
        global_rounds: int,
    ) -> None:
        self.metrics = list(metrics)
        self.global_rounds = global_rounds
        self._asked = {"central": central_schedule, "grad_norm": grad_norm_schedule}
        self.needs_central = any(m.startswith(CENTRAL_PREFIX) for m in self.metrics)
        self.needs_grad_norm = GRAD_NORM_COLUMN in self.metrics
        self.rounds = 0
        self._sums = {metric: ExactSum() for metric in self.metrics}
        self._not_finite = dict.fromkeys(self.metrics, False)

    @classmethod
    def from_config(
        cls,
        convergence: Any,
        *,
        central_schedule: int | None,
        grad_norm_schedule: int | None,
        global_rounds: int,
    ) -> RunningMeans | None:
        """The run's means, or None -- nothing to build -- when ``convergence.metrics`` is empty."""

        if convergence is None or not convergence.metrics:
            return None
        return cls(
            convergence.metrics,
            central_schedule=central_schedule,
            grad_norm_schedule=grad_norm_schedule,
            global_rounds=global_rounds,
        )

    def schedules(self) -> tuple[int | None, int | None]:
        """The (central, grad_norm) schedules the loop evaluates with: every round where needed."""

        central, grad_norm = self._asked["central"], self._asked["grad_norm"]
        return (1 if self.needs_central else central, 1 if self.needs_grad_norm else grad_norm)

    def observe(self, round_id: int, metrics: dict[str, float]) -> None:
        """Fold round ``round_id``'s iterate in, and write the means into ``metrics``.

        A value the round did not produce, or that is not finite, makes the
        mean not finite from then on: written as NaN, which is what the mean
        of a set with such a member is.
        """

        if round_id != self.rounds + 1:
            raise RuntimeError(
                f"the running means were asked for round {round_id} after round {self.rounds}: "
                "an iterate was skipped or repeated, and the mean would not be over the run's "
                "iterates"
            )
        self.rounds = round_id
        for metric in self.metrics:
            value = metrics.get(metric)
            if value is None or isinstance(value, bool) or not math.isfinite(value):
                self._not_finite[metric] = True
            elif not self._not_finite[metric]:
                self._sums[metric].add(float(value))
            mean = (
                math.nan if self._not_finite[metric] else self._sums[metric].value() / self.rounds
            )
            metrics[running_mean_column(metric)] = mean
        self._drop_unasked(round_id, metrics)

    def _drop_unasked(self, round_id: int, metrics: dict[str, float]) -> None:
        """Remove what a pass forced for the mean wrote, where its own schedule did not ask."""

        if self.needs_central and not evaluates_round(
            self._asked["central"], round_id, self.global_rounds
        ):
            means = {running_mean_column(metric) for metric in self.metrics}
            for name in [n for n in metrics if n.startswith(CENTRAL_PREFIX) and n not in means]:
                del metrics[name]
        if self.needs_grad_norm and not evaluates_round(
            self._asked["grad_norm"], round_id, self.global_rounds
        ):
            metrics.pop(GRAD_NORM_COLUMN, None)

    def state(self) -> dict[str, Any]:
        """The partial sums, for a checkpoint: what ``restore`` continues from."""

        return {
            "rounds": self.rounds,
            "metrics": {
                metric: {
                    "partials": list(self._sums[metric].partials),
                    "not_finite": self._not_finite[metric],
                }
                for metric in self.metrics
            },
        }

    def restore(self, checkpoint: Mapping[str, Any], round_id: int, where: str) -> None:
        """Continue from the partial sums a checkpoint of round ``round_id`` holds, or refuse."""

        held = checkpoint.get(CHECKPOINT_KEY)
        reason = None
        if not isinstance(held, Mapping):
            reason = (
                "carries no convergence state (it was written by a run without convergence.metrics)"
            )
        elif held.get("rounds") != round_id:
            reason = f"holds the convergence state of round {held.get('rounds')}, not {round_id}"
        elif set(held.get("metrics", {})) != set(self.metrics):
            reason = (
                f"holds the running means of {sorted(held.get('metrics', {}))}, and this run "
                f"asks for {sorted(self.metrics)}"
            )
        if reason is not None:
            raise RunRefused(
                f"cannot resume from {where}: the checkpoint {reason}, so the running mean over "
                f"the iterates of rounds 1 to {round_id} cannot be continued. Nothing was run "
                "and nothing was changed. Run again from round 1."
            )
        assert isinstance(held, Mapping)
        self.rounds = round_id
        for metric in self.metrics:
            entry = held["metrics"][metric]
            self._sums[metric] = ExactSum(entry["partials"])
            self._not_finite[metric] = bool(entry["not_finite"])


# What the round loops call. Each takes the run's ``RunningMeans`` or None, and
# does nothing for None: a config without the section leaves the loops'
# own code to run exactly as it ran before the section existed.


def set_up(
    convergence: Any,
    central_schedule: int | None,
    grad_norm_schedule: int | None,
    global_rounds: int,
) -> tuple[RunningMeans | None, int | None, int | None]:
    """The run's means (None if the section is off) and the schedules the loop evaluates with."""

    means = RunningMeans.from_config(
        convergence,
        central_schedule=central_schedule,
        grad_norm_schedule=grad_norm_schedule,
        global_rounds=global_rounds,
    )
    if means is None:
        return None, central_schedule, grad_norm_schedule
    return (means, *means.schedules())


def continue_from(
    means: RunningMeans | None,
    checkpoint: Mapping[str, Any] | None,
    start_round: int,
    resume_from: Any,
) -> None:
    """A resumed run continues the means from the checkpoint's partial sums, or is refused."""

    if means is not None and checkpoint is not None:
        means.restore(checkpoint, start_round - 1, str(resume_from))


def observe_round(means: RunningMeans | None, round_id: int, metrics: dict[str, float]) -> None:
    """Round ``round_id``'s iterate folded into the means, which are written into ``metrics``."""

    if means is not None:
        means.observe(round_id, metrics)


def state_of(means: RunningMeans | None) -> dict[str, Any] | None:
    """What a checkpoint of the round just observed holds; None when the section is off."""

    return None if means is None else means.state()
