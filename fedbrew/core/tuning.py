"""``fedbrew tune``: choose a run's dials by running and scoring a grid, on sweep and analyze.

The ``tuning`` config section names a method, a metric and the dials to tune.
``fedbrew tune --config X`` writes one run config per candidate and seed from
X's own (the dials set, the seed set, nothing else changed), runs them with
``fedbrew sweep`` -- candidates that differ only in numeric hyperparameters run
as one group, their clients trained together -- scores each run from its
``round_metrics.csv`` (``fedbrew/core/analysis.py``), aggregates the scores over
seeds, selects, and extends the grid past an edge until the best is interior.

Two named methods, each a fixed way to score, to tie and to stop:

``grid_and_edge``
    Dials on a grid: a step size on powers of two (``base: 2, exponents: [lo,
    hi]``), any other dial on a stated grid (``values: [...]``). A run's score
    is the **mean of log10 of the metric** over the rounds it is evaluated,
    values clamped below at ``floor`` (default 1e-16) so an exact zero is a
    finite score; a run that did not complete, or whose metric is not finite, is
    worst. The score of a candidate is the median over seeds. Candidates within
    ``tie`` (0.05 decades) of the best are **tied and all reported**; the pick is
    the best score. Where the pick is on an edge of a dial the dial is extended
    one value past it (a ``base`` dial by its base, a stated grid that is
    geometric by its ratio) and every new candidate scored. **Stop rule:** the
    search stops when the pick is interior on every dial; or when an extension
    gains no more than ``tie`` over the pick before it, which is then a tie
    and the earlier pick, closer to the grid's centre, is kept; or when a dial
    at the pick's edge cannot be extended; or after ``max_steps`` extensions.

``pilot``
    A short pilot run: ``rounds`` is the pilot horizon, every dial is powers of
    ten around a stated centre (``centre: c, decades: n`` is ``c * 10^k`` for
    ``k`` in ``-n..n``), and the run turns ``convergence`` on for the metric, so
    its score is the **final exact running mean** of the metric over the
    pilot's iterates (``fedbrew/core/convergence.py``). Candidates within
    ``tie`` (1e-3, relative) of the best are tied and the pick is the tied one
    **closest to the grid's centre** (summed distance in grid steps; the lower
    score breaks a remaining tie). The search is extended past an edge and stops
    on the same four rules. The resolved config restores the full horizon.

Every dial is a dotted path into the config as it is written
(``client.learning_rate``, ``server.beta1``). A tune writes into its output
directory the candidates' configs and runs, ``evidence.csv`` (a row per
candidate and seed), ``selection.json`` (the settings, every step's grid and
tied set, the pick, why the search stopped), ``analysis/`` (the tables of
``fedbrew analyze`` over every run) and ``selected.yaml``, the config with the
chosen values and the full horizon. Run again, it reuses the runs that finished.
"""

from __future__ import annotations

import copy
import csv
import itertools
import json
import math
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from fedbrew.core.analysis import Run, default_direction, find_runs, read_run, write_tables
from fedbrew.core.convergence import running_mean_column
from fedbrew.core.metrics import json_safe
from fedbrew.core.refusal import RunRefused


class TuneError(Exception):
    """A tune that cannot go on and why: the user's to fix. Never a traceback."""


@dataclass(frozen=True)
class Method:
    """How a named method scores, ties, breaks ties and wants its dials written."""

    name: str
    score: str
    tie: float
    tie_kind: str
    tie_break: str
    needs_rounds: bool
    dials: str


METHODS: dict[str, Method] = {
    "grid_and_edge": Method("grid_and_edge", "mean_log10", 0.05, "absolute", "best", False, "grid"),
    "pilot": Method("pilot", "running_mean", 1e-3, "relative", "centre", True, "centre"),
}

AGGREGATES = ("median", "mean")
TIE_KINDS = ("absolute", "relative")
FINISHED = frozenset({"completed", "diverged", "stalled"})
DEFAULT_FLOOR = 1e-16
DEFAULT_MAX_STEPS = 6

#: The keys of ``tuning`` and of each dial; anything else is refused.
SECTION_KEYS = frozenset(
    {
        "method", "metric", "dials", "seeds", "rounds", "aggregate", "tie", "tie_kind",
        "max_steps", "floor", "direction", "output_dir",
    }
)  # fmt: skip
DIAL_KEYS = frozenset({"base", "exponents", "values", "centre", "decades", "extend"})


# ---------------------------------------------------------------------------
# The section
# ---------------------------------------------------------------------------


def validate_section(section: Any) -> None:
    """Refuse a ``tuning`` section that cannot be tuned; a section with no method is off."""

    if section.method is None:
        if section.dials or section.metric:
            raise RunRefused("tuning names dials or a metric but no method; set tuning.method")
        return
    method = METHODS.get(section.method)
    if method is None:
        raise RunRefused(
            f"tuning.method is {section.method!r}; the methods are {', '.join(METHODS)}"
        )
    if not section.metric or not isinstance(section.metric, str):
        raise RunRefused("tuning.metric must name the round column the dials are scored on")
    if not section.dials:
        raise RunRefused("tuning.dials must name at least one dial to tune")
    for key, spec in section.dials.items():
        parse_axis(key, spec, method)
    _validate_choices(section)
    _validate_horizon_and_seeds(section, method)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_choices(section: Any) -> None:
    """The aggregate, the tie, the step cap, the floor and the direction."""

    if section.aggregate not in AGGREGATES:
        raise RunRefused(f"tuning.aggregate must be one of {', '.join(AGGREGATES)}")
    if section.tie is not None and not (
        isinstance(section.tie, int | float)
        and not isinstance(section.tie, bool)
        and section.tie >= 0
    ):
        raise RunRefused("tuning.tie must be a number, 0 or more")
    if section.tie_kind is not None and section.tie_kind not in TIE_KINDS:
        raise RunRefused(f"tuning.tie_kind must be one of {', '.join(TIE_KINDS)}")
    if not _is_int(section.max_steps):
        raise RunRefused("tuning.max_steps must be an integer, 0 or more")
    if section.max_steps < 0:
        raise RunRefused("tuning.max_steps must be 0 or more")
    if not (isinstance(section.floor, int | float) and section.floor > 0):
        raise RunRefused("tuning.floor must be positive")
    if section.direction is not None and section.direction not in {"min", "max"}:
        raise RunRefused("tuning.direction must be min or max")


def _validate_horizon_and_seeds(section: Any, method: Method) -> None:
    """The pilot horizon a method needs, and the seeds."""

    if method.needs_rounds and not (_is_int(section.rounds) and section.rounds > 0):
        raise RunRefused(
            f"tuning.method {method.name} runs a pilot: set tuning.rounds, the pilot horizon"
        )
    if section.rounds is not None and not (_is_int(section.rounds) and section.rounds > 0):
        raise RunRefused("tuning.rounds must be a positive integer")
    seeds = section.seeds
    if not isinstance(seeds, list) or not all(_is_int(s) and s >= 0 for s in seeds):
        raise RunRefused("tuning.seeds must be a list of non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise RunRefused("tuning.seeds repeats a seed")


# ---------------------------------------------------------------------------
# Dials and their grids
# ---------------------------------------------------------------------------


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _decimal(value: float | int) -> Decimal:
    return Decimal(repr(value))


@dataclass
class Axis:
    """One dial's grid: ascending values, and how far past either end it may go.

    ``first`` is the index of ``values[0]`` in the coordinates of the initial
    grid (0 to begin with, -1 after one extension below), so a value's distance
    from the grid's centre is the same number of steps however far the grid grew.
    """

    key: str
    values: list[float | int]
    ratio: float | None
    first: int = 0
    initial: int = field(init=False)

    def __post_init__(self) -> None:
        self.initial = len(self.values)

    @property
    def centre(self) -> float:
        """The centre of the initial grid, in index coordinates (between two for an even grid)."""

        return (self.initial - 1) / 2

    def index(self, value: float | int) -> int:
        return self.first + self.values.index(value)

    def edge(self, value: float | int) -> str | None:
        position = self.values.index(value)
        if position == 0:
            return "low"
        return "high" if position == len(self.values) - 1 else None

    def extend(self, side: str) -> float:
        """Add one value past the ``low`` or ``high`` end and return it."""

        if self.ratio is None:
            raise TuneError(f"{self.key} has a stated grid that is not geometric: not extendable")
        ratio = _decimal(self.ratio)
        if side == "high":
            new = float(_decimal(self.values[-1]) * ratio)
            self.values.append(new)
        else:
            new = float(_decimal(self.values[0]) / ratio)
            self.values.insert(0, new)
            self.first -= 1
        return new


def parse_axis(key: str, spec: Any, method: Method) -> Axis:
    """A dial's spec as its initial grid; ``RunRefused`` for one the method does not take."""

    where = f"tuning.dials.{key}"
    form = _form_of(where, key, spec, method)
    values, ratio = _GRID_FORMS[form](where, spec)
    if "extend" in spec:
        ratio = _extension(where, spec["extend"])
    if len(values) < 3:
        raise RunRefused(f"{where} has {len(values)} values; an interior best needs at least 3")
    return Axis(key=key, values=values, ratio=ratio)


def _form_of(where: str, key: Any, spec: Any, method: Method) -> str:
    """Which of the three grid forms a dial is written in, refusing one the method does not take."""

    if not isinstance(key, str) or not key or key.startswith(".") or ".." in key:
        raise RunRefused(f"tuning.dials key {key!r} must be a dotted path into the config")
    if not isinstance(spec, Mapping):
        raise RunRefused(f"{where} must be a mapping: base and exponents, values, or centre")
    unknown = sorted(set(spec) - DIAL_KEYS)
    if unknown:
        raise RunRefused(f"{where} has unknown keys {unknown}; it may write {sorted(DIAL_KEYS)}")
    forms = [form for form in ("exponents", "values", "centre") if form in spec]
    if len(forms) != 1:
        raise RunRefused(f"{where} must give exactly one of exponents (with base), values, centre")
    form = forms[0]
    if method.dials == "centre" and form != "centre":
        raise RunRefused(
            f"{where}: method {method.name} takes powers of 10 around a stated centre; "
            "write centre (and decades)"
        )
    if method.dials == "grid" and form == "centre":
        raise RunRefused(
            f"{where}: method {method.name} takes a grid; write base and exponents, or values"
        )
    return form


def _extension(where: str, extend: Any) -> float | None:
    """A dial's ``extend``: false is not extendable, a ratio above 1 is its step."""

    if extend is False:
        return None
    if _number(extend) and extend > 1:
        return float(extend)
    raise RunRefused(f"{where}.extend must be false or a ratio above 1")


def _powers(where: str, spec: Mapping[str, Any]) -> tuple[list[float | int], float | None]:
    given = spec.get("base")
    exponents = spec["exponents"]
    if not isinstance(given, int | float) or not _number(given) or given <= 1:
        raise RunRefused(f"{where}.base must be a number above 1")
    base = given
    if not (
        isinstance(exponents, list)
        and len(exponents) == 2
        and all(isinstance(e, int) and not isinstance(e, bool) for e in exponents)
        and exponents[0] <= exponents[1]
    ):
        raise RunRefused(f"{where}.exponents must be [lowest, highest], two integers")
    values = [float(_decimal(base) ** e) for e in range(exponents[0], exponents[1] + 1)]
    return values, float(base)


def _stated(where: str, spec: Mapping[str, Any]) -> tuple[list[float | int], float | None]:
    values = spec["values"]
    if not isinstance(values, list) or not all(_number(v) for v in values):
        raise RunRefused(f"{where}.values must be a list of numbers")
    ordered = sorted(set(values))
    if len(ordered) != len(values):
        raise RunRefused(f"{where}.values repeats a value")
    ratio = None
    if len(ordered) >= 2 and all(v > 0 for v in ordered):
        ratios = [ordered[i + 1] / ordered[i] for i in range(len(ordered) - 1)]
        if all(abs(r - ratios[0]) <= 1e-9 * ratios[0] for r in ratios):
            ratio = ratios[0]
    return list(ordered), ratio


def _around(where: str, spec: Mapping[str, Any]) -> tuple[list[float | int], float | None]:
    centre = spec["centre"]
    decades = spec.get("decades", 1)
    if not _number(centre) or centre <= 0:
        raise RunRefused(f"{where}.centre must be a positive number")
    if not isinstance(decades, int) or isinstance(decades, bool) or decades < 1:
        raise RunRefused(f"{where}.decades must be an integer, 1 or more")
    values = [float(_decimal(centre).scaleb(k)) for k in range(-decades, decades + 1)]
    return values, 10.0


#: Each grid form and how it is built: ``(values, extension ratio)``.
_GRID_FORMS: dict[
    str, Callable[[str, Mapping[str, Any]], tuple[list[float | int], float | None]]
] = {
    "exponents": _powers,
    "values": _stated,
    "centre": _around,
}


# ---------------------------------------------------------------------------
# Scoring, ties and the pick (pure: the tests drive these with hand-made scores)
# ---------------------------------------------------------------------------


def internal(score: float | None, direction: str) -> float:
    """A score as a number to minimise: ``inf`` for a run that could not be scored."""

    if score is None or not math.isfinite(score):
        return math.inf
    return score if direction == "min" else -score


def _scale(reference: float, kind: str) -> float:
    return abs(reference) if kind == "relative" else 1.0


def within(best: float, value: float, tie: float, kind: str) -> bool:
    """Whether ``value`` ties with ``best`` (both to minimise)."""

    return value <= best + tie * _scale(best, kind)


def improves(old: float, new: float, tie: float, kind: str) -> bool:
    """Whether ``new`` beats ``old`` by more than a tie."""

    return new < old - tie * _scale(old, kind)


def aggregate_scores(values: Sequence[float], how: str) -> float:
    """The median or mean over seeds of scores to minimise; ``inf``, a failed run, ranks worst."""

    ordered = sorted(values)
    if not ordered:
        return math.inf
    if how == "mean":
        return (
            math.inf if any(math.isinf(v) for v in ordered) else math.fsum(ordered) / len(ordered)
        )
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    low, high = ordered[middle - 1], ordered[middle]
    if low == high:
        return low
    return low + (high - low) / 2 if math.isfinite(high) else math.inf


def distance_from_centre(axes: Sequence[Axis], point: Sequence[float | int]) -> float:
    """Steps between a candidate and the initial grid's centre, summed over the dials."""

    return math.fsum(
        abs(axis.index(value) - axis.centre) for axis, value in zip(axes, point, strict=True)
    )


def pick(
    scores: Mapping[tuple[Any, ...], float],
    axes: Sequence[Axis],
    method: Method,
    tie: float,
    kind: str,
) -> tuple[tuple[Any, ...], list[tuple[Any, ...]]]:
    """The pick among scored candidates and the candidates that tie with the best.

    ``best`` takes the lowest score; ``centre`` takes the tied candidate closest to the
    grid's centre. Either way the lower score and then the point itself settle a remainder.
    """

    best = min(scores.values())
    tied = [c for c, s in scores.items() if within(best, s, tie, kind)]
    tied.sort(key=lambda c: (scores[c], distance_from_centre(axes, c), c))
    if method.tie_break == "centre":
        tied.sort(key=lambda c: (distance_from_centre(axes, c), scores[c], c))
    return tied[0], tied


# ---------------------------------------------------------------------------
# One tune
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    """What a tune is run with, resolved: the section, its method's defaults filled in."""

    method: Method
    metric: str
    direction: str
    seeds: list[int]
    rounds: int | None
    aggregate: str
    tie: float
    tie_kind: str
    max_steps: int
    floor: float


@dataclass
class RunScore:
    seed: int
    run_dir: Path | None
    status: str | None
    score: float | None
    note: str = ""


@dataclass
class Candidate:
    point: tuple[Any, ...]
    label: str
    step: int
    runs: list[RunScore] = field(default_factory=list)
    aggregate: float = math.inf


@dataclass
class Selection:
    """The outcome of a tune: the pick and everything that led to it."""

    settings: Settings
    keys: list[str]
    selected: Candidate
    tied: list[Candidate]
    candidates: list[Candidate]
    steps: list[dict[str, Any]]
    stop: dict[str, str]
    interior: bool
    initial: dict[str, list[float | int]]
    final: dict[str, list[float | int]]
    out: Path
    config: Path | None = None


def _fmt(value: float | int) -> str:
    return str(value) if isinstance(value, int) else repr(value)


def label_of(keys: Sequence[str], point: Sequence[float | int]) -> str:
    return "+".join(f"{key}={_fmt(value)}" for key, value in zip(keys, point, strict=True))


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=+-]", "_", text)


def set_path(mapping: dict[str, Any], key: str, value: Any) -> None:
    """``mapping[a][b][c] = value`` for the dotted ``key`` a.b.c, creating blocks on the way."""

    parts = key.split(".")
    node = mapping
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise TuneError(f"dial {key}: {part} is not a block in the config")
        node = child
    node[parts[-1]] = value


def _names_column(name: str, column: str) -> bool:
    """Whether a ``convergence.metrics`` entry, bare or prefixed, is the column ``column``."""

    return name == column or f"central_test_{name.removeprefix('central_test_')}" == column


def resolve_metric(config: Any, name: str, method: Method) -> str:
    """The round column ``name`` means for this run, checked against what the run evaluates."""

    from fedbrew.core.config import (
        parse_evaluation_schedule,
        task_grad_norm_gloss,
        task_reported_metrics,
    )
    from fedbrew.core.convergence import resolve_convergence_metrics
    from fedbrew.core.metrics import GRAD_NORM_COLUMN

    central = task_reported_metrics(config).central
    grad_norm = task_grad_norm_gloss(config) is not None
    if method.score == "running_mean":
        try:
            (column,) = resolve_convergence_metrics(
                [name], central=central, grad_norm=grad_norm, task=config.task.name
            )
        except RunRefused as error:
            raise TuneError(f"tuning.metric: {error}") from None
        return column
    column = name
    if name != GRAD_NORM_COLUMN and name.removeprefix("central_test_") in central:
        column = f"central_test_{name.removeprefix('central_test_')}"
    if column == GRAD_NORM_COLUMN:
        asked = parse_evaluation_schedule(config.evaluation.grad_norm.every, "evaluation.grad_norm")
        if asked is None:
            raise TuneError(
                f"tuning.metric is {name}, which the run does not evaluate: set "
                "evaluation.grad_norm.every"
            )
    elif column.startswith("central_test_"):
        asked = parse_evaluation_schedule(
            config.evaluation.central_test.every, "evaluation.central_test"
        )
        if asked is None:
            raise TuneError(
                f"tuning.metric is {name}, which the run does not evaluate: set "
                "evaluation.central_test.every"
            )
    return column


def settings_from(section: Any, config: Any, method: Method, column: str) -> Settings:
    return Settings(
        method=method,
        metric=column,
        direction=section.direction or default_direction(column),
        seeds=list(section.seeds) or [config.experiment.seed],
        rounds=section.rounds,
        aggregate=section.aggregate,
        tie=float(method.tie if section.tie is None else section.tie),
        tie_kind=section.tie_kind or method.tie_kind,
        max_steps=section.max_steps,
        floor=float(section.floor),
    )


def score_run(run: Run, settings: Settings) -> tuple[float | None, str]:
    """A run's score by the method's rule, and why it has none: (None, reason)."""

    if run.status is not None and run.status != "completed":
        return None, f"status {run.status}"
    metric = settings.metric
    if metric not in run.columns:
        raise TuneError(f"{run.path}: round_metrics.csv has no {metric} column to score")
    if settings.method.score == "running_mean":
        held = [v for v in run.columns.get(running_mean_column(metric), []) if v is not None]
        if not held:
            raise TuneError(f"{run.path}: no {running_mean_column(metric)} column (convergence)")
        value = held[-1]
        return (value, "") if math.isfinite(value) else (None, "running mean not finite")
    values = [v for v in run.columns[metric] if v is not None]
    if not values:
        raise TuneError(f"{run.path}: {metric} is never evaluated")
    if not all(math.isfinite(v) for v in values):
        return None, f"{metric} not finite"
    clamped = [max(v, settings.floor) for v in values]
    return math.fsum(math.log10(v) for v in clamped) / len(clamped), ""


def run_sweep(configs: Sequence[Path]) -> int:
    """``fedbrew sweep`` over ``configs`` in a child process; its exit status."""

    command = [sys.executable, "-m", "fedbrew.cli.dispatch", "sweep", *map(str, configs)]
    return subprocess.run(command, check=False).returncode


class Tuner:
    """One tune: the grid, the runs it needs, the scores, the pick, the extensions."""

    def __init__(
        self,
        base: Path,
        base_mapping: dict[str, Any],
        section: Any,
        config: Any,
        out: Path,
        *,
        execute: Callable[[Sequence[Path]], int] = run_sweep,
        log: Callable[[str], None] = print,
    ) -> None:
        method = METHODS[section.method]
        self.base = base
        self.mapping = base_mapping
        self.out = out
        self.execute = execute
        self.log = log
        self.axes = [parse_axis(key, spec, method) for key, spec in section.dials.items()]
        self.keys = [axis.key for axis in self.axes]
        column = resolve_metric(config, section.metric, method)
        self.settings = settings_from(section, config, method, column)
        self.initial = {axis.key: list(axis.values) for axis in self.axes}
        self.name = str((base_mapping.get("experiment") or {}).get("name") or base.stem)
        self.candidates: dict[tuple[Any, ...], Candidate] = {}
        self.steps: list[dict[str, Any]] = []

    # -- configs and runs -------------------------------------------------

    def run_dir(self, label: str, seed: int) -> Path:
        return self.out / "runs" / _safe(label) / f"seed{seed}"

    def config_path(self, label: str, seed: int) -> Path:
        return self.out / "configs" / f"{_safe(label)}-seed{seed}.yaml"

    def mapping_for(self, point: Sequence[float | int], label: str, seed: int) -> dict[str, Any]:
        settings = self.settings
        mapping = copy.deepcopy(self.mapping)
        mapping.pop("tuning", None)
        for key, value in zip(self.keys, point, strict=True):
            set_path(mapping, key, value)
        experiment = dict(mapping.get("experiment") or {})
        experiment.update(
            seed=seed,
            name=_safe(f"{self.name}-{label}-seed{seed}"),
            output_dir=str(self.run_dir(label, seed)),
            use_run_subdir=False,
        )
        mapping["experiment"] = experiment
        if settings.rounds is not None:
            schedule = dict(mapping.get("schedule") or {})
            schedule["rounds"] = settings.rounds
            mapping["schedule"] = schedule
        if settings.method.score == "running_mean":
            convergence = dict(mapping.get("convergence") or {})
            held = list(convergence.get("metrics") or [])
            if not any(_names_column(name, settings.metric) for name in held):
                held.append(settings.metric)
            convergence["metrics"] = held
            mapping["convergence"] = convergence
        return mapping

    def _write_config(self, label: str, seed: int, mapping: dict[str, Any]) -> tuple[Path, bool]:
        """Write the candidate's config; whether it is the config a finished run already ran."""

        path = self.config_path(label, seed)
        text = yaml.safe_dump(mapping, sort_keys=False)
        same = path.is_file() and path.read_text(encoding="utf-8") == text
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        record = self.run_dir(label, seed) / "run.json"
        finished = False
        if same and record.is_file():
            try:
                status = json.loads(record.read_text(encoding="utf-8")).get("status")
            except (OSError, ValueError):
                status = None
            finished = status in FINISHED
        return path, finished

    def points(self) -> list[tuple[Any, ...]]:
        return list(itertools.product(*(axis.values for axis in self.axes)))

    def plan_step(self, step: int) -> list[Candidate]:
        new = []
        for point in self.points():
            if point not in self.candidates:
                candidate = Candidate(point, label_of(self.keys, point), step)
                self.candidates[point] = candidate
                new.append(candidate)
        return new

    def run_candidates(self, new: Sequence[Candidate]) -> None:
        pending: list[Path] = []
        for candidate in new:
            for seed in self.settings.seeds:
                path, finished = self._write_config(
                    candidate.label, seed, self.mapping_for(candidate.point, candidate.label, seed)
                )
                if not finished:
                    pending.append(path)
        reused = len(new) * len(self.settings.seeds) - len(pending)
        self.log(
            f"fedbrew tune: {len(pending)} runs to do"
            + (f", {reused} finished and reused" if reused else "")
        )
        if pending:
            code = self.execute(pending)
            if code == 2:
                raise TuneError("fedbrew sweep refused a run (exit 2): see its message above")
            if code != 0:
                raise TuneError(f"fedbrew sweep failed (exit {code}): see its message above")
        for candidate in new:
            self.score(candidate)

    def score(self, candidate: Candidate) -> None:
        values = []
        for seed in self.settings.seeds:
            directory = self.run_dir(candidate.label, seed)
            runs, _ = find_runs([directory])
            if not runs:
                raise TuneError(f"{candidate.label} seed {seed}: no run was written to {directory}")
            run = read_run(runs[0])
            value, note = score_run(run, self.settings)
            candidate.runs.append(RunScore(seed, runs[0], run.status, value, note))
            values.append(internal(value, self.settings.direction))
        candidate.aggregate = aggregate_scores(values, self.settings.aggregate)

    # -- the search -------------------------------------------------------

    def tune(self) -> Selection:
        settings = self.settings
        tie, kind = settings.tie, settings.tie_kind
        selected: tuple[Any, ...] | None = None
        stop = {"code": "", "reason": ""}
        step = 0
        while True:
            new = self.plan_step(step)
            self.log(f"fedbrew tune: step {step}: {len(new)} new candidates")
            self.run_candidates(new)
            scores = {point: c.aggregate for point, c in self.candidates.items()}
            if not any(math.isfinite(s) for s in scores.values()):
                raise TuneError(
                    "every candidate failed to score (diverged, did not complete, or not finite)"
                )
            challenger, tied = pick(scores, self.axes, settings.method, tie, kind)
            held = selected
            if selected is None or improves(scores[selected], scores[challenger], tie, kind):
                selected = challenger
            record = {
                "step": step,
                "new": [c.label for c in new],
                "grid": {axis.key: list(axis.values) for axis in self.axes},
                "best": self.candidates[min(scores, key=lambda p: scores[p])].label,
                "tied": [self.candidates[p].label for p in tied],
                "selected": self.candidates[selected].label,
                "kept": held is not None and selected == held,
            }
            self.steps.append(record)
            if held is not None and selected == held:
                stop = {
                    "code": "tie",
                    "reason": (
                        f"the extension at step {step} gained no more than the tie "
                        f"({tie:g}, {kind}) over {self.candidates[held].label}: a tie, so the "
                        "pick before it, closer to the grid's centre, is kept"
                    ),
                }
                break
            edges = {
                axis.key: side
                for axis, value in zip(self.axes, selected, strict=True)
                if (side := axis.edge(value)) is not None
            }
            record["edges"] = edges
            if not edges:
                stop = {"code": "interior", "reason": "the pick is interior on every dial"}
                break
            fixed = [axis.key for axis in self.axes if axis.key in edges and axis.ratio is None]
            if fixed:
                stop = {
                    "code": "not_extendable",
                    "reason": f"the pick is on an edge of {', '.join(fixed)}, a stated grid "
                    "that is not geometric and has no extend ratio",
                }
                break
            if step >= settings.max_steps:
                stop = {
                    "code": "max_steps",
                    "reason": f"{settings.max_steps} extensions made and the pick is still on an "
                    f"edge of {', '.join(edges)}",
                }
                break
            for axis in self.axes:
                if axis.key in edges:
                    added = axis.extend(edges[axis.key])
                    self.log(f"fedbrew tune: {axis.key}: extended {edges[axis.key]} to {added!r}")
            step += 1
        return self.finish(selected, scores, tied, stop)

    def finish(
        self,
        selected: tuple[Any, ...],
        scores: Mapping[tuple[Any, ...], float],
        tied: Sequence[tuple[Any, ...]],
        stop: dict[str, str],
    ) -> Selection:
        chosen = self.candidates[selected]
        interior = all(
            axis.edge(value) is None for axis, value in zip(self.axes, selected, strict=True)
        )
        selection = Selection(
            settings=self.settings,
            keys=self.keys,
            selected=chosen,
            tied=[self.candidates[p] for p in tied],
            candidates=list(self.candidates.values()),
            steps=self.steps,
            stop=stop,
            interior=interior,
            initial=self.initial,
            final={axis.key: list(axis.values) for axis in self.axes},
            out=self.out,
        )
        return selection

    # -- the record -------------------------------------------------------

    def write(self, selection: Selection) -> None:
        write_selection(selection, self.base, self.mapping)


def write_selection(selection: Selection, base: Path, mapping: dict[str, Any]) -> None:
    """Write evidence.csv, selection.json, analysis/ and selected.yaml into the tune's directory."""

    out = selection.out
    out.mkdir(parents=True, exist_ok=True)
    settings = selection.settings
    tied_labels = {c.label for c in selection.tied}
    header = [
        "step", "candidate", *selection.keys, "seed", "run", "status", "score", "note",
        "aggregate", "tied", "selected",
    ]  # fmt: skip
    with (out / "evidence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for candidate in selection.candidates:
            sign = 1.0 if settings.direction == "min" else -1.0
            aggregate = "" if math.isinf(candidate.aggregate) else repr(sign * candidate.aggregate)
            for run in candidate.runs:
                writer.writerow(
                    [
                        candidate.step,
                        candidate.label,
                        *(_fmt(v) for v in candidate.point),
                        run.seed,
                        "" if run.run_dir is None else str(run.run_dir),
                        run.status or "",
                        "" if run.score is None else repr(run.score),
                        run.note,
                        aggregate,
                        candidate.label in tied_labels,
                        candidate is selection.selected,
                    ]
                )
    chosen = selection.selected
    resolved = copy.deepcopy(mapping)
    resolved.pop("tuning", None)
    for key, value in zip(selection.keys, chosen.point, strict=True):
        set_path(resolved, key, value)
    resolved_path = out / "selected.yaml"
    header_lines = (
        f"# {base}, with the dials fedbrew tune chose "
        f"({settings.method.name} on {settings.metric}):\n"
        + "".join(
            f"#   {k} = {_fmt(v)}\n" for k, v in zip(selection.keys, chosen.point, strict=True)
        )
        + "# The horizon and everything else are the base config's.\n"
    )
    resolved_path.write_text(
        header_lines + yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    selection.config = resolved_path
    sign = 1.0 if settings.direction == "min" else -1.0
    document = {
        "method": settings.method.name,
        "base_config": str(base),
        "metric": settings.metric,
        "direction": settings.direction,
        "score": settings.method.score,
        "aggregate": settings.aggregate,
        "tie": {"value": settings.tie, "kind": settings.tie_kind},
        "tie_break": settings.method.tie_break,
        "seeds": settings.seeds,
        "rounds": settings.rounds,
        "floor": settings.floor,
        "max_steps": settings.max_steps,
        "dials": {
            key: {"initial": selection.initial[key], "final": selection.final[key]}
            for key in selection.keys
        },
        "steps": selection.steps,
        "stop": selection.stop,
        "selected": {
            "label": chosen.label,
            "values": dict(zip(selection.keys, chosen.point, strict=True)),
            "score": None if math.isinf(chosen.aggregate) else sign * chosen.aggregate,
            "interior": selection.interior,
            "tied_with": sorted(c.label for c in selection.tied if c is not chosen),
            "config": str(resolved_path),
        },
        "candidates": [
            {
                "label": c.label,
                "step": c.step,
                "values": dict(zip(selection.keys, c.point, strict=True)),
                "aggregate": None if math.isinf(c.aggregate) else sign * c.aggregate,
                "runs": [
                    {
                        "seed": r.seed,
                        "run": None if r.run_dir is None else str(r.run_dir),
                        "status": r.status,
                        "score": r.score,
                        "note": r.note,
                    }
                    for r in c.runs
                ],
            }
            for c in selection.candidates
        ],
    }
    (out / "selection.json").write_text(
        json.dumps(json_safe(document), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    runs = [r.run_dir for c in selection.candidates for r in c.runs if r.run_dir is not None]
    if runs:
        from fedbrew.core.analysis import analyze

        try:
            result = analyze(runs, metrics=[settings.metric])
            write_tables(result, out / "analysis")
        except Exception as error:  # noqa: BLE001 - the tables are an extra; the selection stands
            (out / "analysis-error.txt").write_text(f"{error}\n", encoding="utf-8")
