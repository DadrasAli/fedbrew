"""Summaries of finished runs, read from their round_metrics.csv: what ``fedbrew analyze`` writes.

A run's curve says little until it is a number a protocol can compare and a
spread over seeds. This module reads each run's ``round_metrics.csv`` -- and
from the run's ``run.json`` only its config and seed, to group the runs -- and
computes, for each chosen metric ``m``:

- per run: the **last iterate** (``m`` on the last round it is evaluated), the
  **best so far** (the best value over the rounds, by the metric's direction),
  the **mean of log10(m)** over the evaluated rounds, and the **running mean**,
  ``(1/t) sum_{s<=t} m(x_s)``, which is the column ``m_running_mean`` where the
  run wrote one (``convergence.metrics``, exact over every round) and
  otherwise a reconstruction from the evaluated rows only, with a warning;
- across the runs of a group -- the runs whose configs are the same but for
  their seed -- the median, minimum, maximum, the chosen quantiles, and the
  mean with its upward and downward RMS deviations, of each of those, per
  round and at the end.

Evaluated rows. A cell the run left blank (a split not evaluated that round)
is not a value: the last iterate is the last non-blank, the best and the
mean of log10 are over the non-blank, and a reconstruction of the running mean
averages only those. Exactly one thing is then not what it was named for: the
reconstructed mean is over the evaluated rounds, equal to the mean over the
run's iterates only when every round is evaluated; ``running_mean_source``
says which, and a run that reconstructs is warned about.

Statistics. Quantiles interpolate linearly between order statistics (Hyndman
and Fan's type 7, ``numpy.quantile``'s default); the median of an even count
is the mean of the two middle values. Sums are ``math.fsum``.

RMS deviations. With ``m`` the mean of the values, the upward RMS deviation
is the root mean square of ``v - m`` over the values above ``m``, the
downward one that of ``m - v`` over the values below it; 0 where there is no
such value. ``[m - rms_down, m + rms_up]`` is the band they draw: inside
``[min, max]``, so positive wherever every value is, and wider on the side the
seeds spread to.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fedbrew.core.metrics import GRAD_NORM_COLUMN, RUNNING_MEAN_SUFFIX, json_safe

ROUND_METRICS = "round_metrics.csv"
RUN_JSON = "run.json"

#: The metrics analyzed when none is named: those of them a run wrote.
DEFAULT_METRICS = (
    GRAD_NORM_COLUMN,
    "central_test_optimality_gap",
    "central_test_loss",
    "central_test_accuracy",
)

#: What each per-run summary is called, in the order the outputs list them.
STATISTICS = ("last", "best", "mean_log10", "running_mean")

#: The variables of a per-round curve.
CURVES = ("value", "best_so_far", "running_mean")

DEFAULT_QUANTILES = (0.25, 0.75)

#: The across-runs statistics every spread has, before the chosen quantiles.
SPREAD = ("n", "median", "min", "max", "mean", "rms_up", "rms_down")


class AnalysisError(Exception):
    """Something the user must fix: no run found, an unknown metric. Never a traceback."""


@dataclass
class Run:
    """One run's rows and what identifies it."""

    path: Path
    rounds: list[int]
    columns: dict[str, list[float | None]]
    seed: int | None = None
    config: dict[str, Any] | None = None
    status: str | None = None
    name: str = ""

    @property
    def label(self) -> str:
        return self.name or self.path.name


@dataclass
class RunSummary:
    """One run's numbers for one metric."""

    run: str
    group: str
    seed: int | None
    status: str | None
    metric: str
    direction: str
    rounds: int
    evaluated: int
    last_round: int | None
    last: float
    best_round: int | None
    best: float
    mean_log10: float
    nonpositive: int
    running_mean: float
    running_mean_source: str
    curves: dict[str, dict[int, float]] = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def parse_cell(text: str | None) -> float | None:
    """A CSV cell as a float: None for a blank (not evaluated), NaN/inf as they are written."""

    if text is None or text.strip() == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_run(path: Path) -> Run:
    """The run in directory ``path``: its round_metrics.csv, and its config and seed if recorded."""

    table = path / ROUND_METRICS
    with table.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        names = list(reader.fieldnames or [])
        if "round_id" not in names:
            raise ValueError(f"{table} has no round_id column")
        rounds: list[int] = []
        columns: dict[str, list[float | None]] = {name: [] for name in names if name != "round_id"}
        for row in reader:
            rounds.append(int(float(row["round_id"])))
            for name in columns:
                columns[name].append(parse_cell(row.get(name)))
    run = Run(path=path, rounds=rounds, columns=columns)
    record = path / RUN_JSON
    if record.is_file():
        try:
            data = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        config = data.get("config")
        if isinstance(config, dict):
            run.config = config
            seed = config.get("experiment", {}).get("seed")
            run.seed = seed if isinstance(seed, int) and not isinstance(seed, bool) else None
            run.name = str(config.get("experiment", {}).get("name") or "")
        run.status = data.get("status")
    return run


def find_runs(paths: Sequence[str | Path]) -> tuple[list[Path], list[str]]:
    """The run directories ``paths`` name or hold, and why anything named was not one.

    A directory with a ``round_metrics.csv`` is a run; any other directory is
    searched below for them; a config file (``.yaml``) is the run of its
    ``experiment.output_dir`` (or the directories below it, for a config that
    writes one per run).
    """

    found: list[Path] = []
    notes: list[str] = []
    for item in paths:
        path = Path(item)
        if path.suffix in {".yaml", ".yml"} and path.is_file():
            directory = _config_output_dir(path)
            if directory is None or not directory.exists():
                notes.append(f"{path}: its output directory does not exist (nothing was run?)")
                continue
            path = directory
        if not path.exists():
            notes.append(f"{path}: no such file or directory")
            continue
        if (path / ROUND_METRICS).is_file():
            found.append(path)
            continue
        below = sorted(table.parent for table in path.rglob(ROUND_METRICS))
        if not below:
            notes.append(f"{path}: no {ROUND_METRICS} in it or below it")
        found.extend(below)
    seen: set[Path] = set()
    unique = []
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique, notes


def _config_output_dir(path: Path) -> Path | None:
    """The output directory a config file names or infers, without validating the config."""

    from fedbrew.core.config import load_config_mapping
    from fedbrew.core.inferred import infer_experiment
    from fedbrew.core.paths import resolve_output_dir

    try:
        mapping = load_config_mapping(path)
        experiment = dict(mapping.get("experiment") or {})
        infer_experiment(experiment, path, {})
        output = experiment.get("output_dir")
    except Exception:  # noqa: BLE001 - a config that does not load is a note, not a crash
        return None
    return resolve_output_dir(output) if output else None


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        flat: dict[str, Any] = {}
        for key, inner in value.items():
            flat.update(_flatten(inner, f"{prefix}{key}."))
        if not value and prefix:
            flat[prefix[:-1]] = {}
        return flat
    return {prefix[:-1]: value}


def _shown(value: Any) -> str:
    return value if isinstance(value, str) else repr(value)


def group_runs(runs: Sequence[Run]) -> dict[str, list[Run]]:
    """Runs grouped by config, the seed and what only names a run left out; label -> runs.

    The label is the settings that differ between the groups (``key=value``,
    comma separated); with one group, its experiment name; a run without a
    recorded config is its directory's own group.
    """

    from fedbrew.core.runner import _NOT_CONFIGURATION

    ignored = set(_NOT_CONFIGURATION) | {"experiment.seed"}
    keyed: dict[str, list[Run]] = {}
    flats: dict[str, dict[str, Any]] = {}
    for run in runs:
        if run.config is None:
            key = f"no-config:{run.path.parent.resolve()}"
            flat: dict[str, Any] = {}
        else:
            flat = {k: v for k, v in _flatten(run.config).items() if k not in ignored}
            key = repr(sorted((k, repr(v)) for k, v in flat.items()))
        keyed.setdefault(key, []).append(run)
        flats[key] = flat
    if len(keyed) == 1:
        (only,) = keyed.values()
        return {only[0].label if only[0].config is not None else only[0].path.parent.name: only}
    varying = sorted(
        key
        for key in set().union(*flats.values())
        if len({repr(flat.get(key, "<absent>")) for flat in flats.values()}) > 1
    )
    labelled: dict[str, list[Run]] = {}
    for key, members in keyed.items():
        flat = flats[key]
        if members[0].config is None:
            label = members[0].path.parent.name
        else:
            label = ", ".join(f"{name}={_shown(flat.get(name, '<absent>'))}" for name in varying)
        while label in labelled:
            label += "'"
        labelled[label] = members
    return labelled


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def quantile(values: Sequence[float], q: float) -> float:
    """The ``q`` quantile by linear interpolation of the order statistics (Hyndman-Fan type 7)."""

    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def median(values: Sequence[float]) -> float:
    return quantile(values, 0.5)


def spread(values: Sequence[float], quantiles: Sequence[float]) -> dict[str, float]:
    """n, median, min, max, the mean and its RMS deviations, and each quantile (finite values)."""

    finite = [v for v in values if math.isfinite(v)]
    out: dict[str, float] = {
        "n": float(len(finite)),
        "median": median(finite),
        "min": min(finite) if finite else math.nan,
        "max": max(finite) if finite else math.nan,
        **mean_and_rms_deviations(finite),
    }
    for q in quantiles:
        out[quantile_name(q)] = quantile(finite, q)
    return out


def mean_and_rms_deviations(values: Sequence[float]) -> dict[str, float]:
    """The mean, and the RMS deviation of the values above it and of those below it."""

    if not values:
        return {"mean": math.nan, "rms_up": math.nan, "rms_down": math.nan}
    mean = math.fsum(values) / len(values)
    above = [v - mean for v in values if v > mean]
    below = [mean - v for v in values if v < mean]
    return {"mean": mean, "rms_up": _rms(above), "rms_down": _rms(below)}


def _rms(deviations: Sequence[float]) -> float:
    if not deviations:
        return 0.0
    return math.sqrt(math.fsum(d * d for d in deviations) / len(deviations))


def quantile_name(q: float) -> str:
    """``q25`` for 0.25, ``q2.5`` for 0.025: a column name."""

    return f"q{q * 100:g}"


def default_direction(metric: str) -> str:
    """ "max" for an accuracy or an F1, "min" for everything else (losses, gaps, norms)."""

    lowered = metric.lower()
    return "max" if "accuracy" in lowered or "_f1" in lowered else "min"


def summarize_run(
    run: Run, group: str, metric: str, direction: str
) -> tuple[RunSummary, list[str]]:
    """One run's numbers for ``metric``, and the warnings reading it raised."""

    warnings: list[str] = []
    values = run.columns[metric]
    evaluated = [(r, v) for r, v in zip(run.rounds, values, strict=True) if v is not None]
    finite = [(r, v) for r, v in evaluated if math.isfinite(v)]
    last_round, last = evaluated[-1] if evaluated else (None, math.nan)
    pick = min if direction == "min" else max
    best_round, best = pick(finite, key=lambda item: item[1]) if finite else (None, math.nan)
    positive = [v for _, v in finite if v > 0]
    nonpositive = len(finite) - len(positive)
    if nonpositive:
        warnings.append(
            f"{run.label} {metric}: {nonpositive} evaluated rounds are not positive and are "
            "left out of the mean of log10"
        )
    mean_log10 = (
        math.fsum(math.log10(v) for v in positive) / len(positive) if positive else math.nan
    )

    mean_column = f"{metric}{RUNNING_MEAN_SUFFIX}"
    curves: dict[str, dict[int, float]] = {"value": {}, "best_so_far": {}, "running_mean": {}}
    curves["value"] = dict(evaluated)
    best_so_far = math.nan
    for r, v in finite:
        best_so_far = v if math.isnan(best_so_far) else pick(best_so_far, v)
        curves["best_so_far"][r] = best_so_far
    if mean_column in run.columns:
        source = "column"
        held = [
            (r, v)
            for r, v in zip(run.rounds, run.columns[mean_column], strict=True)
            if v is not None
        ]
        curves["running_mean"] = dict(held)
        running = held[-1][1] if held else math.nan
    else:
        source = "reconstructed"
        total = 0.0
        count = 0
        partial: list[float] = []
        reconstructed: dict[int, float] = {}
        bad = False
        for r, v in evaluated:
            if not math.isfinite(v):
                bad = True
            partial.append(v)
            count += 1
            total = math.nan if bad else math.fsum(partial) / count
            reconstructed[r] = total
        curves["running_mean"] = reconstructed
        running = reconstructed[evaluated[-1][0]] if evaluated else math.nan
        if len(evaluated) < len(run.rounds):
            warnings.append(
                f"{run.label} {metric}: no {mean_column} column, so the running mean is "
                f"reconstructed from the {len(evaluated)} of {len(run.rounds)} rounds evaluated: "
                "it is the mean over the evaluated rounds, not over every iterate"
            )
    summary = RunSummary(
        run=run.label,
        group=group,
        seed=run.seed,
        status=run.status,
        metric=metric,
        direction=direction,
        rounds=len(run.rounds),
        evaluated=len(evaluated),
        last_round=last_round,
        last=last,
        best_round=best_round,
        best=best,
        mean_log10=mean_log10,
        nonpositive=nonpositive,
        running_mean=running,
        running_mean_source=source,
        curves=curves,
    )
    return summary, warnings


@dataclass
class Analysis:
    """Everything ``analyze`` computed; ``rows`` are the tidy tables, ready to write."""

    metrics: list[str]
    quantiles: list[float]
    directions: dict[str, str]
    runs: list[RunSummary]
    groups: dict[str, list[Run]]
    warnings: list[str]
    notes: list[str]

    def run_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "group": s.group,
                "run": s.run,
                "seed": s.seed,
                "status": s.status,
                "metric": s.metric,
                "direction": s.direction,
                "rounds": s.rounds,
                "evaluated": s.evaluated,
                "last_round": s.last_round,
                "last": s.last,
                "best_round": s.best_round,
                "best": s.best,
                "mean_log10": s.mean_log10,
                "nonpositive": s.nonpositive,
                "running_mean": s.running_mean,
                "running_mean_source": s.running_mean_source,
            }
            for s in self.runs
        ]

    def group_rows(self) -> list[dict[str, Any]]:
        """Per group, metric and statistic: the spread across the group's runs."""

        rows: list[dict[str, Any]] = []
        for group in self.groups:
            for metric in self.metrics:
                members = [s for s in self.runs if s.group == group and s.metric == metric]
                if not members:
                    continue
                for statistic in STATISTICS:
                    values = [float(getattr(s, statistic)) for s in members]
                    rows.append(
                        {
                            "group": group,
                            "metric": metric,
                            "statistic": statistic,
                            **spread(values, self.quantiles),
                        }
                    )
        return rows

    def curve_rows(self) -> list[dict[str, Any]]:
        """Per group, metric, curve and round: the spread across the runs that have that round."""

        rows: list[dict[str, Any]] = []
        for group in self.groups:
            for metric in self.metrics:
                members = [s for s in self.runs if s.group == group and s.metric == metric]
                if not members:
                    continue
                for variable in CURVES:
                    rounds = sorted({r for s in members for r in s.curves[variable]})
                    for round_id in rounds:
                        values = [
                            s.curves[variable][round_id]
                            for s in members
                            if round_id in s.curves[variable]
                        ]
                        rows.append(
                            {
                                "group": group,
                                "metric": metric,
                                "variable": variable,
                                "round_id": round_id,
                                **spread(values, self.quantiles),
                            }
                        )
        return rows


def analyze(
    paths: Sequence[str | Path],
    *,
    metrics: Sequence[str] | None = None,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    directions: Mapping[str, str] | None = None,
) -> Analysis:
    """Read the runs ``paths`` name and summarize them; ``AnalysisError`` if there is nothing."""

    for q in quantiles:
        if not 0.0 <= q <= 1.0:
            raise AnalysisError(f"a quantile must lie in [0, 1], got {q}")
    runs, notes = _read_runs(paths)
    chosen = resolve_metrics(runs, metrics)
    chosen_directions = _directions_of(chosen, directions)
    groups = group_runs(runs)
    summaries, warnings = _summaries(groups, chosen, chosen_directions)
    return Analysis(
        metrics=chosen,
        quantiles=list(quantiles),
        directions=chosen_directions,
        runs=summaries,
        groups=groups,
        warnings=_unique(warnings),
        notes=notes,
    )


def _read_runs(paths: Sequence[str | Path]) -> tuple[list[Run], list[str]]:
    """Every run ``paths`` name or hold, and a note for each that was not one or not read."""

    directories, notes = find_runs(paths)
    runs: list[Run] = []
    for directory in directories:
        try:
            runs.append(read_run(directory))
        except (OSError, ValueError, csv.Error) as error:
            notes.append(f"{directory}: not read ({error})")
    if not runs:
        raise AnalysisError("no run to analyze. " + "; ".join(notes))
    return runs, notes


def _directions_of(metrics: Sequence[str], directions: Mapping[str, str] | None) -> dict[str, str]:
    chosen = {
        metric: (directions or {}).get(metric, default_direction(metric)) for metric in metrics
    }
    for metric, direction in chosen.items():
        if direction not in {"min", "max"}:
            raise AnalysisError(f"the direction of {metric} must be min or max, got {direction!r}")
    return chosen


def _summaries(
    groups: Mapping[str, list[Run]], metrics: Sequence[str], directions: Mapping[str, str]
) -> tuple[list[RunSummary], list[str]]:
    """Each run's summary per metric, the runs of a group in seed order, and the warnings."""

    summaries: list[RunSummary] = []
    warnings: list[str] = []
    for group, members in groups.items():
        for run in sorted(members, key=lambda r: (r.seed is None, r.seed, str(r.path))):
            if run.config is None:
                warnings.append(
                    f"{run.path}: no {RUN_JSON} with a config, so its group is its parent directory"
                )
            for metric in metrics:
                if metric not in run.columns:
                    warnings.append(f"{run.label}: has no {metric} column")
                    continue
                summary, found = summarize_run(run, group, metric, directions[metric])
                summaries.append(summary)
                warnings.extend(found)
    return summaries, warnings


def _unique(items: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return list(seen)


def resolve_metrics(runs: Sequence[Run], names: Sequence[str] | None) -> list[str]:
    """The metrics to analyze: those named (``central_test_`` may be left off), else defaults."""

    present = {
        name
        for run in runs
        for name, values in run.columns.items()
        if any(v is not None for v in values)
    }
    if not names:
        chosen = [name for name in DEFAULT_METRICS if name in present]
        if not chosen:
            raise AnalysisError(
                "none of the default metrics "
                f"({', '.join(DEFAULT_METRICS)}) is in these runs; name one with --metrics"
            )
        return chosen
    resolved = []
    for name in names:
        column = (
            name
            if name in present
            else f"central_test_{name}"
            if f"central_test_{name}" in present
            else None
        )
        if column is None:
            raise AnalysisError(
                f"no run has a {name!r} column with values. "
                f"Columns with values: {', '.join(sorted(present - {'round_id'}))}"
            )
        if column not in resolved:
            resolved.append(column)
    return resolved


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

#: The files ``write_tables`` writes, in ``out``.
OUTPUT_FILES = ("runs.csv", "groups.csv", "curves.csv", "analysis.json")


def _cell(value: Any) -> Any:
    """A value as a CSV cell: blank for None and NaN, which a figure tool reads as missing."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float):
        return repr(value)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], header: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _cell(row.get(name)) for name in header})


def write_tables(result: Analysis, out: str | Path) -> list[Path]:
    """Write ``runs.csv``, ``groups.csv``, ``curves.csv`` and ``analysis.json`` into ``out``.

    Tidy: one observation per row, the group, metric and statistic or curve
    in columns of their own, a blank cell where a value does not exist.
    ``analysis.json`` holds the same tables with the settings, the groups'
    members and every warning.
    """

    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    spread_names = [*SPREAD, *(quantile_name(q) for q in result.quantiles)]
    run_rows, group_rows, curve_rows = result.run_rows(), result.group_rows(), result.curve_rows()
    _write_csv(
        directory / "runs.csv",
        run_rows,
        [
            "group", "run", "seed", "status", "metric", "direction", "rounds", "evaluated",
            "last_round", "last", "best_round", "best", "mean_log10", "nonpositive",
            "running_mean", "running_mean_source",
        ],
    )  # fmt: skip
    _write_csv(
        directory / "groups.csv", group_rows, ["group", "metric", "statistic", *spread_names]
    )
    _write_csv(
        directory / "curves.csv",
        curve_rows,
        ["group", "metric", "variable", "round_id", *spread_names],
    )
    document = {
        "settings": {
            "metrics": result.metrics,
            "directions": result.directions,
            "quantiles": result.quantiles,
        },
        "groups": {
            group: [
                {"run": run.label, "path": str(run.path), "seed": run.seed, "status": run.status}
                for run in members
            ]
            for group, members in result.groups.items()
        },
        "runs": run_rows,
        "group_statistics": group_rows,
        "curves": curve_rows,
        "warnings": result.warnings,
        "notes": result.notes,
    }
    (directory / "analysis.json").write_text(
        json.dumps(json_safe(document), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return [directory / name for name in OUTPUT_FILES]
