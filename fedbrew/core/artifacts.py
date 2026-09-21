"""Experiment artifact persistence helpers."""

from __future__ import annotations

import csv
import io
import json
import os
import statistics
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fedbrew.core.config import FullConfig
from fedbrew.core.metrics import json_safe
from fedbrew.core.state import (
    ClientEvaluationRecord,
    ClientHistorySummary,
    ClientMetricRecord,
    MetricRecord,
    RoundTimings,
)

# client_metrics and client_update_metrics are CSV-only: the .jsonl twins held
# byte-identical records in a larger format, and results.json held a third copy
# of all of it. The readers for both still exist so older runs stay loadable.
DEFAULT_ARTIFACT_FILES = [
    "round_metrics.csv",
    "client_metrics.csv",
    "client_update_metrics.csv",
    "run.json",
]

_CLIENT_EVALUATION_FIELDS = [
    "round_id",
    "client_id",
    "participated",
    "train_num_examples",
    "val_num_examples",
    "test_num_examples",
    "global_model_train_loss",
    "global_model_val_loss",
    "global_model_test_loss",
    "global_model_train_accuracy",
    "global_model_val_accuracy",
    "global_model_test_accuracy",
    # Appended, not inserted, so no existing column shifts position. The
    # global_model_* names predate evaluation.model_scope; this says which
    # model they actually describe.
    "model_scope",
]

# CSV column name -> RoundTimings attribute. Appended after the metric columns
# so that adding timings never shifts an existing column's position.
_ROUND_TIMING_FIELDS = {
    "duration_sec": "total",
    "fit_sec": "fit",
    "aggregate_sec": "aggregate",
    "client_eval_sec": "client_eval",
    "global_eval_sec": "global_eval",
    "checkpoint_sec": "checkpoint",
}


def prepare_output_dir(output_dir: str | Path) -> Path:
    """Create and return the experiment output directory."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


@contextmanager
def _atomic_text_writer(path: Path) -> Iterator[Any]:
    """Yield a file handle for `path`, renamed into place only on clean exit.

    The CSVs and run.json are rewritten at the end of every round, so a process
    killed mid-write is no longer a rare event -- it is how a preempted or
    timed-out run normally ends. A plain open("w") truncates first, so a kill
    inside that window leaves a half-written file that a later resume cannot
    replay from. Writing to a sibling ".tmp" and renaming makes the visible file
    always either the previous complete version or the new complete one. The
    checkpoints get the same treatment in checkpointing._save_atomically.
    """

    temp_path = path.with_name(path.name + ".tmp")
    file = temp_path.open("w", encoding="utf-8", newline="")
    try:
        yield file
        file.flush()
        os.fsync(file.fileno())
        file.close()
    except BaseException:
        file.close()
        temp_path.unlink(missing_ok=True)
        raise
    os.replace(temp_path, path)


def save_round_metrics_csv(
    history: list[MetricRecord],
    output_dir: str | Path,
) -> Path:
    """Write flat round-level metrics for plotting and analysis."""

    metrics_path = prepare_output_dir(output_dir) / "round_metrics.csv"
    metric_names = _round_metric_names(history)
    fieldnames = [
        "round_id",
        "num_clients",
        "num_examples",
        *metric_names,
        *_ROUND_TIMING_FIELDS,
    ]
    with _atomic_text_writer(metrics_path) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            row: dict[str, object] = {
                "round_id": record.round_id,
                "num_clients": record.num_clients,
                "num_examples": record.num_examples,
            }
            row.update({name: record.metrics.get(name, "") for name in metric_names})
            # Rounds replayed from an artifact written before timings existed
            # leave these blank rather than claiming a zero-second round.
            row.update(
                {
                    column: (
                        ""
                        if record.timings is None
                        else round(getattr(record.timings, attribute), 4)
                    )
                    for column, attribute in _ROUND_TIMING_FIELDS.items()
                }
            )
            writer.writerow(row)
    return metrics_path


def save_client_metrics_csv(
    client_history: list[ClientEvaluationRecord],
    output_dir: str | Path,
) -> Path:
    """Write one fixed-schema post-aggregation row per client and round."""

    metrics_path = prepare_output_dir(output_dir) / "client_metrics.csv"
    with _atomic_text_writer(metrics_path) as file:
        writer = csv.DictWriter(file, fieldnames=_CLIENT_EVALUATION_FIELDS)
        writer.writeheader()
        for record in client_history:
            writer.writerow(_client_evaluation_payload(record))
    return metrics_path


def save_client_update_metrics_csv(
    client_history: list[ClientMetricRecord],
    output_dir: str | Path,
) -> Path:
    """Write flat selected-client fit/update diagnostics for analysis."""

    metrics_path = prepare_output_dir(output_dir) / "client_update_metrics.csv"
    metric_names = _client_update_metric_names(client_history)
    fieldnames = ["round_id", "client_id", "phase", "num_examples", *metric_names]
    with _atomic_text_writer(metrics_path) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in client_history:
            writer.writerow(_client_update_row(record, metric_names))
    return metrics_path


def _client_update_row(
    record: ClientMetricRecord,
    metric_names: Sequence[str],
) -> dict[str, object]:
    row: dict[str, object] = {
        "round_id": record.round_id,
        "client_id": record.client_id,
        "phase": record.phase,
        "num_examples": record.num_examples,
    }
    row.update({name: record.metrics.get(name, "") for name in metric_names})
    return row


def _append_csv_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    """Append rows to an existing CSV in one buffered write, then fsync.

    One write() rather than one per row: the exposure to a kill is the width
    of that single call, and these files are appended to every round of every
    run. It is a smaller window than the full rewrite it replaces, which spent
    a large part of each round streaming the whole history into a temp file --
    but unlike the rename, it is not atomic, so the last row can be short if
    the process dies inside it. The two readers drop an unparseable final row
    for exactly that reason.
    """

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames))
    for row in rows:
        writer.writerow(row)
    payload = buffer.getvalue()
    if not payload:
        return
    with path.open("a", encoding="utf-8", newline="") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())


def _csv_header(path: Path) -> list[str] | None:
    """The header of an existing CSV, or None when there is nothing usable."""

    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as file:
        header = next(csv.reader(file), None)
    return header or None


def retired_metric_columns(output_dir: str | Path | None) -> dict[str, list[str]]:
    """The run's CSVs whose header carries a retired metric name, and which.

    A resume replays these files into its history and writes the resumed
    rounds under the current names, so a file with a retired column would end
    up holding both spellings of one quantity. ``RETIRED_METRIC_NAMES``,
    FINDINGS.csv POST-F14.
    """

    from fedbrew.core.metrics import RETIRED_METRIC_NAMES

    if output_dir is None:
        return {}
    found: dict[str, list[str]] = {}
    for name in ("round_metrics.csv", "client_metrics.csv", "client_update_metrics.csv"):
        header = _csv_header(Path(output_dir) / name)
        retired = [column for column in header or [] if column in RETIRED_METRIC_NAMES]
        if retired:
            found[name] = retired
    return found


def flush_client_csvs(
    client_history: list[ClientEvaluationRecord],
    client_update_history: list[ClientMetricRecord],
    output_dir: str | Path,
    cursor: dict[str, Any],
) -> None:
    """Bring the two per-client CSVs up to date by appending the new rows.

    These files are one row per client per round: 1.8M rows over a 500-round
    FEMNIST run at clients "all". Rewriting all of that on every round -- which
    is what happened, because the gate keyed off "a checkpoint was written" and
    save_last makes that true every round -- makes the bytes written grow with
    the square of the round count, for files that grow linearly.

    Throttling the rewrite instead is not an option: a resume rewinds to
    latest.pt, save_last writes latest.pt every round, and the resumed run
    rebuilds its per-client history from these files. Any round the CSV lagged
    behind would lose its rows permanently. So the files have to be current
    every round, and the only way to make that affordable is to write each
    round's rows once instead of rewriting the run each time.

    ``cursor`` carries how many records of each history are already on disk,
    and the column set the update file's header was written with. A full
    rewrite still happens whenever appending cannot be trusted: no file yet, a
    history that shrank (a resume replaced it), or a new metric name widening
    the schema.
    """

    directory = prepare_output_dir(output_dir)
    evaluation_path = directory / "client_metrics.csv"
    written = int(cursor.get("client_metrics", 0))
    if written > len(client_history) or _csv_header(evaluation_path) != _CLIENT_EVALUATION_FIELDS:
        save_client_metrics_csv(client_history, directory)
    else:
        _append_csv_rows(
            evaluation_path,
            _CLIENT_EVALUATION_FIELDS,
            (_client_evaluation_payload(record) for record in client_history[written:]),
        )
    cursor["client_metrics"] = len(client_history)

    update_path = directory / "client_update_metrics.csv"
    metric_names = _client_update_metric_names(client_update_history)
    fieldnames = ["round_id", "client_id", "phase", "num_examples", *metric_names]
    written = int(cursor.get("client_update_metrics", 0))
    if (
        written > len(client_update_history)
        or cursor.get("client_update_fields") != fieldnames
        or _csv_header(update_path) != fieldnames
    ):
        save_client_update_metrics_csv(client_update_history, directory)
    else:
        _append_csv_rows(
            update_path,
            fieldnames,
            (
                _client_update_row(record, metric_names)
                for record in client_update_history[written:]
            ),
        )
    cursor["client_update_metrics"] = len(client_update_history)
    cursor["client_update_fields"] = fieldnames


def flush_round_artifacts(
    history: list[MetricRecord],
    client_history: list[ClientEvaluationRecord],
    client_update_history: list[ClientMetricRecord],
    output_dir: str | Path | None,
    per_client_csv: bool,
    cursor: dict[str, Any] | None = None,
) -> None:
    """Persist the metric CSVs mid-run, so a killed job keeps what it computed.

    Checkpoints are written inside the round loop but the CSVs used to be
    written once, after the last round returned. A scancel, a wall-clock
    timeout or a preemption therefore left `checkpoints/latest.pt` at round N
    and no round_metrics.csv at all, which is exactly the history a later
    --resume-latest needs to replay. Calling this every round closes that gap:
    the file on disk always covers every round that finished.

    round_metrics.csv is one row per round, so it is rewritten in full every
    time. The per-client CSVs are 1.8M rows over a 500-round FEMNIST run at
    clients "all", so they are appended to instead -- see flush_client_csvs.
    They used to be rewritten in full on rounds that "wrote a checkpoint",
    which save_last makes every round, so turning per_client_csv on meant
    rewriting every earlier round's rows on every round.

    ``cursor`` is the per-run bookkeeping flush_client_csvs needs to know what
    is already on disk. Without one, the per-client files are rewritten in
    full, which is what a caller outside the round loop wants.
    """

    if output_dir is None:
        return
    save_round_metrics_csv(history, output_dir)
    if not per_client_csv:
        return
    if cursor is None:
        save_client_metrics_csv(client_history, output_dir)
        save_client_update_metrics_csv(client_update_history, output_dir)
        return
    flush_client_csvs(client_history, client_update_history, output_dir, cursor)


def round_metrics_gap(
    output_dir: str | Path | None,
    through_round: int,
) -> str | None:
    """Describe why round_metrics.csv cannot back a resume, or None if it can.

    A resume replays every recorded round before the checkpoint's, so the CSV
    has to hold rounds 1..through_round with nothing missing. Rows after that
    are fine -- they belong to work the checkpoint predates and are dropped.
    """

    if output_dir is None:
        return None

    path = Path(output_dir) / "round_metrics.csv"
    if not path.is_file():
        return f"{path.name} is missing"

    try:
        recorded = {record.round_id for record in load_round_metrics_csv(output_dir)}
    except ValueError as exc:
        return f"{path.name} could not be parsed ({exc})"

    missing = sorted(set(range(1, through_round + 1)) - recorded)
    if not missing:
        return None
    return (
        f"{path.name} is missing {len(missing)} of rounds 1-{through_round} "
        f"({_summarize_round_gaps(missing)})"
    )


def _summarize_round_gaps(missing: list[int]) -> str:
    """Render missing round ids as ranges, so the message stays one line."""

    ranges: list[str] = []
    start = previous = missing[0]
    for round_id in missing[1:]:
        if round_id == previous + 1:
            previous = round_id
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = round_id
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    if len(ranges) > 5:
        return ", ".join(ranges[:5]) + f", ... (+{len(ranges) - 5} more)"
    return ", ".join(ranges)


#: Everything one attempt owns inside its output directory: what the flush
#: above writes, and whose killed writes clear_stale_temp_files sweeps.
_RUN_ARTIFACTS = (
    "checkpoints",
    "round_metrics.csv",
    "client_metrics.csv",
    "client_update_metrics.csv",
    "run.json",
)


def clear_stale_temp_files(output_dir: str | Path | None) -> None:
    """Remove ".tmp" files a previously killed write left behind.

    _atomic_text_writer reuses one temp name per artifact, so a run that keeps
    writing the same file cleans up after itself. What lingers is the temp file
    of an artifact this run no longer writes -- per-client CSVs after the flag
    was turned off, say -- and a checkpoint's: each numbered round has its own
    name, so a round_NNN.pt.tmp a kill left behind is never written again.
    Swept once at the start of a run, when nothing is holding one open.
    """

    if output_dir is None:
        return
    directory = Path(output_dir)
    if not directory.is_dir():
        return
    for name in _RUN_ARTIFACTS:
        (directory / f"{name}.tmp").unlink(missing_ok=True)
    checkpoints = directory / "checkpoints"
    if checkpoints.is_dir():
        for stale in checkpoints.glob("*.pt.tmp"):
            stale.unlink(missing_ok=True)


def load_round_metrics_csv(output_dir: str | Path) -> list[MetricRecord]:
    """Load round history from the CSV a previous run wrote.

    This is the only round-level format now. A metric column is blank on rounds
    where its split was not scheduled, and blanks are dropped rather than read
    as zero -- "not measured" and "measured as 0.0" are different facts.
    """

    path = Path(output_dir) / "round_metrics.csv"
    if not path.is_file():
        return []

    records: list[MetricRecord] = []
    timing_columns = dict(_ROUND_TIMING_FIELDS)
    with path.open("r", encoding="utf-8", newline="") as file:
        for line_number, row in enumerate(csv.DictReader(file), start=2):
            try:
                metrics = {
                    name: float(value)
                    for name, value in row.items()
                    if name not in {"round_id", "num_clients", "num_examples", *timing_columns}
                    and value not in (None, "")
                }
                timings = {
                    attribute: float(row[column] or 0.0)
                    for column, attribute in timing_columns.items()
                    if column in row
                }
                records.append(
                    MetricRecord(
                        round_id=int(row["round_id"]),
                        metrics=metrics,
                        num_clients=int(row["num_clients"] or 0),
                        num_examples=int(row["num_examples"] or 0),
                        timings=RoundTimings(**timings) if timings else None,
                    )
                )
            except Exception as exc:
                raise ValueError(f"Could not parse {path}:{line_number}: {exc}") from exc
    return records


def _optional_csv_float(value: str | None) -> float | None:
    """Parse a CSV cell that is blank when the metric was not produced."""

    if value is None or value == "":
        return None
    return float(value)


def _row_is_short(row: Mapping[str, Any]) -> bool:
    """Whether a CSV row has fewer fields than its header.

    csv.DictReader fills a short row's missing columns with None rather than
    failing, and every reader below coerces with `or 0` or str(), so a torn
    row parsed as a record of zeros with client_id "None" -- a silently wrong
    row, not a loud one. Detecting it has to be explicit.
    """

    return any(value is None for value in row.values())


def _refuse_or_drop_short_row(
    path: Path,
    line_number: int,
    total_rows: int,
) -> bool:
    """Drop an incomplete final row; refuse an incomplete row anywhere else.

    These files are appended to once per round, in one buffered write, so the
    only corruption an interrupted run can leave is a short final line.
    Dropping it costs that round's rows -- work a resume redoes anyway --
    whereas raising refuses the whole resume, because the resume path will not
    rewrite a history it could not read. A short row earlier in the file is
    real corruption and still raises.
    """

    if line_number != total_rows + 1:
        raise ValueError(
            f"{path}:{line_number} has fewer fields than its header. Only the "
            "last row of these files can be a torn write; a short row before "
            "it means the file is corrupt."
        )
    print(
        f"Warning: dropping the incomplete last row of {path}. A row is "
        "written per client per round and the file is appended to each round, "
        "so a short final line is how an interrupted run ends. Every earlier "
        "row is intact.",
        flush=True,
    )
    return True


def load_client_metrics_csv(
    output_dir: str | Path,
) -> list[ClientEvaluationRecord]:
    """Load client evaluations from the CSV written by the run.

    CSV is the only format these records are written in. The reader exists
    because resume replays completed rounds out of it: without it, a requeued
    run rebuilds no client history and rewrites the file with post-resume
    rounds only, silently truncating everything the first attempt produced.
    """

    path = Path(output_dir) / "client_metrics.csv"
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    records: list[ClientEvaluationRecord] = []
    for line_number, row in enumerate(rows, start=2):
        if _row_is_short(row):
            _refuse_or_drop_short_row(path, line_number, len(rows))
            break
        try:
            records.append(
                ClientEvaluationRecord(
                    round_id=int(row["round_id"]),
                    client_id=str(row["client_id"]),
                    participated=str(row["participated"]).strip().lower() in {"true", "1"},
                    train_num_examples=int(row["train_num_examples"] or 0),
                    test_num_examples=int(row["test_num_examples"] or 0),
                    # .get, not []: runs written before validation existed
                    # have no val columns and must stay loadable on resume.
                    val_num_examples=int(row.get("val_num_examples") or 0),
                    global_model_train_loss=_optional_csv_float(row.get("global_model_train_loss")),
                    global_model_train_accuracy=_optional_csv_float(
                        row.get("global_model_train_accuracy")
                    ),
                    global_model_test_loss=_optional_csv_float(row.get("global_model_test_loss")),
                    global_model_test_accuracy=_optional_csv_float(
                        row.get("global_model_test_accuracy")
                    ),
                    global_model_val_loss=_optional_csv_float(row.get("global_model_val_loss")),
                    global_model_val_accuracy=_optional_csv_float(
                        row.get("global_model_val_accuracy")
                    ),
                )
            )
        except Exception as exc:
            raise ValueError(f"Could not parse {path}:{line_number}: {exc}") from exc
    return sorted(records, key=lambda record: (record.round_id, record.client_id))


def load_client_update_metrics_csv(
    output_dir: str | Path,
) -> list[ClientMetricRecord]:
    """Load client fit/update diagnostics from the CSV written by the run."""

    path = Path(output_dir) / "client_update_metrics.csv"
    if not path.exists():
        return []

    fixed = {"round_id", "client_id", "phase", "num_examples"}
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    records: list[ClientMetricRecord] = []
    for line_number, row in enumerate(rows, start=2):
        if _row_is_short(row):
            _refuse_or_drop_short_row(path, line_number, len(rows))
            break
        try:
            metrics = {
                name: float(value)
                for name, value in row.items()
                if name not in fixed and value not in (None, "")
            }
            records.append(
                ClientMetricRecord(
                    round_id=int(row["round_id"]),
                    client_id=str(row["client_id"]),
                    phase=str(row["phase"]),
                    num_examples=int(row["num_examples"] or 0),
                    metrics=metrics,
                )
            )
        except Exception as exc:
            raise ValueError(f"Could not parse {path}:{line_number}: {exc}") from exc
    # File order, not sorted: the CSV is written in history order (selection
    # order within each round), and the .jsonl reader this replaces preserved
    # it too. Sorting here would silently reorder a resumed run's rows.
    return records


def append_run_index(
    metadata: Mapping[str, Any],
    root_output_dir: str | Path = "outputs",
) -> Path:
    """Append one compact run record to the global runs index."""

    root = Path(root_output_dir)
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "runs_index.jsonl"
    code_state = metadata.get("code_state", {})
    if not isinstance(code_state, Mapping):
        code_state = {}
    compact = {
        "run_id": metadata.get("run_id"),
        "experiment_name": metadata.get("experiment_name"),
        "seed": metadata.get("seed"),
        "output_dir": metadata.get("output_dir"),
        "created_at": metadata.get("created_at"),
        "tags": metadata.get("tags", []),
        "git_commit": code_state.get("git_commit"),
        "git_dirty": code_state.get("git_dirty"),
        # Which of capture_code_state's three answers this row carries. Without
        # it a null git_dirty beside a real commit is unreadable: it means "an
        # archive, so there was nothing to diff" and cannot be told from a
        # lookup that half-failed.
        "commit_source": code_state.get("commit_source"),
        # The sweep-level record of what was tried and what survived: this file
        # is what answers "which hyperparameters diverged" months later.
        "status": metadata.get("status", "completed"),
        "stopped_round": metadata.get("stopped_round"),
        "termination": metadata.get("termination") or None,
        "final_metrics": metadata.get("final_metrics", {}),
    }
    # Same reason save_run_json sanitises: a diverged run's final_metrics and
    # termination.value are inf or nan, and json.dumps would write the bare
    # tokens NaN / Infinity, which RFC 8259 does not define. Python's json and
    # pandas.read_json read them back and jq 1.6 rewrites them; Go, serde_json
    # and JSON.parse refuse them, and this file is the sweep-level record meant
    # to outlive the run.
    # allow_nan=False makes a field that ever skips json_safe fail loudly
    # instead of writing invalid JSON.
    with index_path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                json_safe(compact),
                sort_keys=True,
                default=str,
                allow_nan=False,
            )
            + "\n"
        )
    return index_path


def save_run_json(
    history: list[MetricRecord],
    output_dir: str | Path,
    config: FullConfig,
    client_history: list[ClientEvaluationRecord] | None = None,
    artifact_files: list[str] | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    client_update_history: list[ClientMetricRecord] | None = None,
) -> Path:
    """Write run.json: results, setup, config and provenance in one file.

    Replaces summary.json + resolved_config.json + run_metadata.json, which
    were three files nobody read separately. Section order is deliberate and
    `sort_keys` is off: identity and results first, the full config and the
    reproducibility record last, so the interesting part is what you see when
    the file opens rather than whatever sorts to the top alphabetically.
    """

    final_record = history[-1] if history else None
    checkpoint_dir = Path(output_dir) / "checkpoints"
    evaluation_records = client_history or []
    update_records = client_update_history or []
    clients = _unique_clients(evaluation_records, update_records)
    phase_counts = _client_update_metric_phase_counts(update_records)
    total_client_fits = _total_client_fits(history, client_update_history)
    total_client_evaluations = len(evaluation_records)
    final_metrics = dict(final_record.metrics) if final_record else {}

    metadata = dict(run_metadata or {})
    scale: dict[str, Any] = {
        "total_client_fits": total_client_fits,
        "total_client_evaluations": total_client_evaluations,
        "total_client_update_metric_records": len(update_records),
        "client_update_metric_phase_counts": phase_counts,
        "total_client_train_examples_evaluated": _total_client_train_examples(evaluation_records),
        "total_client_test_examples_evaluated": _total_client_test_examples(evaluation_records),
        "total_client_examples_processed": _total_client_examples_processed(
            history,
            client_history,
            client_update_history,
        ),
        # A count, not the list. On OpenImage's 13,771 clients the list alone
        # was larger than every other section of this file combined, and the
        # per-client CSVs already hold the identities for anyone who needs them.
        "unique_clients": len(clients),
    }

    artifacts: dict[str, Any] = {
        "files": artifact_files or list(DEFAULT_ARTIFACT_FILES),
    }
    if checkpoint_dir.is_dir() and any(checkpoint_dir.glob("*.pt")):
        artifacts["checkpoint_dir"] = str(checkpoint_dir)
    # Which checkpoints exist and which round won -- results, not policy. The
    # policy fields (enabled, best_metric, interval...) are dropped because
    # config.runtime.extra.checkpointing already holds them.
    checkpointing = metadata.get("checkpointing")
    if isinstance(checkpointing, Mapping):
        results_only = {
            key: value
            for key, value in checkpointing.items()
            if key
            in {
                "best_checkpoint",
                "best_metric_value",
                "best_round_id",
                "latest_checkpoint",
                "kept_checkpoints",
            }
        }
        if results_only:
            artifacts["checkpoints"] = results_only

    # What this run recorded about itself: the code that produced it, what was
    # seeded, what torch actually held, and the model/adapter provenance. All
    # of it was already computed and none of it reached disk -- code_state was
    # read here and by runs_index.jsonl but never written by anything, and the
    # seeding, runtime and llm records were built and discarded. Empty entries
    # are dropped rather than written as null, so a missing key means the run
    # never had one instead of meaning the lookup returned nothing.
    reproducibility: dict[str, Any] = {}
    for key in (
        "code_state",
        "dataset",
        "seeding",
        "runtime",
        "llm",
        "federated_model_state",
        "extensions",
    ):
        value = metadata.get(key)
        if isinstance(value, Mapping) and value:
            reproducibility[key] = dict(value)
    # seed_everything reads torch's flags at seeding time, which is before
    # configure_runtime applies the performance block: its matmul_precision
    # says "highest" on a run that trains at "high". Keeping both would put a
    # stale value beside the effective one for the single key that changes
    # every fp32 matmul. The runtime record is the effective one, so seeding
    # keeps only what seeding itself decided plus the versions -- and only
    # gives up a key that runtime actually reports, so the exception path in
    # configure_runtime cannot leave a flag recorded nowhere.
    seeding = reproducibility.get("seeding")
    runtime_record = reproducibility.get("runtime")
    if isinstance(seeding, dict) and isinstance(runtime_record, dict):
        for key in set(seeding) & set(runtime_record):
            seeding.pop(key)
    # model_state_scope is already inside federated_model_state whenever the
    # task reports one; this file states each thing once.
    model_state_scope = metadata.get("model_state_scope")
    if isinstance(model_state_scope, str) and "model_state_scope" not in (
        reproducibility.get("federated_model_state") or {}
    ):
        reproducibility["model_state_scope"] = model_state_scope

    # Written config, minus the one field that is not configuration: run_id is
    # generated per run and appears at the top of this file instead.
    written_config = asdict(config)
    written_config.get("experiment", {}).pop("run_id", None)

    # Nothing here is repeated anywhere else in the file. Identity and outcome
    # first, the full configuration last; every setting lives once, in "config".
    run: dict[str, Any] = {
        "run_id": metadata.get("run_id"),
        "created_at": metadata.get("created_at"),
        "started_at": metadata.get("started_at"),
        "finished_at": metadata.get("finished_at"),
        # Across every attempt; attempt_duration_sec is this process alone.
        # A requeued run that reported only the latter under-counted itself by
        # however long the killed attempt had already burned.
        "duration_sec": metadata.get("duration_sec"),
        "attempt_duration_sec": metadata.get("attempt_duration_sec"),
        "attempts": metadata.get("attempts", 1),
        # Whether this run continued an earlier attempt, and from which
        # checkpoint. Generated on both resume paths and, until now, dropped --
        # so nothing on disk distinguished a resumed run from a fresh one.
        #
        # `resumed` is what happened, not what was asked for (POST-F05). A
        # checkpoint whose metric history cannot back a resume used to be
        # rejected and the run restarted from round 1, with `resume_restart`
        # saying why; since POST-F25 that resume is refused before anything is
        # written, so no run.json records it and the key is gone.
        "resumed": bool(metadata.get("resumed")),
        "resume_from": metadata.get("resume_from"),
        "num_rounds": len(history),
        # num_rounds counts the records present, which equals the rounds run
        # only when first_round is 1. A resume that could not replay its
        # history leaves the two disagreeing, and that is the signal.
        "first_round": history[0].round_id if history else None,
        "final_round": final_record.round_id if final_record else None,
        # A run that stopped early is not a failed run. Without this, a run cut
        # short by divergence or a walltime kill is indistinguishable from one
        # that finished, short of comparing final_round to global_rounds.
        "status": metadata.get("status", "completed"),
        # Which detector fired, on which round, and against what threshold.
        # None for a run that completed normally.
        "termination": metadata.get("termination") or None,
        "results": {
            "final_metrics": dict(sorted(final_metrics.items())),
        },
        "timing": _timing_summary(history, run_metadata),
        "scale": scale,
        "artifacts": artifacts,
        "reproducibility": reproducibility,
        "config": written_config,
    }

    run_path = prepare_output_dir(output_dir) / "run.json"
    # A diverged run reports inf or nan; json.dumps would emit a bare Infinity,
    # which is not valid JSON and which json.load in another language refuses.
    run = json_safe(run)
    # sort_keys is off on purpose: the section order above IS the organisation.
    text = json.dumps(run, indent=2, default=str, allow_nan=False) + "\n"
    # Replaced, not rewritten in place: it is rewritten every round, a resume
    # reads it for the attempts and the duration so far and falls back to "one
    # attempt, no time spent" on a file it cannot parse. POST-F23.
    with _atomic_text_writer(run_path) as file:
        file.write(text)
    return run_path


def _timing_summary(
    history: list[MetricRecord],
    run_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Condense per-round wall clock into the numbers a job budget needs.

    Median as well as mean because the first round of a run pays one-off costs
    (shard cache fill, CUDA context, lazily built clients) that make the mean a
    poor predictor of the remaining rounds on a long job.
    """

    metadata = run_metadata or {}
    timed = [record.timings for record in history if record.timings is not None]
    summary: dict[str, Any] = {
        "run_duration_sec": metadata.get("duration_sec"),
        "timed_rounds": len(timed),
    }
    if not timed:
        return summary
    durations = [timings.total for timings in timed]
    summary.update(
        {
            "total_round_sec": round(sum(durations), 3),
            "mean_sec_per_round": round(statistics.fmean(durations), 3),
            "median_sec_per_round": round(statistics.median(durations), 3),
            "min_sec_per_round": round(min(durations), 3),
            "max_sec_per_round": round(max(durations), 3),
            "phase_sec": {
                attribute: round(sum(getattr(timings, attribute) for timings in timed), 3)
                for attribute in _ROUND_TIMING_FIELDS.values()
                if attribute != "total"
            },
        }
    )
    return summary


def _client_evaluation_payload(
    record: ClientEvaluationRecord,
) -> dict[str, Any]:
    return {
        "round_id": record.round_id,
        "client_id": record.client_id,
        "participated": record.participated,
        "train_num_examples": record.train_num_examples,
        "val_num_examples": record.val_num_examples,
        "test_num_examples": record.test_num_examples,
        "global_model_train_loss": record.global_model_train_loss,
        "global_model_val_loss": record.global_model_val_loss,
        "global_model_test_loss": record.global_model_test_loss,
        "global_model_train_accuracy": record.global_model_train_accuracy,
        "global_model_val_accuracy": record.global_model_val_accuracy,
        "global_model_test_accuracy": record.global_model_test_accuracy,
        "model_scope": record.model_scope,
    }


def _round_metric_names(history: list[MetricRecord]) -> list[str]:
    return sorted({name for record in history for name in record.metrics})


def _client_update_metric_names(history: list[ClientMetricRecord]) -> list[str]:
    summary = _summary(history)
    if summary is not None:
        return sorted(summary.metric_names)
    return sorted({name for record in history for name in record.metrics})


def _summary(history: Any) -> ClientHistorySummary | None:
    """The running totals a self-summarising history carries, if it is one.

    run.json is rewritten on every round, and every helper below used to
    re-scan the whole history each time -- O(records so far) per round, so
    quadratic over a run. ExperimentState's histories now maintain these
    totals as records arrive; a plain list still gets the full scan, so every
    other caller, including the tests, is unaffected.
    """

    summary = getattr(history, "summary", None)
    return summary if isinstance(summary, ClientHistorySummary) else None


def _unique_clients(
    client_history: list[ClientEvaluationRecord],
    client_update_history: list[ClientMetricRecord],
) -> set[str]:
    """Distinct clients the run touched, from the per-client records.

    Round records only carry a count now, so this is the only source of client
    identity -- and it is the better one: it includes clients that were
    evaluated without ever being selected for training.
    """

    evaluated = _summary(client_history)
    updated = _summary(client_update_history)
    if evaluated is not None and updated is not None:
        return evaluated.client_ids | updated.client_ids
    clients = {record.client_id for record in client_history}
    clients.update(record.client_id for record in client_update_history)
    return clients


def _client_update_metric_phase_counts(
    client_history: list[ClientMetricRecord],
) -> dict[str, int]:
    summary = _summary(client_history)
    if summary is not None:
        return {"fit": 0, **summary.phase_counts}
    counts: dict[str, int] = {"fit": 0}
    for record in client_history:
        counts[record.phase] = counts.get(record.phase, 0) + 1
    return counts


def _total_client_fits(
    history: list[MetricRecord],
    client_history: list[ClientMetricRecord] | None,
) -> int:
    if client_history is None:
        return sum(record.num_clients for record in history)
    summary = _summary(client_history)
    if summary is not None:
        return summary.phase_counts.get("fit", 0)
    return sum(1 for record in client_history if record.phase == "fit")


def _total_client_train_examples(
    client_history: list[ClientEvaluationRecord],
) -> int:
    summary = _summary(client_history)
    if summary is not None:
        return summary.train_examples
    return sum(record.train_num_examples for record in client_history)


def _total_client_test_examples(
    client_history: list[ClientEvaluationRecord],
) -> int:
    summary = _summary(client_history)
    if summary is not None:
        return summary.test_examples
    return sum(record.test_num_examples for record in client_history)


def _total_client_examples_processed(
    history: list[MetricRecord],
    client_history: list[ClientEvaluationRecord] | None,
    client_update_history: list[ClientMetricRecord] | None,
) -> int:
    if client_history is None and client_update_history is None:
        return sum(record.num_examples for record in history)
    evaluation_examples = (
        _total_client_train_examples(client_history) + _total_client_test_examples(client_history)
        if client_history is not None
        else 0
    )
    update_summary = _summary(client_update_history) if client_update_history is not None else None
    if update_summary is not None:
        update_examples = update_summary.num_examples
    else:
        update_examples = sum(record.num_examples for record in (client_update_history or []))
    return evaluation_examples + update_examples
