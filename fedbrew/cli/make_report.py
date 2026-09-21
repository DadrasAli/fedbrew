"""Export a compact Markdown report for one completed run."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def main(argv: Sequence[str] | None = None) -> None:
    """Run the report export CLI."""

    args = parse_args(argv)
    input_path = Path(args.run_dir)
    report_path = make_run_report(input_path)
    _print_generated(input_path, report_path.parent, [report_path])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse report CLI arguments."""

    parser = argparse.ArgumentParser(description="Create fedbrew Markdown reports.")
    parser.add_argument("--run-dir", required=True, help="Directory containing one completed run.")
    return parser.parse_args(argv)


def make_run_report(run_dir: Path) -> Path:
    """Write report.md for one completed run."""

    # One file now: run.json carries the outcome and the full config.
    run = _read_json_if_exists(run_dir / "run.json")
    summary = run
    config = _mapping(run.get("config"))
    experiment = _mapping(config.get("experiment"))
    scale = _mapping(run.get("scale"))
    metadata = {
        "run_id": run.get("run_id"),
        "experiment_name": experiment.get("name"),
        "seed": experiment.get("seed"),
    }
    server = _mapping(config.get("server"))
    client = _mapping(config.get("client"))
    data = _mapping(config.get("data"))
    model = _mapping(config.get("model"))

    final_metrics = _mapping(_mapping(run.get("results")).get("final_metrics"))
    artifacts = _artifact_files(run_dir, summary)
    status = run.get("status")

    run_name = _first(experiment.get("name"), metadata.get("experiment_name"), run_dir.name)
    lines = [
        f"# Run Report: {run_name}",
        "",
        "## Overview",
        "",
        f"- Experiment: {_first(experiment.get('name'), metadata.get('experiment_name'), '')}",
        f"- Run ID: {_first(metadata.get('run_id'), summary.get('run_id'), '')}",
        f"- Seed: {_first(metadata.get('seed'), experiment.get('seed'), '')}",
        f"- Server strategy: {server.get('strategy', '')}",
        f"- Client update rule: {client.get('update_rule', '')}",
        f"- Data: {data.get('name', '')}",
        f"- Model: {model.get('name', '')}",
        f"- Status: {_first(status, 'unknown')}",
        *_termination_lines(run),
        "",
        "## Run Size",
        "",
        f"- Rounds: {_round_span(summary, server)}",
        *_scale_lines(scale),
        "",
        _metrics_heading(status),
        "",
        _metric_table(final_metrics),
        "",
        "## Artifacts",
        "",
        _bullet_list(artifacts),
        "",
    ]
    return _write_report(run_dir / "report.md", lines)


#: The one status whose last recorded round is the run's actual final round.
#: A diverged or still-running run's last recorded round is where it stopped,
#: so "Final Metrics" would name the metric at the round the run was killed.
COMPLETED_STATUS = "completed"


def _metrics_heading(status: object) -> str:
    """Name the metrics table for what it actually holds.

    report.md is the documented post-run summary (docs/09-artifacts.md),
    and the last row of a diverged run's history is the round it blew up at,
    not a final result. The status sits two keys away in the same JSON object.
    """

    if status == COMPLETED_STATUS:
        return "## Final Metrics"
    if status in (None, ""):
        return "## Last Evaluated Metrics (run status unknown)"
    return f"## Last Evaluated Metrics (run {status})"


def _termination_lines(run: Mapping[str, Any]) -> list[str]:
    """Why the run stopped, when it did not simply finish.

    The divergence monitor already records the detector, the round, the metric
    and the value that tripped it, in a sentence written for exactly this. A
    reader who sees a short run and no explanation has to open run.json.
    """

    termination = _mapping(run.get("termination"))
    if not termination:
        return []
    reason = termination.get("reason")
    if reason:
        return [f"- Stopped: {reason}"]
    detector = _first(termination.get("detector"), "unknown")
    return [f"- Stopped: {detector} at round {_first(termination.get('round_id'), '')}"]


def _round_span(summary: Mapping[str, Any], server: Mapping[str, Any]) -> str:
    """Rounds recorded, the span they cover, and the number configured.

    A bare count read as "this run ran N rounds" is wrong for any run whose
    history does not start at round 1; printing the span alongside is what lets
    a reader see that without opening run.json. The configured total is what
    makes "1" legible as a run that died immediately rather than a run of one.
    """

    recorded = summary.get("num_rounds")
    first = summary.get("first_round")
    final = summary.get("final_round")
    if recorded is None:
        return ""
    if isinstance(first, int) and isinstance(final, int) and first != 1:
        span = f"{recorded} recorded, rounds {first}-{final}"
    else:
        span = str(recorded)
    configured = server.get("global_rounds")
    if isinstance(configured, int) and configured != recorded:
        return f"{span} of {configured} configured"
    return span


#: The scale keys this report prints, in order, with the label each gets.
#: Names have to match what artifacts.py:save_run_json writes -- two of them did
#: not ("total_client_updates", "total_client_evaluation_records"), so those two
#: lines rendered as a heading with nothing after it for every run.
#: tests/test_report_run_size.py pins the table against a real run.
SCALE_LINES: tuple[tuple[str, str], ...] = (
    ("total_client_fits", "Client fits"),
    ("total_client_evaluations", "Client evaluation records"),
    ("total_client_update_metric_records", "Client update metric records"),
    ("unique_clients", "Unique clients"),
    ("total_client_examples_processed", "Client examples processed"),
)


def _scale_lines(scale: Mapping[str, Any]) -> list[str]:
    """One line per scale key, saying so when a key is absent.

    A key that run.json does not carry used to render as an empty string after
    its label. An old run.json legitimately predates a key; a typo does not.
    Neither is distinguishable from the other in a blank, so say which value is
    missing instead.
    """

    lines = []
    for key, label in SCALE_LINES:
        value = scale.get(key)
        lines.append(f"- {label}: {value if value is not None else 'not recorded'}")
    return lines


def _metric_table(metrics: Mapping[str, Any]) -> str:
    if not metrics:
        return "No final metrics found."
    lines = ["| Metric | Value |", "| --- | --- |"]
    for key in sorted(metrics):
        lines.append(f"| {key} | {metrics[key]} |")
    return "\n".join(lines)


def _bullet_list(paths: list[str]) -> str:
    if not paths:
        return "- none"
    return "\n".join(f"- {path}" for path in paths)


def _artifact_files(run_dir: Path, summary: Mapping[str, Any]) -> list[str]:
    declared = _mapping(summary.get("artifacts")).get("files")
    artifacts = [str(item) for item in declared] if isinstance(declared, list) else []
    existing = sorted(path.name for path in run_dir.iterdir() if path.is_file())
    for name in existing:
        if name not in artifacts and name != "report.md":
            artifacts.append(name)
    return artifacts


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first(*values: object) -> object:
    for value in values:
        if value not in (None, ""):
            return value
    return ""


def _write_report(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _print_generated(input_path: Path, output_dir: Path, generated: list[Path]) -> None:
    print(f"Input       : {input_path}")
    print(f"Output dir  : {output_dir}")
    print("Generated files:")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()
