"""Inspect and validate one generated federated dataset.

Renders through fedbrew.core.console like every other command: the palette,
the markers and the TTY gate are decided there, and the findings go through
the same issue renderer the experiment preflight uses, so a dataset finding
and a config finding do not look like output from two different tools.
"""

import json
import os
import sys
from dataclasses import replace

from fedbrew.core.console import AMBER, DIM, DONE, FAIL, GOLD, RED, WARN, Row, build_surface
from fedbrew.core.validation import print_issues
from fedbrew.data.manifest_validation import validate_manifest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def _load_json(path):
    with open(path) as file:
        return json.load(file)


def _client_values(clients, key):
    values = []
    for client in clients:
        value = client.get(key)
        if isinstance(value, int):
            values.append(value)
    return values


def main(argv=None):
    """Validate and inspect one generated federated dataset."""

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv in (["-h"], ["--help"]):
        print("usage: fedbrew inspect-data <manifest.json>")
        return
    if len(argv) != 1:
        print("Usage: fedbrew inspect-data <manifest.json>")
        raise SystemExit(2)

    surface = build_surface()
    manifest_path, used_local_fallback = _resolve_manifest_path(argv[0])
    issues = validate_manifest(manifest_path)
    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warning"]

    surface.rule("FEDERATED DATASET REPORT")
    surface.rows([Row("Manifest", _display_path(manifest_path))])
    if used_local_fallback:
        # Amber, by the palette's rule for a value that was defaulted rather
        # than chosen: which tree this report describes is exactly what a
        # reader would otherwise assume.
        surface.line(
            "FL_DATA_ROOT is unset; using local data/generated/.",
            tone=AMBER,
            marker=WARN,
            marker_tone=AMBER,
        )
    if errors:
        surface.line("INVALID DATASET", tone=RED, marker=FAIL, marker_tone=RED)
        surface.line(f"{len(errors)} errors  {len(warnings)} warnings", tone=DIM)
        surface.blank()
        print_issues(surface, _shortened(issues))
        raise SystemExit(1)

    surface.line("VALID DATASET", tone=GOLD, marker=DONE, marker_tone=GOLD)
    surface.line(f"0 errors  {len(warnings)} warnings", tone=DIM)
    try:
        manifest = _load_json(manifest_path)
        root = os.path.dirname(manifest_path)
        stats_path = os.path.join(
            root, manifest.get("partition_stats_file", "partition_stats.json")
        )
        stats = _load_json(stats_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        surface.line(f"Could not render dataset summary: {exc}", tone=RED, marker=FAIL)
        raise SystemExit(1) from exc

    _print_summary(surface, manifest_path, manifest, stats, stats_path)
    if warnings:
        surface.blank()
        print_issues(surface, _shortened(warnings))
    surface.rule("DATASET READY FOR EXPERIMENTS")


def _print_summary(surface, manifest_path, manifest, stats, stats_path):
    """One measured block: what the dataset is, then where its files are.

    Three bordered tables and two panels before this. They said the same
    things, in five differently sized boxes, none of which survived a
    redirected stream legibly.
    """

    clients = stats.get("clients", [])
    clients = clients if isinstance(clients, list) else []
    train_sizes = _client_values(clients, "num_train_examples")
    eval_sizes = _client_values(clients, "num_eval_examples")
    splits = manifest.get("client_splits", stats.get("client_splits", {}))
    splits = splits if isinstance(splits, dict) else {}

    rows = [
        Row("Dataset", str(manifest.get("dataset_name", "unknown"))),
        Row("Partition", str(manifest.get("partition_strategy", "unknown"))),
        Row("Clients", str(stats.get("num_clients", manifest.get("num_clients", "unknown")))),
        Row("Total examples", str(stats.get("total_examples", "unknown"))),
    ]
    if "input_shape" in manifest:
        rows.append(Row("Input shape", str(manifest["input_shape"])))
    elif "input_dim" in manifest:
        rows.append(Row("Input dimension", str(manifest["input_dim"])))
    train_ratio = splits.get("train_ratio", "unknown")
    eval_ratio = splits.get("eval_ratio", "unknown")
    rows.append(Row("Client splits", f"train={train_ratio}, eval={eval_ratio}"))
    rows.append(Row("Train examples", _size_summary(train_sizes)))
    rows.append(Row("Eval examples", _size_summary(eval_sizes)))
    if "global_label_counts" in stats:
        rows.append(Row("Label counts", str(stats["global_label_counts"])))
    if "total_tokens" in stats:
        rows.append(Row("Training tokens", str(stats["total_tokens"])))
        rows.append(Row("Global test tokens", str(stats.get("global_test_tokens", "unknown"))))

    client_stats_path = os.path.join(
        os.path.dirname(manifest_path),
        manifest.get("client_stats_file", "client_stats.csv"),
    )
    shards_path = os.path.join(
        os.path.dirname(manifest_path),
        manifest.get("shards_dir", "shards"),
    )
    rows.extend(
        (
            Row("Partition statistics", _display_path(stats_path)),
            Row("Client statistics", _display_path(client_stats_path)),
            Row("Client shards", _display_path(shards_path)),
        )
    )
    surface.blank()
    surface.rows(rows)


def _shortened(issues):
    """Issue messages with this repo's own absolute paths folded back to the
    relative form the rest of the report uses. A finding that names a shard is
    unreadable at 120 characters of leading path."""

    return [replace(issue, message=_relative_text(issue.message)) for issue in issues]


def _size_summary(values):
    if not values:
        return "unknown"
    mean = float(sum(values)) / len(values)
    return f"total={sum(values)}, min={min(values)}, max={max(values)}, mean={mean:.2f}"


def _resolve_manifest_path(path):
    """Resolve a manifest path without requiring FL_DATA_ROOT for local data."""

    resolved_path = os.path.abspath(os.path.expanduser(path))
    if os.environ.get("FL_DATA_ROOT") or os.path.exists(resolved_path):
        return resolved_path, False
    if resolved_path.startswith(os.sep):
        local_path = os.path.join(
            PROJECT_ROOT,
            "data",
            "generated",
            resolved_path.lstrip(os.sep),
        )
        if os.path.exists(local_path):
            return local_path, True
    return resolved_path, False


def _display_path(path):
    """Show project paths relatively without changing the path used on disk."""

    path = os.path.abspath(path)
    data_root = os.environ.get("FL_DATA_ROOT")
    if data_root:
        data_root = os.path.abspath(os.path.expanduser(data_root))
        try:
            if os.path.commonpath((path, data_root)) == data_root:
                return chr(36) + "FL_DATA_ROOT/" + os.path.relpath(path, data_root)
        except ValueError:
            pass

    try:
        relative_path = os.path.relpath(path, PROJECT_ROOT)
        if not relative_path.startswith(".." + os.sep):
            return relative_path
    except ValueError:
        pass
    return os.path.basename(path)


def _relative_text(value):
    value = value.replace(PROJECT_ROOT + os.sep, "")
    data_root = os.environ.get("FL_DATA_ROOT")
    if data_root:
        value = value.replace(
            os.path.abspath(os.path.expanduser(data_root)) + os.sep,
            chr(36) + "FL_DATA_ROOT/",
        )
    return value


if __name__ == "__main__":
    main()
