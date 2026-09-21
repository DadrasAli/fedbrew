"""Dataset partition statistics helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def compute_label_counts(labels: Iterable[Any]) -> dict[str, int]:
    """Count labels using string keys for JSON/CSV stability."""

    counts: dict[str, int] = {}
    for label in labels:
        key = str(int(label))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def compute_client_stats(
    client_id: str,
    indices: Sequence[int],
    labels: Sequence[Any],
) -> dict[str, object]:
    """Compute per-client label and quantity statistics."""

    local_labels = [labels[index] for index in indices]
    label_counts = compute_label_counts(local_labels)
    num_examples = len(indices)
    dominant_label = None
    dominant_count = 0
    if label_counts:
        dominant_label, dominant_count = max(
            label_counts.items(),
            key=lambda item: (item[1], -int(item[0])),
        )
    return {
        "client_id": client_id,
        "num_examples": num_examples,
        "label_counts": label_counts,
        "num_labels": len(label_counts),
        "dominant_label": dominant_label,
        "dominant_label_fraction": (dominant_count / num_examples if num_examples else 0.0),
    }


def compute_partition_summary(
    client_stats: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Compute aggregate partition statistics across clients."""

    sizes: list[int] = []
    for stats in client_stats:
        raw_size = stats["num_examples"]
        if isinstance(raw_size, bool) or not isinstance(raw_size, int | str):
            raise ValueError("num_examples must be an integer")
        sizes.append(int(raw_size))

    global_counts: dict[str, int] = {}
    for stats in client_stats:
        counts = stats.get("label_counts", {})
        if not isinstance(counts, dict):
            continue
        for label, count in counts.items():
            if isinstance(count, bool) or not isinstance(count, int | str):
                raise ValueError("label counts must be integers")
            key = str(label)
            global_counts[key] = global_counts.get(key, 0) + int(count)

    labels = sorted(global_counts, key=int)
    total_examples = sum(sizes)
    return {
        "num_clients": len(client_stats),
        "total_examples": total_examples,
        "min_examples_per_client": min(sizes) if sizes else 0,
        "max_examples_per_client": max(sizes) if sizes else 0,
        "mean_examples_per_client": total_examples / len(sizes) if sizes else 0.0,
        "labels": labels,
        "global_label_counts": {label: global_counts[label] for label in labels},
    }
