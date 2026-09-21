"""Partition an official test set to match realized client heterogeneity."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence


def partition_test_indices_like_train(
    *,
    test_labels: Sequence[int],
    train_labels: Sequence[int],
    train_partitions: Mapping[str, Sequence[int]],
    strategy: str,
    seed: int,
) -> dict[str, list[int]]:
    """Assign every official-test example using the training partition profile."""

    client_ids = sorted(train_partitions)
    if not client_ids:
        raise ValueError("test partitioning requires at least one client")
    if len(test_labels) < len(client_ids):
        raise ValueError("official test set must contain at least one example per client")

    rng = random.Random(seed)
    assignments = {client_id: [] for client_id in client_ids}
    test_indices = list(range(len(test_labels)))

    if strategy == "iid":
        _assign_by_weights(
            test_indices,
            dict.fromkeys(client_ids, 1),
            assignments,
            rng,
        )
    elif strategy == "quantity_skew":
        _assign_by_weights(
            test_indices,
            {client_id: len(train_partitions[client_id]) for client_id in client_ids},
            assignments,
            rng,
        )
    elif strategy in {"dirichlet", "label_skew"}:
        train_label_counts = _training_label_counts(
            train_labels,
            train_partitions,
        )
        test_indices_by_label: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(test_labels):
            test_indices_by_label[int(label)].append(index)
        for label in sorted(test_indices_by_label):
            weights = {
                client_id: train_label_counts[client_id].get(label, 0) for client_id in client_ids
            }
            if sum(weights.values()) == 0:
                weights = {client_id: len(train_partitions[client_id]) for client_id in client_ids}
            _assign_by_weights(
                test_indices_by_label[label],
                weights,
                assignments,
                rng,
            )
    else:
        raise ValueError(f"Unknown partition strategy: {strategy}")

    _fill_empty_clients(
        assignments,
        test_labels,
        train_label_counts=_training_label_counts(
            train_labels,
            train_partitions,
        ),
    )
    for indices in assignments.values():
        indices.sort()
    return assignments


def _training_label_counts(
    train_labels: Sequence[int],
    train_partitions: Mapping[str, Sequence[int]],
) -> dict[str, dict[int, int]]:
    counts: dict[str, dict[int, int]] = {}
    for client_id, indices in train_partitions.items():
        local_counts: dict[int, int] = {}
        for index in indices:
            label = int(train_labels[index])
            local_counts[label] = local_counts.get(label, 0) + 1
        counts[client_id] = local_counts
    return counts


def _assign_by_weights(
    indices: Sequence[int],
    weights: Mapping[str, int],
    assignments: dict[str, list[int]],
    rng: random.Random,
) -> None:
    shuffled = list(indices)
    rng.shuffle(shuffled)
    client_ids = list(assignments)
    total_weight = sum(max(0, int(weights.get(client_id, 0))) for client_id in client_ids)
    if total_weight == 0:
        normalized = dict.fromkeys(client_ids, 1)
        total_weight = len(client_ids)
    else:
        normalized = {client_id: max(0, int(weights.get(client_id, 0))) for client_id in client_ids}

    raw_counts = {
        client_id: len(shuffled) * normalized[client_id] / total_weight for client_id in client_ids
    }
    counts = {client_id: math.floor(raw_counts[client_id]) for client_id in client_ids}
    remainder_order = list(client_ids)
    rng.shuffle(remainder_order)
    remainder_order.sort(
        key=lambda client_id: raw_counts[client_id] - counts[client_id],
        reverse=True,
    )
    for client_id in remainder_order[: len(shuffled) - sum(counts.values())]:
        counts[client_id] += 1

    offset = 0
    for client_id in client_ids:
        stop = offset + counts[client_id]
        assignments[client_id].extend(shuffled[offset:stop])
        offset = stop


def _fill_empty_clients(
    assignments: dict[str, list[int]],
    test_labels: Sequence[int],
    train_label_counts: Mapping[str, Mapping[int, int]],
) -> None:
    for empty_client in [client_id for client_id, indices in assignments.items() if not indices]:
        allowed_labels = set(train_label_counts[empty_client])
        donor_and_position: tuple[str, int] | None = None
        for donor in sorted(assignments, key=lambda key: len(assignments[key]), reverse=True):
            if len(assignments[donor]) <= 1:
                continue
            position = next(
                (
                    index
                    for index, test_index in enumerate(assignments[donor])
                    if int(test_labels[test_index]) in allowed_labels
                ),
                None,
            )
            if position is not None:
                donor_and_position = (donor, position)
                break
        if donor_and_position is None:
            donor = max(assignments, key=lambda key: len(assignments[key]))
            if len(assignments[donor]) <= 1:
                raise ValueError("could not assign a test example to every client")
            donor_and_position = (donor, 0)
        donor, position = donor_and_position
        assignments[empty_client].append(assignments[donor].pop(position))
