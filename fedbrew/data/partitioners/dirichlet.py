"""Dirichlet label-skew partitioning helpers."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from fedbrew.data.partitioners.empty_clients import fill_empty_clients


def partition_dirichlet(
    labels: Sequence[int],
    num_clients: int,
    alpha: float,
    seed: int,
) -> dict[str, list[int]]:
    """Partition indices by class-label Dirichlet proportions."""

    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    rng = random.Random(seed)
    labels_by_class: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        labels_by_class[int(label)].append(index)

    partitions: dict[str, list[int]] = {f"client_{index}": [] for index in range(num_clients)}
    for class_indices in labels_by_class.values():
        rng.shuffle(class_indices)
        proportions = _sample_dirichlet(num_clients, alpha, rng)
        for data_index in class_indices:
            client_index = _sample_categorical(proportions, rng)
            partitions[f"client_{client_index}"].append(data_index)

    fill_empty_clients(partitions, rng)
    for values in partitions.values():
        values.sort()
    return partitions


def _sample_dirichlet(
    num_clients: int,
    alpha: float,
    rng: random.Random,
) -> list[float]:
    values = [rng.gammavariate(alpha, 1.0) for _ in range(num_clients)]
    total = sum(values)
    if total == 0.0:
        return [1.0 / num_clients] * num_clients
    return [value / total for value in values]


def _sample_categorical(proportions: Sequence[float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(proportions):
        cumulative += probability
        if threshold <= cumulative:
            return index
    return len(proportions) - 1
