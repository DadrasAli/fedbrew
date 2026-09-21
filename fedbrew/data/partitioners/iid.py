"""IID partitioning helpers for generated federated data."""

from __future__ import annotations

import random
from collections.abc import Sequence


def partition_iid(
    indices: Sequence[int],
    num_clients: int,
    seed: int,
) -> dict[str, list[int]]:
    """Shuffle and split indices nearly equally across clients."""

    if num_clients <= 0:
        raise ValueError("num_clients must be positive")

    shuffled = list(indices)
    random.Random(seed).shuffle(shuffled)
    partitions: dict[str, list[int]] = {f"client_{index}": [] for index in range(num_clients)}
    for position, data_index in enumerate(shuffled):
        partitions[f"client_{position % num_clients}"].append(data_index)
    return partitions
