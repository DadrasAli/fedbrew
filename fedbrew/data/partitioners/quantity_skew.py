"""Quantity-skew partitioning helpers."""

from __future__ import annotations

import random
from collections.abc import Sequence

#: Default shape of the client-size distribution. Sizes are drawn as lognormal
#: weights over the budget above min_size, the same family
#: tools/generate_openimage_shaped_synthetic.py::client_sample_counts uses to
#: reproduce FedScale's OpenImage skew: a few prolific clients and a long tail.
#: sigma is the only knob on the tail -- 0 gives equal sizes, 1.0 gives a Gini
#: around 0.5 on a 100-client split. Overridable via partition.sigma; this is
#: the value every quantity_skew dataset used before that key existed.
LOGNORMAL_SIGMA = 1.0


def partition_quantity_skew(
    indices: Sequence[int],
    num_clients: int,
    min_size: int,
    max_size: int,
    seed: int,
    sigma: float = LOGNORMAL_SIGMA,
) -> dict[str, list[int]]:
    """Partition examples into deterministic unequal client quantities.

    Sizes are lognormal within [min_size, max_size] and sum to len(indices).
    Every client gets at least min_size examples, so none is empty.
    """

    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if min_size < 1:
        raise ValueError(
            "min_size must be at least 1: a client with no examples is not a "
            "client, and min_size=0 let the allocator write empty shards "
            "(91 of 100 clients at min_size=0, max_size=60000, N=60000)"
        )
    if max_size < min_size:
        raise ValueError("max_size must be greater than or equal to min_size")

    total = len(indices)
    if total < num_clients * min_size:
        raise ValueError("not enough examples for requested min_size")
    if total > num_clients * max_size:
        raise ValueError("too many examples for requested max_size")

    rng = random.Random(seed)
    sizes = _lognormal_sizes(num_clients, total, min_size, max_size, sigma, rng)
    _make_sizes_differ_when_possible(sizes, min_size, max_size)
    if sum(sizes) != total:
        # The slicing below would silently hand short or empty lists to the
        # tail clients, which is the failure this partitioner is being fixed
        # for. Say so instead.
        raise AssertionError(f"client sizes sum to {sum(sizes)}, not the {total} examples given")

    shuffled = list(indices)
    rng.shuffle(shuffled)
    partitions: dict[str, list[int]] = {}
    offset = 0
    for client_index, size in enumerate(sizes):
        values = shuffled[offset : offset + size]
        partitions[f"client_{client_index}"] = sorted(values)
        offset += size
    return partitions


def _lognormal_sizes(
    num_clients: int,
    total: int,
    min_size: int,
    max_size: int,
    sigma: float,
    rng: random.Random,
) -> list[int]:
    """Client sizes drawn from a stated law, inside [min_size, max_size].

    The previous allocator handed the budget out greedily: pick a client at
    random, give it rng.randint(1, everything that is left). Uniform over the
    whole remaining budget means the first few draws take most of the corpus,
    and the realized distribution was a by-product of that loop rather than
    anything the docstring or the config schema named. At N=60000 over 20
    clients with [100, 20000] it produced
    [16594, 16367, 11834, 8585, 2633, 1890, 674, 217, 103, 103, 100 x 10]:
    two clients holding 55% and thirteen sitting on the floor.

    Every client starts at min_size; the budget above that is split in
    proportion to lognormal weights, water-filled so that a client which would
    exceed max_size is capped and its overflow goes to the others. The integer
    remainder goes to the largest fractional shares, so the sizes sum to
    exactly `total`.
    """

    sizes = [min_size] * num_clients
    budget = total - num_clients * min_size
    if budget <= 0:
        return sizes

    room = [max_size - min_size] * num_clients
    weights = [rng.lognormvariate(0.0, sigma) for _ in range(num_clients)]
    if sum(weights) <= 0.0:
        weights = [1.0] * num_clients

    extra = [0.0] * num_clients
    open_clients = [index for index in range(num_clients) if room[index] > 0]
    remaining = float(budget)
    while remaining > 0.0 and open_clients:
        weight_total = sum(weights[index] for index in open_clients)
        if weight_total <= 0.0:
            weight_total = float(len(open_clients))
            for index in open_clients:
                weights[index] = 1.0
        shares = {index: remaining * weights[index] / weight_total for index in open_clients}
        capped = [index for index in open_clients if shares[index] >= room[index] - extra[index]]
        if not capped:
            for index in open_clients:
                extra[index] += shares[index]
            break
        for index in capped:
            remaining -= room[index] - extra[index]
            extra[index] = float(room[index])
            open_clients.remove(index)

    whole = [int(value) for value in extra]
    shortfall = budget - sum(whole)
    # Largest remainder, skipping anyone already at their cap. sum(room) >=
    # budget is guaranteed by the max_size check above, so this terminates.
    order = sorted(
        range(num_clients),
        key=lambda index: extra[index] - whole[index],
        reverse=True,
    )
    position = 0
    while shortfall > 0:
        index = order[position % num_clients]
        position += 1
        if whole[index] < room[index]:
            whole[index] += 1
            shortfall -= 1

    return [min_size + value for value in whole]


def _make_sizes_differ_when_possible(
    sizes: list[int],
    min_size: int,
    max_size: int,
) -> None:
    if len(set(sizes)) > 1 or len(sizes) < 2 or min_size == max_size:
        return
    if sizes[0] > min_size and sizes[1] < max_size:
        sizes[0] -= 1
        sizes[1] += 1
    elif sizes[0] < max_size and sizes[1] > min_size:
        sizes[0] += 1
        sizes[1] -= 1
