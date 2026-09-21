"""Label-skew partitioning helpers."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from fedbrew.data.partitioners.empty_clients import fill_empty_clients


def partition_label_skew(
    labels: Sequence[int],
    num_clients: int,
    labels_per_client: int,
    seed: int,
) -> dict[str, list[int]]:
    """Partition examples so each client sees at most `labels_per_client` labels.

    The bound is a guarantee, not a preference. A configuration that cannot
    honour it is refused rather than quietly relaxed.
    """

    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if labels_per_client <= 0:
        raise ValueError("labels_per_client must be positive")

    rng = random.Random(seed)
    labels_by_class: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        labels_by_class[int(label)].append(index)
    for values in labels_by_class.values():
        rng.shuffle(values)

    unique_labels = sorted(labels_by_class)
    if labels_per_client > len(unique_labels):
        raise ValueError("labels_per_client must not exceed the number of unique labels")
    _require_label_coverage(len(unique_labels), num_clients, labels_per_client)
    allowed_labels = _assign_allowed_labels(
        unique_labels,
        num_clients,
        labels_per_client,
        rng,
    )
    partitions: dict[str, list[int]] = {f"client_{index}": [] for index in range(num_clients)}

    for label in unique_labels:
        eligible_clients = [
            client_index for client_index, allowed in enumerate(allowed_labels) if label in allowed
        ]
        if not eligible_clients:
            # _require_label_coverage guarantees every label is claimed, so
            # this is an invariant break rather than a configuration the caller
            # can fix. Spreading the label over all clients instead -- what this
            # did -- is what turned label skew into a near-IID partition.
            raise AssertionError(
                f"label {label} was assigned to no client despite "
                f"{num_clients} x {labels_per_client} >= {len(unique_labels)} "
                "allowed-label slots"
            )
        rng.shuffle(eligible_clients)
        for position, data_index in enumerate(labels_by_class[label]):
            client_index = eligible_clients[position % len(eligible_clients)]
            partitions[f"client_{client_index}"].append(data_index)

    fill_empty_clients(partitions, rng)
    for values in partitions.values():
        values.sort()
    return partitions


def _require_label_coverage(
    num_labels: int,
    num_clients: int,
    labels_per_client: int,
) -> None:
    """Refuse a configuration that cannot give every label a client.

    Each client may hold at most `labels_per_client` distinct labels, so the
    clients between them offer `num_clients * labels_per_client` slots. Below
    `num_labels` slots the two halves of the contract -- every label lands
    somewhere, and no client exceeds its label budget -- cannot both hold, and
    the partition that came out was a near-IID one still written to disk under
    `partition_strategy: label_skew` with `labels_per_client` in its config.
    At K=62 with 5 clients and labels_per_client 2, clients held up to 56
    distinct labels.
    """

    slots = num_clients * labels_per_client
    if slots >= num_labels:
        return
    raise ValueError(
        "label_skew cannot cover every label: num_clients * "
        f"labels_per_client = {num_clients} * {labels_per_client} = {slots}, "
        f"below the {num_labels} unique labels in the data. Every label left "
        "unclaimed would be spread across all clients, which silently produces "
        "a near-IID partition labelled as label skew. Raise num_clients or "
        "labels_per_client so their product is at least "
        f"{num_labels}, or use a partitioner that does not bound the labels "
        "per client."
    )


def _assign_allowed_labels(
    labels: list[int],
    num_clients: int,
    labels_per_client: int,
    rng: random.Random,
) -> list[set[int]]:
    allowed: list[set[int]] = [set() for _ in range(num_clients)]
    if not labels:
        return allowed

    slots = [client_index for client_index in range(num_clients) for _ in range(labels_per_client)]
    rng.shuffle(slots)
    shuffled_labels = list(labels)
    rng.shuffle(shuffled_labels)

    # One slot per label, so every label is claimed by exactly one client.
    # _require_label_coverage has already established there are enough slots;
    # the branch that used to run when there were not gave each client an
    # independent sample and left most labels claimed by nobody.
    #
    # strict=False, deliberately: the invariant is num_clients *
    # labels_per_client >= len(unique_labels), not ==, so surplus slots are the
    # normal case and are meant to be dropped here -- the loop below tops each
    # client back up to labels_per_client. strict=True would raise on every
    # configuration that is not exactly tight, including the shipped
    # synthetic_label_skew one (3 labels, 10 slots).
    for label, client_index in zip(shuffled_labels, slots, strict=False):
        allowed[client_index].add(label)

    for client_index, values in enumerate(allowed):
        while len(values) < labels_per_client:
            values.add(rng.choice(labels))
        if not values:
            values.add(labels[client_index % len(labels)])
    return allowed
