"""Give every client at least one example, or say why that is impossible."""

from __future__ import annotations

import random


def fill_empty_clients(
    partitions: dict[str, list[int]],
    rng: random.Random,
) -> None:
    """Move one example into each empty client, in place.

    Both label-skewing partitioners can leave a client with nothing: dirichlet
    draws each example's client independently, and label_skew hands a label to
    a subset of the clients that could hold it. An empty partition becomes a
    zero-row shard that ``fedbrew generate`` writes without complaint, and the
    run only fails much later, at evaluation, with "has no non-empty train
    split".

    Both exits therefore raise. They used to ``return``, and the first of them
    returned exactly when the corpus was too small to give everyone an example
    -- the case the fill exists for. Measured at alpha=0.5 with 100 clients and
    99 examples: 38 clients came out empty, and 38 zero-row shards were written
    as a successful generation. One example more and the same call filled every
    client.

    ``partition_test_indices_like_train`` has refused on these two conditions
    since it was written -- a corpus smaller than the client count up front, and
    no donor with more than one example in the loop. This is the training side
    saying the same thing.

    Raises:
        ValueError: There are fewer examples than clients. The caller fixes this
            by asking for fewer clients or supplying more data.
        AssertionError: No client holds more than one example while another
            holds none. Unreachable once the count above passes -- it is kept as
            an invariant check, in the spelling ``partition_label_skew`` already
            uses for the same distinction.
    """

    total = sum(len(values) for values in partitions.values())
    if total < len(partitions):
        raise ValueError(
            f"cannot give every client an example: {total} examples for "
            f"{len(partitions)} clients. Lower num_clients to at most {total}, "
            "or partition a larger corpus."
        )

    empty_clients = [name for name, values in partitions.items() if not values]
    for empty_client in empty_clients:
        donors = [name for name, values in partitions.items() if len(values) > 1]
        if not donors:
            raise AssertionError(
                f"no client holds more than one example, yet {empty_client!r} "
                f"holds none, across {total} examples and {len(partitions)} clients"
            )
        donor = max(donors, key=lambda name: len(partitions[name]))
        moved_position = rng.randrange(len(partitions[donor]))
        partitions[empty_client].append(partitions[donor].pop(moved_position))
