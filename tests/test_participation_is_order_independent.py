"""The participation schedule is a function of the ids, not of the file order.

`loop._sampled_client_infos` sorts the roster before drawing and its
docstring says why: "so it does not depend on the order the dataset happens to
list clients in". `sample_clients` -- the same operation, on the same roster,
400 lines away -- did not. It called `rng.sample(list(clients), k)` on whatever
order `clients.jsonl` was read in.

That was correct, by luck. Both generators write the file sorted:
`generate.py` iterates `enumerate(sorted(partitions))` and `femnist.py` selects
through `_select_writer_ids`, which returns `sorted(...)`. Every roster they
write is in id order, so sorting changes nothing about a dataset either produced --
which is what `test_sorting_changes_nothing_on_a_roster_that_is_already_sorted`
pins, because a fix that silently moved a shipped baseline's schedule would be
a worse defect than the one it fixed.

The coupling it removes is one function call wide. FEMNIST's `_client_id`
passes a filename-safe writer id through unchanged and rewrites anything else
to `writer_<sha256[:16]>`. Sorted writer ids therefore produce sorted client
ids only while every writer id is safe; one that is not is enough to make the
file order differ from the id order. At that point the old code drew a
completely different set every round -- measured below at 0 of 36 clients in
common at `participation_rate: 0.01` -- with nothing to signal it, because
both schedules are equally valid-looking draws from the same seed.
"""

from __future__ import annotations

import random
import unittest

import pytest

from fedbrew.core.protocol import ClientInfo
from fedbrew.core.seeding import derive_seed
from fedbrew.data.femnist import _client_id
from fedbrew.servers.fedavg import FedAvgServer

pytestmark = pytest.mark.fast

ROSTER = 3597
RATE = 0.01


def _server(rate: float = RATE) -> FedAvgServer:
    return FedAvgServer(participation_rate=rate, seed=42)


def _roster(client_ids: list[str]) -> list[ClientInfo]:
    return [ClientInfo(client_id=cid, num_examples=10) for cid in client_ids]


def _drawn(server: FedAvgServer, roster: list[ClientInfo], round_id: int) -> list[str]:
    return [client.client_id for client in server.sample_clients(roster, round_id)]


class OrderIndependenceTest(unittest.TestCase):
    def test_shuffling_the_roster_does_not_move_the_schedule(self) -> None:
        ids = [f"client_{index}" for index in range(ROSTER)]
        shuffled = list(ids)
        random.Random(7).shuffle(shuffled)
        self.assertNotEqual(shuffled, ids, "the shuffle must actually reorder")

        for round_id in range(5):
            with self.subTest(round=round_id):
                self.assertEqual(
                    sorted(_drawn(_server(), _roster(ids), round_id)),
                    sorted(_drawn(_server(), _roster(shuffled), round_id)),
                )

    def test_the_femnist_id_rewrite_no_longer_moves_the_schedule(self) -> None:
        """The concrete way the roster stops being id-sorted.

        `_client_id` hashes any writer id that is not filename-safe, so a
        sorted writer list maps to an unsorted client list. Before this, that
        rewrite changed who trained.
        """

        writers = sorted(f"writer #{index}" for index in range(ROSTER))
        file_order = [_client_id(writer) for writer in writers]
        self.assertNotEqual(file_order, sorted(file_order), "the rewrite must break the order")

        for round_id in range(3):
            with self.subTest(round=round_id):
                self.assertEqual(
                    sorted(_drawn(_server(), _roster(file_order), round_id)),
                    sorted(_drawn(_server(), _roster(sorted(file_order)), round_id)),
                )


class NoShippedScheduleMovesTest(unittest.TestCase):
    def test_sorting_changes_nothing_on_a_roster_that_is_already_sorted(self) -> None:
        """Both generators write clients.jsonl sorted, so this fix is free.

        Compared against the old implementation directly rather than against a
        recorded expectation: what is being claimed is that the two agree on an
        id-sorted roster, which is every roster this repository can produce.
        """

        for ids in (
            [f"client_{index}" for index in range(ROSTER)],
            sorted(f"client_{index}" for index in range(ROSTER)),
            sorted(_client_id(f"f{index:04d}_{index % 50:02d}") for index in range(ROSTER)),
        ):
            roster = _roster(sorted(ids))
            for round_id in range(5):
                with self.subTest(first=roster[0].client_id, round=round_id):
                    size = max(1, -(-len(roster) * 1 // 100))
                    previous = random.Random(derive_seed(42, "participation", round_id)).sample(
                        list(roster), size
                    )
                    self.assertEqual(
                        _drawn(_server(), roster, round_id),
                        [client.client_id for client in previous],
                    )


class WhatTheCouplingCostTest(unittest.TestCase):
    def test_the_old_draw_and_the_new_one_share_nothing_on_an_unsorted_roster(self) -> None:
        """The measurement the module docstring quotes, kept runnable.

        Not an assertion that the new schedule is better -- neither is -- but
        that the difference is total, so "which order was clients.jsonl in"
        silently decided every participant of every round.
        """

        writers = sorted(f"writer #{index}" for index in range(ROSTER))
        file_order = [_client_id(writer) for writer in writers]
        roster = _roster(file_order)
        size = max(1, -(-ROSTER * 1 // 100))

        for round_id in range(3):
            with self.subTest(round=round_id):
                previous = {
                    client.client_id
                    for client in random.Random(derive_seed(42, "participation", round_id)).sample(
                        list(roster), size
                    )
                }
                current = set(_drawn(_server(), roster, round_id))
                self.assertEqual(len(current), size)
                self.assertEqual(len(previous & current), 0)


class DuplicateIdTest(unittest.TestCase):
    def test_a_repeated_client_id_is_refused_by_name(self) -> None:
        """Keying by id is what makes a duplicate matter, so it says so.

        Left to `random.sample` it surfaces as "Sample larger than population",
        which names neither the roster nor the id.
        """

        roster = _roster(["a", "b", "a"] + [f"c{index}" for index in range(200)])
        with self.assertRaisesRegex(ValueError, "distinct client_ids"):
            _server().sample_clients(roster, 1)


if __name__ == "__main__":
    unittest.main()
