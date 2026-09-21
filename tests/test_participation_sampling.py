"""Which clients train in which round, across seeds.

In a cross-device regime the participation schedule is the dominant source of
run-to-run variance: at participation_rate 0.01 over 3,597 FEMNIST writers, a
client is sampled about five times in a 500-round run. Replicates that share
that schedule are not replicates of the thing a federated-optimizer comparison
is averaging over.
"""

from __future__ import annotations

import random
import unittest

import pytest

from fedbrew.core.protocol import ClientInfo
from fedbrew.core.seeding import derive_seed
from fedbrew.servers.fedavg import FedAvgServer

pytestmark = pytest.mark.fast


def _clients(count: int) -> list[ClientInfo]:
    return [ClientInfo(client_id=f"c{index}", num_examples=10) for index in range(count)]


class ParticipationScheduleTests(unittest.TestCase):
    """Neighbouring seeds must not share a participation schedule.

    Seeds s and s+1 once produced the same schedule shifted by one round, so a
    3-seed spread measured a phase offset rather than replicate variance.
    """

    def _schedule(self, seed: int, rounds: int = 100, count: int = 20) -> list[tuple[str, ...]]:
        server = FedAvgServer(seed=seed, participation_rate=0.25)
        clients = _clients(count)
        return [
            tuple(sorted(info.client_id for info in server.sample_clients(clients, r)))
            for r in range(1, rounds + 1)
        ]

    def test_adjacent_seeds_do_not_share_a_shifted_schedule(self) -> None:
        """Random(42 + 2) and Random(43 + 1) are the same generator.

        Under `seed + round_id`, seed 43 saw at round r exactly the clients seed
        42 saw at round r+1, for 99 of 100 rounds -- so a 3-seed spread was the
        spread over three runs sharing their participation schedule.
        """

        base = self._schedule(42)
        for offset in (1, 2):
            other = self._schedule(42 + offset)
            with self.subTest(offset=offset):
                shifted = sum(1 for r in range(len(base) - offset) if other[r] == base[r + offset])
                self.assertLess(
                    shifted,
                    len(base) // 10,
                    f"seed {42 + offset} replays seed 42 shifted by {offset} "
                    f"in {shifted} of {len(base) - offset} rounds",
                )

    def test_the_old_additive_derivation_is_what_that_test_would_catch(self) -> None:
        """Pins the defect itself, so the test above cannot pass vacuously."""

        clients = list(range(20))
        additive = lambda seed, r: seed + r  # noqa: E731 - the old derivation
        schedule = lambda seed: [  # noqa: E731
            tuple(sorted(random.Random(additive(seed, r)).sample(clients, 5)))
            for r in range(1, 101)
        ]
        base, other = schedule(42), schedule(43)

        self.assertTrue(all(other[r] == base[r + 1] for r in range(99)))

    def test_a_schedule_is_still_reproducible_from_its_seed(self) -> None:
        self.assertEqual(self._schedule(42), self._schedule(42))

    def test_different_seeds_give_different_schedules(self) -> None:
        self.assertNotEqual(self._schedule(42), self._schedule(43))

    def test_the_round_is_still_what_varies_within_a_run(self) -> None:
        schedule = self._schedule(42, rounds=20)
        self.assertGreater(len(set(schedule)), 1)

    def test_full_participation_still_returns_everyone(self) -> None:
        server = FedAvgServer(seed=42, participation_rate=1.0)
        clients = _clients(8)
        self.assertEqual(len(server.sample_clients(clients, 3)), 8)

    def test_the_sample_size_is_unchanged(self) -> None:
        """ceil(n * rate), at least 1 -- the paper's m = max(C*K, 1)."""

        for count, rate, expected in ((20, 0.25, 5), (3597, 0.01, 36), (10, 0.01, 1)):
            with self.subTest(count=count, rate=rate):
                server = FedAvgServer(seed=42, participation_rate=rate)
                self.assertEqual(len(server.sample_clients(_clients(count), 1)), expected)

    def test_derive_seed_fields_do_not_collide(self) -> None:
        """Length-prefixed encoding, so ("ab","c") and ("a","bc") differ."""

        self.assertNotEqual(derive_seed(1, "ab", "c"), derive_seed(1, "a", "bc"))


if __name__ == "__main__":
    unittest.main()
