"""quantity_skew must draw client sizes from a stated law, and never empty ones.

min_size: 0 passed validation and the greedy allocator then wrote 91 empty
shards out of 100 while handing one client 83% of the corpus. Even with a
sensible floor the shape was a by-product of the loop -- rng.randint(1,
everything left) -- so the first draws took most of the data and the rest of
the clients sat pinned on min_size. Nothing in the docstring or the config
schema said what distribution to expect.
"""

from __future__ import annotations

import unittest

import pytest

from fedbrew.data.partitioners.quantity_skew import partition_quantity_skew

pytestmark = pytest.mark.fast

#: (total, num_clients, min_size, max_size) covering a large split, a small
#: one, a tight band, and the two boundaries the size checks allow.
SHAPES = (
    (60000, 100, 100, 2000),
    (60000, 20, 100, 20000),
    (60000, 10, 1000, 20000),
    (1000, 10, 50, 500),
    (100, 10, 10, 10),
    (100, 10, 10, 100),
    (7, 7, 1, 1),
)


def _sizes(total: int, num_clients: int, min_size: int, max_size: int, seed: int):
    partitions = partition_quantity_skew(list(range(total)), num_clients, min_size, max_size, seed)
    return partitions, [len(values) for values in partitions.values()]


def _gini(values):
    ordered = sorted(values)
    n = len(ordered)
    total = sum(ordered)
    if total == 0:
        return 0.0
    weighted = sum((rank + 1) * value for rank, value in enumerate(ordered))
    return 2 * weighted / (n * total) - (n + 1) / n


class SizeContractTests(unittest.TestCase):
    def test_sizes_are_inside_the_band_and_sum_to_the_corpus(self) -> None:
        for shape in SHAPES:
            total, num_clients, min_size, max_size = shape
            for seed in (0, 1, 13):
                with self.subTest(shape=shape, seed=seed):
                    _, sizes = _sizes(*shape, seed)
                    self.assertEqual(len(sizes), num_clients)
                    self.assertEqual(sum(sizes), total)
                    self.assertGreaterEqual(min(sizes), min_size)
                    self.assertLessEqual(max(sizes), max_size)

    def test_no_client_is_empty(self) -> None:
        for shape in SHAPES:
            with self.subTest(shape=shape):
                _, sizes = _sizes(*shape, 0)
                self.assertGreater(min(sizes), 0)

    def test_every_example_is_assigned_exactly_once(self) -> None:
        partitions, _ = _sizes(1000, 10, 50, 500, seed=0)
        assigned = [index for values in partitions.values() for index in values]
        self.assertEqual(sorted(assigned), list(range(1000)))

    def test_the_same_seed_gives_the_same_split(self) -> None:
        first, _ = _sizes(1000, 10, 50, 500, seed=4)
        again, _ = _sizes(1000, 10, 50, 500, seed=4)
        other, _ = _sizes(1000, 10, 50, 500, seed=5)
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_a_band_of_one_size_gives_equal_clients(self) -> None:
        _, sizes = _sizes(100, 10, 10, 10, seed=0)
        self.assertEqual(sizes, [10] * 10)


class SizeShapeTests(unittest.TestCase):
    """The distribution is skewed, but not by pinning most clients on the floor."""

    SHAPE = (60000, 20, 100, 20000)

    def test_most_clients_are_not_stuck_at_the_minimum(self) -> None:
        # The greedy allocator put 13 of these 20 clients within three examples
        # of min_size, because the first draws had already taken the corpus.
        for seed in (0, 1, 13):
            with self.subTest(seed=seed):
                _, sizes = _sizes(*self.SHAPE, seed)
                floor = self.SHAPE[2]
                at_floor = sum(1 for size in sizes if size <= floor * 1.5)
                self.assertLess(at_floor, len(sizes) // 4)

    def test_the_split_is_still_unequal(self) -> None:
        # Quantity skew that is not skewed is not quantity skew.
        for seed in (0, 1, 13):
            with self.subTest(seed=seed):
                _, sizes = _sizes(*self.SHAPE, seed)
                self.assertGreater(_gini(sizes), 0.2)
                self.assertLess(_gini(sizes), 0.75)


class RefusalTests(unittest.TestCase):
    def test_min_size_zero_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            partition_quantity_skew(list(range(60000)), 100, 0, 60000, seed=0)
        message = str(caught.exception)
        self.assertIn("min_size must be at least 1", message)
        # Say what it did, not just that it is disallowed.
        self.assertIn("empty", message)

    def test_a_negative_min_size_is_still_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_size must be at least 1"):
            partition_quantity_skew(list(range(100)), 10, -1, 50, seed=0)

    def test_the_existing_capacity_checks_still_hold(self) -> None:
        with self.assertRaisesRegex(ValueError, "not enough examples"):
            partition_quantity_skew(list(range(10)), 10, 5, 50, seed=0)
        with self.assertRaisesRegex(ValueError, "too many examples"):
            partition_quantity_skew(list(range(1000)), 10, 1, 50, seed=0)


if __name__ == "__main__":
    unittest.main()
