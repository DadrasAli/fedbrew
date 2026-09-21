"""Every partitioner must produce a partition: disjoint, exhaustive, complete.

Each strategy is tested elsewhere for the property it exists for -- label
budgets in test_label_skew_coverage.py, size laws in test_quantity_skew_sizes.py
-- and each of those files checks exhaustiveness for its own strategy on one
shape. Nothing checked the partition property itself across every strategy, so
dirichlet and iid had no coverage at all.

The property is the one that makes a client's data mean anything: an index that
lands on two clients is a duplicated example inflating both clients' weights and
the corpus size; an index that lands on none is data silently dropped from the
experiment. Both are invisible downstream -- the manifest counts what was
written, not what was asked for -- and both change the FedAvg denominator.

Shapes cover the shipped configs (MNIST at 10/100/1000 clients, FEMNIST's 62
classes) plus the awkward cases: more clients than classes, a corpus that does
not divide evenly, and the smallest split each partitioner accepts.
"""

from __future__ import annotations

import unittest
from collections import Counter

import pytest

from fedbrew.data.partitioners.dirichlet import partition_dirichlet
from fedbrew.data.partitioners.iid import partition_iid
from fedbrew.data.partitioners.label_skew import partition_label_skew
from fedbrew.data.partitioners.quantity_skew import partition_quantity_skew

pytestmark = pytest.mark.fast

SEEDS = (0, 1, 17)


def _labels(num_classes: int, total: int) -> list[int]:
    """Round-robin labels, so every class is present and the counts are known."""

    return [index % num_classes for index in range(total)]


def _assert_is_a_partition(
    case: unittest.TestCase,
    partitions: dict[str, list[int]],
    universe: list[int],
    num_clients: int,
) -> None:
    """Disjoint, exhaustive, and one entry per requested client."""

    case.assertEqual(len(partitions), num_clients)
    case.assertEqual(sorted(partitions), sorted(f"client_{index}" for index in range(num_clients)))
    assigned = [index for values in partitions.values() for index in values]

    duplicated = [index for index, count in Counter(assigned).items() if count > 1]
    case.assertEqual(duplicated, [], f"{len(duplicated)} indices assigned twice")

    missing = sorted(set(universe) - set(assigned))
    case.assertEqual(missing, [], f"{len(missing)} indices assigned to no client")

    # Counter equality is what actually pins disjoint AND exhaustive together:
    # the two checks above can both pass on a multiset that is short and long
    # in compensating places.
    case.assertEqual(Counter(assigned), Counter(universe))


class DirichletTests(unittest.TestCase):
    #: (num_classes, total, num_clients, alpha). alpha 0.1 is the pathological
    #: end the FEMNIST/MNIST configs use, where whole classes land on one
    #: client and _fill_empty_clients has to move indices between clients --
    #: the code path most likely to lose or duplicate one.
    SHAPES = (
        (10, 600, 10, 0.1),
        (10, 600, 10, 1.0),
        (10, 600, 100, 0.1),
        (10, 1000, 1000, 0.5),
        (62, 620, 31, 0.1),
        (2, 7, 3, 0.5),
    )

    def test_dirichlet_returns_a_partition(self) -> None:
        for shape in self.SHAPES:
            num_classes, total, num_clients, alpha = shape
            for seed in SEEDS:
                with self.subTest(shape=shape, seed=seed):
                    labels = _labels(num_classes, total)
                    partitions = partition_dirichlet(labels, num_clients, alpha, seed)
                    _assert_is_a_partition(self, partitions, list(range(total)), num_clients)

    def test_filling_empty_clients_moves_indices_without_copying_them(self) -> None:
        """The donor must lose exactly what the empty client gains."""

        # alpha this small concentrates each class on one client, so with more
        # clients than examples-per-class the empty-client path runs.
        labels = _labels(10, 600)
        partitions = partition_dirichlet(labels, 60, 0.01, 0)
        _assert_is_a_partition(self, partitions, list(range(600)), 60)

    def test_indices_come_back_sorted(self) -> None:
        partitions = partition_dirichlet(_labels(10, 600), 10, 0.5, 0)
        for name, values in partitions.items():
            with self.subTest(client=name):
                self.assertEqual(values, sorted(values))


class LabelSkewTests(unittest.TestCase):
    #: Only satisfiable configurations: the unsatisfiable ones are refused, and
    #: that refusal is test_label_skew_coverage.py's subject.
    SHAPES = (
        (10, 600, 10, 2),
        (10, 600, 100, 1),
        (10, 1000, 1000, 5),
        (62, 620, 31, 2),
        (10, 601, 10, 3),
        (3, 30, 5, 2),
    )

    def test_label_skew_returns_a_partition(self) -> None:
        for shape in self.SHAPES:
            num_classes, total, num_clients, per_client = shape
            for seed in SEEDS:
                with self.subTest(shape=shape, seed=seed):
                    labels = _labels(num_classes, total)
                    partitions = partition_label_skew(labels, num_clients, per_client, seed)
                    _assert_is_a_partition(self, partitions, list(range(total)), num_clients)


class QuantitySkewTests(unittest.TestCase):
    #: (total, num_clients, min_size, max_size).
    SHAPES = (
        (600, 10, 10, 200),
        (600, 100, 1, 60),
        (1000, 10, 50, 500),
        (601, 10, 10, 200),
        (100, 10, 10, 10),
        (7, 7, 1, 1),
    )

    def test_quantity_skew_returns_a_partition(self) -> None:
        for shape in self.SHAPES:
            total, num_clients, min_size, max_size = shape
            for seed in SEEDS:
                with self.subTest(shape=shape, seed=seed):
                    partitions = partition_quantity_skew(
                        list(range(total)), num_clients, min_size, max_size, seed
                    )
                    _assert_is_a_partition(self, partitions, list(range(total)), num_clients)

    def test_a_non_contiguous_universe_is_preserved_exactly(self) -> None:
        """It slices a shuffled copy of what it is given, not range(len(...)).

        quantity_skew and iid take index VALUES, unlike dirichlet and
        label_skew which take labels and return positions. Handing them a
        universe that is not 0..N-1 is what tells the two apart.
        """

        universe = [index * 3 + 5 for index in range(120)]
        partitions = partition_quantity_skew(universe, 8, 5, 40, 0)
        _assert_is_a_partition(self, partitions, universe, 8)


class IIDTests(unittest.TestCase):
    #: (total, num_clients), including totals that do not divide evenly.
    SHAPES = ((600, 10), (601, 10), (1000, 1000), (7, 3), (5, 5), (100, 3))

    def test_iid_returns_a_partition(self) -> None:
        for total, num_clients in self.SHAPES:
            for seed in SEEDS:
                with self.subTest(total=total, clients=num_clients, seed=seed):
                    partitions = partition_iid(list(range(total)), num_clients, seed)
                    _assert_is_a_partition(self, partitions, list(range(total)), num_clients)

    def test_a_non_contiguous_universe_is_preserved_exactly(self) -> None:
        universe = [index * 7 + 2 for index in range(93)]
        _assert_is_a_partition(self, partition_iid(universe, 6, 0), universe, 6)

    def test_client_sizes_differ_by_at_most_one(self) -> None:
        # Not the partition property, but the reason iid is the baseline the
        # other strategies are compared against.
        _, sizes = 601, [len(values) for values in partition_iid(list(range(601)), 10, 0).values()]
        self.assertLessEqual(max(sizes) - min(sizes), 1)


class EmptyClientTests(unittest.TestCase):
    """A partition with an empty part is exhaustive and still unusable.

    An empty client trains on nothing, reports num_examples 0, and is folded
    into the weighted mean at weight 0 -- or divides by zero, depending on the
    strategy. quantity_skew refuses it outright; the
    others must at least not produce one on a corpus that can support them.
    """

    def test_no_strategy_leaves_a_client_empty_when_it_need_not(self) -> None:
        labels = _labels(10, 600)
        cases = {
            "dirichlet": partition_dirichlet(labels, 10, 0.5, 0),
            "label_skew": partition_label_skew(labels, 10, 2, 0),
            "quantity_skew": partition_quantity_skew(list(range(600)), 10, 10, 200, 0),
            "iid": partition_iid(list(range(600)), 10, 0),
        }
        for name, partitions in cases.items():
            with self.subTest(strategy=name):
                empty = [key for key, values in partitions.items() if not values]
                self.assertEqual(empty, [])


if __name__ == "__main__":
    unittest.main()
