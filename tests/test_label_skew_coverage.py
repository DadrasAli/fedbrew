"""label_skew must honour labels_per_client or refuse the configuration.

_assign_allowed_labels guaranteed each CLIENT got labels_per_client labels; it
never guaranteed each LABEL got a client. Below num_clients * labels_per_client
< num_classes every uncovered label was spread round-robin across ALL clients,
so a partition asked for 2 labels per client came out with up to 56 -- a
near-IID partition written to disk under partition_strategy: label_skew, with
labels_per_client still in its config and nothing printed.
"""

from __future__ import annotations

import unittest

import pytest

from fedbrew.data.partitioners.label_skew import partition_label_skew

pytestmark = pytest.mark.fast


def _labels(num_classes: int, per_class: int = 40) -> list[int]:
    return [index % num_classes for index in range(num_classes * per_class)]


def _corpus_for(num_classes: int, num_clients: int) -> list[int]:
    """A corpus with at least one example per client, which is now required.

    The COVERED rows below use up to 1000 clients against what was a fixed
    400-example fixture, so 600 to 668 of those clients came out empty and this
    guard asserted the label budget over a partition two thirds of which was
    zero-row shards. `fill_empty_clients` refuses that shape now (chapter 05
    §3.7), and the rows meant to stand for the shipped MNIST configs -- 1000
    clients over 60,000 examples -- are better served by a corpus that at least
    has an example per client.
    """

    return _labels(num_classes, per_class=max(40, -(-num_clients // num_classes)))


def _realized(labels: list[int], partitions: dict[str, list[int]]) -> tuple[int, int]:
    """(most labels any one client holds, number of labels held by nobody)."""

    per_client = [{labels[index] for index in values} for values in partitions.values()]
    covered = set().union(*per_client) if per_client else set()
    return max((len(seen) for seen in per_client), default=0), len(set(labels) - covered)


class LabelBudgetIsHonouredTests(unittest.TestCase):
    #: (num_classes, num_clients, labels_per_client) with enough slots to cover
    #: every label. The three shipped MNIST configs are the K=10/1000 rows and
    #: data/configs/synthetic_label_skew.yaml is K=3/5/2.
    COVERED = (
        (10, 1000, 1),
        (10, 1000, 5),
        (10, 1000, 10),
        (3, 5, 2),
        (10, 5, 2),
        (62, 31, 2),
        (10, 10, 1),
    )

    def test_no_client_exceeds_its_label_budget(self) -> None:
        for num_classes, num_clients, per_client in self.COVERED:
            for seed in (0, 1, 7):
                with self.subTest(K=num_classes, clients=num_clients, lpc=per_client, seed=seed):
                    labels = _corpus_for(num_classes, num_clients)
                    partitions = partition_label_skew(labels, num_clients, per_client, seed=seed)
                    worst, uncovered = _realized(labels, partitions)
                    self.assertLessEqual(worst, per_client)
                    self.assertEqual(uncovered, 0)
                    # The budget held before over partitions that were two
                    # thirds empty shards. It has to hold over real clients.
                    self.assertEqual([n for n, v in partitions.items() if not v], [])

    def test_every_example_is_assigned_exactly_once(self) -> None:
        labels = _labels(10)
        partitions = partition_label_skew(labels, 5, 2, seed=0)
        assigned = [index for values in partitions.values() for index in values]
        self.assertEqual(sorted(assigned), list(range(len(labels))))


class UnsatisfiableConfigurationTests(unittest.TestCase):
    #: Below the threshold the budget and full coverage cannot both hold. The
    #: last two are label_skew pointed at FEMNIST's 62 classes and OpenImage's
    #: class count with a small client roster -- the experiment the strategy
    #: exists for.
    UNCOVERED = ((10, 8, 1), (10, 4, 2), (62, 10, 3), (62, 5, 2), (100, 20, 2))

    def test_a_configuration_that_cannot_cover_every_label_is_refused(self) -> None:
        for num_classes, num_clients, per_client in self.UNCOVERED:
            with self.subTest(K=num_classes, clients=num_clients, lpc=per_client):
                with self.assertRaises(ValueError) as caught:
                    partition_label_skew(_labels(num_classes), num_clients, per_client, seed=0)
                message = str(caught.exception)
                # Both sides of the comparison, so the fix needs no arithmetic.
                self.assertIn(str(num_clients * per_client), message)
                self.assertIn(str(num_classes), message)
                self.assertIn("near-IID", message)

    def test_the_boundary_is_exactly_enough_slots(self) -> None:
        labels = _labels(12)
        # 12 slots for 12 labels: satisfiable, and satisfied.
        partitions = partition_label_skew(labels, 6, 2, seed=0)
        worst, uncovered = _realized(labels, partitions)
        self.assertLessEqual(worst, 2)
        self.assertEqual(uncovered, 0)
        # 10 slots for 12 labels: refused.
        with self.assertRaises(ValueError):
            partition_label_skew(labels, 5, 2, seed=0)

    def test_labels_per_client_above_the_class_count_is_still_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            partition_label_skew(_labels(3), 10, 4, seed=0)


if __name__ == "__main__":
    unittest.main()
