"""A partition either gives every client examples or says why it cannot.

`_fill_empty_clients` -- one copy in dirichlet.py, a byte-identical one in
label_skew.py -- had two bare `return`s, and the first of them fired exactly
when the corpus was too small to give everyone an example, which is the case
the fill exists for. Measured on the pre-fix code at alpha=0.5, 100 clients:
99 examples returned 38 empty partitions, 100 examples returned none. Those 38
became zero-row shards, written as a successful generation; the run failed
hours later with "has no non-empty train split".

The second behaviour is the fill's own: a rescued client holds exactly one
example, and `split_client_indices` maps a one-element partition to
`([x], [])`. That is legal -- the per-split `evaluate()` reports zero examples
and drops the client from the val aggregate rather than failing the round -- so
generation may not refuse it. It may not stay silent about it either. The
shipped `data/configs/mnist_dirichlet.yaml` (1000 clients, alpha 0.1, seed 42)
leaves 29 clients with no val split, so every `val_*` figure it produces is an
average over 971 of the 1000 clients its own config names.
"""

from __future__ import annotations

import io
import json
import random
import tempfile
import unittest
from pathlib import Path

import pytest
import torch

from fedbrew.core.console import build_surface
from fedbrew.data.generate import _write_torch_shard_dataset
from fedbrew.data.partitioners.dirichlet import partition_dirichlet
from fedbrew.data.partitioners.empty_clients import fill_empty_clients
from fedbrew.data.partitioners.label_skew import partition_label_skew


def _labels(count: int, num_classes: int = 10) -> list[int]:
    return [index % num_classes for index in range(count)]


@pytest.mark.fast
class ACorpusTooSmallForItsRosterIsRefusedTests(unittest.TestCase):
    """Both partitioners, at the shape where the fill used to give up."""

    #: (examples, clients). The first row is the measured boundary: one example
    #: fewer than clients returned 38 empty partitions, and one more filled
    #: every client.
    TOO_SMALL = ((99, 100), (50, 100), (5, 10), (1, 2))

    def test_dirichlet_refuses_rather_than_returning_empty_clients(self) -> None:
        for count, num_clients in self.TOO_SMALL:
            with self.subTest(examples=count, clients=num_clients):
                with self.assertRaises(ValueError) as caught:
                    partition_dirichlet(_labels(count), num_clients, alpha=0.5, seed=0)
                message = str(caught.exception)
                # Both sides of the comparison, so the fix needs no arithmetic.
                self.assertIn(str(count), message)
                self.assertIn(str(num_clients), message)

    def test_label_skew_refuses_on_the_same_condition(self) -> None:
        for count, num_clients in self.TOO_SMALL:
            with self.subTest(examples=count, clients=num_clients):
                with self.assertRaises(ValueError) as caught:
                    partition_label_skew(
                        _labels(count, num_classes=min(2, count)),
                        num_clients=num_clients,
                        labels_per_client=1,
                        seed=0,
                    )
                self.assertIn(str(num_clients), str(caught.exception))

    def test_one_example_per_client_is_the_boundary_and_it_fills(self) -> None:
        """The row above the refusal, so the guard cannot pass by refusing
        everything."""

        partitions = partition_dirichlet(_labels(100), 100, alpha=0.5, seed=0)
        self.assertEqual(len(partitions), 100)
        self.assertEqual([name for name, values in partitions.items() if not values], [])
        self.assertEqual(sum(len(values) for values in partitions.values()), 100)


@pytest.mark.fast
class TheFillItselfTests(unittest.TestCase):
    def test_it_moves_one_example_into_each_empty_client_from_the_largest_donor(self) -> None:
        partitions = {
            "client_0": [0, 1, 2, 3, 4],
            "client_1": [5, 6],
            "client_2": [],
            "client_3": [],
        }

        fill_empty_clients(partitions, random.Random(0))

        self.assertEqual(sorted(len(values) for values in partitions.values()), [1, 1, 2, 3])
        self.assertEqual(
            sorted(index for values in partitions.values() for index in values),
            list(range(7)),
        )
        self.assertEqual(len(partitions["client_0"]), 3)
        self.assertEqual(len(partitions["client_1"]), 2)

    def test_a_partition_that_already_has_no_empty_client_is_untouched(self) -> None:
        partitions = {"client_0": [0, 1], "client_1": [2, 3]}

        fill_empty_clients(partitions, random.Random(0))

        self.assertEqual(partitions, {"client_0": [0, 1], "client_1": [2, 3]})


class TheShardWriterAssertsThePostconditionTests(unittest.TestCase):
    """The backstop, so the next partitioner cannot reintroduce this quietly."""

    def _write(self, partitions: dict[str, list[int]], eval_ratio: float, rail=None) -> Path:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        directory = Path(holder.name)
        _write_torch_shard_dataset(
            output_dir=directory,
            dataset_name="mnist",
            train_x=torch.arange(6, dtype=torch.float32).reshape(6, 1),
            train_y=torch.tensor([0, 0, 0, 1, 1, 1]),
            test_x=torch.arange(100, 106, dtype=torch.float32).reshape(6, 1),
            test_y=torch.tensor([0, 0, 0, 1, 1, 1]),
            partitions=partitions,
            num_clients=len(partitions),
            metadata={"input_dim": 1, "num_classes": 2, "source": "test fixture"},
            partition_strategy="dirichlet",
            client_splits={"train_ratio": 1.0 - eval_ratio, "eval_ratio": eval_ratio},
            seed=17,
            rail=rail,
        )
        return directory

    def test_an_empty_partition_is_refused_where_the_shard_would_be_written(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._write({"client_0": [0, 1, 2, 3, 4, 5], "client_1": []}, eval_ratio=0.5)
        message = str(caught.exception)
        self.assertIn("client_1", message)
        self.assertIn("dirichlet", message)

    def test_the_clients_left_without_a_val_split_are_counted_and_announced(self) -> None:
        buffer = io.StringIO()
        rail = build_surface(file=buffer).rail(["clients", "labels", "val split"])

        directory = self._write(
            {"client_0": [0, 1, 2, 3], "client_1": [4], "client_2": [5]},
            eval_ratio=0.5,
            rail=rail,
        )

        stats = json.loads((directory / "partition_stats.json").read_text(encoding="utf-8"))
        self.assertEqual(stats["clients_without_eval_split"], 2)
        # The two are the one-example clients, and only those.
        without = {row["client_id"] for row in stats["clients"] if row["num_eval_examples"] == 0}
        self.assertEqual(without, {"client_1", "client_2"})

        rendered = buffer.getvalue()
        self.assertIn("val split", rendered)
        self.assertIn("2 of 3", rendered)
        self.assertEqual(rail.warnings, 1)

    def test_nothing_is_announced_when_the_config_asks_for_no_val_split(self) -> None:
        """eval_ratio 0.0 gives every client an empty val slice by request, so
        the count would be num_clients and the warning would be noise."""

        buffer = io.StringIO()
        rail = build_surface(file=buffer).rail(["clients", "labels"])

        directory = self._write(
            {"client_0": [0, 1, 2, 3], "client_1": [4], "client_2": [5]},
            eval_ratio=0.0,
            rail=rail,
        )

        stats = json.loads((directory / "partition_stats.json").read_text(encoding="utf-8"))
        self.assertEqual(stats["clients_without_eval_split"], 0)
        self.assertNotIn("val split", buffer.getvalue())
        self.assertEqual(rail.warnings, 0)


if __name__ == "__main__":
    unittest.main()
