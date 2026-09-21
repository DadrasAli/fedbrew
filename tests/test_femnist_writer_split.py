"""The natural writer split, at the unit where the three slices are cut.

tests/test_femnist_support.py::FEMNISTHeldOutTestSplitTests covers this end to
end on one fake corpus: 3 writers, 10 examples each, ratios 0.6/0.2/0.2. That
pins that fix on the shape it was fixed against. What it
cannot reach is the boundary -- writers with 3, 4 or 5 examples, where the
ratios alone would round a slice to zero and the max(1, ...) floors in
_split_writer_examples are the only thing keeping all three non-empty.

FEMNIST's real writer sizes run from 16 to 525 examples, so the floors are not
exercised by the shipped data; they are exercised by min_samples_per_client,
which any config may lower to 3. A writer whose eval or test slice rounds away
produces a client that is silently absent from val_* or from global_test.pt
while still counting in the manifest.

Two properties, over every size and ratio combination:

  disjoint     no example is in two slices
  exhaustive   every example is in one -- the union is exactly range(n)

Exhaustiveness is the half the end-to-end test cannot see: a dropped index
looks like a slightly smaller writer, and the manifest records the sizes that
were written rather than the sizes that were asked for.
"""

from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

import pytest
import torch

from fedbrew.data.femnist import _split_writer_examples, generate_femnist_from_config
from fedbrew.data.manifest_dataset import ManifestFederatedDataset
from tests.test_femnist_support import (
    _SPLITS,
    _fake_femnist_source,
    _generator_config,
)

#: Writer sizes: the three smallest legal ones, then the ends of FEMNIST's own
#: range (16 to 525 in the real corpus) and a couple in between.
SIZES = (3, 4, 5, 7, 16, 41, 100, 525)

#: (eval_ratio, test_ratio). The shipped femnist_natural.yaml is 0.1/0.1; the
#: rest span from ratios that round away on small writers to ratios that leave
#: training with barely anything.
RATIOS = ((0.1, 0.1), (0.2, 0.2), (0.05, 0.05), (0.4, 0.4), (0.01, 0.3))

SEEDS = (0, 3, 99)


@pytest.mark.fast
class WriterSplitIsAPartitionTests(unittest.TestCase):
    def test_the_three_slices_are_disjoint_and_exhaustive(self) -> None:
        for size in SIZES:
            for eval_ratio, test_ratio in RATIOS:
                for seed in SEEDS:
                    with self.subTest(size=size, ratios=(eval_ratio, test_ratio), seed=seed):
                        train, evaluation, test = _split_writer_examples(
                            num_examples=size,
                            eval_ratio=eval_ratio,
                            test_ratio=test_ratio,
                            seed=seed,
                        )
                        assigned = Counter(
                            int(index)
                            for slice_ in (train, evaluation, test)
                            for index in slice_.tolist()
                        )
                        # One Counter comparison pins both properties at once:
                        # a duplicate raises a count to 2, a drop leaves a key
                        # missing, and a compensating pair of both still fails.
                        self.assertEqual(assigned, Counter(range(size)))

    def test_no_slice_is_ever_empty(self) -> None:
        """The floors, which is the whole reason a writer needs three examples.

        An empty eval slice is a client that never appears in val_*; an empty
        test slice is a client missing from global_test.pt. Neither raises.
        """

        for size in SIZES:
            for eval_ratio, test_ratio in RATIOS:
                for seed in SEEDS:
                    with self.subTest(size=size, ratios=(eval_ratio, test_ratio), seed=seed):
                        slices = _split_writer_examples(
                            num_examples=size,
                            eval_ratio=eval_ratio,
                            test_ratio=test_ratio,
                            seed=seed,
                        )
                        for name, slice_ in zip(("train", "eval", "test"), slices, strict=True):
                            self.assertGreater(len(slice_), 0, f"{name} slice is empty")

    def test_a_writer_with_two_examples_is_refused(self) -> None:
        # Three slices cannot come out of two examples; the generator's
        # min_samples_per_client >= 3 is enforced here as well.
        for size in (0, 1, 2):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, "at least three examples"):
                    _split_writer_examples(
                        num_examples=size, eval_ratio=0.1, test_ratio=0.1, seed=0
                    )

    def test_the_split_is_a_function_of_the_seed_alone(self) -> None:
        first = _split_writer_examples(num_examples=41, eval_ratio=0.1, test_ratio=0.1, seed=7)
        again = _split_writer_examples(num_examples=41, eval_ratio=0.1, test_ratio=0.1, seed=7)
        other = _split_writer_examples(num_examples=41, eval_ratio=0.1, test_ratio=0.1, seed=8)
        for left, right in zip(first, again, strict=True):
            self.assertTrue(torch.equal(left, right))
        self.assertFalse(
            all(torch.equal(left, right) for left, right in zip(first, other, strict=True))
        )

    def test_ratios_that_leave_no_room_to_train_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "room for training"):
            _split_writer_examples(num_examples=100, eval_ratio=0.5, test_ratio=0.5, seed=0)

    def test_a_ratio_outside_the_unit_interval_is_refused(self) -> None:
        for eval_ratio, test_ratio in ((0.0, 0.1), (0.1, 0.0), (1.0, 0.1), (-0.1, 0.1)):
            with self.subTest(ratios=(eval_ratio, test_ratio)):
                with self.assertRaises(ValueError):
                    _split_writer_examples(
                        num_examples=100,
                        eval_ratio=eval_ratio,
                        test_ratio=test_ratio,
                        seed=0,
                    )


class WriterPartitionTests(unittest.TestCase):
    """Across writers, not within one: the natural strategy is also a partition.

    A writer is a client, so "every source example reaches exactly one client"
    is the same property the other four strategies are held to in
    tests/test_partition_disjointness.py. The fake corpus encodes each example's
    identity in its pixel value, so the union can be compared to the source.
    """

    def _dataset(self, directory: Path) -> ManifestFederatedDataset:
        summary = generate_femnist_from_config(
            config=_generator_config(),
            output_dir=directory,
            seed=11,
            client_splits=_SPLITS,
            source_dataset=_fake_femnist_source(),
        )
        return ManifestFederatedDataset(summary.manifest_path)

    def test_every_source_example_reaches_exactly_one_client(self) -> None:
        source = _fake_femnist_source()
        expected = Counter(int(record["image"].flatten()[0]) for record in source.records)

        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory) / "femnist")
            assigned: Counter[int] = Counter()
            for client_id in dataset.list_clients():
                data = dataset.get_client_data(client_id)
                for split in ("train", "eval", "test"):
                    assigned.update(
                        int(value) for value in data[split]["x"].flatten(1)[:, 0].tolist()
                    )
            self.assertEqual(assigned, expected)

    def test_clients_share_no_examples_with_each_other(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory) / "femnist")
            seen: dict[int, str] = {}
            for client_id in dataset.list_clients():
                data = dataset.get_client_data(client_id)
                for split in ("train", "eval", "test"):
                    for value in data[split]["x"].flatten(1)[:, 0].tolist():
                        value = int(value)
                        self.assertNotIn(
                            value,
                            seen,
                            f"example {value} is on {seen.get(value)} and {client_id}",
                        )
                        seen[value] = client_id


if __name__ == "__main__":
    unittest.main()
