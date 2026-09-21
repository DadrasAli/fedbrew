"""Regression tests for persisted client partitions of official test data."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pytest
import torch

from fedbrew.data.generate import _write_torch_shard_dataset
from fedbrew.data.manifest_dataset import ManifestFederatedDataset
from fedbrew.data.manifest_validation import validate_manifest
from fedbrew.data.official_test_partitioning import partition_test_indices_like_train
from fedbrew.data.writers.torch_shards import load_client_shard


class ClassificationClientTestPartitionTests(unittest.TestCase):
    def test_official_test_is_partitioned_without_changing_global_test(self) -> None:
        train_x = torch.arange(12, dtype=torch.float32).reshape(12, 1)
        train_y = torch.tensor([0] * 6 + [1] * 6)
        official_test_x = torch.arange(100, 108, dtype=torch.float32).reshape(8, 1)
        official_test_y = torch.tensor([0] * 4 + [1] * 4)
        train_partitions = {
            "client_0": list(range(0, 6)),
            "client_1": list(range(6, 12)),
        }

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            manifest_path = _write_torch_shard_dataset(
                output_dir=output_dir,
                dataset_name="mnist",
                train_x=train_x,
                train_y=train_y,
                test_x=official_test_x,
                test_y=official_test_y,
                partitions=train_partitions,
                num_clients=2,
                metadata={
                    "input_dim": 1,
                    "num_classes": 2,
                    "source": "test fixture",
                },
                partition_strategy="label_skew",
                client_splits={"train_ratio": 0.5, "eval_ratio": 0.5},
                seed=17,
            )
            dataset = ManifestFederatedDataset(manifest_path)

            client_0 = dataset.get_client_data("client_0")
            client_1 = dataset.get_client_data("client_1")
            self.assertEqual(
                set(client_0),
                {
                    "train",
                    "eval",
                    "test",
                    "num_examples",
                    "num_train_examples",
                    "num_eval_examples",
                    "num_test_examples",
                    "metadata",
                },
            )
            self.assertTrue(torch.all(client_0["test"]["y"] == 0))
            self.assertTrue(torch.all(client_1["test"]["y"] == 1))
            self.assertTrue(torch.all(client_0["eval"]["x"] < 100))
            self.assertTrue(torch.all(client_1["eval"]["x"] < 100))

            local_test_x = torch.cat([client_0["test"]["x"], client_1["test"]["x"]]).flatten()
            self.assertTrue(
                torch.equal(
                    torch.sort(local_test_x).values,
                    torch.sort(official_test_x.flatten()).values,
                )
            )
            global_test = dataset.get_global_data("test")
            self.assertTrue(torch.equal(global_test["x"], official_test_x))
            self.assertTrue(torch.equal(global_test["y"], official_test_y))

            metadata = dataset.get_client_metadata("client_0")
            self.assertEqual(
                metadata["num_train_examples"] + metadata["num_eval_examples"],
                6,
            )
            self.assertEqual(metadata["num_test_examples"], 4)
            # num_examples is every split, so the client's six training-corpus
            # rows plus the four official test rows partitioned to it.
            self.assertEqual(metadata["num_examples"], 10)
            self.assertEqual(
                metadata["num_examples"],
                metadata["num_train_examples"]
                + metadata["num_eval_examples"]
                + metadata["num_test_examples"],
            )
            self.assertFalse(
                any(
                    issue.severity == "error"
                    for issue in validate_manifest(
                        manifest_path,
                        require_client_test=True,
                    )
                )
            )

    def test_validation_rejects_eval_only_client_shards_for_client_test(self) -> None:
        train_x = torch.arange(8, dtype=torch.float32).reshape(8, 1)
        train_y = torch.tensor([0] * 4 + [1] * 4)
        test_x = torch.arange(20, 24, dtype=torch.float32).reshape(4, 1)
        test_y = torch.tensor([0, 0, 1, 1])

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            manifest_path = _write_torch_shard_dataset(
                output_dir=output_dir,
                dataset_name="cifar10",
                train_x=train_x,
                train_y=train_y,
                test_x=test_x,
                test_y=test_y,
                partitions={
                    "client_0": list(range(0, 4)),
                    "client_1": list(range(4, 8)),
                },
                num_clients=2,
                metadata={
                    "input_shape": [1],
                    "num_classes": 2,
                    "source": "test fixture",
                },
                partition_strategy="label_skew",
                client_splits={"train_ratio": 0.5, "eval_ratio": 0.5},
                seed=23,
            )
            shard_path = output_dir / "shards" / "client_0.pt"
            shard = load_client_shard(shard_path)
            del shard["test"]
            torch.save(shard, shard_path)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["client_shard_format"] = "split_v1"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            strict_codes = {
                issue.code
                for issue in validate_manifest(
                    manifest_path,
                    require_client_test=True,
                )
            }
            server_codes = {
                issue.code
                for issue in validate_manifest(
                    manifest_path,
                    require_client_test=False,
                )
            }
            self.assertIn("manifest.shard_test_missing", strict_codes)
            self.assertNotIn("manifest.shard_test_missing", server_codes)


@pytest.mark.fast
class DegenerateTestPartitionsTests(unittest.TestCase):
    """The two edge cases docs/05 section 4.1 describes and nothing reached.

    The chapter says a label no client trained on falls back to
    size-proportional weighting, and a client left with nothing is filled
    rather than allowed to have an empty test split. Both are implemented;
    neither test above gets near either, because both go through the happy
    path where every client trained on every label present in the test set.

    That is the same shape as the defect this file's chapter was corrected
    for: a chapter describing behaviour, an implementation of it, and nothing
    joining them. These call the partitioner directly rather than going
    through the generator, because forcing an orphan label or an empty draw
    through a whole dataset write would obscure which behaviour is under test.
    """

    def _assert_is_a_partition(
        self,
        assignments: dict[str, list[int]],
        test_count: int,
    ) -> None:
        """Whatever the fallbacks do, every test example lands exactly once."""

        assigned = [index for indices in assignments.values() for index in indices]
        self.assertEqual(sorted(assigned), list(range(test_count)))

    def test_a_label_no_client_trained_on_is_dealt_by_training_size(self) -> None:
        """Not dropped, and not dealt equally -- which is the other fallback.

        `_assign_by_weights` has a second fallback of its own: an all-zero
        weight vector becomes uniform. If the size-proportional step were
        removed, the orphan label would reach that one and be split 20/20/20
        instead of 10/20/30, so the proportions are what distinguishes the
        behaviour the chapter describes from the one underneath it.
        """

        train_labels = [0] * 10 + [1] * 20 + [1] * 30
        train_partitions = {
            "c0": list(range(0, 10)),
            "c1": list(range(10, 30)),
            "c2": list(range(30, 60)),
        }
        # Label 9 appears in the official test set and in no client's training
        # data, which is ordinary under label_skew with many classes.
        test_labels = [9] * 60

        for seed in (0, 1, 7):
            with self.subTest(seed=seed):
                assignments = partition_test_indices_like_train(
                    test_labels=test_labels,
                    train_labels=train_labels,
                    train_partitions=train_partitions,
                    strategy="label_skew",
                    seed=seed,
                )
                self.assertEqual(
                    [len(assignments[client]) for client in sorted(assignments)],
                    [10, 20, 30],
                    "the orphan label was not dealt in proportion to training size",
                )
                self._assert_is_a_partition(assignments, len(test_labels))

    def test_a_client_that_would_draw_nothing_is_filled(self) -> None:
        """An empty test split reports zero accuracy indistinguishably."""

        train_labels = [0] * 4 + [1] * 4
        train_partitions = {"c0": list(range(0, 4)), "c1": list(range(4, 8))}
        # Only c1's label is in the test set, so c0's weight for it is zero.
        test_labels = [1] * 6

        for seed in (0, 3):
            with self.subTest(seed=seed):
                assignments = partition_test_indices_like_train(
                    test_labels=test_labels,
                    train_labels=train_labels,
                    train_partitions=train_partitions,
                    strategy="label_skew",
                    seed=seed,
                )
                self.assertEqual(len(assignments["c0"]), 1, "c0 was left with no test split")
                self.assertEqual(len(assignments["c1"]), 5, "the donor gave more than one")
                self._assert_is_a_partition(assignments, len(test_labels))

    def test_the_fill_prefers_a_label_the_empty_client_trained_on(self) -> None:
        """The donor holds two labels and gives the one that means something.

        A client tested only on a class it never saw is a worse measurement
        than one tested on its own class, so the fill looks for an allowed
        label before taking whatever is first. Here c0 trained on label 1
        alone and the donor holds two examples of label 0 it could have given.
        """

        train_labels = [1, 1] + [0] * 100 + [1] * 100
        train_partitions = {"c0": [0, 1], "c1": list(range(2, 202))}
        test_labels = [0, 0, 1, 1]

        for seed in range(4):
            with self.subTest(seed=seed):
                assignments = partition_test_indices_like_train(
                    test_labels=test_labels,
                    train_labels=train_labels,
                    train_partitions=train_partitions,
                    strategy="label_skew",
                    seed=seed,
                )
                self.assertEqual(len(assignments["c0"]), 1)
                self.assertEqual(
                    [test_labels[index] for index in assignments["c0"]],
                    [1],
                    "the fill donated a label c0 never trained on",
                )
                self.assertEqual(
                    sorted(test_labels[index] for index in assignments["c1"]),
                    [0, 0, 1],
                    "the donor should have kept both label-0 examples",
                )
                self._assert_is_a_partition(assignments, len(test_labels))

    def test_a_donor_with_nothing_to_spare_is_refused_rather_than_silent(self) -> None:
        """One example per client is the documented floor, and it is enforced."""

        with self.assertRaises(ValueError):
            partition_test_indices_like_train(
                test_labels=[1],
                train_labels=[0, 1],
                train_partitions={"c0": [0], "c1": [1]},
                strategy="label_skew",
                seed=0,
            )


if __name__ == "__main__":
    unittest.main()
