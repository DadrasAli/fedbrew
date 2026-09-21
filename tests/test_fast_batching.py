"""The fast batching path must stay observationally identical to DataLoader.

These tests pin the optimization that keeps client mini-batching off the CPU:
if the resident-tensor batcher ever diverges from ``DataLoader`` in batch
contents, batch order, or shuffling RNG, seeded experiments stop reproducing.
"""

from __future__ import annotations

import unittest

import pytest
import torch

from fedbrew.tasks.classification.torch_classification import (
    TorchClassificationTask,
    _DeviceTensorBatches,
)


def _client_data(num_examples: int = 53) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "x": torch.randint(
            0, 255, (num_examples, 1, 28, 28), dtype=torch.uint8, generator=generator
        ),
        "y": torch.randint(0, 62, (num_examples,), generator=generator),
    }


def _collect(task: TorchClassificationTask, data: dict, config: dict) -> list:
    return [
        (features.clone(), targets.clone())
        for features, targets in task.build_dataloader(data, config)
    ]


class FastBatchingEquivalenceTests(unittest.TestCase):
    """Fast batching must match the DataLoader path batch for batch."""

    def setUp(self) -> None:
        self.data = _client_data()

    def _assert_same_batches(self, config: dict) -> None:
        fast = _collect(
            TorchClassificationTask(device="cpu", fast_batching=True),
            self.data,
            config,
        )
        reference = _collect(
            TorchClassificationTask(device="cpu", fast_batching=False),
            self.data,
            config,
        )
        self.assertEqual(len(fast), len(reference), f"batch count differs for {config}")
        for index, ((x1, y1), (x2, y2)) in enumerate(zip(fast, reference, strict=True)):
            self.assertTrue(torch.equal(x1, x2), f"features differ in batch {index} for {config}")
            self.assertTrue(torch.equal(y1, y2), f"targets differ in batch {index} for {config}")

    def test_sequential_batches_match_dataloader(self) -> None:
        for batch_size in (16, 53, 64):
            for drop_last in (False, True):
                with self.subTest(batch_size=batch_size, drop_last=drop_last):
                    self._assert_same_batches(
                        {
                            "batch_size": batch_size,
                            "shuffle": False,
                            "drop_last": drop_last,
                        }
                    )

    def test_seeded_shuffle_matches_dataloader(self) -> None:
        for batch_size in (16, 53, 64):
            for drop_last in (False, True):
                with self.subTest(batch_size=batch_size, drop_last=drop_last):
                    self._assert_same_batches(
                        {
                            "batch_size": batch_size,
                            "shuffle": True,
                            "drop_last": drop_last,
                            "seed": 1234,
                        }
                    )

    def test_repeated_epochs_match_dataloader(self) -> None:
        for drop_last in (False, True):
            with self.subTest(drop_last=drop_last):
                config = {
                    "batch_size": 16,
                    "shuffle": True,
                    "seed": 7,
                    "drop_last": drop_last,
                }
                fast_loader = TorchClassificationTask(
                    device="cpu", fast_batching=True
                ).build_dataloader(self.data, config)
                reference_loader = TorchClassificationTask(
                    device="cpu", fast_batching=False
                ).build_dataloader(self.data, config)

                for epoch in range(4):
                    fast = [targets.clone() for _, targets in fast_loader]
                    reference = [targets.clone() for _, targets in reference_loader]
                    self.assertEqual(len(fast), len(reference))
                    for index, (left, right) in enumerate(zip(fast, reference, strict=True)):
                        self.assertTrue(
                            torch.equal(left, right),
                            f"epoch {epoch} batch {index} order differs",
                        )

    def test_early_exit_keeps_generator_aligned_with_dataloader(self) -> None:
        config = {"batch_size": 16, "shuffle": True, "seed": 11}
        fast_loader = TorchClassificationTask(device="cpu", fast_batching=True).build_dataloader(
            self.data, config
        )
        reference_loader = TorchClassificationTask(
            device="cpu", fast_batching=False
        ).build_dataloader(self.data, config)

        # Abandon the first epoch after one batch, as max_local_steps does.
        next(iter(fast_loader))
        next(iter(reference_loader))

        fast = torch.cat([targets.clone() for _, targets in fast_loader])
        reference = torch.cat([targets.clone() for _, targets in reference_loader])
        self.assertTrue(torch.equal(fast, reference))

    @pytest.mark.fast
    def test_successive_epochs_use_different_orders(self) -> None:
        loader = TorchClassificationTask(device="cpu", fast_batching=True).build_dataloader(
            self.data, {"batch_size": 16, "shuffle": True, "seed": 7}
        )
        first = torch.cat([targets.clone() for _, targets in loader])
        second = torch.cat([targets.clone() for _, targets in loader])
        self.assertFalse(torch.equal(first, second))

    @pytest.mark.fast
    def test_uint8_features_are_widened_to_float(self) -> None:
        loader = TorchClassificationTask(device="cpu", fast_batching=True).build_dataloader(
            self.data, {"batch_size": 8, "shuffle": False}
        )
        features, targets = next(iter(loader))
        self.assertEqual(features.dtype, torch.float32)
        self.assertEqual(targets.dtype, torch.int64)

    @pytest.mark.fast
    def test_length_matches_emitted_batch_count(self) -> None:
        for batch_size, drop_last in ((16, False), (16, True), (64, True)):
            with self.subTest(batch_size=batch_size, drop_last=drop_last):
                loader = TorchClassificationTask(device="cpu", fast_batching=True).build_dataloader(
                    self.data,
                    {
                        "batch_size": batch_size,
                        "shuffle": False,
                        "drop_last": drop_last,
                    },
                )
                self.assertEqual(len(loader), len(list(loader)))

    @pytest.mark.fast
    def test_worker_processes_fall_back_to_dataloader(self) -> None:
        task = TorchClassificationTask(
            device="cpu",
            fast_batching=True,
            dataloader_config={"num_workers": 2},
        )
        loader = task.build_dataloader(self.data, {"batch_size": 8, "shuffle": False})
        self.assertNotIsInstance(loader, _DeviceTensorBatches)

    @pytest.mark.fast
    def test_mismatched_lengths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _DeviceTensorBatches(
                features=torch.zeros(4, 2),
                targets=torch.zeros(3, dtype=torch.long),
                batch_size=2,
                shuffle=False,
                drop_last=False,
                device=torch.device("cpu"),
            )


@pytest.mark.fast
class ModelReuseTests(unittest.TestCase):
    """Model reuse must hand back one instance without leaking stale state."""

    model_config = {"name": "mlp", "input_dim": 4, "hidden_dim": 8, "num_classes": 3}

    def test_reuse_returns_the_same_instance(self) -> None:
        task = TorchClassificationTask(model_config=self.model_config, device="cpu")
        self.assertIs(task.build_model({}), task.build_model({}))

    def test_reuse_disabled_returns_distinct_instances(self) -> None:
        task = TorchClassificationTask(
            model_config=self.model_config, device="cpu", reuse_model=False
        )
        self.assertIsNot(task.build_model({}), task.build_model({}))

    def test_distinct_architectures_are_cached_separately(self) -> None:
        task = TorchClassificationTask(model_config=self.model_config, device="cpu")
        wide = task.build_model({"hidden_dim": 16})
        narrow = task.build_model({"hidden_dim": 8})
        self.assertIsNot(wide, narrow)
        self.assertIs(wide, task.build_model({"hidden_dim": 16}))

    def test_metadata_is_cached_per_model_and_still_correct(self) -> None:
        task = TorchClassificationTask(model_config=self.model_config, device="cpu")
        model = task.build_model({})
        first = task.federated_model_state_metadata(model)
        second = task.federated_model_state_metadata(model)
        self.assertEqual(first, second)
        self.assertEqual(
            first["total_parameters"],
            sum(int(p.numel()) for p in model.parameters()),
        )

    def test_cached_metadata_is_not_shared_mutably(self) -> None:
        task = TorchClassificationTask(model_config=self.model_config, device="cpu")
        model = task.build_model({})
        task.federated_model_state_metadata(model)["total_parameters"] = -1
        self.assertNotEqual(task.federated_model_state_metadata(model)["total_parameters"], -1)


if __name__ == "__main__":
    unittest.main()
