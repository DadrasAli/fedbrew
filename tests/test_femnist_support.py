"""Tests for FEMNIST generation, modeling, and scalable evaluation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from fedbrew.clients.torch_sgd_client import TorchSGDClient
from fedbrew.core.config import load_config
from fedbrew.core.loop import _evaluation_client_selector
from fedbrew.core.protocol import ClientInfo
from fedbrew.core.registry import models, register_builtin_components
from fedbrew.data.femnist import generate_femnist_from_config
from fedbrew.data.manifest_dataset import ManifestFederatedDataset
from fedbrew.data.manifest_validation import validate_manifest
from fedbrew.models.femnist_resnet import FEMNISTResNet, build_femnist_resnet18
from fedbrew.tasks.classification.torch_classification import (
    TorchClassificationTask,
)


class _FakeDataset:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.column_names = ["image", "writer_id", "character"]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, str):
            return [record[key] for record in self.records]
        return self.records[key]

    def select(self, indices: list[int]) -> _FakeDataset:
        return _FakeDataset([self.records[index] for index in indices])


def _fake_femnist_source() -> _FakeDataset:
    records = []
    for writer_index in range(3):
        for example_index in range(10):
            records.append(
                {
                    "image": torch.full(
                        (1, 28, 28),
                        fill_value=writer_index * 10 + example_index,
                        dtype=torch.uint8,
                    ),
                    "writer_id": f"writer_{writer_index}",
                    "character": (writer_index * 10 + example_index) % 62,
                }
            )
    return _FakeDataset(records)


#: Three disjoint slices. On the 10-example fake writers: 6 train, 2 eval,
#: 2 test -- and global_test.pt is built from the test slices alone.
_SPLITS = {"train_ratio": 0.6, "eval_ratio": 0.2, "test_ratio": 0.2}


def _generator_config() -> dict[str, Any]:
    return {
        "dataset": {"name": "femnist"},
        "femnist": {"min_samples_per_client": 3, "revision": None},
        "partition": {"strategy": "natural", "num_clients": None},
    }


class FEMNISTGeneratorTests(unittest.TestCase):
    def test_natural_writers_become_split_compatible_manifest_clients(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "femnist"
            summary = generate_femnist_from_config(
                config=_generator_config(),
                output_dir=output_dir,
                seed=17,
                client_splits=_SPLITS,
                source_dataset=_fake_femnist_source(),
            )

            self.assertEqual(summary.num_clients, 3)
            self.assertEqual(summary.num_examples, 30)
            self.assertEqual(summary.num_test_examples, 6)
            self.assertFalse(
                [
                    issue
                    for issue in validate_manifest(summary.manifest_path)
                    if issue.severity == "error"
                ]
            )

            manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["partition_strategy"], "natural")
            self.assertEqual(manifest["partition_key"], "writer_id")
            self.assertEqual(manifest["input_dtype"], "uint8")
            self.assertEqual(manifest["source_revision"], None)

            dataset = ManifestFederatedDataset(summary.manifest_path)
            self.assertEqual(dataset.list_clients(), ["writer_0", "writer_1", "writer_2"])
            for client_id in dataset.list_clients():
                client_data = dataset.get_client_data(client_id)
                self.assertEqual(len(client_data["train"]["y"]), 6)
                self.assertEqual(len(client_data["eval"]["y"]), 2)
                self.assertEqual(len(client_data["test"]["y"]), 2)
                self.assertEqual(client_data["train"]["x"].dtype, torch.uint8)
            self.assertEqual(len(dataset.get_global_data()["y"]), 6)
            self.assertEqual(manifest["client_shard_format"], "split_v2")
            self.assertEqual(
                manifest["client_test_source"],
                "within_client_holdout_disjoint_from_eval",
            )
            self.assertNotIn("test_split", manifest)

    def test_writer_splits_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summaries = [
                generate_femnist_from_config(
                    config=_generator_config(),
                    output_dir=root / name,
                    seed=23,
                    client_splits=_SPLITS,
                    source_dataset=_fake_femnist_source(),
                )
                for name in ("first", "second")
            ]
            datasets = [ManifestFederatedDataset(summary.manifest_path) for summary in summaries]
            for client_id in datasets[0].list_clients():
                first = datasets[0].get_client_data(client_id)
                second = datasets[1].get_client_data(client_id)
                self.assertTrue(torch.equal(first["train"]["x"], second["train"]["x"]))
                self.assertTrue(torch.equal(first["eval"]["y"], second["eval"]["y"]))
                self.assertTrue(torch.equal(first["test"]["y"], second["test"]["y"]))


class FEMNISTHeldOutTestSplitTests(unittest.TestCase):
    """Regression tests for the pooled test set being the validation set.

    global_test.pt used to be the concatenation of every writer's EVAL slice,
    so central_test_* and val_* were the same 81,502 examples: an honest
    estimate for one fixed model at one fixed round, but the shipped configs
    select ~100 checkpoints on it with best_metric: val_accuracy_* and then
    report it, which is selection on the reported number.
    """

    def _generate(self, directory: Path, num_clients=None, seed: int = 11):
        config = _generator_config()
        config["partition"]["num_clients"] = num_clients
        return generate_femnist_from_config(
            config=config,
            output_dir=directory,
            seed=seed,
            client_splits=_SPLITS,
            source_dataset=_fake_femnist_source(),
        )

    def test_the_test_slice_is_disjoint_from_the_eval_slice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = self._generate(Path(directory) / "femnist")
            dataset = ManifestFederatedDataset(summary.manifest_path)

            for client_id in dataset.list_clients():
                data = dataset.get_client_data(client_id)
                with self.subTest(client=client_id):
                    for a, b in (("train", "eval"), ("train", "test"), ("eval", "test")):
                        overlap = {int(v) for v in data[a]["x"].flatten(1)[:, 0].tolist()} & {
                            int(v) for v in data[b]["x"].flatten(1)[:, 0].tolist()
                        }
                        self.assertFalse(overlap, f"{a} and {b} share {overlap}")

    def test_the_global_test_set_pools_test_slices_not_eval_slices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = self._generate(Path(directory) / "femnist")
            dataset = ManifestFederatedDataset(summary.manifest_path)

            pooled = {int(v) for v in dataset.get_global_data()["x"].flatten(1)[:, 0].tolist()}
            expected_test: set[int] = set()
            eval_values: set[int] = set()
            for client_id in dataset.list_clients():
                data = dataset.get_client_data(client_id)
                expected_test |= {int(v) for v in data["test"]["x"].flatten(1)[:, 0].tolist()}
                eval_values |= {int(v) for v in data["eval"]["x"].flatten(1)[:, 0].tolist()}

            self.assertEqual(pooled, expected_test)
            self.assertFalse(pooled & eval_values)

    def test_a_writers_split_does_not_move_when_num_clients_changes(self) -> None:
        """The seed keyed the split by list position, so it did."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            everyone = ManifestFederatedDataset(
                self._generate(root / "all", num_clients=None).manifest_path
            )
            subset = ManifestFederatedDataset(
                self._generate(root / "subset", num_clients=2).manifest_path
            )

            for client_id in subset.list_clients():
                with self.subTest(client=client_id):
                    for split in ("train", "eval", "test"):
                        self.assertTrue(
                            torch.equal(
                                everyone.get_client_data(client_id)[split]["y"],
                                subset.get_client_data(client_id)[split]["y"],
                            )
                        )

    @pytest.mark.fast
    def test_a_config_without_a_test_ratio_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as caught:
                generate_femnist_from_config(
                    config=_generator_config(),
                    output_dir=Path(directory) / "femnist",
                    seed=11,
                    client_splits={"train_ratio": 0.8, "eval_ratio": 0.2},
                    source_dataset=_fake_femnist_source(),
                )
            self.assertIn("test_ratio", str(caught.exception))

    def test_pre_fix_data_is_refused_when_central_test_is_evaluated(self) -> None:
        """The generated shards are not regenerated by this fix, so the stale
        marker has to be caught rather than silently believed."""

        with tempfile.TemporaryDirectory() as directory:
            summary = self._generate(Path(directory) / "femnist")
            manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
            manifest.pop("client_test_source")
            manifest["test_split"] = "within_client_holdout"
            summary.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            codes = {
                issue.code
                for issue in validate_manifest(summary.manifest_path, require_global_test=True)
                if issue.severity == "error"
            }
            self.assertIn("manifest.global_test_is_the_eval_split", codes)

    @pytest.mark.fast
    def test_two_examples_per_writer_is_no_longer_enough(self) -> None:
        config = _generator_config()
        config["femnist"]["min_samples_per_client"] = 2
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as caught:
                generate_femnist_from_config(
                    config=config,
                    output_dir=Path(directory) / "femnist",
                    seed=11,
                    client_splits=_SPLITS,
                    source_dataset=_fake_femnist_source(),
                )
            self.assertIn("at least 3", str(caught.exception))


@pytest.mark.fast
class FEMNISTModelTests(unittest.TestCase):
    def test_resnet_accepts_raw_grayscale_pixels_and_returns_62_logits(self) -> None:
        model = build_femnist_resnet18(
            {
                "input_channels": 1,
                "base_channels": 16,
                "group_norm_groups": 4,
                "num_classes": 62,
                "dropout": 0.0,
            }
        )
        inputs = torch.randint(0, 256, (4, 1, 28, 28), dtype=torch.uint8)
        outputs = model(inputs)

        self.assertEqual(tuple(outputs.shape), (4, 62))
        self.assertFalse(any(isinstance(module, nn.BatchNorm2d) for module in model.modules()))
        self.assertTrue(any(isinstance(module, nn.GroupNorm) for module in model.modules()))

    def test_input_normalization_is_fixed_and_not_configurable(self) -> None:
        """The [0, 255] contract is a property of how the shards are written.

        input_scale / input_mean / input_std used to be model config keys.
        All 13 configs that set them set 255.0 / 0.5 / 0.5, and changing one
        without regenerating the shards trains on the wrong units -- so they
        are constants now, and a config naming one is refused rather than
        quietly accepted at the value it already had.
        """

        model = build_femnist_resnet18(
            {"base_channels": 16, "group_norm_groups": 4, "dropout": 0.0}
        )
        model.eval()

        # A uniform mid-grey image lands exactly on the mean, so the whole
        # normalized tensor is zero: 127.5/255 = 0.5, (0.5 - 0.5)/0.5 = 0.
        grey = torch.full((1, 1, 28, 28), 127.5)
        zeros = torch.zeros((1, 1, 28, 28))
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(grey), model(zeros + 127.5)))
            # White maps to +1, black to -1: the [-1, 1] range the stem expects.
            white = model(torch.full((1, 1, 28, 28), 255.0))
            black = model(torch.zeros((1, 1, 28, 28)))
        self.assertFalse(torch.allclose(white, black))

        for name in ("input_scale", "input_mean", "input_std"):
            with self.subTest(key=name):
                with self.assertRaisesRegex(ValueError, f"does not read: model.{name}"):
                    build_femnist_resnet18({name: 1.0})

    def test_default_model_is_registered_and_compact(self) -> None:
        register_builtin_components()
        self.assertTrue(models.exists("femnist_resnet18"))
        model = models.get("femnist_resnet18")({"name": "femnist_resnet18"})
        self.assertIsInstance(model, FEMNISTResNet)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.assertGreater(parameter_count, 2_000_000)
        self.assertLess(parameter_count, 4_000_000)


@pytest.mark.fast
class FEMNISTRuntimeTests(unittest.TestCase):
    def test_cosine_local_learning_rate_reaches_configured_minimum(self) -> None:
        task = TorchClassificationTask(
            model_config={
                "name": "mlp",
                "input_dim": 2,
                "hidden_dim": 4,
                "num_classes": 2,
            }
        )
        client = TorchSGDClient(
            client_id="client_0",
            task=task,
            model_config=task.model_config,
            client_data={
                "x": torch.zeros(2, 2),
                "y": torch.zeros(2, dtype=torch.long),
            },
            local_iterations=1,
            batch_size=2,
            learning_rate=0.1,
            momentum=0.9,
            weight_decay=0.0,
            nesterov=True,
            learning_rate_schedule="cosine",
            min_learning_rate=0.01,
            total_rounds=5,
        )

        self.assertAlmostEqual(client._round_learning_rate(1), 0.1)
        self.assertAlmostEqual(client._round_learning_rate(5), 0.01)

    def test_research_config_enables_scalable_evaluation_and_optimizer(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        config = load_config(repository_root / "configs/dev/femnist_fedadam_cosine.yaml")

        self.assertEqual(config.model.name, "femnist_resnet18")
        self.assertEqual(config.model.num_classes, 62)
        self.assertEqual(config.evaluation.central_test.every, 10)
        self.assertEqual(config.evaluation.train.clients, "participating")
        self.assertEqual(config.client.extra["learning_rate_schedule"], "cosine")
        self.assertEqual(config.client.extra["momentum"], 0.9)

    def test_participating_evaluation_scope_preserves_server_selection_order(self) -> None:
        client_infos = [
            ClientInfo(client_id=client_id, num_examples=10) for client_id in ("a", "b", "c")
        ]
        selected = _evaluation_client_selector(client_infos, "participating", 0, "val")(
            1, ["c", "a", "c"]
        )

        self.assertEqual([info.client_id for info in selected], ["c", "a"])
        self.assertIs(
            _evaluation_client_selector(client_infos, "all", 0, "val")(1, ["c"]),
            client_infos,
        )


if __name__ == "__main__":
    unittest.main()
