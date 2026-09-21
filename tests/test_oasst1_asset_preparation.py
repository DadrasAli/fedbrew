"""Focused network-free tests for pinned OASST1 asset preparation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml  # type: ignore[import-untyped]

from fedbrew.data.asset_config import (
    DatasetAssetConfigError,
    DatasetAssetPreparationConfig,
    load_dataset_asset_config,
)
from fedbrew.data.asset_manifest import (
    DatasetAssetManifestError,
    load_dataset_asset_manifest,
)
from fedbrew.data.oasst1 import _download_split, prepare_oasst1_assets

pytestmark = pytest.mark.fast

try:
    from datasets import Dataset  # type: ignore[import-untyped]
except ModuleNotFoundError:  # pragma: no cover - base install omits LLM extra.
    Dataset = None  # type: ignore[assignment,misc]

OASST1_REVISION = "37d790373da332a4c8be24bd29ec1550e7f04c3f"
QWEN_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


class OASST1AssetPreparationTests(unittest.TestCase):
    def test_shipped_configs_pin_exact_immutable_revisions(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        data_config = yaml.safe_load(
            (project_root / "data/configs/assets/oasst1.yaml").read_text(encoding="utf-8")
        )
        model_config = yaml.safe_load(
            (project_root / "configs/llm_assets/qwen2_5_0_5b_instruct.yaml").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(data_config["dataset_identifier"], "OpenAssistant/oasst1")
        self.assertEqual(data_config["revision"], OASST1_REVISION)
        self.assertEqual(model_config["model_identifier"], "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(model_config["tokenizer_identifier"], model_config["model_identifier"])
        self.assertEqual(model_config["revision"], QWEN_REVISION)
        # trust_remote_code used to be asserted here, from the config file.
        # The config never carried it anywhere: load_asset_config read four
        # keys and dropped the rest, so the line asserted nothing. Preparation
        # hard-codes False and the manifest reader refuses a manifest that says
        # otherwise, which is where the guarantee actually lives.
        self.assertNotIn("trust_remote_code", model_config)

    def test_config_rejects_a_moving_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "oasst1.yaml"
            config_path.write_text(
                "dataset_identifier: OpenAssistant/oasst1\n"
                "revision: main\n"
                f"cache_dir: {Path(directory) / 'cache'}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                DatasetAssetConfigError,
                "concrete 40-character Hugging Face commit SHA",
            ):
                load_dataset_asset_config(config_path)

    def test_missing_manifest_names_exact_preparation_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "data/raw/datasets/oasst1/asset_manifest.json"
            with self.assertRaisesRegex(
                DatasetAssetManifestError,
                "fedbrew prepare-oasst1 --config data/configs/assets/oasst1.yaml",
            ):
                load_dataset_asset_manifest(manifest_path)

    @unittest.skipIf(Dataset is None, "Datasets LLM dependency unavailable")
    def test_download_uses_only_the_pinned_explicit_preparation_path(self) -> None:
        fixture = Dataset.from_dict({"message_id": ["one"]})
        config = DatasetAssetPreparationConfig(
            dataset_identifier="OpenAssistant/oasst1",
            revision=OASST1_REVISION,
            cache_dir=Path("unused"),
            config_path=Path("oasst1.yaml"),
        )
        download_cache = Path("unused/downloads")

        with patch("datasets.load_dataset", return_value=fixture) as loader:
            self.assertIs(_download_split(config, "train", download_cache), fixture)

        loader.assert_called_once_with(
            "OpenAssistant/oasst1",
            revision=OASST1_REVISION,
            split="train",
            cache_dir=str(download_cache),
            trust_remote_code=False,
            streaming=False,
        )

    @unittest.skipIf(Dataset is None, "Datasets LLM dependency unavailable")
    def test_prepares_both_splits_and_reuses_only_verified_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "cache/oasst1"
            config_path = root / "oasst1.yaml"
            config_path.write_text(
                "dataset_identifier: OpenAssistant/oasst1\n"
                f"revision: {OASST1_REVISION}\n"
                f"cache_dir: {cache_dir}\n",
                encoding="utf-8",
            )
            fixtures = {
                "train": Dataset.from_dict({"message_id": ["t1", "t2"], "text": ["one", "two"]}),
                "validation": Dataset.from_dict({"message_id": ["v1"], "text": ["three"]}),
            }

            with patch(
                "fedbrew.data.oasst1._download_split",
                side_effect=lambda _config, split, _cache: fixtures[split],
            ) as downloader:
                manifest_path = prepare_oasst1_assets(config_path)
            self.assertEqual(downloader.call_count, 2)

            manifest = load_dataset_asset_manifest(manifest_path)
            self.assertEqual(manifest.dataset_identifier, "OpenAssistant/oasst1")
            self.assertEqual(manifest.requested_revision, OASST1_REVISION)
            self.assertEqual(manifest.resolved_revision, OASST1_REVISION)
            self.assertEqual(manifest.storage_format, "parquet")
            self.assertEqual(manifest.row_counts, {"train": 2, "validation": 1})
            self.assertEqual(set(manifest.source_hashes), {"train", "validation"})
            self.assertTrue(manifest.file_for_split("train").local_path.is_file())

            with patch(
                "fedbrew.data.oasst1._download_split",
                side_effect=AssertionError("verified cache should avoid downloads"),
            ):
                self.assertEqual(prepare_oasst1_assets(config_path), manifest_path)

            serialized = json.loads(manifest_path.read_text(encoding="utf-8"))
            serialized["files"]["validation"]["row_count"] = 99
            manifest_path.write_text(
                json.dumps(serialized),
                encoding="utf-8",
            )
            with patch(
                "fedbrew.data.oasst1._download_split",
                side_effect=lambda _config, split, _cache: fixtures[split],
            ) as downloader:
                prepare_oasst1_assets(config_path)
            self.assertEqual(downloader.call_count, 2)
            self.assertEqual(
                load_dataset_asset_manifest(manifest_path).row_counts,
                {"train": 2, "validation": 1},
            )

            train_path = (
                load_dataset_asset_manifest(manifest_path).file_for_split("train").local_path
            )
            with train_path.open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(
                DatasetAssetManifestError,
                "failed SHA-256 verification",
            ):
                load_dataset_asset_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main()
