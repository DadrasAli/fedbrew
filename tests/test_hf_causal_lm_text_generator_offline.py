"""Focused offline tests for the cached Hugging Face text generator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest
import torch
import yaml  # type: ignore[import-untyped]

from fedbrew.data.generate import generate_from_config
from fedbrew.data.hf_causal_lm_text import (
    build_next_token_examples,
    tokenize_records,
)
from fedbrew.data.llm_assets.manifest import AssetManifestError
from fedbrew.data.manifest_dataset import ManifestFederatedDataset
from fedbrew.data.manifest_validation import validate_manifest

HAS_TRANSFORMERS = all(
    importlib.util.find_spec(package) is not None for package in ("tokenizers", "transformers")
)


@unittest.skipUnless(HAS_TRANSFORMERS, "requires the llm optional dependencies")
class HFCausalLMTextGeneratorOfflineTests(unittest.TestCase):
    def test_local_tokenizer_generation_is_deterministic_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset_manifest, tokenizer_path, vocabulary_size = _write_assets(root)
            corpus_path = root / "corpus.txt"
            corpus_text = "\n".join(
                "alpha beta gamma delta epsilon zeta eta theta" for _ in range(48)
            )
            corpus_path.write_text(corpus_text, encoding="utf-8")

            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=True,
                trust_remote_code=False,
            )
            first_tokens = tokenize_records(["alpha beta", "gamma delta"], tokenizer, True)
            second_tokens = tokenize_records(["alpha beta", "gamma delta"], tokenizer, True)
            self.assertTrue(torch.equal(first_tokens, second_tokens))
            self.assertEqual(first_tokens.dtype, torch.long)
            self.assertEqual(first_tokens.tolist(), [2, 3, 1, 4, 5])

            first_output = root / "generated_first"
            first_config = _generator_config(
                corpus_path=corpus_path,
                asset_manifest=asset_manifest,
                output_dir=first_output,
            )
            first_config_path = root / "first.yaml"
            first_config_path.write_text(
                yaml.safe_dump(first_config, sort_keys=False), encoding="utf-8"
            )
            second_output = root / "generated_second"
            second_config = _generator_config(
                corpus_path=corpus_path,
                asset_manifest=asset_manifest,
                output_dir=second_output,
            )
            second_config_path = root / "second.yaml"
            second_config_path.write_text(
                yaml.safe_dump(second_config, sort_keys=False), encoding="utf-8"
            )

            with mock.patch.dict(
                os.environ,
                {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            ):
                first_manifest_path = generate_from_config(first_config_path)
                second_manifest_path = generate_from_config(second_config_path)

            errors = [
                issue
                for issue in validate_manifest(first_manifest_path)
                if issue.severity == "error"
            ]
            self.assertEqual(errors, [])
            first_manifest = json.loads(first_manifest_path.read_text(encoding="utf-8"))
            second_manifest = json.loads(second_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(first_manifest["task"], "causal_lm")
            self.assertEqual(first_manifest["tokenizer_identifier"], "local/test")
            self.assertEqual(first_manifest["tokenizer_revision"], "a" * 40)
            self.assertEqual(first_manifest["tokenizer_asset_manifest"], str(asset_manifest))
            self.assertEqual(first_manifest["vocab_size"], vocabulary_size)
            self.assertEqual(first_manifest["sequence_length"], 8)
            self.assertEqual(first_manifest["stride"], 12)
            self.assertEqual(first_manifest["eos_token_id"], 1)
            self.assertIsNone(first_manifest["padding_token_id"])
            self.assertEqual(
                first_manifest["corpus_hash"],
                hashlib.sha256(corpus_text.encode("utf-8")).hexdigest(),
            )
            self.assertEqual(first_manifest["source_split_strategy"], "seeded_record_holdout")
            self.assertTrue(first_manifest["local_files_only"])
            self.assertFalse(first_manifest["trust_remote_code"])

            stats = json.loads((first_output / "partition_stats.json").read_text(encoding="utf-8"))
            self.assertNotIn("global_label_counts", stats)
            self.assertNotIn("label", (first_output / "client_stats.csv").read_text())

            first_dataset = ManifestFederatedDataset(first_manifest_path)
            second_dataset = ManifestFederatedDataset(second_manifest_path)
            self.assertEqual(first_dataset.list_clients(), second_dataset.list_clients())
            for client_id in first_dataset.list_clients():
                first_client = first_dataset.get_client_data(client_id)
                second_client = second_dataset.get_client_data(client_id)
                for split_name in ("train", "eval"):
                    first_split = first_client[split_name]
                    second_split = second_client[split_name]
                    self.assertGreater(len(first_split["x"]), 0)
                    self.assertEqual(first_split["x"].dtype, torch.long)
                    self.assertEqual(first_split["y"].dtype, torch.long)
                    self.assertEqual(tuple(first_split["x"].shape[1:]), (8,))
                    self.assertTrue(torch.equal(first_split["x"][:, 1:], first_split["y"][:, :-1]))
                    self.assertGreaterEqual(int(first_split["x"].min()), 0)
                    self.assertLess(int(first_split["x"].max()), vocabulary_size)
                    self.assertTrue(torch.equal(first_split["x"], second_split["x"]))
                    self.assertTrue(torch.equal(first_split["y"], second_split["y"]))

            first_test = first_dataset.get_global_data()
            second_test = second_dataset.get_global_data()
            self.assertGreater(len(first_test["x"]), 0)
            self.assertEqual(first_test["x"].dtype, torch.long)
            self.assertEqual(tuple(first_test["x"].shape[1:]), (8,))
            self.assertTrue(torch.equal(first_test["x"], second_test["x"]))
            self.assertTrue(torch.equal(first_test["y"], second_test["y"]))

    @pytest.mark.fast
    def test_missing_manifest_names_exact_preparation_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = root / "corpus.txt"
            corpus_path.write_text("first record\nsecond record\n", encoding="utf-8")
            config = _generator_config(
                corpus_path=corpus_path,
                asset_manifest=root / "missing" / "asset_manifest.json",
                output_dir=root / "generated",
            )
            config["causal_lm"]["preparation_config"] = "configs/llm_assets/local_test.yaml"
            config_path = root / "missing.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            with self.assertRaisesRegex(
                AssetManifestError,
                r"fedbrew prepare-llm --config configs/llm_assets/local_test.yaml",
            ):
                generate_from_config(config_path)

    @pytest.mark.fast
    def test_padding_uses_fixed_length_long_next_token_tensors(self) -> None:
        inputs, targets = build_next_token_examples(
            torch.tensor([2, 3, 4], dtype=torch.long),
            sequence_length=4,
            stride=4,
            pad_incomplete_window=True,
            padding_token_id=1,
        )
        self.assertEqual(inputs.dtype, torch.long)
        self.assertEqual(targets.dtype, torch.long)
        self.assertEqual(inputs.tolist(), [[2, 3, 4, 1]])
        self.assertEqual(targets.tolist(), [[3, 4, 1, 1]])


def _write_assets(root: Path) -> tuple[Path, Path, int]:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocabulary = {
        "<unk>": 0,
        "<eos>": 1,
        "alpha": 2,
        "beta": 3,
        "gamma": 4,
        "delta": 5,
        "epsilon": 6,
        "zeta": 7,
        "eta": 8,
        "theta": 9,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<eos>",
    )

    cache_path = root / "assets"
    model_path = cache_path / "model"
    tokenizer_path = cache_path / "tokenizer"
    model_path.mkdir(parents=True)
    tokenizer_path.mkdir(parents=True)
    tokenizer.save_pretrained(tokenizer_path)
    manifest_path = cache_path / "asset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_identifier": "local/test",
                "tokenizer_identifier": "local/test",
                "requested_revision": "test-revision",
                "resolved_revision": "a" * 40,
                "cache_path": str(cache_path),
                "model_path": "model",
                "tokenizer_path": "tokenizer",
                "vocabulary_size": len(tokenizer),
                "model_type": "gpt2",
                "preparation_timestamp": "2026-01-01T00:00:00+00:00",
                "trust_remote_code": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path, tokenizer_path, len(tokenizer)


def _generator_config(
    corpus_path: Path, asset_manifest: Path, output_dir: Path
) -> dict[str, object]:
    return {
        "dataset": {
            "name": "hf_causal_lm_text",
            "output_dir": str(output_dir),
            "seed": 17,
        },
        "causal_lm": {
            "corpus_path": str(corpus_path),
            "asset_manifest": str(asset_manifest),
            "preparation_config": "configs/llm_assets/local.yaml",
            "sequence_length": 8,
            # Not below sequence_length: overlapping windows alongside a client
            # eval split are refused (P10-F21). Not equal to it either: that is
            # also the default stride, so the manifest's stride assertion could
            # not tell a configured stride from a defaulted one.
            "stride": 12,
            "append_eos_between_records": True,
            "pad_incomplete_window": False,
            "local_files_only": True,
            "trust_remote_code": False,
        },
        "splits": {"train_ratio": 0.75, "test_ratio": 0.25},
        "client_splits": {"train_ratio": 0.75, "eval_ratio": 0.25},
        "partition": {"strategy": "iid", "num_clients": 3},
    }


if __name__ == "__main__":
    unittest.main()
