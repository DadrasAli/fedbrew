"""Focused tests for the local causal-LM data generator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pytest
import torch

from fedbrew.data.generate import generate_from_config
from fedbrew.data.manifest_dataset import ManifestFederatedDataset
from fedbrew.data.manifest_validation import validate_manifest
from fedbrew.data.tiny_causal_lm import tokenize_records


class TinyCausalLMGeneratorTests(unittest.TestCase):
    @pytest.mark.fast
    def test_byte_tokenizer_inserts_eot_between_utf8_records(self) -> None:
        tokens = tokenize_records(["A", "é"])
        self.assertEqual(tokens.dtype, torch.long)
        self.assertEqual(tokens.tolist(), [67, 1, 197, 171])

    def test_selector_writes_non_empty_long_sequence_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = root / "corpus.txt"
            corpus_path.write_text(
                "\n".join(
                    f"record {index:03d} contains deterministic local text" for index in range(40)
                ),
                encoding="utf-8",
            )
            output_dir = root / "data" / "generated"
            config_path = root / "generator.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "dataset:",
                        "  name: tiny_causal_lm",
                        f"  output_dir: {output_dir}",
                        "  seed: 17",
                        "causal_lm:",
                        f"  corpus_path: {corpus_path}",
                        "  sequence_length: 8",
                        "splits:",
                        "  train_ratio: 0.75",
                        "  test_ratio: 0.25",
                        "client_splits:",
                        "  train_ratio: 0.75",
                        "  eval_ratio: 0.25",
                        "partition:",
                        "  strategy: iid",
                        "  num_clients: 3",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            manifest_path = generate_from_config(config_path)

            errors = [
                issue for issue in validate_manifest(manifest_path) if issue.severity == "error"
            ]
            self.assertEqual(errors, [])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["task"], "causal_lm")
            self.assertEqual(manifest["tokenizer"], "byte_level")
            self.assertEqual(manifest["vocab_size"], 258)
            self.assertEqual(manifest["sequence_length"], 8)
            self.assertEqual(manifest["partition_strategy"], "iid")

            stats = json.loads((output_dir / "partition_stats.json").read_text(encoding="utf-8"))
            self.assertNotIn("global_label_counts", stats)
            dataset = ManifestFederatedDataset(manifest_path)
            self.assertEqual(len(dataset.list_clients()), 3)
            for client_id in dataset.list_clients():
                client_data = dataset.get_client_data(client_id)
                for split_name in ("train", "eval"):
                    split = client_data[split_name]
                    self.assertGreater(len(split["x"]), 0)
                    self.assertEqual(split["x"].dtype, torch.long)
                    self.assertEqual(split["y"].dtype, torch.long)
                    self.assertEqual(tuple(split["x"].shape[1:]), (8,))
                    self.assertTrue(torch.equal(split["x"][:, 1:], split["y"][:, :-1]))
                    self.assertGreaterEqual(int(split["x"].min()), 1)
                    self.assertLess(int(split["x"].max()), 258)

            global_test = dataset.get_global_data()
            self.assertGreater(len(global_test["x"]), 0)
            self.assertEqual(global_test["x"].dtype, torch.long)
            self.assertEqual(tuple(global_test["x"].shape[1:]), (8,))


if __name__ == "__main__":
    unittest.main()
