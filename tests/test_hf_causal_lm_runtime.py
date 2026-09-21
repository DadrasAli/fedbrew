"""Focused tests for strict offline Hugging Face causal-LM runtime loading."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch

from fedbrew.core.runner import run
from fedbrew.data.llm_assets.manifest import AssetManifestError
from fedbrew.data.writers.torch_shards import (
    save_client_shard,
    save_split_client_shard,
)
from fedbrew.models.hf_causal_lm import (
    HFCausalLMConfigError,
    build_hf_causal_lm,
)
from fedbrew.tasks.causal_lm import TorchCausalLMTask


@unittest.skipUnless(
    importlib.util.find_spec("transformers") is not None,
    "transformers is an optional LLM dependency",
)
class HFCausalLMRuntimeTests(unittest.TestCase):
    @pytest.mark.fast
    def test_loads_pretrained_weights_strictly_locally_and_keeps_ties(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = _write_local_model_fixture(Path(directory))
            with patch.dict(
                os.environ,
                {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            ):
                model = build_hf_causal_lm(
                    {
                        "asset_manifest": str(manifest_path),
                        "preparation_config": "configs/llm_assets/tiny_gpt2.yaml",
                        "sequence_length": 8,
                        "dataset_vocab_size": 32,
                        "dataset_padding_token_id": None,
                        "local_files_only": True,
                        "trust_remote_code": False,
                    }
                )

            loaded_model = cast(Any, model)
            input_weight = loaded_model.get_input_embeddings().weight
            output_weight = loaded_model.get_output_embeddings().weight
            self.assertAlmostEqual(float(input_weight[0, 0].item()), 0.125)
            self.assertEqual(input_weight.data_ptr(), output_weight.data_ptr())
            self.assertEqual(loaded_model.config.vocab_size, 32)

    @pytest.mark.fast
    def test_missing_assets_names_exact_preparation_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "asset_manifest.json"
            with self.assertRaisesRegex(
                AssetManifestError,
                "fedbrew prepare-llm --config configs/llm_assets/tiny_gpt2.yaml",
            ):
                build_hf_causal_lm(
                    {
                        "asset_manifest": str(missing),
                        "preparation_config": "configs/llm_assets/tiny_gpt2.yaml",
                        "sequence_length": 8,
                    }
                )

    @pytest.mark.fast
    def test_rejects_generated_vocabulary_or_sequence_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = _write_local_model_fixture(Path(directory))
            base = {
                "asset_manifest": str(manifest_path),
                "sequence_length": 8,
                "dataset_vocab_size": 32,
            }
            with self.assertRaisesRegex(
                HFCausalLMConfigError,
                "vocabulary sizes",
            ):
                build_hf_causal_lm({**base, "dataset_vocab_size": 31})
            with self.assertRaisesRegex(
                HFCausalLMConfigError,
                "exceeds the loaded model capacity",
            ):
                build_hf_causal_lm(
                    {
                        **base,
                        "sequence_length": 32,
                        "dataset_sequence_length": 32,
                    }
                )

    @pytest.mark.fast
    def test_allows_padded_model_vocabulary_but_rejects_undersized_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = _write_local_model_fixture(Path(directory))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["vocabulary_size"] = 31
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            model = build_hf_causal_lm(
                {
                    "asset_manifest": str(manifest_path),
                    "sequence_length": 8,
                    "dataset_vocab_size": 31,
                }
            )
            self.assertEqual(cast(Any, model).get_input_embeddings().weight.shape[0], 32)

            manifest["vocabulary_size"] = 33
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                HFCausalLMConfigError,
                "smaller than the prepared tokenizer",
            ):
                build_hf_causal_lm(
                    {
                        "asset_manifest": str(manifest_path),
                        "sequence_length": 8,
                        "dataset_vocab_size": 33,
                    }
                )

    def test_one_pretrained_fedavg_round_completes_offline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset_manifest = _write_local_model_fixture(root / "assets")
            data_manifest = _write_federated_fixture(root / "generated", asset_manifest)
            config_path, output_dir = _write_run_config(
                root,
                asset_manifest,
                data_manifest,
            )

            with patch.dict(
                os.environ,
                {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            ):
                state = run(config_path)

            self.assertEqual(len(state.metrics_history), 1)
            metrics = state.metrics_history[0].metrics
            for name in (
                "train_loss_sample_weighted_avg",
                "central_test_loss",
                "train_accuracy_sample_weighted_avg",
                "central_test_accuracy",
            ):
                self.assertIn(name, metrics)
                self.assertTrue(math.isfinite(metrics[name]))

            run_metadata = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))[
                "reproducibility"
            ]
            self.assertEqual(
                run_metadata["llm"]["model_identifier"],
                "local/tiny-gpt2",
            )
            self.assertEqual(run_metadata["llm"]["corpus_hash"], "b" * 64)
            self.assertTrue(run_metadata["llm"]["local_files_only"])
            self.assertEqual(
                run_metadata["llm"]["offline_environment"],
                {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            )

    @pytest.mark.fast
    def test_null_padding_keeps_token_zero_in_metrics_and_attention(self) -> None:
        task = TorchCausalLMTask(model_config={"pad_token_id": None})
        targets = torch.tensor([[0, 1]], dtype=torch.long)
        logits = torch.zeros((1, 2, 4), dtype=torch.float32)

        _, _, total = task._loss_and_counts(logits, targets)

        self.assertEqual(total, 2)
        self.assertEqual(
            set(task._model_inputs(targets)),
            {"input_ids"},
        )


def _write_federated_fixture(root: Path, asset_manifest: Path) -> Path:
    root.mkdir(parents=True)
    clients = []
    for index, client_id in enumerate(("client_0", "client_1")):
        offset = index * 4
        train_x = torch.tensor(
            [
                [2 + offset, 3 + offset, 4 + offset, 5 + offset],
                [3 + offset, 4 + offset, 5 + offset, 6 + offset],
            ],
            dtype=torch.long,
        )
        train_y = torch.tensor(
            [
                [3 + offset, 4 + offset, 5 + offset, 6 + offset],
                [4 + offset, 5 + offset, 6 + offset, 7 + offset],
            ],
            dtype=torch.long,
        )
        eval_x = train_x[:1].clone()
        eval_y = train_y[:1].clone()
        shard = f"shards/{client_id}.pt"
        save_split_client_shard(
            root / shard,
            train_x,
            train_y,
            eval_x,
            eval_y,
        )
        clients.append(
            {
                "client_id": client_id,
                "shard": shard,
                "num_examples": 3,
                "num_train_examples": 2,
                "num_eval_examples": 1,
            }
        )

    global_x = torch.tensor([[2, 4, 6, 8], [3, 5, 7, 9]], dtype=torch.long)
    global_y = torch.tensor([[4, 6, 8, 10], [5, 7, 9, 11]], dtype=torch.long)
    save_client_shard(root / "shards/global_test.pt", global_x, global_y)
    (root / "clients.jsonl").write_text(
        "".join(json.dumps(client) + "\n" for client in clients),
        encoding="utf-8",
    )
    manifest = {
        "dataset_name": "local_hf_causal_lm",
        "format": "torch_shards",
        "task": "causal_lm",
        "num_clients": 2,
        "clients_file": "clients.jsonl",
        "global_test": "shards/global_test.pt",
        "partition_strategy": "iid",
        "client_splits": {"train_ratio": 0.67, "eval_ratio": 0.33},
        "vocab_size": 32,
        "tokenizer_vocabulary_size": 32,
        "sequence_length": 4,
        "stride": 4,
        "padding_token_id": None,
        "eos_token_id": 1,
        "tokenizer_identifier": "local/tiny-gpt2",
        "tokenizer_revision": "a" * 40,
        "tokenizer_requested_revision": "a" * 40,
        "tokenizer_resolved_revision": "a" * 40,
        "tokenizer_asset_manifest": str(asset_manifest),
        "asset_manifest": str(asset_manifest),
        "corpus_hash": "b" * 64,
        "source_sha256": "b" * 64,
        "local_files_only": True,
        "trust_remote_code": False,
        "offline": True,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _write_run_config(
    root: Path,
    asset_manifest: Path,
    data_manifest: Path,
) -> tuple[Path, Path]:
    server_path = root / "server.yaml"
    server_path.write_text(
        "strategy: fedavg\nparticipation_rate: 1\nmetrics:\n  - fit_loss\n  - fit_accuracy\n",
        encoding="utf-8",
    )
    client_path = root / "client.yaml"
    client_path.write_text(
        "update_rule: local_adamw\n"
        "batch_size: 2\n"
        "learning_rate: 0.001\n"
        "learning_rate_schedule: constant\n"
        "min_learning_rate: 0.0\n"
        "weight_decay: 0.0\n"
        "beta1: 0.9\n"
        "beta2: 0.999\n"
        "epsilon: 1.0e-8\n"
        "metrics:\n"
        "  - fit_loss\n"
        "  - fit_accuracy\n",
        encoding="utf-8",
    )
    output_dir = root / "output"
    config_path = root / "experiment.yaml"
    config_path.write_text(
        f"""experiment:
  name: local_hf_fedavg
  seed: 9
  output_dir: {output_dir}
server_config: {server_path}
client_config: {client_path}
data:
  name: manifest_dataset
  path: {data_manifest}
model:
  name: hf_causal_lm
  asset_manifest: {asset_manifest}
  preparation_config: configs/llm_assets/tiny_gpt2.yaml
  sequence_length: 4
  local_files_only: true
  trust_remote_code: false
runtime:
  deterministic: true
  deterministic_warn_only: true
  device: cpu
  use_amp: false
evaluation:
  train:
    every: 1
    clients: all
  val:
    every: never
  test:
    every: never
  central_test:
    every: 1
defaults:
  global_rounds: 1
  local_iterations: 1
""",
        encoding="utf-8",
    )
    return config_path, output_dir


def _write_local_model_fixture(root: Path) -> Path:
    from transformers import GPT2Config, GPT2LMHeadModel

    root.mkdir(parents=True, exist_ok=True)

    model_path = root / "model"
    tokenizer_path = root / "tokenizer"
    model_path.mkdir()
    tokenizer_path.mkdir()
    config = GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_ctx=16,
        n_embd=8,
        n_layer=1,
        n_head=2,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=None,
        tie_word_embeddings=True,
    )
    model = GPT2LMHeadModel(config)
    with torch.no_grad():
        cast(Any, model).get_input_embeddings().weight.fill_(0.125)
    model.save_pretrained(model_path)

    manifest_path = root / "asset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_identifier": "local/tiny-gpt2",
                "tokenizer_identifier": "local/tiny-gpt2",
                "requested_revision": "a" * 40,
                "resolved_revision": "a" * 40,
                "cache_path": str(root),
                "model_path": "model",
                "tokenizer_path": "tokenizer",
                "vocabulary_size": 32,
                "model_type": "gpt2",
                "preparation_timestamp": "2026-01-01T00:00:00+00:00",
                "trust_remote_code": False,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


if __name__ == "__main__":
    unittest.main()
