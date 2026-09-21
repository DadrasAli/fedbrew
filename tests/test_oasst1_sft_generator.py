"""Focused, network-free tests for the OASST1 federated SFT generator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch
import yaml  # type: ignore[import-untyped]

from fedbrew.data.generate import generate_from_config
from fedbrew.data.manifest_validation import validate_manifest
from fedbrew.data.oasst1_sft import (
    IGNORE_INDEX,
    ChatMessage,
    TokenizedSFTExample,
    anonymize_client_id,
    assign_tree_split,
    pack_sft_examples,
    reconstruct_assistant_conversations,
    tokenize_assistant_target,
)

HAS_LLM_DEPENDENCIES = all(
    importlib.util.find_spec(package) is not None
    for package in ("pyarrow", "tokenizers", "transformers")
)


@pytest.mark.fast
class OASST1ConversationTests(unittest.TestCase):
    def test_reconstructs_valid_path_and_rejects_invalid_or_broken_paths(
        self,
    ) -> None:
        rows = [
            _message("root-ok", None, "tree-ok", "prompter", "human-ok", "Question"),
            _message(
                "target-ok",
                "root-ok",
                "tree-ok",
                "assistant",
                "assistant-ok",
                "Answer",
            ),
            _message(
                "root-invalid",
                None,
                "tree-invalid",
                "prompter",
                "human-invalid",
                "Question",
                review_result=False,
            ),
            _message(
                "target-invalid",
                "root-invalid",
                "tree-invalid",
                "assistant",
                "assistant-invalid",
                "Answer",
            ),
            _message(
                "target-missing",
                "does-not-exist",
                "tree-missing",
                "assistant",
                "assistant-missing",
                "Answer",
            ),
            _message(
                "cycle-assistant",
                "cycle-user",
                "tree-cycle",
                "assistant",
                "assistant-cycle",
                "Answer",
            ),
            _message(
                "cycle-user",
                "cycle-assistant",
                "tree-cycle",
                "prompter",
                "human-cycle",
                "Question",
            ),
        ]

        targets = reconstruct_assistant_conversations(rows)

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].message_id, "target-ok")
        self.assertEqual(
            [(message.role, message.content) for message in targets[0].messages],
            [("user", "Question"), ("assistant", "Answer")],
        )

    def test_tree_split_is_deterministic_and_client_id_is_anonymized(self) -> None:
        assignments: dict[str, set[str]] = {
            "train": set(),
            "client_eval": set(),
            "global_test": set(),
        }
        for index in range(1000):
            tree_id = f"tree-{index}"
            first = assign_tree_split(tree_id, seed=42)
            second = assign_tree_split(tree_id, seed=42)
            self.assertEqual(first, second)
            assignments[first].add(tree_id)
        self.assertFalse(assignments["train"] & assignments["client_eval"])
        self.assertFalse(assignments["train"] & assignments["global_test"])
        self.assertFalse(assignments["client_eval"] & assignments["global_test"])
        self.assertTrue(all(assignments.values()))

        raw_id = "raw-contributor@example.test"
        client_id = anonymize_client_id(raw_id)
        self.assertNotIn(raw_id, client_id)
        self.assertEqual(client_id, anonymize_client_id(raw_id))
        self.assertEqual(len(client_id), len("client_") + 64)


@unittest.skipUnless(HAS_LLM_DEPENDENCIES, "requires the llm dependencies")
class OASST1TokenizationAndGenerationTests(unittest.TestCase):
    @pytest.mark.fast
    def test_chat_template_masks_prompt_and_packing_keeps_active_targets(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tokenizer, vocabulary_size = _write_tokenizer(Path(directory))
            encoded = tokenize_assistant_target(
                (
                    ChatMessage("user", "question alpha"),
                    ChatMessage("assistant", "answer beta"),
                ),
                tokenizer,
            )

            self.assertEqual(encoded.input_ids.dtype, torch.long)
            self.assertEqual(encoded.labels.dtype, torch.long)
            self.assertTrue(torch.all(encoded.labels[: encoded.prompt_length - 1] == IGNORE_INDEX))
            self.assertGreater(encoded.target_token_count, 0)
            active = encoded.labels[encoded.labels != IGNORE_INDEX]
            self.assertGreater(len(active), 0)
            self.assertGreaterEqual(int(active.min()), 0)
            self.assertLess(int(active.max()), vocabulary_size)

            inputs, labels = pack_sft_examples(
                [encoded, encoded],
                eos_token_id=int(tokenizer.eos_token_id),
                sequence_length=4,
            )
            self.assertEqual(inputs.dtype, torch.long)
            self.assertEqual(labels.dtype, torch.long)
            self.assertEqual(tuple(inputs.shape[1:]), (4,))
            self.assertTrue(torch.all((labels == IGNORE_INDEX) | (labels >= 0)))
            self.assertTrue(torch.all((labels != IGNORE_INDEX).any(dim=1)))
            self.assertGreaterEqual(int(inputs.min()), 0)
            self.assertLess(int(inputs.max()), vocabulary_size)

            inactive = TokenizedSFTExample(
                input_ids=torch.tensor([2, 3, 1], dtype=torch.long),
                labels=torch.full((3,), IGNORE_INDEX, dtype=torch.long),
                token_ids=(2, 3, 1, 1, 1),
                active_token_mask=(False, False, False, False, False),
                prompt_length=5,
                target_token_count=0,
            )
            empty_x, empty_y = pack_sft_examples([inactive], eos_token_id=1, sequence_length=4)
            self.assertEqual(tuple(empty_x.shape), (0, 4))
            self.assertEqual(tuple(empty_y.shape), (0, 4))

    def test_full_local_generation_has_four_anonymized_leakage_free_clients(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer, vocabulary_size = _write_tokenizer(root / "tokenizer")
            llm_manifest = _write_llm_manifest(root, tokenizer, vocabulary_size)
            rows = _federated_fixture_rows(seed=42)
            dataset_manifest = _write_dataset_manifest(root, rows)
            output_dir = root / "generated"
            config = {
                "dataset": {
                    "name": "oasst1_sft",
                    "output_dir": str(output_dir),
                    "seed": 42,
                },
                "oasst1_sft": {
                    "dataset_asset_manifest": str(dataset_manifest),
                    "dataset_preparation_config": "data/configs/assets/oasst1.yaml",
                    "tokenizer_asset_manifest": str(llm_manifest),
                    "tokenizer_preparation_config": (
                        "configs/llm_assets/qwen2_5_0_5b_instruct.yaml"
                    ),
                    "sequence_length": 4,
                    "ignore_index": -100,
                    "local_files_only": True,
                    "trust_remote_code": False,
                },
                "tree_splits": {
                    "train_ratio": 0.8,
                    "client_eval_ratio": 0.1,
                    "global_test_ratio": 0.1,
                },
                # No client_splits: oasst1_sft does not declare the section,
                # and a config that sets it is refused rather than having it
                # validated and deleted.
                "partition": {
                    "strategy": ("natural_assistant_contributor_top_target_tokens"),
                    "num_clients": 4,
                },
                "pilot_caps": {
                    "maximum_qualifying_assistant_responses_per_client": 10,
                    "maximum_client_evaluation_responses_per_client": 5,
                    "maximum_global_test_responses": 10,
                    "maximum_generated_windows_per_split": {
                        "train_per_client": 10,
                        "client_eval_per_client": 5,
                        "global_test": 10,
                    },
                },
            }
            config_path = root / "generator.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            with mock.patch.dict(
                os.environ,
                {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            ):
                manifest_path = generate_from_config(config_path)

            errors = [
                issue for issue in validate_manifest(manifest_path) if issue.severity == "error"
            ]
            self.assertEqual(errors, [])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["task"], "causal_lm_sft")
            self.assertEqual(manifest["num_clients"], 4)
            self.assertEqual(manifest["sequence_length"], 4)
            self.assertEqual(manifest["ignore_index"], -100)
            self.assertTrue(manifest["local_files_only"])
            self.assertFalse(manifest["trust_remote_code"])
            self.assertEqual(
                set(manifest["tree_counts"]),
                {"train", "client_eval", "global_test"},
            )

            public_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in (
                    manifest_path,
                    output_dir / "clients.jsonl",
                    output_dir / "client_stats.csv",
                    output_dir / "partition_stats.json",
                )
            )
            for raw_id in (f"assistant-{index}" for index in range(4)):
                self.assertNotIn(raw_id, public_text)

            clients = [
                json.loads(line)
                for line in (output_dir / "clients.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(len(clients), 4)
            self.assertTrue(
                all(
                    record["client_id"].startswith("client_")
                    and len(record["client_id"]) == len("client_") + 64
                    for record in clients
                )
            )
            for record in clients:
                shard = torch.load(
                    output_dir / record["shard"],
                    map_location="cpu",
                    weights_only=False,
                )
                for split in ("train", "eval"):
                    inputs = shard[split]["x"]
                    labels = shard[split]["y"]
                    self.assertGreater(len(labels), 0)
                    self.assertEqual(inputs.dtype, torch.long)
                    self.assertEqual(labels.dtype, torch.long)
                    self.assertEqual(tuple(inputs.shape[1:]), (4,))
                    self.assertTrue(torch.all((labels != IGNORE_INDEX).any(dim=1)))
                    self.assertGreaterEqual(int(inputs.min()), 0)
                    self.assertLess(int(inputs.max()), vocabulary_size)

            global_test = torch.load(
                output_dir / manifest["global_test"],
                map_location="cpu",
                weights_only=False,
            )
            self.assertGreater(len(global_test["y"]), 0)
            self.assertTrue(torch.all((global_test["y"] != IGNORE_INDEX).any(dim=1)))


def _message(
    message_id: str,
    parent_id: str | None,
    tree_id: str,
    role: str,
    user_id: str,
    text: str,
    *,
    review_result: bool = True,
) -> dict[str, object]:
    return {
        "message_id": message_id,
        "parent_id": parent_id,
        "message_tree_id": tree_id,
        "role": role,
        "user_id": user_id,
        "text": text,
        "lang": "en",
        "deleted": False,
        "review_result": review_result,
        "synthetic": False,
    }


def _write_tokenizer(root: Path) -> tuple[Any, int]:
    from tokenizers import Tokenizer  # type: ignore[import-untyped]
    from tokenizers.models import WordLevel  # type: ignore[import-untyped]
    from tokenizers.pre_tokenizers import (  # type: ignore[import-untyped]
        WhitespaceSplit,
    )
    from transformers import PreTrainedTokenizerFast

    root.mkdir(parents=True, exist_ok=True)
    vocabulary = {
        "<unk>": 0,
        "<eos>": 1,
        "<user>": 2,
        "<assistant>": 3,
        "question": 4,
        "answer": 5,
        "alpha": 6,
        "beta": 7,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<eos>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '<user> ' + message['content'] + ' <eos> ' }}"
        "{% else %}"
        "{{ '<assistant> ' + message['content'] + ' <eos> ' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<assistant> ' }}{% endif %}"
    )
    tokenizer.save_pretrained(root)
    return tokenizer, len(tokenizer)


def _write_llm_manifest(root: Path, tokenizer: object, vocabulary_size: int) -> Path:
    cache_path = root / "llm_assets"
    model_path = cache_path / "model"
    tokenizer_path = cache_path / "tokenizer"
    model_path.mkdir(parents=True)
    tokenizer_path.mkdir(parents=True)
    tokenizer.save_pretrained(tokenizer_path)  # type: ignore[attr-defined]
    manifest_path = cache_path / "asset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_identifier": "local/qwen-fixture",
                "tokenizer_identifier": "local/qwen-fixture",
                "requested_revision": "fixture-revision",
                "resolved_revision": "a" * 40,
                "cache_path": str(cache_path),
                "model_path": "model",
                "tokenizer_path": "tokenizer",
                "vocabulary_size": vocabulary_size,
                "model_type": "qwen2",
                "preparation_timestamp": "2026-01-01T00:00:00+00:00",
                "trust_remote_code": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _write_dataset_manifest(root: Path, rows: list[dict[str, object]]) -> Path:
    import pyarrow as arrow  # type: ignore[import-untyped]
    import pyarrow.parquet as parquet  # type: ignore[import-untyped]

    cache_path = root / "dataset_assets"
    snapshots = cache_path / "snapshots"
    snapshots.mkdir(parents=True)
    split_at = len(rows) // 2
    split_rows = {"train": rows[:split_at], "validation": rows[split_at:]}
    files: dict[str, dict[str, object]] = {}
    for split, values in split_rows.items():
        path = snapshots / f"{split}.parquet"
        parquet.write_table(arrow.Table.from_pylist(values), path)
        files[split] = {
            "path": f"snapshots/{split}.parquet",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "row_count": len(values),
        }
    manifest_path = cache_path / "asset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_identifier": "OpenAssistant/oasst1",
                "requested_revision": ("37d790373da332a4c8be24bd29ec1550e7f04c3f"),
                "resolved_revision": ("37d790373da332a4c8be24bd29ec1550e7f04c3f"),
                "cache_path": str(cache_path),
                "storage_format": "parquet",
                "files": files,
                "preparation_timestamp": "2026-01-01T00:00:00+00:00",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _federated_fixture_rows(seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for contributor_position in range(4):
        contributor = f"assistant-{contributor_position}"
        for split, count in (("train", 2), ("client_eval", 2), ("global_test", 1)):
            for response_position in range(count):
                prefix = f"fixture-{contributor_position}-{split}-{response_position}"
                tree_id = _find_tree_id(prefix, split, seed)
                root_id = f"{prefix}-root"
                target_id = f"{prefix}-target"
                rows.extend(
                    [
                        _message(
                            root_id,
                            None,
                            tree_id,
                            "prompter",
                            f"human-{prefix}",
                            "question alpha",
                        ),
                        _message(
                            target_id,
                            root_id,
                            tree_id,
                            "assistant",
                            contributor,
                            "answer beta",
                        ),
                    ]
                )
    return rows


def _find_tree_id(prefix: str, split: str, seed: int) -> str:
    for position in range(10000):
        value = f"{prefix}-{position}"
        if assign_tree_split(value, seed=seed) == split:
            return value
    raise AssertionError(f"could not construct a {split} tree fixture")
