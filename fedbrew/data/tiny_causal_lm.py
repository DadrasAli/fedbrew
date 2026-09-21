"""Generate a tiny federated causal-language-model corpus."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from fedbrew.core.paths import expand_path
from fedbrew.data.partitioners.iid import partition_iid
from fedbrew.data.writers.manifest import save_clients_jsonl, save_manifest
from fedbrew.data.writers.torch_shards import (
    save_client_shard,
    save_split_client_shard,
)

PADDING_TOKEN_ID = 0
END_OF_TEXT_TOKEN_ID = 1
BYTE_TOKEN_OFFSET = 2
VOCAB_SIZE = 258
TOKENIZER_NAME = "byte_level"


@dataclass(frozen=True, slots=True)
class TinyCausalLMGenerationSummary:
    """Summary returned to the generic data-generator CLI."""

    manifest_path: Path
    num_clients: int
    num_examples: int
    num_test_examples: int


def generate_tiny_causal_lm_from_config(
    config: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    client_splits: Mapping[str, float],
) -> TinyCausalLMGenerationSummary:
    """Generate deterministic split-aware shards from a local text corpus."""

    partition_config = _mapping(config["partition"])
    causal_lm_config = _mapping(config.get("causal_lm", {}))
    split_config = _mapping(config.get("splits", {}))

    strategy = str(partition_config.get("strategy", "iid"))
    if strategy != "iid":
        raise ValueError("tiny_causal_lm requires partition.strategy=iid")
    num_clients = _positive_int(partition_config.get("num_clients"), "partition.num_clients")
    sequence_length = _positive_int(
        causal_lm_config.get("sequence_length", 32),
        "causal_lm.sequence_length",
    )
    source_path = _source_path(causal_lm_config)
    source_train_ratio, source_test_ratio = _source_split_ratios(split_config)

    corpus_text = source_path.read_text(encoding="utf-8")
    records = corpus_text.splitlines()
    if not records:
        raise ValueError(f"causal-LM corpus contains no text records: {source_path}")
    tokens = tokenize_records(records)
    examples_x, examples_y = build_next_token_examples(tokens, sequence_length)
    source_train_indices, source_test_indices = _split_source_examples(
        len(examples_y), source_train_ratio, seed
    )
    train_x = _select_rows(examples_x, source_train_indices)
    train_y = _select_rows(examples_y, source_train_indices)
    test_x = _select_rows(examples_x, source_test_indices)
    test_y = _select_rows(examples_y, source_test_indices)

    train_ratio = float(client_splits["train_ratio"])
    eval_ratio = float(client_splits["eval_ratio"])
    required_per_client = 2 if eval_ratio > 0.0 else 1
    if len(train_y) < num_clients * required_per_client:
        raise ValueError(
            "tiny_causal_lm source-train split is too small for non-empty client "
            "splits; add corpus text, shorten causal_lm.sequence_length, or use "
            "fewer clients"
        )
    partitions = partition_iid(list(range(len(train_y))), num_clients, seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    clients_metadata: list[dict[str, Any]] = []
    client_stats: list[dict[str, Any]] = []

    for client_position, client_id in enumerate(sorted(partitions)):
        local_indices = partitions[client_id]
        local_train_indices, local_eval_indices = _split_client_examples(
            local_indices,
            eval_ratio=eval_ratio,
            seed=seed + client_position + 1009,
        )
        shard = f"shards/{client_id}.pt"
        save_split_client_shard(
            output_dir / shard,
            _select_rows(train_x, local_train_indices),
            _select_rows(train_y, local_train_indices),
            _select_rows(train_x, local_eval_indices),
            _select_rows(train_y, local_eval_indices),
        )
        stats = _client_stats(
            client_id=client_id,
            num_examples=len(local_indices),
            num_train_examples=len(local_train_indices),
            num_eval_examples=len(local_eval_indices),
            sequence_length=sequence_length,
        )
        client_stats.append(stats)
        clients_metadata.append(
            {
                "client_id": client_id,
                "split": "train",
                "num_examples": stats["num_examples"],
                "num_train_examples": stats["num_train_examples"],
                "num_eval_examples": stats["num_eval_examples"],
                "num_tokens": stats["num_tokens"],
                "num_train_tokens": stats["num_train_tokens"],
                "num_eval_tokens": stats["num_eval_tokens"],
                "shard": shard,
            }
        )

    save_client_shard(shards_dir / "global_test.pt", test_x, test_y)
    _write_partition_stats(
        output_dir=output_dir,
        client_stats=client_stats,
        sequence_length=sequence_length,
        num_test_examples=len(test_y),
        source_train_ratio=source_train_ratio,
        source_test_ratio=source_test_ratio,
        client_splits=client_splits,
    )
    manifest = {
        "dataset_name": "tiny_causal_lm",
        "format": "torch_shards",
        "task": "causal_lm",
        "tokenizer": TOKENIZER_NAME,
        "tokenizer_config": {
            "encoding": "utf-8",
            "padding_token_id": PADDING_TOKEN_ID,
            "end_of_text_token_id": END_OF_TEXT_TOKEN_ID,
            "byte_token_offset": BYTE_TOKEN_OFFSET,
        },
        "padding_token_id": PADDING_TOKEN_ID,
        "end_of_text_token_id": END_OF_TEXT_TOKEN_ID,
        "vocab_size": VOCAB_SIZE,
        "sequence_length": sequence_length,
        "num_clients": num_clients,
        "clients_file": "clients.jsonl",
        "global_test": "shards/global_test.pt",
        "shards_dir": "shards",
        "partition_strategy": strategy,
        "partition_stats_file": "partition_stats.json",
        "client_stats_file": "client_stats.csv",
        "client_shard_format": "split_v1",
        "client_splits": {
            "train_ratio": train_ratio,
            "eval_ratio": eval_ratio,
        },
        "source_splits": {
            "train_ratio": source_train_ratio,
            "test_ratio": source_test_ratio,
        },
        "test_split": "source_holdout",
        "input_shape": [sequence_length],
        "input_dtype": "int64",
        "target_dtype": "int64",
        "source_path": str(source_path),
        "source_sha256": hashlib.sha256(corpus_text.encode("utf-8")).hexdigest(),
        "source_num_records": len(records),
        "source_num_tokens": len(tokens),
        "source_num_examples": len(examples_y),
        "source_train_num_examples": len(train_y),
        "global_test_num_examples": len(test_y),
    }
    manifest_path = save_manifest(output_dir, manifest)
    save_clients_jsonl(output_dir, clients_metadata)
    return TinyCausalLMGenerationSummary(
        manifest_path=manifest_path,
        num_clients=num_clients,
        num_examples=len(train_y),
        num_test_examples=len(test_y),
    )


def tokenize_records(records: Sequence[str]) -> Tensor:
    """Encode UTF-8 bytes and insert an end-of-text token between records."""

    token_ids: list[int] = []
    for position, record in enumerate(records):
        if position:
            token_ids.append(END_OF_TEXT_TOKEN_ID)
        token_ids.extend(byte + BYTE_TOKEN_OFFSET for byte in record.encode("utf-8"))
    return torch.tensor(token_ids, dtype=torch.long)


def build_next_token_examples(tokens: Tensor, sequence_length: int) -> tuple[Tensor, Tensor]:
    """Build non-overlapping, fixed-length next-token examples."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if tokens.ndim != 1:
        raise ValueError("token tensor must be one-dimensional")
    num_examples = max(0, (len(tokens) - 1) // sequence_length)
    if num_examples == 0:
        raise ValueError("causal-LM corpus is too short to build one next-token example")
    usable = tokens[: num_examples * sequence_length + 1]
    inputs = usable[:-1].reshape(num_examples, sequence_length).clone()
    targets = usable[1:].reshape(num_examples, sequence_length).clone()
    return inputs.to(dtype=torch.long), targets.to(dtype=torch.long)


def _source_path(causal_lm_config: Mapping[str, Any]) -> Path:
    raw_path = causal_lm_config.get("corpus_path", causal_lm_config.get("source_path"))
    if raw_path is None or not str(raw_path).strip():
        raise ValueError("tiny_causal_lm requires causal_lm.corpus_path")
    path = expand_path(str(raw_path))
    if not path.is_file():
        raise FileNotFoundError(f"causal-LM corpus file does not exist: {path}")
    return path


def _source_split_ratios(split_config: Mapping[str, Any]) -> tuple[float, float]:
    train_ratio = _ratio(split_config.get("train_ratio", 0.8), "splits.train_ratio")
    test_ratio = _ratio(split_config.get("test_ratio", 1.0 - train_ratio), "splits.test_ratio")
    if train_ratio <= 0.0 or test_ratio <= 0.0:
        raise ValueError("source train and test split ratios must both be positive")
    if abs((train_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("splits.train_ratio + splits.test_ratio must equal 1.0")
    return train_ratio, test_ratio


def _split_source_examples(
    num_examples: int, train_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    if num_examples < 2:
        raise ValueError("causal-LM corpus must produce at least two examples")
    generator = torch.Generator().manual_seed(seed + 1)
    order = torch.randperm(num_examples, generator=generator).tolist()
    train_size = int(num_examples * train_ratio)
    train_size = max(1, min(train_size, num_examples - 1))
    return sorted(order[:train_size]), sorted(order[train_size:])


def _split_client_examples(
    indices: Sequence[int], eval_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    values = list(indices)
    if not values:
        return [], []
    if eval_ratio == 0.0:
        return sorted(values), []
    if len(values) < 2:
        raise ValueError("each causal-LM client needs at least two source examples")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(values), generator=generator).tolist()
    shuffled = [values[index] for index in order]
    eval_size = max(1, int(round(len(values) * eval_ratio)))
    eval_size = min(eval_size, len(values) - 1)
    return sorted(shuffled[eval_size:]), sorted(shuffled[:eval_size])


def _select_rows(values: Tensor, indices: Sequence[int]) -> Tensor:
    index = torch.tensor(list(indices), dtype=torch.long)
    return values.index_select(0, index)


def _client_stats(
    client_id: str,
    num_examples: int,
    num_train_examples: int,
    num_eval_examples: int,
    sequence_length: int,
) -> dict[str, Any]:
    return {
        "client_id": client_id,
        "num_examples": num_examples,
        "num_train_examples": num_train_examples,
        "num_eval_examples": num_eval_examples,
        "num_tokens": num_examples * sequence_length,
        "num_train_tokens": num_train_examples * sequence_length,
        "num_eval_tokens": num_eval_examples * sequence_length,
    }


def _write_partition_stats(
    output_dir: Path,
    client_stats: list[dict[str, Any]],
    sequence_length: int,
    num_test_examples: int,
    source_train_ratio: float,
    source_test_ratio: float,
    client_splits: Mapping[str, float],
) -> None:
    sizes = [int(stats["num_examples"]) for stats in client_stats]
    total_examples = sum(sizes)
    payload = {
        "dataset_name": "tiny_causal_lm",
        "task": "causal_lm",
        "partition_strategy": "iid",
        "num_clients": len(client_stats),
        "total_examples": total_examples,
        "total_tokens": total_examples * sequence_length,
        "global_test_examples": num_test_examples,
        "global_test_tokens": num_test_examples * sequence_length,
        "min_examples_per_client": min(sizes) if sizes else 0,
        "max_examples_per_client": max(sizes) if sizes else 0,
        "mean_examples_per_client": (total_examples / len(sizes) if sizes else 0.0),
        "source_splits": {
            "train_ratio": source_train_ratio,
            "test_ratio": source_test_ratio,
        },
        "client_splits": {
            "train_ratio": float(client_splits["train_ratio"]),
            "eval_ratio": float(client_splits["eval_ratio"]),
        },
        "clients": client_stats,
    }
    (output_dir / "partition_stats.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_client_stats_csv(output_dir / "client_stats.csv", client_stats)


def _write_client_stats_csv(path: Path, client_stats: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "client_id",
        "num_examples",
        "num_train_examples",
        "num_eval_examples",
        "num_tokens",
        "num_train_tokens",
        "num_eval_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(client_stats)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise ValueError(f"{name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _ratio(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, float | int):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return parsed


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected config section to be a mapping")
    return cast(Mapping[str, Any], value)
