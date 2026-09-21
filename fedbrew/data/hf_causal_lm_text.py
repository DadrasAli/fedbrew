"""Generate federated causal-LM shards with a prepared HF tokenizer."""

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


@dataclass(frozen=True, slots=True)
class HFCausalLMTextGenerationSummary:
    """Summary returned to the generic data-generator CLI."""

    manifest_path: Path
    num_clients: int
    num_examples: int
    num_test_examples: int


@dataclass(frozen=True, slots=True)
class TokenizerAssets:
    """Resolved tokenizer assets and trace metadata."""

    path: Path
    identifier: str
    requested_revision: str
    resolved_revision: str | None
    vocabulary_size: int
    asset_manifest_path: Path | None
    preparation_config: str | None


def generate_hf_causal_lm_text_from_config(
    config: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    client_splits: Mapping[str, float],
) -> HFCausalLMTextGenerationSummary:
    """Tokenize local text offline and write deterministic federated shards."""

    partition_config = _mapping(config["partition"])
    causal_lm_config = _mapping(config.get("hf_causal_lm_text", config.get("causal_lm", {})))
    split_config = _mapping(config.get("source_splits", config.get("splits", {})))

    strategy = str(partition_config.get("strategy", "iid"))
    if strategy != "iid":
        raise ValueError("hf_causal_lm_text requires partition.strategy=iid")
    num_clients = _positive_int(partition_config.get("num_clients"), "partition.num_clients")
    sequence_length = _positive_int(
        causal_lm_config.get("sequence_length", 32),
        "causal_lm.sequence_length",
    )
    stride = _positive_int(causal_lm_config.get("stride", sequence_length), "causal_lm.stride")
    train_ratio = float(client_splits["train_ratio"])
    eval_ratio = float(client_splits["eval_ratio"])
    _require_disjoint_client_windows(
        sequence_length=sequence_length,
        stride=stride,
        eval_ratio=eval_ratio,
    )
    append_eos = _boolean(
        causal_lm_config.get("append_eos_between_records", True),
        "append_eos_between_records",
    )
    pad_incomplete = _boolean(
        causal_lm_config.get(
            "pad_incomplete_window",
            causal_lm_config.get("pad_incomplete", causal_lm_config.get("padding", False)),
        ),
        "pad_incomplete_window",
    )
    _require_offline_options(causal_lm_config)

    source_path = _source_path(causal_lm_config)
    source_bytes = source_path.read_bytes()
    try:
        corpus_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"causal-LM corpus is not valid UTF-8: {source_path}") from exc
    records = corpus_text.splitlines()
    if not records:
        raise ValueError(f"causal-LM corpus contains no text records: {source_path}")
    source_train_ratio, source_test_ratio = _source_split_ratios(split_config)
    train_records, test_records = _split_source_records(
        records, train_ratio=source_train_ratio, seed=seed
    )

    assets = _resolve_tokenizer_assets(causal_lm_config)
    tokenizer = _load_tokenizer(assets)
    loaded_vocabulary_size = len(tokenizer)
    if loaded_vocabulary_size != assets.vocabulary_size:
        raise ValueError(
            "cached tokenizer vocabulary does not match its asset manifest: "
            f"loaded {loaded_vocabulary_size}, manifest {assets.vocabulary_size}"
        )
    eos_token_id = _optional_token_id(tokenizer.eos_token_id, "EOS")
    if append_eos and eos_token_id is None:
        raise ValueError("append_eos_between_records=true requires a tokenizer EOS token")
    padding_token_id = _padding_token_id(
        tokenizer=tokenizer,
        eos_token_id=eos_token_id,
        pad_incomplete=pad_incomplete,
    )

    train_tokens = tokenize_records(
        train_records,
        tokenizer=tokenizer,
        append_eos_between_records=append_eos,
    )
    test_tokens = tokenize_records(
        test_records,
        tokenizer=tokenizer,
        append_eos_between_records=append_eos,
    )
    for token_split in (train_tokens, test_tokens):
        if int(token_split.min()) < 0 or int(token_split.max()) >= loaded_vocabulary_size:
            raise ValueError("cached tokenizer produced a token ID outside its declared vocabulary")
    train_x, train_y = build_next_token_examples(
        train_tokens,
        sequence_length=sequence_length,
        stride=stride,
        pad_incomplete_window=pad_incomplete,
        padding_token_id=padding_token_id,
    )
    test_x, test_y = build_next_token_examples(
        test_tokens,
        sequence_length=sequence_length,
        stride=stride,
        pad_incomplete_window=pad_incomplete,
        padding_token_id=padding_token_id,
    )

    required_per_client = 2 if eval_ratio > 0.0 else 1
    if len(train_y) < num_clients * required_per_client:
        raise ValueError(
            "hf_causal_lm_text source-train split is too small for non-empty "
            "client splits; add corpus text, shorten causal_lm.sequence_length, "
            "reduce causal_lm.stride, or use fewer clients"
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
    corpus_hash = hashlib.sha256(source_bytes).hexdigest()
    asset_manifest_text = (
        str(assets.asset_manifest_path) if assets.asset_manifest_path is not None else None
    )
    tokenizer_revision = assets.resolved_revision or assets.requested_revision
    manifest = {
        "dataset_name": "hf_causal_lm_text",
        "format": "torch_shards",
        "task": "causal_lm",
        "tokenizer": assets.identifier,
        "tokenizer_identifier": assets.identifier,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_requested_revision": assets.requested_revision,
        "tokenizer_resolved_revision": assets.resolved_revision,
        "tokenizer_asset_manifest": asset_manifest_text,
        "asset_manifest": asset_manifest_text,
        "tokenizer_asset_path": str(assets.path),
        "vocab_size": loaded_vocabulary_size,
        "tokenizer_vocabulary_size": loaded_vocabulary_size,
        "sequence_length": sequence_length,
        "stride": stride,
        "append_eos_between_records": append_eos,
        "pad_incomplete_window": pad_incomplete,
        "eos_token_id": eos_token_id,
        "padding_token_id": padding_token_id,
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
        "source_split_strategy": "seeded_record_holdout",
        "test_split": "source_holdout",
        "input_shape": [sequence_length],
        "input_dtype": "int64",
        "target_dtype": "int64",
        "source_path": str(source_path),
        "source_sha256": corpus_hash,
        "corpus_hash": corpus_hash,
        "source_num_records": len(records),
        "source_train_num_records": len(train_records),
        "source_test_num_records": len(test_records),
        "source_num_tokens": len(train_tokens) + len(test_tokens),
        "source_train_num_tokens": len(train_tokens),
        "source_test_num_tokens": len(test_tokens),
        "source_num_examples": len(train_y) + len(test_y),
        "source_train_num_examples": len(train_y),
        "global_test_num_examples": len(test_y),
        "local_files_only": True,
        "trust_remote_code": False,
        "offline": True,
    }
    manifest_path = save_manifest(output_dir, manifest)
    save_clients_jsonl(output_dir, clients_metadata)
    return HFCausalLMTextGenerationSummary(
        manifest_path=manifest_path,
        num_clients=num_clients,
        num_examples=len(train_y),
        num_test_examples=len(test_y),
    )


def tokenize_records(
    records: Sequence[str],
    tokenizer: Any,
    append_eos_between_records: bool,
) -> Tensor:
    """Encode records without modifying the prepared tokenizer vocabulary."""

    eos_token_id = _optional_token_id(tokenizer.eos_token_id, "EOS")
    if append_eos_between_records and eos_token_id is None:
        raise ValueError("append_eos_between_records=true requires a tokenizer EOS token")
    token_ids: list[int] = []
    for position, record in enumerate(records):
        if position and append_eos_between_records:
            token_ids.append(cast(int, eos_token_id))
        encoded = tokenizer.encode(record, add_special_tokens=False)
        token_ids.extend(int(token_id) for token_id in encoded)
    if not token_ids:
        raise ValueError("causal-LM corpus produced no tokenizer tokens")
    return torch.tensor(token_ids, dtype=torch.long)


def build_next_token_examples(
    tokens: Tensor,
    sequence_length: int,
    stride: int,
    pad_incomplete_window: bool = False,
    padding_token_id: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Build fixed-length, possibly overlapping next-token examples.

    ``stride`` is the distance between consecutive window starts. At most one
    trailing incomplete window is padded; otherwise it is dropped.
    """

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if tokens.ndim != 1:
        raise ValueError("token tensor must be one-dimensional")
    if tokens.dtype != torch.long:
        tokens = tokens.to(dtype=torch.long)
    if pad_incomplete_window and padding_token_id is None:
        raise ValueError("padding_token_id is required when padding is enabled")

    inputs: list[Tensor] = []
    targets: list[Tensor] = []
    start = 0
    while start + sequence_length + 1 <= len(tokens):
        inputs.append(tokens[start : start + sequence_length].clone())
        targets.append(tokens[start + 1 : start + sequence_length + 1].clone())
        start += stride

    if pad_incomplete_window and start < len(tokens) - 1:
        window = tokens[start : start + sequence_length + 1]
        padded = torch.full(
            (sequence_length + 1,),
            cast(int, padding_token_id),
            dtype=torch.long,
        )
        padded[: len(window)] = window
        inputs.append(padded[:-1].clone())
        targets.append(padded[1:].clone())

    if not inputs:
        raise ValueError(
            "causal-LM corpus is too short to build one next-token example; "
            "enable causal_lm.pad_incomplete_window or shorten sequence_length"
        )
    return torch.stack(inputs).to(dtype=torch.long), torch.stack(targets).to(dtype=torch.long)


def _resolve_tokenizer_assets(config: Mapping[str, Any]) -> TokenizerAssets:
    preparation_config_value = config.get("preparation_config")
    preparation_config = (
        str(preparation_config_value)
        if preparation_config_value is not None and str(preparation_config_value).strip()
        else None
    )
    manifest_value = config.get("asset_manifest", config.get("tokenizer_asset_manifest"))
    if manifest_value is not None and str(manifest_value).strip():
        manifest_path = expand_path(str(manifest_value))
        from fedbrew.data.llm_assets.manifest import load_asset_manifest

        asset_manifest = load_asset_manifest(
            manifest_path,
            preparation_config=preparation_config,
            require_assets=True,
        )
        return TokenizerAssets(
            path=asset_manifest.tokenizer_asset_path,
            identifier=asset_manifest.tokenizer_identifier,
            requested_revision=asset_manifest.requested_revision,
            resolved_revision=asset_manifest.resolved_revision,
            vocabulary_size=asset_manifest.vocabulary_size,
            asset_manifest_path=manifest_path,
            preparation_config=preparation_config,
        )

    tokenizer_path_value = config.get("tokenizer_path")
    if tokenizer_path_value is None or not str(tokenizer_path_value).strip():
        raise ValueError(
            "hf_causal_lm_text requires causal_lm.asset_manifest or causal_lm.tokenizer_path"
        )
    tokenizer_path = expand_path(str(tokenizer_path_value))
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(_missing_assets_message(tokenizer_path, preparation_config))
    identifier = _non_empty_text(
        config.get("tokenizer_identifier"),
        "causal_lm.tokenizer_identifier is required with tokenizer_path",
    )
    requested_revision = _non_empty_text(
        config.get("tokenizer_revision"),
        "causal_lm.tokenizer_revision is required with tokenizer_path",
    )
    vocabulary_size = _positive_int(config.get("vocab_size"), "causal_lm.vocab_size")
    return TokenizerAssets(
        path=tokenizer_path,
        identifier=identifier,
        requested_revision=requested_revision,
        resolved_revision=None,
        vocabulary_size=vocabulary_size,
        asset_manifest_path=None,
        preparation_config=preparation_config,
    )


def _load_tokenizer(assets: TokenizerAssets) -> Any:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "hf_causal_lm_text requires the optional LLM dependencies; "
            'install them with: pip install -e ".[llm]"'
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(
            str(assets.path),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "could not load the prepared tokenizer in offline/local-only mode. "
            + _missing_assets_message(assets.path, assets.preparation_config)
        ) from exc


def _require_disjoint_client_windows(sequence_length: int, stride: int, eval_ratio: float) -> None:
    """Refuse a stride that would put shared tokens in two client splits.

    `build_next_token_examples` starts a window every `stride` tokens, so with
    `stride < sequence_length` consecutive windows overlap. The windows are then
    dealt to clients by index (`partition_iid`) and cut into train and eval by a
    shuffled index (`_split_client_examples`), neither of which knows that two
    windows share tokens -- so a client's eval window can be tokens it also
    trained on, and its loss is optimistic by an amount nothing reports.

    Measured on 4,000 tokens, sequence_length 16, 4 clients, eval_ratio 0.2 --
    how many of the four clients had at least one eval window overlapping their
    own train windows, and the worst single eval window's overlap:

    | stride | windows | clients leaking | worst eval window |
    | --- | --- | --- | --- |
    | 16 | 249 | 0 | 0 of 16 tokens |
    | 12 | 332 | 4 | 8 of 16 |
    | 8 | 498 | 4 | **16 of 16** |
    | 4 | 996 | 4 | 16 of 16 |
    | 1 | 3,984 | 4 | 16 of 16 |

    The audit bounded the leak at `sequence_length - stride`, which is the
    overlap of one pair. Against the *union* of a client's train windows it is
    worse: at stride 8 an eval window's every token is covered by the two train
    windows either side of it, so the eval split can be entirely seen data.

    Refused rather than repaired. Keeping overlapping windows disjoint across
    splits would mean cutting clients and splits at contiguous window blocks
    with a guard band of `ceil(sequence_length / stride) - 1` windows between
    them -- both this partition and the client cut are index-shuffled, so it is
    the whole assignment that would have to change, for a knob no shipped config
    moves off its default. `eval_ratio: 0` keeps overlapping windows available
    for a training-only corpus, where there is no client split to leak across:
    the source test split is cut at the record level before tokenization
    (`_split_source_records`), so no window spans it.
    """

    if eval_ratio <= 0.0 or stride >= sequence_length:
        return
    raise ValueError(
        f"causal_lm.stride ({stride}) below causal_lm.sequence_length "
        f"({sequence_length}) makes consecutive windows share up to "
        f"{sequence_length - stride} tokens, and the client eval split is cut "
        "from those windows by index: an eval window would be tokens the same "
        "client trained on, so its loss would be optimistic with nothing "
        f"reporting it. Set causal_lm.stride to {sequence_length} for "
        "non-overlapping windows, or client_splits.eval_ratio to 0 if the "
        "overlap is wanted and no per-client validation split is."
    )


def _padding_token_id(tokenizer: Any, eos_token_id: int | None, pad_incomplete: bool) -> int | None:
    """The id this dataset's padding is written with, or None if it has none.

    Two decisions, in this order.

    ``pad_incomplete_window: false`` means every window is full, so the dataset
    contains no padding and declares no padding token. The tokenizer's own
    ``pad_token_id`` was returned here regardless, which put a padding id in the
    manifest of a dataset with nothing padded -- and `TorchCausalLMTask` masks
    targets by value, so that id was a live filter over real tokens. The shipped
    dev config (tiny_gpt2, no pad token) escaped it; Qwen2.5, whose tokenizer
    ships ``pad_token == eos_token``, would have declared the EOS id.

    With padding on, an id equal to EOS is refused rather than written. The task
    refuses such a manifest at build time -- masking by value would drop every
    end-of-document target from the loss, the accuracy and the aggregation
    weight -- so writing one produces a dataset that generation calls a success
    and no run can load. Both routes to it are covered: a tokenizer whose pad
    token *is* its EOS token, and the fallback below, which reached for EOS when
    there was no pad token at all.
    """

    if not pad_incomplete:
        return None
    configured = _optional_token_id(tokenizer.pad_token_id, "padding")
    if configured is None:
        if eos_token_id is None:
            raise ValueError(
                "padding an incomplete window requires a tokenizer padding or EOS token"
            )
        configured = eos_token_id
    if eos_token_id is not None and configured == eos_token_id:
        raise ValueError(
            f"causal_lm.pad_incomplete_window is on and the padding token ({configured}) "
            "is this tokenizer's EOS token. The causal-LM task masks targets by value, "
            "so every end-of-document target would be dropped from the loss, the accuracy "
            "and the aggregation weight, and it refuses such a dataset at build time. "
            "Set causal_lm.pad_incomplete_window: false to drop the trailing partial "
            "window instead, or use a tokenizer whose padding token is distinct from "
            "its EOS token."
        )
    return configured


def _optional_token_id(value: object, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"tokenizer {label} token ID must be an integer")
    if value < 0:
        raise ValueError(f"tokenizer {label} token ID must be non-negative")
    return value


def _require_offline_options(config: Mapping[str, Any]) -> None:
    if not _boolean(config.get("local_files_only", True), "local_files_only"):
        raise ValueError("hf_causal_lm_text requires local_files_only=true")
    if _boolean(config.get("trust_remote_code", False), "trust_remote_code"):
        raise ValueError("hf_causal_lm_text requires trust_remote_code=false")


def _missing_assets_message(path: Path, preparation_config: str | None) -> str:
    command = (
        f"fedbrew prepare-llm --config {preparation_config}"
        if preparation_config
        else "fedbrew prepare-llm --config <asset-config.yaml>"
    )
    return f"prepared tokenizer assets are missing or incomplete at {path}. Run: {command}"


def _source_path(causal_lm_config: Mapping[str, Any]) -> Path:
    raw_path = causal_lm_config.get(
        "corpus_path",
        causal_lm_config.get("source_path", causal_lm_config.get("input_path")),
    )
    if raw_path is None or not str(raw_path).strip():
        raise ValueError("hf_causal_lm_text requires causal_lm.corpus_path")
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


def _split_source_records(
    records: Sequence[str], train_ratio: float, seed: int
) -> tuple[list[str], list[str]]:
    """Create a seeded record holdout before token windows are constructed."""

    if len(records) < 2:
        raise ValueError("causal-LM corpus must contain at least two text records")
    generator = torch.Generator().manual_seed(seed + 1)
    order = torch.randperm(len(records), generator=generator).tolist()
    train_size = int(len(records) * train_ratio)
    train_size = max(1, min(train_size, len(records) - 1))
    train_indices = set(order[:train_size])
    train_records = [record for index, record in enumerate(records) if index in train_indices]
    test_records = [record for index, record in enumerate(records) if index not in train_indices]
    return train_records, test_records


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
        "dataset_name": "hf_causal_lm_text",
        "task": "causal_lm",
        "partition_strategy": "iid",
        "num_clients": len(client_stats),
        "total_examples": total_examples,
        "total_tokens": total_examples * sequence_length,
        "global_test_examples": num_test_examples,
        "global_test_tokens": num_test_examples * sequence_length,
        "min_examples_per_client": min(sizes) if sizes else 0,
        "max_examples_per_client": max(sizes) if sizes else 0,
        "mean_examples_per_client": total_examples / len(sizes) if sizes else 0.0,
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


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"causal_lm.{name} must be true or false")
    return value


def _non_empty_text(value: object, message: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(message)
    return str(value)


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected config section to be a mapping")
    return cast(Mapping[str, Any], value)
