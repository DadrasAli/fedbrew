"""Generate leakage-free federated OASST1 supervised fine-tuning shards."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from fedbrew.core.paths import expand_path
from fedbrew.data.writers.manifest import save_clients_jsonl, save_manifest
from fedbrew.data.writers.torch_shards import (
    save_client_shard,
    save_split_client_shard,
)

IGNORE_INDEX = -100
DEFAULT_SEQUENCE_LENGTH = 256
DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)
FILTERING_RULES = {
    "lang": "en",
    "deleted": False,
    "review_result": True,
    "synthetic": False,
    "text": "non-empty",
    "user_id": "non-empty",
    "target_role": "assistant",
    "path": "every message passes these filters",
}


@dataclass(frozen=True, slots=True)
class OASST1SFTGenerationSummary:
    """Summary returned to the generic generator CLI."""

    manifest_path: Path
    num_clients: int
    num_examples: int
    num_test_examples: int


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One validated message in a reconstructed conversation."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ConversationTarget:
    """A valid root-to-assistant-target conversation."""

    message_id: str
    tree_id: str
    contributor_id: str
    messages: tuple[ChatMessage, ...]


@dataclass(frozen=True, slots=True)
class TokenizedSFTExample:
    """Shifted causal-LM tensors plus information required for packing."""

    input_ids: Tensor
    labels: Tensor
    token_ids: tuple[int, ...]
    active_token_mask: tuple[bool, ...]
    prompt_length: int
    target_token_count: int


@dataclass(frozen=True, slots=True)
class PreparedExample:
    """Tokenized target with private, in-memory partition keys."""

    target: ConversationTarget
    encoded: TokenizedSFTExample
    split: str


@dataclass(frozen=True, slots=True)
class TokenizerTrace:
    """Which tokenizer produced a generated dataset, pinned for reuse.

    Recorded in the manifest so a training run can verify it is tokenizing with
    the same vocabulary the shards were built from. ``requested_revision`` is
    what the config asked for and ``resolved_revision`` is what the hub
    returned; they differ when a branch name was requested instead of a commit,
    which is exactly the case where the dataset is not reproducible.
    """

    path: Path
    identifier: str
    requested_revision: str
    resolved_revision: str | None
    vocabulary_size: int
    manifest_path: Path
    preparation_config: str


@dataclass(frozen=True, slots=True)
class PilotCaps:
    """Optional ceilings that shrink a generation run to a pilot-sized one.

    Every field is a maximum count, in items, applied per split; None means no
    cap. ``*_responses_*`` cap qualifying assistant responses before
    tokenization, and ``*_windows_*`` cap the fixed-length token windows those
    responses produce afterwards -- so the two act at different stages and a
    response cap does not imply a window cap.

    Caps make a dataset cheap to build but **not** a subset of the uncapped
    one in any statistically useful sense: a run against capped data is a
    smoke test, not a small experiment.
    """

    train_responses_per_client: int | None
    eval_responses_per_client: int | None
    global_test_responses: int | None
    train_windows_per_client: int | None
    eval_windows_per_client: int | None
    global_test_windows: int | None

    def as_dict(self) -> dict[str, Any]:
        """Render the caps under the names the manifest records them by.

        Returns:
            A nested mapping written into the generated manifest, so a dataset
            carries the caps it was built under. The keys are the manifest's
            spelling, deliberately more explicit than the field names, and the
            window caps are nested one level below the response caps.
        """

        return {
            "maximum_qualifying_assistant_responses_per_client": (self.train_responses_per_client),
            "maximum_client_evaluation_responses_per_client": (self.eval_responses_per_client),
            "maximum_global_test_responses": self.global_test_responses,
            "maximum_generated_windows_per_split": {
                "train_per_client": self.train_windows_per_client,
                "client_eval_per_client": self.eval_windows_per_client,
                "global_test": self.global_test_windows,
            },
        }


def generate_oasst1_sft_from_config(
    config: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    client_splits: Mapping[str, float],
    on_progress: Callable[[str], None] | None = None,
) -> OASST1SFTGenerationSummary:
    """Create local-only OASST1 SFT shards grouped by assistant contributor.

    `on_progress` is called with a short note as each internal phase starts and
    as the two counted loops advance. This function used to run silent from
    start to finish, the longest silence in the whole tool. Three of its phases
    take most of the time: loading the 84,437-row parquet, loading the
    tokenizer, and tokenizing 24,239 candidate responses. The first two are
    single blocking calls with nothing to count, so they get a note; the third
    is a bounded loop over a known total, so it gets the count.

    Called as often as the loops iterate. The caller throttles -- a redraw per
    response would cost more than the tokenization.
    """

    # Part of the shards ABI (chapter 12 §4), and unread here: OASST1's splits
    # are cut at the conversation tree, by SHA, which is what keeps two turns
    # of one conversation out of two splits. A config that sets the section is
    # refused by name rather than reaching this line -- oasst1_sft does not
    # declare `client_splits` on its GeneratorSpec.
    del client_splits
    report = on_progress if on_progress is not None else _ignore_progress
    partition_config = _mapping(config["partition"], "partition")
    sft_config = _mapping(config.get("oasst1_sft", config.get("sft", {})), "oasst1_sft")
    split_config = _mapping(config.get("tree_splits", config.get("splits", {})), "tree_splits")
    strategy = str(
        partition_config.get("strategy", "natural_assistant_contributor_top_target_tokens")
    )
    if strategy not in {
        "assistant_contributor",
        "natural_assistant_contributor",
        "natural_assistant_contributor_top_target_tokens",
    }:
        raise ValueError("oasst1_sft requires an assistant-contributor partition strategy")
    num_clients = _positive_int(partition_config.get("num_clients", 4), "partition.num_clients")
    sequence_length = _positive_int(
        sft_config.get("sequence_length", DEFAULT_SEQUENCE_LENGTH),
        "oasst1_sft.sequence_length",
    )
    ignore_index = _integer(sft_config.get("ignore_index", IGNORE_INDEX), "oasst1_sft.ignore_index")
    if ignore_index != IGNORE_INDEX:
        raise ValueError("oasst1_sft.ignore_index must be -100")
    _require_offline_options(sft_config)
    split_ratios = _parse_tree_split_ratios(split_config)
    caps = _parse_pilot_caps(config.get("pilot_caps"))

    report("verifying dataset assets")
    dataset_manifest_path, dataset_manifest = _load_dataset_assets(sft_config)
    report("loading OASST1 rows")
    rows = _load_dataset_rows(dataset_manifest)
    report(f"reconstructing conversations from {len(rows):,} rows")
    targets, reconstruction_stats = _reconstruct_targets_with_stats(rows)
    if not targets:
        raise ValueError("OASST1 assets contain no valid assistant conversation targets")

    tokenizer_trace = _resolve_tokenizer_assets(sft_config)
    report("loading tokenizer")
    tokenizer = _load_tokenizer(tokenizer_trace)
    if len(tokenizer) != tokenizer_trace.vocabulary_size:
        raise ValueError(
            "cached tokenizer vocabulary does not match its asset manifest: "
            f"loaded {len(tokenizer)}, manifest {tokenizer_trace.vocabulary_size}"
        )
    eos_token_id = _required_token_id(tokenizer.eos_token_id, "EOS")

    prepared: list[PreparedExample] = []
    total_targets = len(targets)
    for position, target in enumerate(targets, start=1):
        report(f"tokenizing {position:,}/{total_targets:,} responses")
        encoded = tokenize_assistant_target(target.messages, tokenizer)
        _validate_token_ids(encoded.token_ids, tokenizer_trace.vocabulary_size)
        prepared.append(
            PreparedExample(
                target=target,
                encoded=encoded,
                split=assign_tree_split(
                    target.tree_id,
                    seed=seed,
                    train_ratio=split_ratios[0],
                    client_eval_ratio=split_ratios[1],
                    global_test_ratio=split_ratios[2],
                ),
            )
        )

    grouped: dict[str, dict[str, list[PreparedExample]]] = defaultdict(
        lambda: {"train": [], "client_eval": [], "global_test": []}
    )
    for example in prepared:
        grouped[example.target.contributor_id][example.split].append(example)
    for split_examples in grouped.values():
        for values in split_examples.values():
            values.sort(key=lambda value: _example_order_key(value, seed))

    eligible: dict[
        str,
        tuple[Tensor, Tensor, Tensor, Tensor, list[PreparedExample], list[PreparedExample]],
    ] = {}
    total_contributors = len(grouped)
    for position, (contributor_id, splits) in enumerate(grouped.items(), start=1):
        report(f"packing {position:,}/{total_contributors:,} contributors")
        train_examples = _cap(splits["train"], caps.train_responses_per_client)
        eval_examples = _cap(splits["client_eval"], caps.eval_responses_per_client)
        train_x, train_y = pack_sft_examples(
            [example.encoded for example in train_examples],
            eos_token_id=eos_token_id,
            sequence_length=sequence_length,
            ignore_index=ignore_index,
            max_windows=caps.train_windows_per_client,
        )
        eval_x, eval_y = pack_sft_examples(
            [example.encoded for example in eval_examples],
            eos_token_id=eos_token_id,
            sequence_length=sequence_length,
            ignore_index=ignore_index,
            max_windows=caps.eval_windows_per_client,
        )
        if len(train_y) and len(eval_y):
            eligible[contributor_id] = (
                train_x,
                train_y,
                eval_x,
                eval_y,
                train_examples,
                eval_examples,
            )

    ranked_contributors = sorted(
        eligible,
        key=lambda contributor_id: (
            -sum(
                example.encoded.target_token_count
                for split_examples in grouped[contributor_id].values()
                for example in split_examples
            ),
            hashlib.sha256(contributor_id.encode("utf-8")).hexdigest(),
        ),
    )
    if len(ranked_contributors) < num_clients:
        raise ValueError(
            "not enough assistant contributors have non-empty packed train and "
            f"client-evaluation data: need {num_clients}, found "
            f"{len(ranked_contributors)}; reduce sequence length or pilot caps"
        )
    selected = ranked_contributors[:num_clients]

    global_examples = [
        example for contributor_id in selected for example in grouped[contributor_id]["global_test"]
    ]
    global_examples.sort(key=lambda value: _example_order_key(value, seed))
    global_examples = _cap(global_examples, caps.global_test_responses)
    global_x, global_y = pack_sft_examples(
        [example.encoded for example in global_examples],
        eos_token_id=eos_token_id,
        sequence_length=sequence_length,
        ignore_index=ignore_index,
        max_windows=caps.global_test_windows,
    )
    if not len(global_y):
        raise ValueError(
            "selected assistant contributors produced no packed global-test "
            "windows; reduce sequence length or increase pilot caps"
        )

    selected_split_trees = {
        "train": {
            example.target.tree_id
            for contributor_id in selected
            for example in eligible[contributor_id][4]
        },
        "client_eval": {
            example.target.tree_id
            for contributor_id in selected
            for example in eligible[contributor_id][5]
        },
        "global_test": {example.target.tree_id for example in global_examples},
    }
    _assert_disjoint_tree_splits(selected_split_trees)

    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    client_records: list[dict[str, Any]] = []
    client_stats: list[dict[str, Any]] = []
    selected_metadata: dict[str, dict[str, Any]] = {}
    for contributor_id in selected:
        client_id = anonymize_client_id(contributor_id)
        train_x, train_y, eval_x, eval_y, train_examples, eval_examples = eligible[contributor_id]
        shard = f"shards/{client_id}.pt"
        save_split_client_shard(output_dir / shard, train_x, train_y, eval_x, eval_y)
        qualifying_counts = {
            split: len(grouped[contributor_id][split])
            for split in ("train", "client_eval", "global_test")
        }
        qualifying_tokens = {
            split: sum(
                example.encoded.target_token_count for example in grouped[contributor_id][split]
            )
            for split in ("train", "client_eval", "global_test")
        }
        stats = _make_client_stats(
            client_id=client_id,
            train_y=train_y,
            eval_y=eval_y,
            train_responses=len(train_examples),
            eval_responses=len(eval_examples),
            qualifying_responses=qualifying_counts,
            qualifying_target_tokens=qualifying_tokens,
            ignore_index=ignore_index,
        )
        client_stats.append(stats)
        client_records.append(
            {
                "client_id": client_id,
                "split": "train",
                "num_examples": stats["num_examples"],
                "num_train_examples": stats["num_train_examples"],
                "num_eval_examples": stats["num_eval_examples"],
                "num_tokens": stats["num_tokens"],
                "num_train_tokens": stats["num_train_tokens"],
                "num_eval_tokens": stats["num_eval_tokens"],
                "active_target_tokens": stats["active_target_tokens"],
                "train_active_target_tokens": stats["train_active_target_tokens"],
                "eval_active_target_tokens": stats["eval_active_target_tokens"],
                "shard": shard,
            }
        )
        selected_metadata[client_id] = {
            "selection_usable_target_tokens": sum(qualifying_tokens.values()),
            "qualifying_responses": qualifying_counts,
            "qualifying_target_tokens": qualifying_tokens,
            "used_train_responses": len(train_examples),
            "used_client_eval_responses": len(eval_examples),
            "generated_train_windows": len(train_y),
            "generated_client_eval_windows": len(eval_y),
            "train_active_target_tokens": stats["train_active_target_tokens"],
            "client_eval_active_target_tokens": stats["eval_active_target_tokens"],
        }

    save_client_shard(shards_dir / "global_test.pt", global_x, global_y)
    global_active_tokens = _active_token_count(global_y, ignore_index)
    split_tree_counts = {split: len(tree_ids) for split, tree_ids in selected_split_trees.items()}
    _write_statistics(
        output_dir=output_dir,
        client_stats=client_stats,
        sequence_length=sequence_length,
        global_test_windows=len(global_y),
        global_test_active_tokens=global_active_tokens,
        global_test_responses=len(global_examples),
        split_tree_counts=split_tree_counts,
        split_ratios=split_ratios,
    )

    source_hashes = dict(dataset_manifest.source_hashes)
    resolved_dataset_revision = (
        dataset_manifest.resolved_revision or dataset_manifest.requested_revision
    )
    resolved_tokenizer_revision = (
        tokenizer_trace.resolved_revision or tokenizer_trace.requested_revision
    )
    selected_client_ids = [anonymize_client_id(value) for value in selected]
    manifest = {
        "dataset_name": "oasst1_sft",
        "format": "torch_shards",
        "task": "causal_lm_sft",
        "source_dataset_identifier": dataset_manifest.dataset_identifier,
        "source_dataset_revision": resolved_dataset_revision,
        "source_dataset_requested_revision": dataset_manifest.requested_revision,
        "source_dataset_resolved_revision": dataset_manifest.resolved_revision,
        "dataset_asset_manifest": str(dataset_manifest_path),
        "source_file_hashes": source_hashes,
        "corpus_hash": _combined_hash(source_hashes),
        "tokenizer": tokenizer_trace.identifier,
        "tokenizer_identifier": tokenizer_trace.identifier,
        "tokenizer_revision": resolved_tokenizer_revision,
        "tokenizer_requested_revision": tokenizer_trace.requested_revision,
        "tokenizer_resolved_revision": tokenizer_trace.resolved_revision,
        "tokenizer_asset_manifest": str(tokenizer_trace.manifest_path),
        "asset_manifest": str(tokenizer_trace.manifest_path),
        "tokenizer_asset_path": str(tokenizer_trace.path),
        "vocab_size": tokenizer_trace.vocabulary_size,
        "tokenizer_vocabulary_size": tokenizer_trace.vocabulary_size,
        "sequence_length": sequence_length,
        "stride": sequence_length,
        "packing": "fixed_non_overlapping_eos_separated_drop_incomplete",
        "padding_token_id": None,
        "eos_token_id": eos_token_id,
        "ignore_index": ignore_index,
        "split_ratios": {
            "train": split_ratios[0],
            "client_eval": split_ratios[1],
            "global_test": split_ratios[2],
        },
        "tree_split_policy": "sha256(seed:message_tree_id), deterministic 80/10/10",
        "tree_split_key": "message_tree_id",
        "tree_counts": split_tree_counts,
        "filtering_rules": FILTERING_RULES,
        "filtering_summary": reconstruction_stats,
        "client_selection": {
            "method": "highest usable assistant-target token count",
            "personalization_unit": "target assistant contributor",
            "eligibility": "non-empty packed train and client-evaluation data",
            "tie_breaker": "SHA-256 of contributor identifier",
            "anonymization": "client_ + SHA-256(contributor identifier)",
        },
        "selected_client_ids": selected_client_ids,
        "selected_client_statistics": selected_metadata,
        "num_clients": num_clients,
        "partition_strategy": strategy,
        "pilot_caps": caps.as_dict(),
        "global_test_responses": len(global_examples),
        "global_test_num_examples": len(global_y),
        "global_test_active_target_tokens": global_active_tokens,
        "clients_file": "clients.jsonl",
        "global_test": "shards/global_test.pt",
        "shards_dir": "shards",
        "partition_stats_file": "partition_stats.json",
        "client_stats_file": "client_stats.csv",
        "client_shard_format": "split_v1",
        "input_shape": [sequence_length],
        "input_dtype": "int64",
        "target_dtype": "int64",
        "local_files_only": True,
        "trust_remote_code": False,
        "offline": True,
    }
    manifest_path = save_manifest(output_dir, manifest)
    save_clients_jsonl(output_dir, client_records)
    return OASST1SFTGenerationSummary(
        manifest_path=manifest_path,
        num_clients=num_clients,
        num_examples=sum(int(row["num_examples"]) for row in client_stats),
        num_test_examples=len(global_y),
    )


def is_valid_message(row: Mapping[str, Any]) -> bool:
    """Return whether an OASST1 message satisfies the public validity filters."""

    return (
        row.get("lang") == "en"
        and row.get("deleted") is False
        and row.get("review_result") is True
        and row.get("synthetic") is False
        and _non_empty(row.get("text"))
        and _non_empty(row.get("user_id"))
    )


def reconstruct_conversation_path(
    target: Mapping[str, Any],
    messages_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[ChatMessage, ...] | None:
    """Safely reconstruct one valid root-to-target path, or reject it."""

    if target.get("role") != "assistant" or not is_valid_message(target):
        return None
    target_tree_id = _text_or_none(target.get("message_tree_id"))
    if target_tree_id is None:
        return None
    path: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    current: Mapping[str, Any] = target
    while True:
        message_id = _text_or_none(current.get("message_id"))
        if (
            message_id is None
            or message_id in seen
            or not is_valid_message(current)
            or _text_or_none(current.get("message_tree_id")) != target_tree_id
        ):
            return None
        seen.add(message_id)
        path.append(current)
        parent_id = _text_or_none(current.get("parent_id"))
        if parent_id is None:
            break
        parent = messages_by_id.get(parent_id)
        if parent is None:
            return None
        current = parent
    path.reverse()

    converted: list[ChatMessage] = []
    expected_role = "prompter"
    for row in path:
        role = row.get("role")
        if role != expected_role:
            return None
        converted.append(
            ChatMessage(
                role="user" if role == "prompter" else "assistant",
                content=str(row["text"]).strip(),
            )
        )
        expected_role = "assistant" if expected_role == "prompter" else "prompter"
    if not converted or converted[-1].role != "assistant":
        return None
    return tuple(converted)


def reconstruct_assistant_conversations(
    rows: Sequence[Mapping[str, Any]],
) -> list[ConversationTarget]:
    """Return every valid assistant target in deterministic order."""

    targets, _ = _reconstruct_targets_with_stats(rows)
    return targets


class TokenizerContractError(ValueError):
    """The tokenizer or its chat template broke a property the masking needs.

    Distinct from a row that simply cannot be tokenized. A row failure is a
    property of that row's text and skipping it is correct; a contract failure
    says the prompt/response boundary cannot be located at all, so every
    example this tokenizer produces has the wrong labels. Generators skip the
    first and must not skip the second.
    """


def _ignore_progress(note: str) -> None:
    """The default sink. A generator called as a library reports to nobody."""


def tokenize_assistant_target(
    messages: Sequence[ChatMessage | Mapping[str, str]],
    tokenizer: Any,
) -> TokenizedSFTExample:
    """Apply the tokenizer chat template and mask every prompt-token label."""

    normalized = [
        {
            "role": (message.role if isinstance(message, ChatMessage) else str(message["role"])),
            "content": (
                message.content if isinstance(message, ChatMessage) else str(message["content"])
            ),
        }
        for message in messages
    ]
    if (
        len(normalized) < 2
        or normalized[-1]["role"] != "assistant"
        or normalized[-2]["role"] != "user"
    ):
        raise ValueError("SFT target must end with a user/assistant exchange")
    prompt_ids = _chat_template_ids(
        tokenizer,
        normalized[:-1],
        add_generation_prompt=True,
    )
    complete_ids = _chat_template_ids(
        tokenizer,
        normalized,
        add_generation_prompt=False,
    )
    if complete_ids[: len(prompt_ids)] != prompt_ids:
        raise TokenizerContractError(
            "tokenizer chat template prompt is not a prefix of the complete "
            "assistant conversation, so prompt tokens cannot be label-masked"
        )
    if len(complete_ids) <= len(prompt_ids):
        raise ValueError("assistant target produced no active tokenizer tokens")
    if len(complete_ids) < 2:
        raise TokenizerContractError("chat template produced fewer than two tokens")
    active_mask = (False,) * len(prompt_ids) + (True,) * (len(complete_ids) - len(prompt_ids))
    inputs = torch.tensor(complete_ids[:-1], dtype=torch.long)
    labels = torch.tensor(complete_ids[1:], dtype=torch.long)
    label_activity = active_mask[1:]
    labels = labels.masked_fill(~torch.tensor(label_activity, dtype=torch.bool), IGNORE_INDEX)
    return TokenizedSFTExample(
        input_ids=inputs,
        labels=labels,
        token_ids=tuple(complete_ids),
        active_token_mask=active_mask,
        prompt_length=len(prompt_ids),
        target_token_count=int((labels != IGNORE_INDEX).sum().item()),
    )


def pack_sft_examples(
    examples: Sequence[TokenizedSFTExample],
    *,
    eos_token_id: int,
    sequence_length: int,
    ignore_index: int = IGNORE_INDEX,
    max_windows: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Pack EOS-separated examples into fixed windows and drop inactive windows."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if eos_token_id < 0:
        raise ValueError("eos_token_id must be non-negative")
    if max_windows is not None and max_windows <= 0:
        raise ValueError("max_windows must be positive when set")
    stream: list[int] = []
    active: list[bool] = []
    for position, example in enumerate(examples):
        if position:
            stream.append(eos_token_id)
            active.append(False)
        if len(example.token_ids) != len(example.active_token_mask):
            raise ValueError("SFT example token and loss-mask lengths differ")
        stream.extend(example.token_ids)
        active.extend(example.active_token_mask)

    inputs: list[Tensor] = []
    labels: list[Tensor] = []
    start = 0
    while start + sequence_length + 1 <= len(stream):
        x = torch.tensor(stream[start : start + sequence_length], dtype=torch.long)
        target_ids = torch.tensor(stream[start + 1 : start + sequence_length + 1], dtype=torch.long)
        target_active = torch.tensor(
            active[start + 1 : start + sequence_length + 1], dtype=torch.bool
        )
        y = target_ids.masked_fill(~target_active, ignore_index)
        if bool(target_active.any()):
            inputs.append(x)
            labels.append(y)
            if max_windows is not None and len(inputs) >= max_windows:
                break
        start += sequence_length
    if not inputs:
        empty = torch.empty((0, sequence_length), dtype=torch.long)
        return empty.clone(), empty
    return torch.stack(inputs), torch.stack(labels)


def anonymize_client_id(contributor_id: str) -> str:
    """Return a deterministic identifier that never exposes the raw user ID."""

    if not contributor_id.strip():
        raise ValueError("contributor_id must be non-empty")
    return "client_" + hashlib.sha256(contributor_id.encode("utf-8")).hexdigest()


def assign_tree_split(
    tree_id: str,
    *,
    seed: int,
    train_ratio: float = DEFAULT_SPLIT_RATIOS[0],
    client_eval_ratio: float = DEFAULT_SPLIT_RATIOS[1],
    global_test_ratio: float = DEFAULT_SPLIT_RATIOS[2],
) -> str:
    """Assign a complete message tree with a stable SHA-256 hash."""

    _validate_split_ratios(train_ratio, client_eval_ratio, global_test_ratio)
    digest = hashlib.sha256(f"{seed}:{tree_id}".encode()).digest()
    value = int.from_bytes(digest, "big") / float(1 << (8 * len(digest)))
    if value < train_ratio:
        return "train"
    if value < train_ratio + client_eval_ratio:
        return "client_eval"
    return "global_test"


def _reconstruct_targets_with_stats(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[ConversationTarget], dict[str, int]]:
    counts = Counter[str]()
    counts["source_rows"] = len(rows)
    ids: dict[str, Mapping[str, Any]] = {}
    duplicates: set[str] = set()
    for row in rows:
        message_id = _text_or_none(row.get("message_id"))
        if message_id is None:
            counts["rows_missing_message_id"] += 1
            continue
        if message_id in ids:
            duplicates.add(message_id)
        else:
            ids[message_id] = row
        if is_valid_message(row):
            counts["rows_passing_validity_filters"] += 1
    for message_id in duplicates:
        ids.pop(message_id, None)
    counts["duplicate_message_ids"] = len(duplicates)

    targets: list[ConversationTarget] = []
    candidates = sorted(
        (
            row
            for row in rows
            if row.get("role") == "assistant"
            and _text_or_none(row.get("message_id")) not in duplicates
        ),
        key=lambda row: (
            str(row.get("message_tree_id", "")),
            str(row.get("message_id", "")),
        ),
    )
    counts["assistant_target_candidates"] = len(candidates)
    for target in candidates:
        path = reconstruct_conversation_path(target, ids)
        if path is None:
            counts["rejected_assistant_targets"] += 1
            continue
        message_id = cast(str, _text_or_none(target.get("message_id")))
        tree_id = cast(str, _text_or_none(target.get("message_tree_id")))
        contributor_id = cast(str, _text_or_none(target.get("user_id")))
        targets.append(
            ConversationTarget(
                message_id=message_id,
                tree_id=tree_id,
                contributor_id=contributor_id,
                messages=path,
            )
        )
    counts["qualifying_assistant_targets"] = len(targets)
    counts["qualifying_message_trees"] = len({target.tree_id for target in targets})
    return targets, dict(counts)


def _load_dataset_assets(sft_config: Mapping[str, Any]) -> tuple[Path, Any]:
    manifest_value = sft_config.get(
        "dataset_asset_manifest", sft_config.get("source_asset_manifest")
    )
    if not _non_empty(manifest_value):
        raise ValueError("oasst1_sft requires sft.dataset_asset_manifest")
    manifest_path = expand_path(str(manifest_value))
    preparation_config = sft_config.get("dataset_preparation_config")
    from fedbrew.data.asset_manifest import load_dataset_asset_manifest

    manifest = load_dataset_asset_manifest(
        manifest_path,
        preparation_config=(str(preparation_config) if _non_empty(preparation_config) else None),
        require_files=True,
        verify_hashes=True,
    )
    if manifest.dataset_identifier != "OpenAssistant/oasst1":
        raise ValueError("oasst1_sft dataset manifest must describe OpenAssistant/oasst1")
    return manifest_path, manifest


def _load_dataset_rows(manifest: Any) -> list[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "oasst1_sft requires the optional LLM dependencies; install them "
            'with: pip install -e ".[llm]"'
        ) from exc
    rows: list[Mapping[str, Any]] = []
    for split in ("train", "validation"):
        source = manifest.file_for_split(split)
        table = parquet.read_table(source.local_path)
        split_rows = cast(list[dict[str, Any]], table.to_pylist())
        if len(split_rows) != source.row_count:
            raise ValueError(
                f"prepared OASST1 {split} row count changed: manifest "
                f"{source.row_count}, loaded {len(split_rows)}"
            )
        rows.extend(split_rows)
    return rows


def _resolve_tokenizer_assets(sft_config: Mapping[str, Any]) -> TokenizerTrace:
    manifest_value = sft_config.get("tokenizer_asset_manifest", sft_config.get("asset_manifest"))
    if not _non_empty(manifest_value):
        raise ValueError("oasst1_sft requires sft.tokenizer_asset_manifest")
    manifest_path = expand_path(str(manifest_value))
    preparation_config = sft_config.get("tokenizer_preparation_config")
    preparation_config_text = (
        str(preparation_config)
        if _non_empty(preparation_config)
        else str(Path("configs") / "llm_assets" / f"{manifest_path.parent.name}.yaml")
    )
    from fedbrew.data.llm_assets.manifest import load_asset_manifest

    manifest = load_asset_manifest(
        manifest_path,
        preparation_config=preparation_config_text,
        require_assets=True,
    )
    return TokenizerTrace(
        path=manifest.tokenizer_asset_path,
        identifier=manifest.tokenizer_identifier,
        requested_revision=manifest.requested_revision,
        resolved_revision=manifest.resolved_revision,
        vocabulary_size=manifest.vocabulary_size,
        manifest_path=manifest_path,
        preparation_config=preparation_config_text,
    )


def _load_tokenizer(trace: TokenizerTrace) -> Any:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "oasst1_sft requires the optional LLM dependencies; install them "
            'with: pip install -e ".[llm]"'
        ) from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(trace.path),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "could not load the prepared tokenizer in local-only mode. Run: "
            f"fedbrew prepare-llm --config {trace.preparation_config}"
        ) from exc
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("prepared tokenizer does not define an official chat template")
    return tokenizer


def _chat_template_ids(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    values = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors=None,
    )
    if isinstance(values, Tensor):
        values = values.tolist()
    if (
        isinstance(values, Sequence)
        and values
        and isinstance(values[0], Sequence)
        and not isinstance(values[0], str | bytes)
    ):
        if len(values) != 1:
            raise TokenizerContractError("chat template returned an unexpected token batch")
        values = values[0]
    if not isinstance(values, Sequence) or isinstance(values, str | bytes):
        raise TokenizerContractError("chat template did not return a token sequence")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TokenizerContractError("chat template returned a non-integer token ID")
        result.append(value)
    return result


def _parse_tree_split_ratios(
    config: Mapping[str, Any],
) -> tuple[float, float, float]:
    train = _ratio(config.get("train_ratio", 0.8), "tree_splits.train_ratio")
    client_eval = _ratio(
        config.get("client_eval_ratio", config.get("eval_ratio", 0.1)),
        "tree_splits.client_eval_ratio",
    )
    global_test = _ratio(
        config.get("global_test_ratio", config.get("test_ratio", 0.1)),
        "tree_splits.global_test_ratio",
    )
    _validate_split_ratios(train, client_eval, global_test)
    return train, client_eval, global_test


def _validate_split_ratios(train: float, client_eval: float, global_test: float) -> None:
    if min(train, client_eval, global_test) <= 0.0:
        raise ValueError("all OASST1 tree split ratios must be positive")
    if abs(train + client_eval + global_test - 1.0) > 1e-8:
        raise ValueError("OASST1 train, client-eval, and global-test ratios must sum to 1")


def _parse_pilot_caps(value: object) -> PilotCaps:
    config = _mapping(value or {}, "pilot_caps")
    window_value = config.get("maximum_generated_windows_per_split", {})
    if isinstance(window_value, Mapping):
        windows = cast(Mapping[str, Any], window_value)
        train_windows = _optional_positive_int(
            windows.get("train_per_client", 100),
            "pilot_caps.maximum_generated_windows_per_split.train_per_client",
        )
        eval_windows = _optional_positive_int(
            windows.get("client_eval_per_client", 20),
            "pilot_caps.maximum_generated_windows_per_split.client_eval_per_client",
        )
        global_windows = _optional_positive_int(
            windows.get("global_test", 80),
            "pilot_caps.maximum_generated_windows_per_split.global_test",
        )
    else:
        shared = _optional_positive_int(
            window_value,
            "pilot_caps.maximum_generated_windows_per_split",
        )
        train_windows = eval_windows = global_windows = shared
    return PilotCaps(
        train_responses_per_client=_optional_positive_int(
            config.get(
                "maximum_qualifying_assistant_responses_per_client",
                config.get("max_train_responses_per_client", 100),
            ),
            "pilot_caps.maximum_qualifying_assistant_responses_per_client",
        ),
        eval_responses_per_client=_optional_positive_int(
            config.get(
                "maximum_client_evaluation_responses_per_client",
                config.get("max_client_eval_responses_per_client", 20),
            ),
            "pilot_caps.maximum_client_evaluation_responses_per_client",
        ),
        global_test_responses=_optional_positive_int(
            config.get(
                "maximum_global_test_responses",
                config.get("max_global_test_responses", 80),
            ),
            "pilot_caps.maximum_global_test_responses",
        ),
        train_windows_per_client=train_windows,
        eval_windows_per_client=eval_windows,
        global_test_windows=global_windows,
    )


def _make_client_stats(
    *,
    client_id: str,
    train_y: Tensor,
    eval_y: Tensor,
    train_responses: int,
    eval_responses: int,
    qualifying_responses: Mapping[str, int],
    qualifying_target_tokens: Mapping[str, int],
    ignore_index: int,
) -> dict[str, Any]:
    sequence_length = int(train_y.shape[1])
    train_examples = len(train_y)
    eval_examples = len(eval_y)
    train_active = _active_token_count(train_y, ignore_index)
    eval_active = _active_token_count(eval_y, ignore_index)
    return {
        "client_id": client_id,
        "num_examples": train_examples + eval_examples,
        "num_train_examples": train_examples,
        "num_eval_examples": eval_examples,
        "num_tokens": (train_examples + eval_examples) * sequence_length,
        "num_train_tokens": train_examples * sequence_length,
        "num_eval_tokens": eval_examples * sequence_length,
        "active_target_tokens": train_active + eval_active,
        "train_active_target_tokens": train_active,
        "eval_active_target_tokens": eval_active,
        "train_responses": train_responses,
        "eval_responses": eval_responses,
        "qualifying_train_responses": qualifying_responses["train"],
        "qualifying_eval_responses": qualifying_responses["client_eval"],
        "qualifying_global_test_responses": qualifying_responses["global_test"],
        "selection_usable_target_tokens": sum(qualifying_target_tokens.values()),
    }


def _write_statistics(
    *,
    output_dir: Path,
    client_stats: Sequence[Mapping[str, Any]],
    sequence_length: int,
    global_test_windows: int,
    global_test_active_tokens: int,
    global_test_responses: int,
    split_tree_counts: Mapping[str, int],
    split_ratios: tuple[float, float, float],
) -> None:
    total_examples = sum(int(row["num_examples"]) for row in client_stats)
    payload = {
        "dataset_name": "oasst1_sft",
        "task": "causal_lm_sft",
        "partition_strategy": "natural_assistant_contributor",
        "num_clients": len(client_stats),
        "total_examples": total_examples,
        "total_tokens": total_examples * sequence_length,
        "total_active_target_tokens": sum(int(row["active_target_tokens"]) for row in client_stats),
        "global_test_examples": global_test_windows,
        "global_test_tokens": global_test_windows * sequence_length,
        "global_test_active_target_tokens": global_test_active_tokens,
        "global_test_responses": global_test_responses,
        "tree_counts": dict(split_tree_counts),
        "split_ratios": {
            "train": split_ratios[0],
            "client_eval": split_ratios[1],
            "global_test": split_ratios[2],
        },
        "clients": list(client_stats),
    }
    (output_dir / "partition_stats.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fieldnames = list(client_stats[0].keys()) if client_stats else ["client_id"]
    with (output_dir / "client_stats.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(client_stats)


def _active_token_count(labels: Tensor, ignore_index: int) -> int:
    return int((labels != ignore_index).sum().item())


def _assert_disjoint_tree_splits(splits: Mapping[str, set[str]]) -> None:
    names = list(splits)
    for position, first in enumerate(names):
        for second in names[position + 1 :]:
            overlap = splits[first] & splits[second]
            if overlap:
                raise RuntimeError(f"message trees leaked between {first} and {second} splits")


def _example_order_key(example: PreparedExample, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{seed}:{example.target.tree_id}:{example.target.message_id}".encode()
    ).hexdigest()
    return digest, example.target.message_id


def _cap(values: Sequence[Any], maximum: int | None) -> list[Any]:
    result = list(values)
    return result if maximum is None else result[:maximum]


def _combined_hash(source_hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for split, value in sorted(source_hashes.items()):
        digest.update(split.encode("utf-8"))
        digest.update(b":")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_token_ids(token_ids: Sequence[int], vocabulary_size: int) -> None:
    if not token_ids:
        raise ValueError("SFT example contains no token IDs")
    if min(token_ids) < 0 or max(token_ids) >= vocabulary_size:
        raise ValueError("chat template produced a token ID outside the vocabulary")


def _required_token_id(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"tokenizer {name} token ID must be a non-negative integer")
    return value


def _require_offline_options(config: Mapping[str, Any]) -> None:
    if config.get("local_files_only", True) is not True:
        raise ValueError("oasst1_sft requires local_files_only=true")
    if config.get("trust_remote_code", False) is not False:
        raise ValueError("oasst1_sft requires trust_remote_code=false")


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _optional_positive_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _ratio(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, float | int):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return result


def _non_empty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _text_or_none(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
