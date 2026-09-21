"""Generate naturally writer-partitioned FEMNIST torch shards."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from fedbrew.core.paths import expand_path
from fedbrew.core.seeding import derive_seed
from fedbrew.data.stats import compute_client_stats, compute_label_counts
from fedbrew.data.writers.manifest import save_clients_jsonl, save_manifest
from fedbrew.data.writers.torch_shards import (
    save_client_shard,
    save_split_client_shard,
)

DEFAULT_FEMNIST_DATASET = "flwrlabs/femnist"
DEFAULT_FEMNIST_REVISION = "5c617740df553a6d3e666035d29fb5d85e1c59a2"
FEMNIST_LABELS = [
    *[str(value) for value in range(10)],
    *[chr(ord("A") + value) for value in range(26)],
    *[chr(ord("a") + value) for value in range(26)],
]
_SAFE_CLIENT_ID = re.compile(r"[A-Za-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class FEMNISTGenerationSummary:
    """Summary returned to the generic data-generator CLI."""

    manifest_path: Path
    num_clients: int
    num_examples: int
    num_test_examples: int


def generate_femnist_from_config(
    config: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    client_splits: Mapping[str, float],
    source_dataset: Any | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> FEMNISTGenerationSummary:
    """Generate one split-aware shard per natural FEMNIST writer.

    `on_progress` replaces the bare print this loop used to make every 250
    writers. Two reasons it had to go: it wrote straight to stdout from inside
    a generator, which is the second console layer the package now forbids,
    and once `generate` wrapped this call in a rail stage it interleaved with
    the stage's own redrawn line. The counter itself was right -- it is the
    same shape the OASST1 tokenizer loop reports -- so it moved rather than
    being deleted, and it now reports every writer instead of every 250th,
    since the caller throttles.
    """

    dataset_config = _mapping(config["dataset"])
    partition_config = _mapping(config["partition"])
    femnist_config = _mapping(config.get("femnist", {}))
    strategy = str(partition_config.get("strategy", "natural"))
    if strategy != "natural":
        raise ValueError("FEMNIST requires partition.strategy=natural")

    requested_clients = _optional_positive_int(partition_config.get("num_clients"))
    # Three, not two: every writer now yields a train, an eval AND a test
    # example, and global_test.pt is built from the third of those.
    min_samples = int(femnist_config.get("min_samples_per_client", 3))
    if min_samples < 3:
        raise ValueError("femnist.min_samples_per_client must be at least 3")

    source_name = str(femnist_config.get("source", DEFAULT_FEMNIST_DATASET))
    source_revision_value = femnist_config.get(
        "revision",
        DEFAULT_FEMNIST_REVISION,
    )
    source_revision = None if source_revision_value is None else str(source_revision_value)
    source_split = str(femnist_config.get("split", "train"))
    writer_column = str(femnist_config.get("writer_column", "writer_id"))
    image_column = str(femnist_config.get("image_column", "image"))
    label_column = str(femnist_config.get("label_column", "character"))
    raw_dir = expand_path(str(dataset_config.get("raw_dir", "data/raw/datasets/femnist")))

    if source_dataset is None:
        source_dataset = _load_source_dataset(
            source_name=source_name,
            source_revision=source_revision,
            source_split=source_split,
            raw_dir=raw_dir,
        )
    _validate_source_columns(
        source_dataset,
        required={writer_column, image_column, label_column},
    )

    writer_indices = _group_indices_by_writer(source_dataset, writer_column)
    eligible_writer_indices = {
        writer_id: indices
        for writer_id, indices in writer_indices.items()
        if len(indices) >= min_samples
    }
    if not eligible_writer_indices:
        raise ValueError("FEMNIST source contains no eligible writers")
    selected_writer_ids = _select_writer_ids(
        eligible_writer_indices,
        requested_clients,
        seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    train_ratio = float(client_splits["train_ratio"])
    eval_ratio = float(client_splits["eval_ratio"])
    test_ratio = float(client_splits.get("test_ratio", 0.0))
    if eval_ratio <= 0.0:
        raise ValueError("FEMNIST requires client_splits.eval_ratio > 0 for model selection")
    if test_ratio <= 0.0:
        raise ValueError(
            "FEMNIST requires client_splits.test_ratio > 0. FEMNIST has no "
            "external test set, and this generator cuts one as a third "
            "per-writer slice; without it global_test.pt is a copy of the eval "
            "data and central_test_* is the validation metric under another name"
        )

    clients_metadata: list[dict[str, Any]] = []
    client_stats: list[dict[str, object]] = []
    global_test_x: list[Tensor] = []
    global_test_y: list[Tensor] = []
    seen_client_ids: set[str] = set()
    total_examples = 0

    for position, writer_id in enumerate(selected_writer_ids):
        source_indices = eligible_writer_indices[writer_id]
        images, labels = _load_writer_tensors(
            source_dataset,
            source_indices,
            image_column=image_column,
            label_column=label_column,
        )
        train_indices, eval_indices, test_indices = _split_writer_examples(
            num_examples=len(labels),
            eval_ratio=eval_ratio,
            test_ratio=test_ratio,
            # Keyed on the writer, not its position in the selected list. With
            # `seed + position` a writer's split depended on how many clients
            # were requested, so the same seed produced different data for the
            # same writer, and one position collided with the seed used to
            # select writers. derive_seed is a hash, so (seed, writer) fixes
            # the split and nothing else touches it.
            seed=derive_seed(seed, "femnist_writer_split", writer_id),
        )
        client_id = _client_id(writer_id)
        if client_id in seen_client_ids:
            raise ValueError(f"FEMNIST client ID collision: {client_id}")
        seen_client_ids.add(client_id)

        train_x = images[train_indices]
        train_y = labels[train_indices]
        eval_x = images[eval_indices]
        eval_y = labels[eval_indices]
        client_test_x = images[test_indices]
        client_test_y = labels[test_indices]
        shard = f"shards/{client_id}.pt"
        save_split_client_shard(
            output_dir / shard,
            train_x,
            train_y,
            eval_x,
            eval_y,
            client_test_x,
            client_test_y,
        )
        # The test slice, never the eval slice. global_test.pt is the pooled
        # per-client test set, so central_test_* and val_* are now measured on
        # disjoint data and selecting on one does not inflate the other.
        global_test_x.append(client_test_x)
        global_test_y.append(client_test_y)

        stats = compute_client_stats(
            client_id,
            list(range(len(labels))),
            labels.tolist(),
        )
        train_label_counts = compute_label_counts(train_y.tolist())
        eval_label_counts = compute_label_counts(eval_y.tolist())
        test_label_counts = compute_label_counts(client_test_y.tolist())
        stats.update(
            {
                "source_client_id": writer_id,
                "num_train_examples": len(train_y),
                "num_eval_examples": len(eval_y),
                "num_test_examples": len(client_test_y),
                "train_label_counts": train_label_counts,
                "eval_label_counts": eval_label_counts,
                "test_label_counts": test_label_counts,
            }
        )
        client_stats.append(stats)
        clients_metadata.append(
            {
                "client_id": client_id,
                "source_client_id": writer_id,
                "split": "train",
                "num_examples": len(labels),
                "num_train_examples": len(train_y),
                "num_eval_examples": len(eval_y),
                "num_test_examples": len(client_test_y),
                "shard": shard,
                "label_counts": stats["label_counts"],
                "train_label_counts": train_label_counts,
                "eval_label_counts": eval_label_counts,
                "test_label_counts": test_label_counts,
                "num_labels": stats["num_labels"],
                "dominant_label": stats["dominant_label"],
                "dominant_label_fraction": stats["dominant_label_fraction"],
            }
        )
        total_examples += len(labels)

        if on_progress is not None:
            on_progress(f"preparing writer {position + 1:,}/{len(selected_writer_ids):,}")

    test_x = torch.cat(global_test_x, dim=0)
    test_y = torch.cat(global_test_y, dim=0)
    save_client_shard(shards_dir / "global_test.pt", test_x, test_y)

    _write_partition_stats(
        output_dir=output_dir,
        client_stats=client_stats,
        source_name=source_name,
        source_revision=source_revision,
        source_num_clients=len(writer_indices),
        client_splits=client_splits,
    )
    manifest = {
        "dataset_name": "femnist",
        "format": "torch_shards",
        "num_clients": len(selected_writer_ids),
        "num_classes": 62,
        "class_names": FEMNIST_LABELS,
        "clients_file": "clients.jsonl",
        "global_test": "shards/global_test.pt",
        "shards_dir": "shards",
        "partition_strategy": "natural",
        "partition_key": writer_column,
        # `natural` has no strategy knobs, but the seed is still a partition
        # input here: it decides which writers are selected and where each
        # writer's own three slices fall (chapter 05 §3.6). The manifest named
        # neither, so a regeneration at a different seed under the same path
        # was undetectable downstream.
        "seed": seed,
        "partition_stats_file": "partition_stats.json",
        "client_stats_file": "client_stats.csv",
        "client_shard_format": "split_v2",
        "client_splits": {
            "train_ratio": train_ratio,
            "eval_ratio": eval_ratio,
            "test_ratio": test_ratio,
        },
        # Named so it cannot be confused with the eval split again: the pooled
        # global test set is the concatenation of the per-writer TEST slices,
        # which no client trains on and no checkpoint selection reads.
        "client_test_source": "within_client_holdout_disjoint_from_eval",
        "input_shape": [1, 28, 28],
        "input_dtype": "uint8",
        "input_range": [0, 255],
        "source": source_name,
        "source_revision": source_revision,
        "source_split": source_split,
        "source_num_examples": len(source_dataset),
        "source_num_clients": len(writer_indices),
        "min_samples_per_client": min_samples,
    }
    manifest_path = save_manifest(output_dir, manifest)
    save_clients_jsonl(output_dir, clients_metadata)
    return FEMNISTGenerationSummary(
        manifest_path=manifest_path,
        num_clients=len(selected_writer_ids),
        num_examples=total_examples,
        num_test_examples=len(test_y),
    )


def _load_source_dataset(
    source_name: str,
    source_revision: str | None,
    source_split: str,
    raw_dir: Path,
) -> Any:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found,import-untyped]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "FEMNIST generation requires the optional datasets dependency. "
            "Install it with: pip install -e '.[vision]'"
        ) from exc

    raw_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "split": source_split,
        "cache_dir": str(raw_dir),
    }
    if source_revision is not None:
        kwargs["revision"] = source_revision
    return load_dataset(source_name, **kwargs)


def _validate_source_columns(source_dataset: Any, required: set[str]) -> None:
    raw_columns = getattr(source_dataset, "column_names", None)
    if raw_columns is None:
        return
    columns = {str(column) for column in raw_columns}
    missing = sorted(required - columns)
    if missing:
        raise ValueError("FEMNIST source is missing required columns: " + ", ".join(missing))


def _group_indices_by_writer(
    source_dataset: Any,
    writer_column: str,
) -> dict[str, list[int]]:
    writer_indices: dict[str, list[int]] = {}
    for index, raw_writer_id in enumerate(source_dataset[writer_column]):
        writer_id = str(raw_writer_id)
        writer_indices.setdefault(writer_id, []).append(index)
    return writer_indices


def _select_writer_ids(
    writer_indices: Mapping[str, list[int]],
    requested_clients: int | None,
    seed: int,
) -> list[str]:
    writer_ids = sorted(writer_indices)
    if requested_clients is None:
        return writer_ids
    if requested_clients > len(writer_ids):
        raise ValueError(
            "partition.num_clients exceeds eligible FEMNIST writers: "
            f"{requested_clients} > {len(writer_ids)}"
        )
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(writer_ids), generator=generator).tolist()
    return sorted(writer_ids[index] for index in order[:requested_clients])


def _load_writer_tensors(
    source_dataset: Any,
    indices: Sequence[int],
    image_column: str,
    label_column: str,
) -> tuple[Tensor, Tensor]:
    if hasattr(source_dataset, "select"):
        selected = source_dataset.select(list(indices))
        raw_images = selected[image_column]
        raw_labels = selected[label_column]
    else:
        records = [source_dataset[index] for index in indices]
        raw_images = [record[image_column] for record in records]
        raw_labels = [record[label_column] for record in records]

    images = torch.stack([_image_to_uint8_tensor(image) for image in raw_images])
    labels = torch.as_tensor(raw_labels, dtype=torch.long)
    if labels.ndim != 1 or len(labels) != len(images):
        raise ValueError("FEMNIST images and labels must have matching row counts")
    if bool(torch.any(labels < 0)) or bool(torch.any(labels >= 62)):
        raise ValueError("FEMNIST labels must be in [0, 62)")
    return images.contiguous(), labels.contiguous()


def _image_to_uint8_tensor(image: Any) -> Tensor:
    if isinstance(image, Tensor):
        tensor = image.detach().cpu()
    elif hasattr(image, "convert"):
        try:
            from torchvision.transforms.functional import (  # type: ignore[import-untyped]
                pil_to_tensor,
            )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "FEMNIST image conversion requires torchvision. Install it with: "
                "pip install -e '.[vision]'"
            ) from exc
        tensor = pil_to_tensor(image.convert("L"))
    else:
        tensor = torch.as_tensor(image)

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.shape != (1, 28, 28):
        raise ValueError(
            f"FEMNIST images must have shape [1, 28, 28], received {list(tensor.shape)}"
        )
    if tensor.dtype.is_floating_point:
        tensor = tensor.float()
        if tensor.numel() and float(tensor.max().item()) <= 1.0:
            tensor = tensor * 255.0
        tensor = tensor.round().clamp(0.0, 255.0)
    return tensor.to(dtype=torch.uint8)


def _split_writer_examples(
    num_examples: int,
    eval_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """One writer's examples cut into three disjoint slices: train, eval, test.

    Only the test slice reaches global_test.pt. Previously there were two
    slices and the eval one was used as both, so val_* and central_test_* were
    the same 81,502 examples -- an honest generalization estimate for a fixed
    model at a fixed round, but the shipped configs select ~100 checkpoints on
    it and then report it, which is selection on the reported number.

    Each slice gets at least one example, so a writer needs at least three.
    """

    if num_examples < 3:
        raise ValueError("each FEMNIST writer must have at least three examples")
    for name, ratio in (("eval_ratio", eval_ratio), ("test_ratio", test_ratio)):
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"client_splits.{name} must be in (0, 1) for FEMNIST")
    if eval_ratio + test_ratio >= 1.0:
        raise ValueError("client_splits eval_ratio + test_ratio must leave room for training")

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(num_examples, generator=generator)
    # Floors that keep all three non-empty on the smallest writers, where the
    # ratios alone would round a slice away.
    eval_size = min(max(1, int(round(num_examples * eval_ratio))), num_examples - 2)
    test_size = min(
        max(1, int(round(num_examples * test_ratio))),
        num_examples - eval_size - 1,
    )
    return (
        order[eval_size + test_size :],
        order[:eval_size],
        order[eval_size : eval_size + test_size],
    )


def _client_id(writer_id: str) -> str:
    if _SAFE_CLIENT_ID.fullmatch(writer_id):
        return writer_id
    digest = hashlib.sha256(writer_id.encode("utf-8")).hexdigest()[:16]
    return f"writer_{digest}"


def _write_partition_stats(
    output_dir: Path,
    client_stats: list[dict[str, object]],
    source_name: str,
    source_revision: str | None,
    source_num_clients: int,
    client_splits: Mapping[str, float],
) -> None:
    sizes = [int(cast(Any, stats["num_examples"])) for stats in client_stats]
    global_counts: dict[str, int] = {}
    for stats in client_stats:
        raw_counts = stats.get("label_counts", {})
        if not isinstance(raw_counts, Mapping):
            continue
        for label, count in raw_counts.items():
            key = str(label)
            global_counts[key] = global_counts.get(key, 0) + int(cast(Any, count))

    payload = {
        "dataset_name": "femnist",
        "partition_strategy": "natural",
        "partition_key": "writer_id",
        "source": source_name,
        "source_revision": source_revision,
        "source_num_clients": source_num_clients,
        "num_clients": len(client_stats),
        "total_examples": sum(sizes),
        "min_examples_per_client": min(sizes) if sizes else 0,
        "max_examples_per_client": max(sizes) if sizes else 0,
        "mean_examples_per_client": sum(sizes) / len(sizes) if sizes else 0.0,
        "labels": sorted(global_counts, key=int),
        "global_label_counts": {
            label: global_counts[label] for label in sorted(global_counts, key=int)
        },
        "client_splits": {
            "train_ratio": float(client_splits["train_ratio"]),
            "eval_ratio": float(client_splits["eval_ratio"]),
            "test_ratio": float(client_splits.get("test_ratio", 0.0)),
        },
        "clients": client_stats,
    }
    (output_dir / "partition_stats.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_client_stats_csv(
        output_dir / "client_stats.csv",
        client_stats,
        sorted(global_counts, key=int),
    )


def _write_client_stats_csv(
    path: Path,
    client_stats: list[dict[str, object]],
    labels: list[str],
) -> None:
    fieldnames = [
        "client_id",
        "source_client_id",
        "num_examples",
        "num_train_examples",
        "num_eval_examples",
        "num_labels",
        "dominant_label",
        "dominant_label_fraction",
        *[f"label_{label}" for label in labels],
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for stats in client_stats:
            raw_counts = stats.get("label_counts", {})
            counts = raw_counts if isinstance(raw_counts, Mapping) else {}
            row: dict[str, object] = {
                "client_id": stats["client_id"],
                "source_client_id": stats["source_client_id"],
                "num_examples": stats["num_examples"],
                "num_train_examples": stats["num_train_examples"],
                "num_eval_examples": stats["num_eval_examples"],
                "num_labels": stats["num_labels"],
                "dominant_label": stats["dominant_label"],
                "dominant_label_fraction": stats["dominant_label_fraction"],
            }
            for label in labels:
                row[f"label_{label}"] = counts.get(label, 0)
            writer.writerow(row)


def _optional_positive_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("partition.num_clients must be a positive integer or null")
    parsed = int(value) if isinstance(value, int | str) else None
    if parsed is None or parsed <= 0:
        raise ValueError("partition.num_clients must be a positive integer or null")
    return parsed


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected config section to be a mapping")
    return cast(Mapping[str, Any], value)
