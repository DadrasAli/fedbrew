"""Generate an OpenImage-SHAPED synthetic dataset for cost measurement.

The pixels are noise. The point is the shape: 13,771 clients, a heavy-tailed
per-client sample count averaging ~94, 596 classes with per-client label skew,
64x64x3 uint8, a 10% global test shard. Round cost depends on those and not on
what the images contain, so this measures the real per-round cost of an
OpenImage run without waiting on the 66 GB download.

Do NOT train anything real on this. It exists to turn a projected round time
into a measured one, and should be deleted afterwards.

Usage: python tools/generate_openimage_shaped_synthetic.py --output-dir DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fedbrew.data.writers.manifest import save_clients_jsonl, save_manifest
from fedbrew.data.writers.torch_shards import save_client_shard, save_split_client_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-clients", type=int, default=13771)
    parser.add_argument("--total-examples", type=int, default=1_300_000)
    parser.add_argument("--num-classes", type=int, default=596)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--global-test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def client_sample_counts(rng, num_clients: int, total: int) -> np.ndarray:
    """Heavy-tailed per-client counts summing to ``total``, min 4.

    FedScale's OpenImage partition is strongly skewed -- a few prolific users
    and a long tail of users with a handful of photos. A lognormal reproduces
    that shape well enough for a cost model, which only needs the tail to exist
    so that per-client overheads and shard sizes are realistic.
    """

    raw = rng.lognormal(mean=0.0, sigma=1.0, size=num_clients)
    counts = np.maximum(4, np.round(raw / raw.sum() * total)).astype(np.int64)
    # Correct the rounding drift on the largest client so the total is exact.
    counts[int(np.argmax(counts))] += total - int(counts.sum())
    return counts


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    shards_dir = args.output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    counts = client_sample_counts(rng, args.num_clients, args.total_examples)
    shape = (3, args.resolution, args.resolution)
    records: list[dict] = []
    total_train = total_eval = 0

    print(
        f"{args.num_clients:,} clients | {counts.sum():,} examples "
        f"| min {counts.min()} med {int(np.median(counts))} max {counts.max():,}",
        flush=True,
    )

    for index, count in enumerate(counts):
        client_id = f"client_{index:05d}"
        # Each client photographs only a few object types, so restrict its
        # labels to a small random subset rather than sampling all 596.
        palette = rng.choice(args.num_classes, size=int(rng.integers(3, 16)), replace=False)
        labels = torch.from_numpy(rng.choice(palette, size=int(count))).long()
        images = torch.randint(0, 256, (int(count), *shape), dtype=torch.uint8, generator=generator)

        num_train = max(1, int(round(int(count) * args.train_ratio)))
        num_train = min(num_train, int(count) - 1)
        save_split_client_shard(
            shards_dir / f"{client_id}.pt",
            images[:num_train],
            labels[:num_train],
            images[num_train:],
            labels[num_train:],
        )
        total_train += num_train
        total_eval += int(count) - num_train
        records.append(
            {
                "client_id": client_id,
                "num_examples": int(count),
                "num_train_examples": num_train,
                "num_eval_examples": int(count) - num_train,
                "shard": f"shards/{client_id}.pt",
                "split": "train",
            }
        )
        if (index + 1) % 2000 == 0:
            print(f"  {index + 1:,}/{args.num_clients:,} shards", flush=True)

    num_global_test = int(args.total_examples * args.global_test_ratio)
    print(f"global test shard: {num_global_test:,} examples", flush=True)
    save_client_shard(
        shards_dir / "global_test.pt",
        torch.randint(0, 256, (num_global_test, *shape), dtype=torch.uint8, generator=generator),
        torch.from_numpy(rng.integers(0, args.num_classes, size=num_global_test)).long(),
    )

    save_clients_jsonl(args.output_dir, records)
    manifest = {
        "client_shard_format": "split_v1",
        "client_splits": {
            "train_ratio": args.train_ratio,
            "eval_ratio": round(1.0 - args.train_ratio, 6),
        },
        "clients_file": "clients.jsonl",
        "dataset_name": "openimage_shaped_synthetic",
        "format": "torch_shards",
        "global_test": "shards/global_test.pt",
        "input_dtype": "uint8",
        "input_range": [0, 255],
        "input_shape": list(shape),
        "min_samples_per_client": int(counts.min()),
        "num_classes": args.num_classes,
        "num_clients": args.num_clients,
        "partition_key": "synthetic_client",
        "partition_strategy": "synthetic_skew",
        "shards_dir": "shards",
        "source": "SYNTHETIC NOISE - cost measurement only, not for training",
        "source_num_clients": args.num_clients,
        "source_num_examples": int(counts.sum()),
        "test_split": "synthetic_global",
    }
    save_manifest(args.output_dir, manifest)
    print(json.dumps({"train": total_train, "eval": total_eval}, indent=1))
    print(f"done -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
