"""Torch shard persistence helpers for generated data."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, cast

import torch

from fedbrew.core.refusal import RunRefused


def save_client_shard(path: str | Path, x: Any, y: Any) -> None:
    """Save feature and target tensors to an old-format torch shard."""

    shard_path = Path(path)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x, "y": y}, shard_path)


def save_split_client_shard(
    path: str | Path,
    train_x: Any,
    train_y: Any,
    eval_x: Any,
    eval_y: Any,
    test_x: Any | None = None,
    test_y: Any | None = None,
) -> None:
    """Save split-aware client tensors, optionally including a test split."""

    shard_path = Path(path)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    if (test_x is None) != (test_y is None):
        raise ValueError("test_x and test_y must either both be provided or both omitted")
    shard = {
        "train": {"x": train_x, "y": train_y},
        "eval": {"x": eval_x, "y": eval_y},
    }
    if test_x is not None and test_y is not None:
        shard["test"] = {"x": test_x, "y": test_y}
    torch.save(shard, shard_path)


def load_client_shard(path: str | Path) -> dict[str, Any]:
    """Load an old-format or split-aware torch shard.

    Raises:
        RunRefused: If the file cannot be read, is not a torch file, or does not
            hold a dictionary -- a shard deleted, truncated or overwritten.
    """

    shard_path = Path(path)
    try:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    except OSError as error:
        raise RunRefused(
            f"cannot read the torch shard {shard_path}: {error.strerror or error}"
        ) from error
    except (RuntimeError, EOFError, pickle.UnpicklingError) as error:
        # What torch.load raises for a truncated archive, an empty file and a
        # file that is not a torch file at all. Caught around the load alone,
        # so nothing after it can be mistaken for a damaged file.
        raise RunRefused(
            f"the torch shard {shard_path} is not a readable torch file: {error}"
        ) from error
    if not isinstance(shard, dict):
        raise RunRefused(f"torch shard {shard_path} must contain a dictionary")
    return cast(dict[str, Any], shard)
