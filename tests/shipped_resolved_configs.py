"""Every shipped run config as the loader resolves it, and the record it must equal.

A config change that claims to change nothing -- a key made optional, a value
inferred instead of written, a block moved, arm files folded onto a family
base -- is held to that claim here: each shipped run config is loaded, its
resolved ``FullConfig`` flattened to dotted keys, and its planned column list
taken from the plan header's own function, and both are compared with
``shipped_resolved_configs.json``, recorded before the change.

The record is of a checkout without generated data, which is what CI loads:
the loader reads a manifest's client records and metadata when one is there,
so every load here goes through ``no_generated_data``, which makes each
manifest read find nothing, whatever ``data/generated`` holds.

Regenerate the record only for a change that is meant to change a resolved
config, and say so in its commit::

    PYTHONPATH=$PWD python -m tests.shipped_resolved_configs
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
RECORD = Path(__file__).with_name("shipped_resolved_configs.json")


def shipped_run_configs() -> list[Path]:
    """Every run config under configs/, as paths relative to the repository.

    Not the family bases, which are not run configs (``is_family_base``), and
    not configs/llm_assets/, a different schema.
    """

    from fedbrew.core.config import is_family_base

    return [
        path.relative_to(REPO)
        for path in sorted((REPO / "configs").rglob("*.yaml"))
        if "llm_assets" not in path.parts and not is_family_base(path)
    ]


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """A nested mapping as dotted keys; lists and tuples stay values, as lists."""

    if isinstance(value, Mapping):
        flat: dict[str, Any] = {}
        for key, item in value.items():
            flat.update(flatten(item, f"{prefix}{key}."))
        if not value and prefix:
            flat[prefix[:-1]] = {}
        return flat
    if isinstance(value, tuple | list):
        return {prefix[:-1]: [_plain(item) for item in value]}
    return {prefix[:-1]: value}


def _plain(value: Any) -> Any:
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


@contextlib.contextmanager
def no_generated_data() -> Iterator[None]:
    """Every manifest the loader reads is absent, as in a checkout without data."""

    with mock.patch("fedbrew.core.inferred.read_manifest", return_value=None):
        yield


def resolved(path: Path) -> dict[str, Any]:
    """The config at ``path`` (relative to the repository) as the loader resolves it."""

    from fedbrew.core.config import load_config
    from fedbrew.core.logging import _planned_metric_names

    with no_generated_data():
        config = load_config(REPO / path)
        return {
            "config": flatten(asdict(config)),
            "planned": list(_planned_metric_names(config)),
        }


def main() -> None:
    record = {str(path): resolved(path) for path in shipped_run_configs()}
    RECORD.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(record)} configs -> {RECORD.relative_to(REPO)}")


if __name__ == "__main__":
    main()
