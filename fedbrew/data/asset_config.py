"""Configuration for explicit dataset asset preparation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

_COMMIT_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")


class DatasetAssetConfigError(ValueError):
    """Raised when a dataset preparation config is invalid."""


@dataclass(frozen=True)
class DatasetAssetPreparationConfig:
    """Validated inputs for one immutable local dataset snapshot."""

    dataset_identifier: str
    revision: str
    cache_dir: Path
    config_path: Path


def load_dataset_asset_config(path: str | Path) -> DatasetAssetPreparationConfig:
    """Load a preparation config and require an immutable commit revision."""

    config_path = Path(path).expanduser()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetAssetConfigError(
            f"dataset asset config does not exist: {config_path}"
        ) from exc
    except yaml.YAMLError as exc:
        raise DatasetAssetConfigError(
            f"dataset asset config is not valid YAML: {config_path}"
        ) from exc

    if not isinstance(raw, Mapping):
        raise DatasetAssetConfigError("dataset asset config must be a YAML mapping")
    values = cast(Mapping[str, Any], raw)
    _reject_unknown_keys(values)

    dataset_identifier = _required_string(values, "dataset_identifier")
    revision = _required_string(values, "revision")
    if not _COMMIT_REVISION.fullmatch(revision):
        raise DatasetAssetConfigError(
            "revision must be a concrete 40-character Hugging Face commit SHA; "
            "moving revisions such as 'main' are not allowed"
        )
    cache_dir = Path(_required_string(values, "cache_dir")).expanduser()

    return DatasetAssetPreparationConfig(
        dataset_identifier=dataset_identifier,
        revision=revision.lower(),
        cache_dir=cache_dir,
        config_path=config_path,
    )


#: Every key this loader reads. Same rule as the model-asset loader in
#: fedbrew/data/llm_assets/config.py: an unread key is dropped in silence, so a
#: misspelling snapshots a different dataset than the file says.
_KNOWN_KEYS = frozenset({"dataset_identifier", "revision", "cache_dir"})


def _reject_unknown_keys(values: Mapping[str, Any]) -> None:
    """Refuse a key this loader will never read."""

    unknown = sorted(set(values) - _KNOWN_KEYS)
    if not unknown:
        return
    raise DatasetAssetConfigError(
        "dataset asset config does not read: "
        + ", ".join(unknown)
        + ". It reads: "
        + ", ".join(sorted(_KNOWN_KEYS))
        + ". An unread key is dropped silently, so a misspelling snapshots a "
        "different dataset than the file says. Dataset preparation always "
        "loads with trust_remote_code=False; it is not configurable."
    )


def _required_string(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetAssetConfigError(f"{key} is required and must be non-empty")
    return value.strip()
