"""Configuration loading for explicit Hugging Face asset preparation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]


class AssetConfigError(ValueError):
    """Raised when an LLM asset preparation config is invalid."""


@dataclass(frozen=True)
class AssetPreparationConfig:
    """Validated inputs for one self-contained model/tokenizer cache."""

    model_identifier: str
    tokenizer_identifier: str
    revision: str
    cache_dir: Path
    config_path: Path


def load_asset_config(path: str | Path) -> AssetPreparationConfig:
    """Load and validate an asset preparation YAML file."""

    config_path = Path(path).expanduser()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AssetConfigError(f"LLM asset config does not exist: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise AssetConfigError(f"LLM asset config is not valid YAML: {config_path}") from exc

    if not isinstance(raw, Mapping):
        raise AssetConfigError("LLM asset config must be a YAML mapping")
    values = cast(Mapping[str, Any], raw)
    _reject_unknown_keys(values)

    model_identifier = _required_string(values, "model_identifier")
    tokenizer_identifier = _required_string(values, "tokenizer_identifier")
    revision = _required_string(values, "revision", explicit=True)
    cache_dir_text = _required_string(values, "cache_dir")

    return AssetPreparationConfig(
        model_identifier=model_identifier,
        tokenizer_identifier=tokenizer_identifier,
        revision=revision,
        cache_dir=Path(cache_dir_text).expanduser(),
        config_path=config_path,
    )


#: Every key this loader reads. Anything else was dropped in silence: three
#: shipped configs carried ``trust_remote_code: false``, which nothing read and
#: nothing rejected, so a key that reads as a security control asserted nothing.
_KNOWN_KEYS = frozenset({"model_identifier", "tokenizer_identifier", "revision", "cache_dir"})

#: Keys that were removed because the value they name is not configurable.
_REMOVED_KEYS = {
    "trust_remote_code": (
        "asset preparation always loads with trust_remote_code=False and "
        "records that in the manifest, which load_asset_manifest then refuses "
        "to read as true. Making it configurable would let preparation execute "
        "Hub code while the run config's own default kept the run looking clean"
    ),
}


def _reject_unknown_keys(values: Mapping[str, Any]) -> None:
    """Refuse a key this loader will never read."""

    unknown = sorted(set(values) - _KNOWN_KEYS)
    if not unknown:
        return
    removed = [
        f"{name} has been removed ({_REMOVED_KEYS[name]})"
        for name in unknown
        if name in _REMOVED_KEYS
    ]
    if removed:
        raise AssetConfigError("; ".join(removed))
    raise AssetConfigError(
        "LLM asset config does not read: "
        + ", ".join(unknown)
        + ". It reads: "
        + ", ".join(sorted(_KNOWN_KEYS))
        + ". An unread key is dropped silently, so a misspelling prepares a "
        "different asset than the file says."
    )


def _required_string(
    values: Mapping[str, Any],
    key: str,
    *,
    explicit: bool = False,
) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        qualifier = " and must be non-empty" if explicit else ""
        raise AssetConfigError(f"{key} is required{qualifier}")
    return value.strip()
