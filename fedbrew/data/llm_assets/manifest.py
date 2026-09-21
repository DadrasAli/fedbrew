"""Machine-readable manifest for prepared Hugging Face assets."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from fedbrew.core.refusal import RunRefused

ASSET_MANIFEST_SCHEMA_VERSION = 1
ASSET_MANIFEST_FILENAME = "asset_manifest.json"


class AssetManifestError(RunRefused):
    """Raised when prepared LLM assets are missing or malformed.

    A refusal, so `fedbrew run` prints its reason rather than a traceback; still a
    ValueError, as it always was, for every caller that catches one.
    """


@dataclass(frozen=True)
class AssetManifest:
    """Validated, locally resolvable prepared-asset metadata."""

    manifest_path: Path
    schema_version: int
    model_identifier: str
    tokenizer_identifier: str
    requested_revision: str
    resolved_revision: str | None
    cache_path: str
    model_path: str
    tokenizer_path: str
    vocabulary_size: int
    model_type: str
    preparation_timestamp: str
    trust_remote_code: bool

    @property
    def model_asset_path(self) -> Path:
        """Return the local model directory recorded by the manifest."""

        return _resolve_asset_path(self.manifest_path, self.model_path)

    @property
    def tokenizer_asset_path(self) -> Path:
        """Return the local tokenizer directory recorded by the manifest."""

        return _resolve_asset_path(self.manifest_path, self.tokenizer_path)

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized v1 manifest fields."""

        return {
            "schema_version": self.schema_version,
            "model_identifier": self.model_identifier,
            "tokenizer_identifier": self.tokenizer_identifier,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "cache_path": self.cache_path,
            "model_path": self.model_path,
            "tokenizer_path": self.tokenizer_path,
            "vocabulary_size": self.vocabulary_size,
            "model_type": self.model_type,
            "preparation_timestamp": self.preparation_timestamp,
            "trust_remote_code": self.trust_remote_code,
        }


def load_asset_manifest(
    path: str | Path,
    *,
    preparation_config: str | Path | None = None,
    require_assets: bool = True,
) -> AssetManifest:
    """Load a prepared-asset manifest and optionally require its local files.

    Relative ``model_path`` and ``tokenizer_path`` values are interpreted relative
    to the manifest directory, not the process working directory.
    """

    manifest_path = _expand_path(path)
    command = preparation_command_for(manifest_path, preparation_config)
    if not manifest_path.is_file():
        raise AssetManifestError(
            f"Prepared Hugging Face assets are missing at {manifest_path}. Run: {command}"
        )

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetManifestError(
            f"Prepared asset manifest is unreadable at {manifest_path}. Run: {command}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise AssetManifestError("prepared asset manifest must be a JSON object")
    values = cast(Mapping[str, Any], raw)

    manifest = AssetManifest(
        manifest_path=manifest_path,
        schema_version=_required_int(values, "schema_version"),
        model_identifier=_required_string(values, "model_identifier"),
        tokenizer_identifier=_required_string(values, "tokenizer_identifier"),
        requested_revision=_required_string(values, "requested_revision"),
        resolved_revision=_optional_string(values, "resolved_revision"),
        cache_path=_required_string(values, "cache_path"),
        model_path=_required_string(values, "model_path"),
        tokenizer_path=_required_string(values, "tokenizer_path"),
        vocabulary_size=_required_int(values, "vocabulary_size"),
        model_type=_required_string(values, "model_type"),
        preparation_timestamp=_required_string(values, "preparation_timestamp"),
        trust_remote_code=_required_bool(values, "trust_remote_code"),
    )
    _validate_manifest(manifest)

    if require_assets:
        missing = [
            asset_path
            for asset_path in (
                manifest.model_asset_path,
                manifest.tokenizer_asset_path,
            )
            if not asset_path.is_dir()
        ]
        if missing:
            missing_text = ", ".join(str(asset_path) for asset_path in missing)
            raise AssetManifestError(
                f"Prepared Hugging Face asset directories are missing: "
                f"{missing_text}. Run: {command}"
            )
    return manifest


def preparation_command_for(
    manifest_path: str | Path,
    preparation_config: str | Path | None = None,
) -> str:
    """Return the exact CLI command that prepares the requested cache."""

    path = _expand_path(manifest_path)
    if preparation_config is None:
        preparation_config = Path("configs") / "llm_assets" / (f"{path.parent.name}.yaml")
    return f"fedbrew prepare-llm --config {preparation_config}"


def _resolve_asset_path(manifest_path: Path, value: str) -> Path:
    path = _expand_path(value)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def _validate_manifest(manifest: AssetManifest) -> None:
    if manifest.schema_version != ASSET_MANIFEST_SCHEMA_VERSION:
        raise AssetManifestError(
            f"unsupported prepared asset manifest schema_version: {manifest.schema_version}"
        )
    if manifest.vocabulary_size <= 0:
        raise AssetManifestError("vocabulary_size must be positive")
    if manifest.trust_remote_code:
        raise AssetManifestError("prepared asset manifest must set trust_remote_code to false")


def _required_string(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AssetManifestError(f"prepared asset manifest requires non-empty {key}")
    return value


def _optional_string(values: Mapping[str, Any], key: str) -> str | None:
    value = values.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AssetManifestError(f"prepared asset manifest {key} must be null or str")
    return value


def _required_int(values: Mapping[str, Any], key: str) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise AssetManifestError(f"prepared asset manifest {key} must be an integer")
    return value


def _required_bool(values: Mapping[str, Any], key: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise AssetManifestError(f"prepared asset manifest {key} must be a boolean")
    return value
