"""Verified, network-free access to prepared dataset snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

DATASET_ASSET_MANIFEST_SCHEMA_VERSION = 1
DATASET_ASSET_MANIFEST_FILENAME = "asset_manifest.json"
REQUIRED_SPLITS = ("train", "validation")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DatasetAssetManifestError(ValueError):
    """Raised when prepared dataset assets are missing or invalid."""


@dataclass(frozen=True)
class DatasetAssetFile:
    """One split snapshot recorded in a dataset asset manifest."""

    manifest_path: Path
    split: str
    path: str
    sha256: str
    row_count: int

    @property
    def local_path(self) -> Path:
        """Return the local path, resolved relative to the manifest."""

        return _resolve_asset_path(self.manifest_path, self.path)

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized file metadata."""

        return {
            "path": self.path,
            "sha256": self.sha256,
            "row_count": self.row_count,
        }


@dataclass(frozen=True)
class DatasetAssetManifest:
    """Validated metadata for immutable train and validation snapshots."""

    manifest_path: Path
    schema_version: int
    dataset_identifier: str
    requested_revision: str
    resolved_revision: str | None
    cache_path: str
    storage_format: str
    files: Mapping[str, DatasetAssetFile]
    preparation_timestamp: str

    def file_for_split(self, split: str) -> DatasetAssetFile:
        """Return a prepared split or raise a clear manifest error."""

        try:
            return self.files[split]
        except KeyError as exc:
            raise DatasetAssetManifestError(
                f"prepared dataset manifest has no {split!r} split"
            ) from exc

    @property
    def source_hashes(self) -> dict[str, str]:
        """Return split-to-SHA-256 metadata for downstream provenance."""

        return {split: asset.sha256 for split, asset in self.files.items()}

    @property
    def row_counts(self) -> dict[str, int]:
        """Return split-to-row-count metadata for downstream provenance."""

        return {split: asset.row_count for split, asset in self.files.items()}

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized v1 manifest fields."""

        return {
            "schema_version": self.schema_version,
            "dataset_identifier": self.dataset_identifier,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "cache_path": self.cache_path,
            "storage_format": self.storage_format,
            "files": {split: asset.as_dict() for split, asset in self.files.items()},
            "preparation_timestamp": self.preparation_timestamp,
        }


def load_dataset_asset_manifest(
    path: str | Path,
    *,
    preparation_config: str | Path | None = None,
    require_files: bool = True,
    verify_hashes: bool = True,
) -> DatasetAssetManifest:
    """Load local dataset metadata without invoking any network-capable API.

    When ``verify_hashes`` is true, both each file's SHA-256 and the Parquet
    metadata row count are verified. Relative paths are resolved from the
    manifest directory, never from the process working directory.
    """

    manifest_path = _expand_path(path)
    command = preparation_command_for(manifest_path, preparation_config)
    if not manifest_path.is_file():
        raise DatasetAssetManifestError(
            f"Prepared OASST1 dataset assets are missing at {manifest_path}. Run: {command}"
        )

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetAssetManifestError(
            f"Prepared dataset asset manifest is unreadable at {manifest_path}. Run: {command}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise DatasetAssetManifestError("prepared dataset asset manifest must be a JSON object")
    values = cast(Mapping[str, Any], raw)
    raw_files = values.get("files")
    if not isinstance(raw_files, Mapping):
        raise DatasetAssetManifestError(
            "prepared dataset asset manifest files must be a JSON object"
        )

    files: dict[str, DatasetAssetFile] = {}
    for split, raw_file in raw_files.items():
        if not isinstance(split, str) or not isinstance(raw_file, Mapping):
            raise DatasetAssetManifestError(
                "prepared dataset asset manifest contains invalid file metadata"
            )
        file_values = cast(Mapping[str, Any], raw_file)
        files[split] = DatasetAssetFile(
            manifest_path=manifest_path,
            split=split,
            path=_required_string(file_values, "path"),
            sha256=_required_string(file_values, "sha256"),
            row_count=_required_int(file_values, "row_count"),
        )

    manifest = DatasetAssetManifest(
        manifest_path=manifest_path,
        schema_version=_required_int(values, "schema_version"),
        dataset_identifier=_required_string(values, "dataset_identifier"),
        requested_revision=_required_string(values, "requested_revision"),
        resolved_revision=_optional_string(values, "resolved_revision"),
        cache_path=_required_string(values, "cache_path"),
        storage_format=_required_string(values, "storage_format"),
        files=files,
        preparation_timestamp=_required_string(values, "preparation_timestamp"),
    )
    _validate_manifest(manifest)

    if require_files:
        for split in REQUIRED_SPLITS:
            asset = manifest.file_for_split(split)
            if not asset.local_path.is_file():
                raise DatasetAssetManifestError(
                    f"Prepared OASST1 {split} snapshot is missing at "
                    f"{asset.local_path}. Run: {command}"
                )
            if verify_hashes:
                _verify_asset_file(asset, command)
    return manifest


def preparation_command_for(
    manifest_path: str | Path,
    preparation_config: str | Path | None = None,
) -> str:
    """Return the exact CLI command for preparing this dataset cache."""

    path = _expand_path(manifest_path)
    if preparation_config is None:
        # Raw-dataset download manifests live in data/configs/assets/; the
        # partition configs beside them are a different kind of file.
        preparation_config = Path("data") / "configs" / "assets" / f"{path.parent.name}.yaml"
    return f"fedbrew prepare-oasst1 --config {preparation_config}"


def sha256_file(path: Path) -> str:
    """Compute a file SHA-256 without reading the whole snapshot into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_manifest(manifest: DatasetAssetManifest) -> None:
    if manifest.schema_version != DATASET_ASSET_MANIFEST_SCHEMA_VERSION:
        raise DatasetAssetManifestError(
            f"unsupported prepared dataset asset manifest schema_version: {manifest.schema_version}"
        )
    if manifest.storage_format != "parquet":
        raise DatasetAssetManifestError("storage_format must be 'parquet'")
    missing_splits = set(REQUIRED_SPLITS).difference(manifest.files)
    if missing_splits:
        raise DatasetAssetManifestError(
            "prepared dataset asset manifest is missing required splits: "
            + ", ".join(sorted(missing_splits))
        )
    for split, asset in manifest.files.items():
        if asset.row_count < 0:
            raise DatasetAssetManifestError(f"{split} row_count cannot be negative")
        if not _SHA256.fullmatch(asset.sha256):
            raise DatasetAssetManifestError(
                f"{split} sha256 must be a lowercase 64-character digest"
            )
        relative_path = Path(asset.path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise DatasetAssetManifestError(
                f"{split} path must remain relative to the asset manifest"
            )


def _verify_asset_file(asset: DatasetAssetFile, command: str) -> None:
    actual_hash = sha256_file(asset.local_path)
    if actual_hash != asset.sha256:
        raise DatasetAssetManifestError(
            f"Prepared OASST1 {asset.split} snapshot failed SHA-256 "
            f"verification at {asset.local_path}. Run: {command}"
        )
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on install mode.
        raise DatasetAssetManifestError(
            "Parquet verification requires the LLM dependencies. Install with: "
            'pip install -e ".[llm]"'
        ) from exc
    try:
        actual_rows = parquet.read_metadata(asset.local_path).num_rows
    except Exception as exc:
        raise DatasetAssetManifestError(
            f"Prepared OASST1 {asset.split} snapshot is not readable Parquet at "
            f"{asset.local_path}. Run: {command}"
        ) from exc
    if actual_rows != asset.row_count:
        raise DatasetAssetManifestError(
            f"Prepared OASST1 {asset.split} snapshot row count is {actual_rows}, "
            f"expected {asset.row_count}. Run: {command}"
        )


def _resolve_asset_path(manifest_path: Path, value: str) -> Path:
    return manifest_path.parent / _expand_path(value)


def _expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def _required_string(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetAssetManifestError(f"prepared dataset asset manifest requires non-empty {key}")
    return value


def _optional_string(values: Mapping[str, Any], key: str) -> str | None:
    value = values.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise DatasetAssetManifestError(
            f"prepared dataset asset manifest {key} must be null or str"
        )
    return value


def _required_int(values: Mapping[str, Any], key: str) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetAssetManifestError(f"prepared dataset asset manifest {key} must be an integer")
    return value
