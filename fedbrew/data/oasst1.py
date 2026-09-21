"""Explicitly prepare pinned OASST1 snapshots for offline generation."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fedbrew.core.console import Rail, add_output_arguments, silent_rail, surface_from_args
from fedbrew.core.download_progress import redirect_datasets_progress
from fedbrew.core.logging import print_download_progress
from fedbrew.data.asset_config import (
    DatasetAssetConfigError,
    DatasetAssetPreparationConfig,
    load_dataset_asset_config,
)
from fedbrew.data.asset_manifest import (
    DATASET_ASSET_MANIFEST_FILENAME,
    DATASET_ASSET_MANIFEST_SCHEMA_VERSION,
    REQUIRED_SPLITS,
    DatasetAssetFile,
    DatasetAssetManifest,
    DatasetAssetManifestError,
    load_dataset_asset_manifest,
    sha256_file,
)


class OASST1PreparationError(RuntimeError):
    """Raised when pinned OASST1 assets cannot be prepared."""


#: The stages worth a line, in order. The config load and the two mkdirs are
#: absent on purpose: they are sub-millisecond and tell a reader nothing they
#: did not already know from typing the command.
STAGES = ("cache", "train split", "validation split", "verify")


def prepare_oasst1_assets(config_path: str | Path, rail: Rail | None = None) -> Path:
    """Download pinned OASST1 splits once and save verified Parquet files.

    `rail` is where the stages report. It defaults to a rail on a quiet
    surface, which renders nothing -- so every existing caller behaves exactly
    as it did, and the CLI passes a real one.
    """

    rail = silent_rail() if rail is None else rail
    config = load_dataset_asset_config(config_path)
    cache_root = config.cache_dir.resolve()
    manifest_path = cache_root / DATASET_ASSET_MANIFEST_FILENAME
    rail.detail("dataset", config.dataset_identifier, note=f"revision {config.revision}")
    rail.detail("cache root", str(cache_root))
    # The single biggest gap in this command: the cache check re-hashes every
    # split file and re-checks its row counts, then either returns immediately
    # or falls through to a full
    # download -- and the final print was identical either way, so the command
    # could not tell a reader which of the two things it had just done.
    with rail.stage("cache") as stage:
        hit = _has_verified_matching_cache(manifest_path, config)
        stage.done("verified, nothing to download" if hit else "no usable cache, preparing")
    if hit:
        return manifest_path

    download_cache = cache_root / "downloads"
    snapshot_dir = cache_root / "snapshots"
    download_cache.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, DatasetAssetFile] = {}
    for split in REQUIRED_SPLITS:
        with rail.stage(f"{split} split") as stage:
            dataset = _download_split(config, split, download_cache)
            try:
                row_count = len(dataset)
            except (TypeError, AttributeError) as exc:
                raise OASST1PreparationError(
                    f"downloaded OASST1 {split} split has no finite row count"
                ) from exc
            if isinstance(row_count, bool) or not isinstance(row_count, int):
                raise OASST1PreparationError(
                    f"downloaded OASST1 {split} split has an invalid row count"
                )
            if row_count <= 0:
                raise OASST1PreparationError(f"downloaded OASST1 {split} split is empty")

            snapshot_path = snapshot_dir / f"{split}.parquet"
            temporary_path = snapshot_dir / f".{split}.parquet.incomplete"
            try:
                temporary_path.unlink(missing_ok=True)
                dataset.to_parquet(str(temporary_path))
                os.replace(temporary_path, snapshot_path)
            except Exception as exc:
                temporary_path.unlink(missing_ok=True)
                raise OASST1PreparationError(
                    f"unable to save the OASST1 {split} Parquet snapshot at {snapshot_path}"
                ) from exc

            relative_path = snapshot_path.relative_to(cache_root).as_posix()
            digest = sha256_file(snapshot_path)
            files[split] = DatasetAssetFile(
                manifest_path=manifest_path,
                split=split,
                path=relative_path,
                sha256=digest,
                row_count=row_count,
            )
            # The row count and the digest are both new facts: one says what
            # was downloaded, the other is what every later run checks it
            # against.
            stage.done(f"{row_count:,} rows", note=f"sha256 {digest[:12]}")

    manifest = DatasetAssetManifest(
        manifest_path=manifest_path,
        schema_version=DATASET_ASSET_MANIFEST_SCHEMA_VERSION,
        dataset_identifier=config.dataset_identifier,
        requested_revision=config.revision,
        resolved_revision=config.revision,
        cache_path=str(cache_root),
        storage_format="parquet",
        files=files,
        preparation_timestamp=datetime.now(timezone.utc).isoformat(),
    )
    _write_manifest(manifest)
    # A second, independent pass over the files just written: re-hashed and
    # re-counted, so "prepared" means "verified as the thing offline
    # generation will consume", not "written without raising".
    with rail.stage("verify") as stage:
        load_dataset_asset_manifest(
            manifest_path,
            preparation_config=config.config_path,
            require_files=True,
            verify_hashes=True,
        )
        stage.done(f"{len(files)} files re-hashed, row counts match")
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the explicit OASST1 preparation command."""

    parser = argparse.ArgumentParser(
        description="Prepare pinned OASST1 train/validation snapshots for offline use."
    )
    parser.add_argument("--config", required=True, help="Dataset asset YAML.")
    add_output_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the OASST1 preparation CLI."""

    args = parse_args(argv)
    surface = surface_from_args(args)
    surface.rule("PREPARE OASST1")
    rail = surface.rail(STAGES)
    try:
        manifest_path = prepare_oasst1_assets(args.config, rail)
    except (
        DatasetAssetConfigError,
        DatasetAssetManifestError,
        OASST1PreparationError,
    ) as exc:
        raise SystemExit(f"fedbrew prepare-oasst1: error: {exc}") from exc
    surface.final(f"Prepared OASST1 dataset assets: {manifest_path}")


def _has_verified_matching_cache(
    manifest_path: Path,
    config: DatasetAssetPreparationConfig,
) -> bool:
    try:
        manifest = load_dataset_asset_manifest(
            manifest_path,
            preparation_config=config.config_path,
            require_files=True,
            verify_hashes=True,
        )
    except DatasetAssetManifestError:
        return False
    return (
        manifest.dataset_identifier == config.dataset_identifier
        and manifest.requested_revision == config.revision
        and manifest.resolved_revision == config.revision
        and Path(manifest.cache_path) == config.cache_dir.resolve()
    )


def _download_split(
    config: DatasetAssetPreparationConfig,
    split: str,
    download_cache: Path,
) -> Any:
    """Invoke the only network-capable dataset operation in this package."""

    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:  # pragma: no cover - install-mode dependent.
        raise OASST1PreparationError(
            'OASST1 preparation requires Datasets. Install with: pip install -e ".[llm]"'
        ) from exc
    try:
        with redirect_datasets_progress(print_download_progress):
            return load_dataset(
                config.dataset_identifier,
                revision=config.revision,
                split=split,
                cache_dir=str(download_cache),
                trust_remote_code=False,
                streaming=False,
            )
    except Exception as exc:
        raise OASST1PreparationError(
            f"unable to resolve OASST1 {split} split for dataset "
            f"{config.dataset_identifier!r} at pinned revision "
            f"{config.revision!r}; verify network access, identifier, and revision"
        ) from exc


def _write_manifest(manifest: DatasetAssetManifest) -> None:
    temporary_path = manifest.manifest_path.with_suffix(".json.incomplete")
    temporary_path.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, manifest.manifest_path)


if __name__ == "__main__":
    main()
