"""Optional manifest dataset staging for job-local scratch storage."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from fedbrew.core.config import FullConfig
from fedbrew.core.paths import expand_path, has_unexpanded_env, resolve_data_path

#: Written into a staged tree once its copy has finished. Its presence is what
#: makes the tree readable; a directory without it is a copy in progress or a
#: copy that died, and either way not a dataset.
STAGING_MARKER = ".fedbrew_staged.json"


def maybe_stage_manifest_dataset(config: FullConfig) -> FullConfig:
    """Optionally copy a manifest dataset directory to local scratch."""

    staging = config.runtime.extra.get("data_staging", {})
    if not isinstance(staging, dict) or not bool(staging.get("enabled", False)):
        return config
    if config.data.name != "manifest_dataset":
        return config

    data_path = config.data.path
    if not data_path:
        print("Data staging skipped: manifest path is empty.")
        return config

    source_manifest = resolve_data_path(data_path)
    source_dir = source_manifest.parent
    if not source_manifest.exists():
        print(f"Data staging skipped: manifest not found: {source_manifest}")
        return config

    local_root = resolve_staging_root(staging)
    if local_root is None:
        print("Data staging skipped: no usable local scratch root was configured.")
        return config

    destination_dir = local_root / staged_directory_name(source_dir)
    try:
        destination_dir = _stage_tree(source_dir, destination_dir)
    except OSError as exc:
        print(f"Data staging skipped: could not copy dataset to {destination_dir}: {exc}")
        return config

    staged_manifest = destination_dir / source_manifest.name
    if not staged_manifest.exists():
        print(f"Data staging skipped: staged manifest missing: {staged_manifest}")
        return config

    data = replace(config.data, path=str(staged_manifest))
    return replace(config, data=data)


def staged_directory_name(source_dir: Path) -> str:
    """The scratch subdirectory one source directory stages into.

    The name alone was the key -- `local_root / source_dir.name` -- so two
    datasets whose directories share a basename staged into the same place.
    Measured: two trees both named `shared_basename`, staged in turn into one
    root, left a directory holding the second's `manifest.json` and *both*
    sets of shards, because `copytree(dirs_exist_ok=True)` overwrites what it
    matches and removes nothing it does not. A run given the first path then
    read the second's manifest. The realistic pairing is not contrived: a
    config pointing at `$FL_DATA_ROOT/oasst1_qwen05b_4clients` and one
    pointing at `data/generated/oasst1_qwen05b_4clients` are two trees with
    one basename. P10-F25.

    Args:
        source_dir: The directory holding the manifest, already resolved.

    Returns:
        `<basename>-<digest>`, the digest taken over the resolved absolute
        source path. Keeping the basename keeps the scratch directory legible
        to whoever is looking at the node; the digest is what makes it
        unambiguous. Two runs of the *same* dataset still share one copy,
        which is the point of staging on a packed node.
    """

    digest = hashlib.sha256(str(source_dir).encode("utf-8")).hexdigest()[:12]
    return f"{source_dir.name}-{digest}"


def _stage_tree(source_dir: Path, destination_dir: Path) -> Path:
    """Copy `source_dir` to `destination_dir`, completely or not at all.

    `copytree` writes into the destination as it goes, and the only
    completeness check was that the manifest existed there afterwards -- a
    file `copytree` may write long before the shards beside it. So a second
    run on the same node could read a tree that was still being written. The
    copy now lands in a private temporary sibling and is renamed into place
    once its marker is written, so a destination either does not exist or is
    a finished dataset.

    Args:
        source_dir: The tree to copy.
        destination_dir: Where it belongs, from `staged_directory_name`.

    Returns:
        `destination_dir`, staged.

    Raises:
        OSError: If the copy fails. The caller reports it and stages nothing,
            which is what it did before.
    """

    if _is_staged(source_dir, destination_dir):
        return destination_dir

    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    # Unique per attempt: two runs of the same dataset racing on one node each
    # copy into their own directory, and the loser of the rename below drops
    # its copy rather than merging into the winner's.
    partial_dir = destination_dir.with_name(f".{destination_dir.name}.{uuid.uuid4().hex}.partial")
    try:
        shutil.copytree(source_dir, partial_dir)
        _write_marker(source_dir, partial_dir)
        try:
            os.rename(partial_dir, destination_dir)
        except OSError:
            if _is_staged(source_dir, destination_dir):
                # Another run finished first. Its tree is a copy of the same
                # source -- the digest in the name says so -- so it is the one
                # to use, and ours is redundant rather than wrong.
                return destination_dir
            # Debris: a copy that died partway, or a tree left by a run from
            # before the digest was part of the name. Nothing reads a
            # directory whose marker does not check out -- that is what
            # _is_staged is for -- so replacing it costs no reader anything,
            # and leaving it would block every later staging of this dataset.
            shutil.rmtree(destination_dir, ignore_errors=True)
            os.rename(partial_dir, destination_dir)
    finally:
        shutil.rmtree(partial_dir, ignore_errors=True)
    return destination_dir


def _is_staged(source_dir: Path, destination_dir: Path) -> bool:
    """Whether `destination_dir` already holds a finished copy of `source_dir`."""

    try:
        marker = json.loads((destination_dir / STAGING_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(marker, Mapping) or marker.get("source") != str(source_dir):
        return False
    return marker.get("files") == _file_count(destination_dir)


def _write_marker(source_dir: Path, staged_dir: Path) -> None:
    """Record what was copied, so a truncated tree is not mistaken for one."""

    marker = {"source": str(source_dir), "files": _file_count(staged_dir) + 1}
    # A path and a count, so allow_nan could not bite -- passed anyway rather
    # than added to test_runs_index_json_validity's exemption list, which is
    # narrower and worth keeping narrow.
    (staged_dir / STAGING_MARKER).write_text(json.dumps(marker, allow_nan=False), encoding="utf-8")


def _file_count(directory: Path) -> int:
    return sum(1 for path in directory.rglob("*") if path.is_file())


def resolve_staging_root(staging: Mapping[str, Any]) -> Path | None:
    """Resolve the staging destination, or None if it cannot be resolved.

    An unset value and one naming an environment variable the job did not
    export are the same answer: there is no usable scratch root, and the caller
    says so and stages nothing rather than writing to a literal "$FL_LOCAL_SCRATCH"
    directory.
    """

    raw_value = staging.get("local_root")
    if not raw_value:
        return None
    expanded = expand_path(raw_value)
    if has_unexpanded_env(str(expanded)):
        return None
    return expanded
