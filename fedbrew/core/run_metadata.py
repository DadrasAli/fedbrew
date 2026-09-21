"""Run identity and reproducibility metadata helpers."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from fedbrew._build_info import source_commit
from fedbrew.core.config import FullConfig
from fedbrew.core.paths import resolve_data_path
from fedbrew.core.refusal import RunRefused


def generate_run_id(experiment_name: str, seed: int | None) -> str:
    """Generate a filesystem-safe run identifier."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = _sanitize_name(experiment_name)
    seed_text = "none" if seed is None else str(seed)
    return f"{timestamp}_{safe_name}_seed{seed_text}"


#: Where the running code lives, not where the process was launched from. A
#: sweep runs every arm from the submit directory, so cwd names the launcher,
#: never the checkout that produced the weights.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: git on a shared filesystem can block; a stalled provenance lookup must not
#: hold up a job that has already been allocated a GPU.
_GIT_TIMEOUT_SEC = 15.0


def capture_code_state(repo_root: Path | None = None) -> dict[str, Any]:
    """Return the commit the run is executing, and whether it was modified.

    runs_index.jsonl has always carried git_commit and git_dirty per row and
    nothing ever produced them, so every run recorded null for both
    inside a git checkout -- the one field that would tie a number back to the
    code that made it. Absence is now distinguishable from failure: git_error
    says why the lookup could not answer, rather than leaving a bare null that
    reads the same as never having looked.

    git_dirty covers tracked files only. Untracked files are ordinary here --
    outputs/, data/generated/ and the scratch a run writes are all ignored or
    untracked -- so counting them would mark every run dirty and the flag
    would stop meaning anything.

    Three answers are possible and commit_source says which one this is:

    - ``"git"`` -- a checkout. The live commit, and a dirty flag that means
      something because there is a tree to diff against.
    - ``"archive"`` -- a release exported by ``git archive``, which has no
      ``.git`` to ask and instead carries the commit stamped into
      ``fedbrew/_build_info.py``. git_dirty is null there and cannot be
      anything else: an extracted tree has nothing to compare itself with, and
      guessing False would assert something no one checked.
    - absent -- neither could answer, and git_error says why.

    A tarball run used to land in the third case unconditionally, which is the
    one place the trail is least recoverable afterwards.

    The lookup reads the checkout and writes nothing to it. ``git status``
    otherwise refreshes the stat cache in .git/index whenever a tracked file's
    mtime has moved, and takes index.lock to do so; a run holding that lock
    makes the user's own ``git add`` or ``git commit`` in the same checkout
    fail. GIT_OPTIONAL_LOCKS=0 keeps that refresh in memory: the same answer,
    nothing written. ``--no-optional-locks`` is the same switch as a flag, but
    git rejects a top-level flag it does not know, so a git older than the
    switch (2.15) would fail the lookup and lose the commit; an unknown
    variable it ignores. ``git diff-index --quiet HEAD`` never writes either,
    but it compares stat data rather than content and calls a touched,
    unchanged file dirty.
    """

    import subprocess

    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    state: dict[str, Any] = {"git_commit": None, "git_dirty": None}

    def _git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SEC,
            check=True,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
        return completed.stdout.strip()

    try:
        commit = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    except Exception as exc:
        # Never fatal. A tarball deployment, a container without git, or a
        # checkout with no commits are all legitimate ways to run this code,
        # and none of them is a reason to lose the run. Both halves are read
        # inside the one try so a half-answered lookup cannot be reported as a
        # checkout: either both git questions answered or neither did.
        stamped = source_commit()
        if stamped is None:
            state["git_error"] = f"{type(exc).__name__}: {exc}".strip()
            return state
        state["git_commit"] = stamped
        state["commit_source"] = "archive"
        return state

    state["git_commit"] = commit
    state["git_dirty"] = dirty
    state["commit_source"] = "git"
    return state


def _package_version(name: str) -> str | None:
    """Installed version of one package, or None when it is not present.

    Kept for the LLM trace, which pins the peft/transformers versions that
    change adapter behaviour. The general environment capture that also used
    this was removed -- nothing read it.
    """

    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python < 3.8 only
        return None
    try:
        return version(name)
    except PackageNotFoundError:
        return None


#: Manifest keys that decide what a number measured on this dataset *means*,
#: as opposed to how big it is. client_shard_format and client_test_source are
#: the pair that matter most: together they say whether a client's test slice
#: is disjoint from the eval slice the checkpoint was selected on, or the same
#: examples under a second name.
#:
#: A key absent from the manifest is recorded as null rather than omitted. That
#: distinction is the point: a manifest predating client_shard_format is a
#: split_v1 shard set, and "the manifest did not declare it" has to be
#: distinguishable from "nobody looked".
_DATASET_PROVENANCE_KEYS = (
    "dataset_name",
    "client_shard_format",
    "client_test_source",
    "client_splits",
    "partition_strategy",
    # The strategy's own knobs and the seed every random choice came from.
    # Without them run.json named the shape of the cut and none of its inputs,
    # so two runs on datasets differing only in alpha recorded the same
    # provenance -- and the generated data is not in the repository, so there
    # was nothing else to check against.
    "partition_parameters",
    "partition_key",
    "seed",
    "num_clients",
    "num_classes",
    "global_test",
    "source",
    "source_revision",
    "source_split",
    "source_num_clients",
    "min_samples_per_client",
    # What a generator says a run on its data is scored against -- a
    # reference optimum, the dials that produced it -- and which extensions,
    # if any, generated the data. Both optional, both copied verbatim, so
    # run.json says what the run was measured against without the manifest.
    "reference",
    "extensions",
)


def build_dataset_provenance(config: FullConfig) -> dict[str, Any] | None:
    """Record which shard set, in which format, produced this run's numbers.

    run.json recorded the manifest *path* and nothing about its content, so a
    FEMNIST accuracy could not be tied back to the split semantics it was
    measured under. The same path holds a different dataset before and after a
    regeneration, and the generated data is not in the repository: a number in
    a table could not be shown to have come from a three-way per-writer split
    rather than from shards whose "test" set was a copy of the eval slice the
    checkpoint was selected on. The LLM runs have had corpus_hash for exactly
    this reason; the classification runs had nothing.

    manifest_sha256 is the anchor -- it settles every question about the
    manifest, including the ones this list does not anticipate -- and the named
    keys are the ones a reader needs without having the file. Both are needed:
    a hash alone cannot be read, and a key list alone goes stale when a
    generator learns a new field.

    Never fatal, for the same reason capture_code_state is not: a provenance
    lookup that fails must not lose a run that has already been allocated a
    GPU. A failure is recorded as manifest_error, so it cannot be mistaken for
    a dataset that had nothing to declare.
    """

    if config.data.name != "manifest_dataset" or not config.data.path:
        return None
    # The LLM runs record their manifest under "llm", down to corpus_hash and
    # the tokenizer revisions. One provenance block per run, not two.
    if config.task.name == "causal_lm":
        return None

    import hashlib

    provenance: dict[str, Any] = {"manifest_path": str(config.data.path)}
    try:
        manifest_path = resolve_data_path(config.data.path)
        provenance["manifest_resolved_path"] = str(manifest_path)
        raw = manifest_path.read_bytes()
        provenance["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        manifest = json.loads(raw.decode("utf-8"))
        if not isinstance(manifest, Mapping):
            raise ValueError("manifest is not a JSON object")
    except Exception as exc:  # noqa: BLE001 - provenance must not lose a run
        provenance["manifest_error"] = f"{type(exc).__name__}: {exc}".strip()
        return provenance

    for key in _DATASET_PROVENANCE_KEYS:
        value = manifest.get(key)
        if isinstance(value, Mapping):
            value = dict(value)
        elif isinstance(value, list):
            value = list(value)
        provenance[key] = value
    return provenance


def build_extension_provenance(config: FullConfig) -> dict[str, Any] | None:
    """Record which version of each extension the run loaded its components from.

    ``experiment.extensions`` names the files; ``code_state`` covers the
    package's own commit and nothing outside it. So each entry is recorded
    with the path it resolved to, the SHA-256 of the file that was imported,
    and the names it registered -- the same record the loader made when
    ``load_config`` called it, keyed by the entry as the config wrote it.
    """

    entries = config.experiment.extensions
    if not entries:
        return None
    from fedbrew.core.extensions import loaded_extensions

    loaded = {record.entry: record for record in loaded_extensions()}
    provenance: dict[str, Any] = {}
    for entry in entries:
        record = loaded.get(entry)
        if record is None:
            # Cannot happen through load_config, which loads before it
            # validates; recorded rather than raised for the same reason a
            # dataset provenance failure is.
            provenance[entry] = {"error": "not loaded in this process"}
            continue
        provenance[entry] = {
            "resolved": record.resolved,
            "sha256": record.sha256,
            "registered": [list(pair) for pair in record.registered],
        }
    return provenance


def build_hf_causal_lm_trace(config: FullConfig) -> dict[str, Any] | None:
    """Build an exact prepared-model, tokenizer, and corpus provenance trace."""

    if config.model.name not in {"hf_causal_lm", "hf_causal_lm_lora"}:
        return None

    model_values = config.model.extra
    asset_manifest_path = model_values.get("asset_manifest")
    if not isinstance(asset_manifest_path, str) or not asset_manifest_path.strip():
        raise RunRefused(f"{config.model.name} requires model.asset_manifest")
    preparation_config = model_values.get("preparation_config")
    if preparation_config is not None and not isinstance(preparation_config, str):
        raise RunRefused("model.preparation_config must be a path string")
    if model_values.get("local_files_only", True) is not True:
        raise RunRefused(f"{config.model.name} requires model.local_files_only=true")
    if model_values.get("trust_remote_code", False) is not False:
        raise RunRefused(f"{config.model.name} requires model.trust_remote_code=false")

    from fedbrew.data.llm_assets.manifest import load_asset_manifest

    asset = load_asset_manifest(
        asset_manifest_path,
        preparation_config=preparation_config,
        require_assets=True,
    )
    data_manifest = _load_llm_data_manifest(config)
    corpus_hash = data_manifest.get(
        "corpus_hash",
        data_manifest.get("source_sha256"),
    )
    if not isinstance(corpus_hash, str) or not corpus_hash:
        raise RunRefused("generated Hugging Face data manifest must contain corpus_hash")

    trace: dict[str, Any] = {
        "model_identifier": asset.model_identifier,
        "tokenizer_identifier": asset.tokenizer_identifier,
        "requested_revision": asset.requested_revision,
        "resolved_revision": asset.resolved_revision,
        "tokenizer_requested_revision": data_manifest.get(
            "tokenizer_requested_revision",
            asset.requested_revision,
        ),
        "tokenizer_resolved_revision": data_manifest.get(
            "tokenizer_resolved_revision",
            asset.resolved_revision,
        ),
        "asset_manifest_path": str(asset_manifest_path),
        "tokenizer_asset_manifest_path": data_manifest.get(
            "tokenizer_asset_manifest",
            data_manifest.get("asset_manifest"),
        ),
        "prepared_cache_path": asset.cache_path,
        "model_type": asset.model_type,
        "preparation_timestamp": asset.preparation_timestamp,
        "corpus_hash": corpus_hash,
        "tokenizer_vocabulary_size": data_manifest.get(
            "tokenizer_vocabulary_size",
            data_manifest.get("vocab_size", asset.vocabulary_size),
        ),
        "sequence_length": data_manifest.get(
            "sequence_length",
            model_values.get("sequence_length"),
        ),
        "stride": data_manifest.get("stride"),
        "dataset_manifest_path": config.data.path,
        "dataset_task": data_manifest.get("task"),
        "dataset_identifier": data_manifest.get("source_dataset_identifier"),
        "dataset_requested_revision": data_manifest.get("source_dataset_requested_revision"),
        "dataset_resolved_revision": data_manifest.get(
            "source_dataset_resolved_revision",
            data_manifest.get("source_dataset_revision"),
        ),
        "dataset_asset_manifest_path": data_manifest.get("dataset_asset_manifest"),
        "source_file_hashes": data_manifest.get("source_file_hashes"),
        "ignore_index": data_manifest.get("ignore_index"),
        "filtering_rules": data_manifest.get("filtering_rules"),
        "filtering_summary": data_manifest.get("filtering_summary"),
        "tree_split_policy": data_manifest.get("tree_split_policy"),
        "tree_split_key": data_manifest.get("tree_split_key"),
        "split_ratios": data_manifest.get("split_ratios"),
        "tree_counts": data_manifest.get("tree_counts"),
        "client_selection": data_manifest.get("client_selection"),
        "selected_client_ids": data_manifest.get("selected_client_ids"),
        "selected_client_statistics": data_manifest.get("selected_client_statistics"),
        "pilot_caps": data_manifest.get("pilot_caps"),
        "peft_version": _package_version("peft"),
        "transformers_version": _package_version("transformers"),
        "offline": True,
        "local_files_only": True,
        "trust_remote_code": False,
        "offline_environment": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
    }
    if config.model.name == "hf_causal_lm_lora":
        # Asked of the builder, not re-derived. This block carried its own
        # copies of 8, 16, 0.05 and "none", so a default changed in one file
        # left run.json describing an adapter that never ran -- and it
        # recorded target_modules unnormalised, where the builder strips and
        # de-duplicates, so a stray space wrote one list and trained on
        # another. P10-F17.
        from fedbrew.models.hf_causal_lm_lora import (
            lora_adapter_name,
            lora_config_from_model_values,
        )

        trace["model_state_scope"] = "adapter"
        trace["adapter_name"] = lora_adapter_name(model_values)
        trace["lora_config"] = lora_config_from_model_values(model_values)
    else:
        trace["model_state_scope"] = "full"
    return trace


def _load_llm_data_manifest(config: FullConfig) -> Mapping[str, Any]:
    if config.data.name != "manifest_dataset" or not config.data.path:
        raise RunRefused("hf_causal_lm requires a generated manifest_dataset")
    manifest_path = resolve_data_path(config.data.path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunRefused(f"could not read generated LLM data manifest: {manifest_path}") from error
    if not isinstance(payload, Mapping):
        raise RunRefused("generated LLM data manifest must be a JSON object")
    return payload


def _sanitize_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    safe = safe.strip("._-")
    return safe or "experiment"
