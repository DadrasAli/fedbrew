"""Lightweight validation for generated manifest datasets."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from fedbrew.core.validation import ValidationIssue

#: The ``client_test_source`` a generator writes when every split of a client
#: holds the same rows. An analytic objective is not estimated from samples --
#: `f_i` *is* the client -- so there is nothing to hold out, and a number read
#: off ``test_*`` or ``central_test_*`` on such data is a training number.
#: Preflight prints a note whenever a run reads a manifest declaring it.
IDENTICAL_TO_TRAIN = "identical_to_train"


def validate_manifest(
    manifest_path: str | Path,
    *,
    require_client_test: bool = False,
    require_global_test: bool = False,
) -> list[ValidationIssue]:
    """Validate manifest metadata and referenced shard files."""

    path = Path(manifest_path)
    issues: list[ValidationIssue] = []
    if not path.exists():
        _add(
            issues,
            "error",
            "manifest.missing",
            f"manifest.json does not exist: {path}",
            "Run fedbrew generate before validating or training on this dataset.",
        )
        return issues
    if not path.is_file():
        _add(issues, "error", "manifest.not_file", f"Manifest path is not a file: {path}")
        return issues

    manifest = _load_manifest(path, issues)
    if manifest is None:
        return issues

    require_client_test = require_client_test or (manifest.get("client_shard_format") == "split_v2")
    root = path.parent
    clients_path = root / str(manifest.get("clients_file", "clients.jsonl"))
    shards_dir = root / str(manifest.get("shards_dir", "shards"))

    if not clients_path.exists():
        _add(
            issues,
            "error",
            "manifest.clients_missing",
            f"clients.jsonl does not exist: {clients_path}",
        )
    if not shards_dir.exists() or not shards_dir.is_dir():
        _add(
            issues,
            "error",
            "manifest.shards_dir_missing",
            f"shards/ directory does not exist: {shards_dir}",
        )

    if require_global_test and not _non_empty(manifest.get("global_test")):
        _add(
            issues,
            "error",
            "manifest.global_test_required",
            "evaluation requests global_test, but the manifest does not declare it",
            "Regenerate the dataset with global test data or remove global_test "
            "from evaluation.test_sets.",
        )
    else:
        _validate_declared_file(root, manifest, "global_test", issues)
    if require_global_test and manifest.get("test_split") == "within_client_holdout":
        # The marker written by FEMNIST generation before its eval and test slices
        # were carved apart:
        # global_test.pt was built by concatenating every client's EVAL slice,
        # so central_test_* is the validation metric under another name. A run
        # against this data reports a number that save_best also selects on.
        _add(
            issues,
            "error",
            "manifest.global_test_is_the_eval_split",
            "this dataset's global_test is a copy of the per-client eval split, "
            "so central_test_* would be the validation metric under another name",
            "Regenerate with a client_splits.test_ratio > 0 (the generator now "
            "carves a third, never-selected-on slice per client), or drop "
            "central_test from evaluation and report val_* as validation.",
        )
    _validate_declared_file(root, manifest, "partition_stats_file", issues)
    _validate_declared_file(root, manifest, "client_stats_file", issues)

    input_claims = _input_claims(manifest, issues)

    clients = _load_clients(clients_path, issues)
    _validate_client_count(manifest, clients, issues)
    seen_client_ids: set[str] = set()
    for index, client in enumerate(clients, start=1):
        _validate_client_entry(
            root,
            index,
            client,
            seen_client_ids,
            issues,
            require_client_test=require_client_test,
            input_claims=input_claims,
        )

    return issues


class _InputClaims(NamedTuple):
    """What a manifest says its feature tensors are, for checking against them.

    Both keys were written and read by nobody: `femnist.py` declared
    `input_dtype: uint8` and `input_range: [0, 255]`, the four LLM generators
    declare `input_dtype: int64`, and `grep -rn "input_range\\|input_dtype"`
    over `fedbrew/` found no consumer. A claim nothing checks is decoration,
    and this one is load-bearing: `femnist_resnet18` normalises internally on
    the `[0, 255]` contract (`input_scale` stopped being a config key for that
    reason), so shards regenerated in `[0, 1]` under the same path would train
    at 1/255 of the intended scale with the manifest still saying `[0, 255]`.

    Checked here rather than in the task because this is where the claim is
    made, and because preflight already loads every shard -- `validate_manifest`
    runs from `core/validation.py` on every run.
    """

    dtype: str | None
    minimum: float | None
    maximum: float | None


def _input_claims(manifest: Mapping[str, Any], issues: list[ValidationIssue]) -> _InputClaims:
    """Read and shape-check the two declarations. Absent means unclaimed."""

    dtype = manifest.get("input_dtype")
    if dtype is not None and not _non_empty(dtype):
        _add(
            issues,
            "error",
            "manifest.input_dtype_invalid",
            f"input_dtype must be a dtype name such as 'uint8', got {dtype!r}",
        )
        dtype = None

    declared_range = manifest.get("input_range")
    if declared_range is None:
        return _InputClaims(dtype=str(dtype) if dtype else None, minimum=None, maximum=None)
    if (
        not isinstance(declared_range, (list, tuple))
        or len(declared_range) != 2
        or any(
            isinstance(bound, bool) or not isinstance(bound, (int, float))
            for bound in declared_range
        )
        or float(declared_range[0]) > float(declared_range[1])
    ):
        _add(
            issues,
            "error",
            "manifest.input_range_invalid",
            f"input_range must be [minimum, maximum], got {declared_range!r}",
        )
        return _InputClaims(dtype=str(dtype) if dtype else None, minimum=None, maximum=None)
    return _InputClaims(
        dtype=str(dtype) if dtype else None,
        minimum=float(declared_range[0]),
        maximum=float(declared_range[1]),
    )


def _validate_input_claims(
    split: Mapping[str, Any],
    label: str,
    path: Path,
    claims: _InputClaims,
    issues: list[ValidationIssue],
) -> None:
    """Hold the shard's feature tensor to what the manifest said it is."""

    features = split.get("x")
    dtype = getattr(features, "dtype", None)
    if dtype is None:
        return

    if claims.dtype is not None:
        actual = str(dtype).removeprefix("torch.")
        if actual != claims.dtype:
            _add(
                issues,
                "error",
                "manifest.input_dtype_mismatch",
                f"{label} features are {actual}, but the manifest declares "
                f"input_dtype {claims.dtype}: {path}",
                "Regenerate the dataset, or correct the manifest to describe the shards it names.",
            )

    if claims.minimum is None or claims.maximum is None or not len(features):
        return
    low = float(features.min())
    high = float(features.max())
    if low < claims.minimum or high > claims.maximum:
        _add(
            issues,
            "error",
            "manifest.input_range_violated",
            f"{label} features span [{low:g}, {high:g}], outside the manifest's "
            f"input_range [{claims.minimum:g}, {claims.maximum:g}]: {path}",
            "A model that normalises on the declared range would train on the "
            "wrong units. Regenerate the dataset, or correct the manifest.",
        )


def _load_manifest(path: Path, issues: list[ValidationIssue]) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        _add(issues, "error", "manifest.unreadable", f"Could not read {path}: {exc}")
        return None
    except json.JSONDecodeError as exc:
        _add(issues, "error", "manifest.invalid_json", f"Invalid manifest JSON: {exc}")
        return None
    if not isinstance(data, dict):
        _add(
            issues,
            "error",
            "manifest.not_object",
            "manifest.json must contain a JSON object",
        )
        return None
    return data


def _load_clients(
    path: Path,
    issues: list[ValidationIssue],
) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []

    clients: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    _add(
                        issues,
                        "error",
                        "manifest.client_json_invalid",
                        f"Invalid clients.jsonl line {line_number}: {exc}",
                    )
                    continue
                if not isinstance(record, Mapping):
                    _add(
                        issues,
                        "error",
                        "manifest.client_not_object",
                        f"clients.jsonl line {line_number} must be a JSON object",
                    )
                    continue
                clients.append(record)
    except OSError as exc:
        _add(
            issues,
            "error",
            "manifest.clients_unreadable",
            f"Could not read {path}: {exc}",
        )
    return clients


def _validate_client_count(
    manifest: Mapping[str, Any],
    clients: list[Mapping[str, Any]],
    issues: list[ValidationIssue],
) -> None:
    declared_count = manifest.get("num_clients")
    if declared_count is None:
        return
    if not isinstance(declared_count, int) or isinstance(declared_count, bool):
        _add(
            issues,
            "error",
            "manifest.num_clients_invalid",
            "manifest.num_clients must be an integer when declared",
        )
        return
    if clients and declared_count != len(clients):
        _add(
            issues,
            "warning",
            "manifest.num_clients_mismatch",
            f"manifest.num_clients={declared_count} but clients.jsonl has {len(clients)} entries",
        )


def _validate_client_entry(
    root: Path,
    index: int,
    client: Mapping[str, Any],
    seen_client_ids: set[str],
    issues: list[ValidationIssue],
    *,
    require_client_test: bool,
    input_claims: _InputClaims,
) -> None:
    client_id = client.get("client_id")
    shard = client.get("shard")
    num_examples = client.get("num_examples")
    label = f"clients.jsonl entry {index}"

    if not _non_empty(client_id):
        _add(
            issues,
            "error",
            "manifest.client_id_missing",
            f"{label} is missing client_id",
        )
    else:
        client_id_text = str(client_id)
        label = f"client {client_id_text}"
        if client_id_text in seen_client_ids:
            # An error, not a warning: ManifestFederatedDataset keys its
            # lookup by client_id, so the last row wins, while list_clients
            # returns every row. The loop then trains one shard twice a round
            # under two separate FedAvg weights and the other client is never
            # seen at all.
            _add(
                issues,
                "error",
                "manifest.client_id_duplicate",
                f"Duplicate client_id in clients.jsonl: {client_id_text}",
                "Two source groups produced the same id. Regenerate with "
                "anonymize_client_ids: true, or make the group values "
                "distinct.",
            )
        seen_client_ids.add(client_id_text)

    if not _non_empty(shard):
        _add(
            issues,
            "error",
            "manifest.client_shard_missing",
            f"{label} is missing shard",
        )
        return

    if not isinstance(num_examples, int) or isinstance(num_examples, bool):
        _add(
            issues,
            "error",
            "manifest.client_num_examples_invalid",
            f"{label} must declare integer num_examples",
        )
    elif num_examples < 0:
        _add(
            issues,
            "error",
            "manifest.client_num_examples_negative",
            f"{label} declares negative num_examples",
        )
    else:
        _validate_declared_total(client, label, issues)

    shard_path = root / str(shard)
    if not shard_path.exists():
        _add(
            issues,
            "error",
            "manifest.client_shard_file_missing",
            f"{label} shard file does not exist: {shard_path}",
        )
        return
    _validate_shard_structure(
        shard_path,
        label,
        issues,
        require_test=require_client_test,
        declared=client,
        input_claims=input_claims,
    )


def _validate_declared_file(
    root: Path,
    manifest: Mapping[str, Any],
    key: str,
    issues: list[ValidationIssue],
) -> None:
    value = manifest.get(key)
    if not value:
        return
    path = root / str(value)
    if not path.exists():
        _add(
            issues,
            "error",
            f"manifest.{key}_missing",
            f"Manifest declares {key}, but file does not exist: {path}",
        )
        return
    if key == "global_test":
        _validate_shard_structure(path, "global_test", issues)


#: Which declared count in clients.jsonl each split's rows have to match.
#: The counts are what the round tables report and what the client pool reads;
#: the shards are what training actually reads. num_examples is the sum of all
#: three -- generate.py used to write it as train + eval while femnist.py wrote
#: train + eval + test -- and _validate_declared_total below holds the two
#: definitions together.
_SPLIT_COUNT_KEYS = {
    "train": "num_train_examples",
    "eval": "num_eval_examples",
    "test": "num_test_examples",
}


def _validate_shard_structure(
    path: Path,
    label: str,
    issues: list[ValidationIssue],
    require_test: bool = False,
    declared: Mapping[str, Any] | None = None,
    input_claims: _InputClaims | None = None,
) -> None:
    claims = input_claims or _InputClaims(None, None, None)
    shard = _load_shard(path, label, issues)
    if shard is None:
        return

    has_split_keys = "train" in shard or "eval" in shard or "test" in shard
    if has_split_keys:
        train = shard.get("train")
        if not isinstance(train, Mapping):
            _add(
                issues,
                "error",
                "manifest.shard_train_missing",
                f"{label} split-aware shard must contain a train split: {path}",
            )
        else:
            _validate_xy(train, f"{label} train split", path, issues)
            _validate_declared_count(train, "train", label, path, declared, issues)
            _validate_input_claims(train, f"{label} train split", path, claims, issues)
        if "eval" in shard:
            eval_split = shard.get("eval")
            if not isinstance(eval_split, Mapping):
                _add(
                    issues,
                    "error",
                    "manifest.shard_eval_invalid",
                    f"{label} eval split must be a mapping: {path}",
                )
            else:
                _validate_xy(eval_split, f"{label} eval split", path, issues)
                _validate_declared_count(eval_split, "eval", label, path, declared, issues)
                _validate_input_claims(eval_split, f"{label} eval split", path, claims, issues)
        if "test" in shard:
            test_split = shard.get("test")
            if not isinstance(test_split, Mapping):
                _add(
                    issues,
                    "error",
                    "manifest.shard_test_invalid",
                    f"{label} test split must be a mapping: {path}",
                )
            else:
                _validate_xy(test_split, f"{label} test split", path, issues)
                _validate_declared_count(test_split, "test", label, path, declared, issues)
                _validate_input_claims(test_split, f"{label} test split", path, claims, issues)
        elif require_test:
            _add(
                issues,
                "error",
                "manifest.shard_test_missing",
                f"{label} must contain a persisted test split: {path}",
                "Regenerate the dataset so official test data is partitioned by client.",
            )

        return
    _validate_xy(shard, label, path, issues)
    _validate_declared_count(shard, None, label, path, declared, issues)
    _validate_input_claims(shard, label, path, claims, issues)


def _load_shard(
    path: Path,
    label: str,
    issues: list[ValidationIssue],
) -> Mapping[str, Any] | None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local environment.
        _add(
            issues,
            "error",
            "manifest.torch_unavailable",
            f"Could not import torch to inspect {label} shard: {exc}",
        )
        return None

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Failed to initialize NumPy:.*",
                category=UserWarning,
            )
            # PyTorch shard keys are only available after unpickling the checkpoint.
            try:
                shard = torch.load(
                    path,
                    map_location="cpu",
                    weights_only=False,
                    mmap=True,
                )
            except TypeError:
                shard = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        _add(
            issues,
            "error",
            "manifest.shard_unreadable",
            f"Could not inspect {label} shard {path}: {exc}",
        )
        return None
    if not isinstance(shard, Mapping):
        _add(
            issues,
            "error",
            "manifest.shard_not_mapping",
            f"{label} shard must contain a mapping: {path}",
        )
        return None
    return shard


def _validate_declared_count(
    split: Mapping[str, Any],
    split_name: str | None,
    label: str,
    path: Path,
    declared: Mapping[str, Any] | None,
    issues: list[ValidationIssue],
) -> None:
    """Check the rows on disk against the count clients.jsonl declares.

    A regeneration that dies part-way leaves new shards beside the previous
    run's clients.jsonl. Every structural check still passes -- the shards have
    x and y, the files all exist -- while FedAvg weights each client by a count
    from the old file and the round tables report it. Comparing the two is the
    only check that sees it, and the shard is already loaded here.
    """

    if declared is None:
        return
    key = _SPLIT_COUNT_KEYS[split_name] if split_name else "num_examples"
    expected = declared.get(key)
    if not isinstance(expected, int) or isinstance(expected, bool):
        return
    actual = _row_count(split)
    if actual is None or actual == expected:
        return
    _add(
        issues,
        "error",
        "manifest.client_count_mismatch",
        f"{label} declares {key}={expected} but its shard holds {actual} rows: {path}",
        "Regenerate the dataset: clients.jsonl and the shards are from different generator runs.",
    )


def _validate_declared_total(
    client: Mapping[str, Any],
    label: str,
    issues: list[ValidationIssue],
) -> None:
    """num_examples has to be the sum of the per-split counts.

    Two generators disagreed about it: generate.py wrote train + eval, leaving
    out the official test rows it had just partitioned to the client, while
    femnist.py wrote all three. A reader cannot tell which convention a record
    follows, so this pins one and fails the other. Checked against the declared
    per-split counts rather than the shard, because _validate_declared_count
    already ties those to the rows on disk -- so the total reaches the shard
    transitively without loading anything extra.
    """

    declared = [
        client.get(key) for key in ("num_train_examples", "num_eval_examples", "num_test_examples")
    ]
    present = [
        value for value in declared if isinstance(value, int) and not isinstance(value, bool)
    ]
    if not present:
        return
    total = client.get("num_examples")
    if not isinstance(total, int) or isinstance(total, bool):
        return
    if total == sum(present):
        return
    parts = ", ".join(
        f"{key}={value}"
        for key, value in zip(
            ("num_train_examples", "num_eval_examples", "num_test_examples"),
            declared,
            strict=True,
        )
        if isinstance(value, int) and not isinstance(value, bool)
    )
    _add(
        issues,
        "error",
        "manifest.client_num_examples_inconsistent",
        f"{label} declares num_examples={total}, but its per-split counts sum "
        f"to {sum(present)} ({parts})",
        "num_examples is every split. Regenerate the dataset.",
    )


def _row_count(split: Mapping[str, Any]) -> int | None:
    targets = split.get("y")
    try:
        return len(targets)
    except TypeError:
        return None


def _validate_xy(
    value: Mapping[str, Any],
    label: str,
    path: Path,
    issues: list[ValidationIssue],
) -> None:
    missing = [name for name in ("x", "y") if name not in value]
    if missing:
        _add(
            issues,
            "error",
            "manifest.shard_xy_missing",
            f"{label} is missing {', '.join(missing)} in shard: {path}",
        )


def _add(
    issues: list[ValidationIssue],
    severity: str,
    code: str,
    message: str,
    hint: str | None = None,
) -> None:
    issues.append(ValidationIssue(severity=severity, code=code, message=message, hint=hint))


def _non_empty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())
