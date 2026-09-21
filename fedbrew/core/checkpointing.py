"""Checkpoint management utilities for experiment state."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch

from fedbrew.core.refusal import RunRefused

_CHECKPOINT_NAME_PATTERN = re.compile(r"round_(\d+)\.pt$")

#: The default metric best.pt is selected on. It is a validation metric, and
#: validate_selection_metric below refuses anything that is not, because
#: selecting the reported checkpoint by its test score is model selection on
#: the test set: the reported number is then optimistically biased, and the
#: bias grows with the number of rounds you select over.
DEFAULT_SELECTION_METRIC = "val_accuracy_sample_weighted_avg"

#: Direction is a property of the metric, not a free choice -- loss goes down,
#: accuracy goes up -- so it is derived rather than configured. A name that
#: matches neither (or both) is a mistake worth failing on.
_MINIMIZED_METRIC_WORDS = frozenset({"loss", "error", "perplexity"})
_MAXIMIZED_METRIC_WORDS = frozenset({"accuracy", "acc", "f1", "auc"})


def selection_mode_for_metric(metric_name: str) -> str:
    """Return "min" or "max" for a metric, from its name alone."""

    words = set(str(metric_name).split("_"))
    minimized = bool(words & _MINIMIZED_METRIC_WORDS)
    maximized = bool(words & _MAXIMIZED_METRIC_WORDS)
    if minimized == maximized:
        raise RunRefused(
            f"cannot tell which direction is better for metric {metric_name!r}; "
            "name it with one of "
            + ", ".join(sorted(_MINIMIZED_METRIC_WORDS | _MAXIMIZED_METRIC_WORDS))
        )
    return "min" if minimized else "max"


#: Prefixes a selection metric may carry. "personal_" is the personalized
#: evaluation pass (evaluation.model_scope), which is still a validation
#: metric measured on data the model was never selected on -- the guard below
#: is about val-vs-test, not about which model produced the number.
SELECTION_METRIC_PREFIXES = ("val_", "personal_val_")


def validate_selection_metric(metric_name: str) -> str:
    """Reject checkpoint-selection metrics that are not validation metrics."""

    name = str(metric_name)
    if not name.startswith(SELECTION_METRIC_PREFIXES):
        raise RunRefused(
            f"checkpointing.best_metric must be a validation metric, got "
            f"{name!r}. Selecting best.pt on a test metric makes the reported "
            "test score optimistically biased. Always available: "
            "val_accuracy_sample_weighted_avg, val_accuracy_avg, "
            "val_loss_sample_weighted_avg, val_loss_avg. Also available when "
            "the matching client_statistics toggle is on: val_accuracy_min, "
            "val_accuracy_max, val_accuracy_std, val_accuracy_variance, "
            "val_accuracy_worst10 (the number follows worst_percent). Each "
            "name also takes a personal_ prefix when evaluation.model_scope "
            "produces the personalized pass."
        )
    # Fails now, at config load, rather than after the first round.
    selection_mode_for_metric(name)
    return name


def checkpoint_config_with_defaults(
    checkpoint_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a normalized checkpoint policy.

    A missing config keeps the historical behavior: save every numbered round
    checkpoint and do not create latest/best aliases.
    """

    if checkpoint_config is None:
        return {
            "enabled": True,
            "interval": 1,
            "save_last": False,
            "save_best": False,
            "best_metric": DEFAULT_SELECTION_METRIC,
            "best_mode": selection_mode_for_metric(DEFAULT_SELECTION_METRIC),
            "keep_last": None,
            "save_every_round": True,
        }

    best_metric = str(checkpoint_config.get("best_metric", DEFAULT_SELECTION_METRIC))
    return {
        "enabled": bool(checkpoint_config.get("enabled", True)),
        "interval": int(checkpoint_config.get("interval", 1)),
        "save_last": bool(checkpoint_config.get("save_last", True)),
        "save_best": bool(checkpoint_config.get("save_best", True)),
        "best_metric": best_metric,
        # Derived, never read from the config. See selection_mode_for_metric.
        "best_mode": selection_mode_for_metric(best_metric),
        "keep_last": checkpoint_config.get("keep_last", 3),
        "save_every_round": bool(checkpoint_config.get("save_every_round", False)),
    }


def should_save_checkpoint(
    round_id: int,
    checkpoint_config: Mapping[str, Any] | None,
) -> bool:
    """Return whether a numbered round checkpoint should be written."""

    config = checkpoint_config_with_defaults(checkpoint_config)
    if not bool(config["enabled"]):
        return False
    if bool(config["save_every_round"]):
        return True
    interval = int(config["interval"])
    return round_id % interval == 0


def get_checkpoint_path(output_dir: str | Path, round_id: int) -> Path:
    """Return the numbered checkpoint path for a round."""

    return Path(output_dir) / "checkpoints" / f"round_{round_id:03d}.pt"


def get_latest_checkpoint_path(output_dir: str | Path) -> Path:
    """Return the latest checkpoint alias path."""

    return Path(output_dir) / "checkpoints" / "latest.pt"


def _stage(payload: Mapping[str, Any], path: Path) -> Path:
    """Write `payload` beside `path` as "<name>.tmp", flushed and fsynced; return it.

    Nothing at `path` changes. A write that fails removes its temp file; one
    killed outright leaves it for clear_stale_temp_files to sweep at the next
    start. The ".tmp" name matches no checkpoint glob -- round_*.pt here, *.pt
    in the artifact record -- so a staged file is never read as a checkpoint.
    """

    temp_path = path.with_name(path.name + ".tmp")
    try:
        with temp_path.open("wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def _save_atomically(payload: Mapping[str, Any], path: Path) -> None:
    """torch.save `payload` to `path` so that no kill can leave `path` damaged. POST-F23.

    torch.save onto the path itself truncates it first, so a process killed
    inside the write leaves a short or empty file where the previous checkpoint
    was. latest.pt is rewritten every round under save_last and is the one file
    --resume-latest reads, so a kill that lands in that window -- a time limit's
    SIGTERM, a preemption -- destroys the only thing the resume needed: on
    2026-09-20 a SCAFFOLD point stopped at round 7235 of 10000 left a zero-byte
    latest.pt and had to restart from round 1.

    The payload is staged to a sibling "<name>.tmp" and then renamed over the
    target with os.replace, which is atomic within one filesystem: the path
    holds the previous complete checkpoint or the new complete one, never part
    of either.
    """

    os.replace(_stage(payload, path), path)


class StagedCheckpoints:
    """A round's checkpoints, written in full but not yet visible. POST-F24.

    A resume replays round_metrics.csv up to the checkpoint's round, so a
    checkpoint may never be visible for a round the CSV does not yet hold. The
    loop used to write a round's checkpoints before that round's CSV row and
    run.json; late in a long run the CSV rewrite is most of each round, so a
    time limit's kill landed between the two almost every time -- ten of ten
    SCAFFOLD points on 2026-09-20 -- and left every checkpoint a round ahead of
    a history it could not be resumed onto.

    The loop now stages the round's checkpoints where it always wrote them --
    so checkpoint_sec still times the write -- and commits them after the CSVs
    and run.json. A kill before the commit leaves the previous round's
    checkpoint beside a history that reaches this round, which a resume takes:
    rows after the checkpoint's round are dropped and recomputed.
    """

    def __init__(self) -> None:
        self._pending: list[tuple[Path, Path]] = []

    def stage(self, payload: Mapping[str, Any], path: Path) -> None:
        self._pending.append((_stage(payload, path), path))

    def commit(self) -> None:
        """Rename each staged file over its target, in the order staged."""

        while self._pending:
            temp_path, path = self._pending.pop(0)
            os.replace(temp_path, path)


def _write(payload: Mapping[str, Any], path: Path, staged: StagedCheckpoints | None) -> None:
    if staged is None:
        _save_atomically(payload, path)
    else:
        staged.stage(payload, path)


def save_checkpoint(
    state: dict[str, Any],
    output_dir: str | Path,
    round_id: int,
    staged: StagedCheckpoints | None = None,
) -> Path:
    """Save a round checkpoint under the experiment output directory.

    With `staged`, the file is written but only becomes visible at its commit.
    """

    checkpoint_path = get_checkpoint_path(output_dir, round_id)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["round_id"] = round_id
    _write(payload, checkpoint_path, staged)
    return checkpoint_path


def save_latest_checkpoint(
    payload: dict[str, Any],
    output_dir: str | Path,
    staged: StagedCheckpoints | None = None,
) -> Path:
    """Save or update the latest checkpoint alias; with `staged`, at its commit."""

    checkpoint_path = get_latest_checkpoint_path(output_dir)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    _write(payload, checkpoint_path, staged)
    return checkpoint_path


def save_best_checkpoint(
    payload: dict[str, Any],
    output_dir: str | Path,
    metric_name: str,
    metric_value: float,
    staged: StagedCheckpoints | None = None,
) -> Path:
    """Save or update the best checkpoint alias; with `staged`, at its commit."""

    checkpoint_path = Path(output_dir) / "checkpoints" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_payload = dict(payload)
    best_payload["best_metric"] = metric_name
    best_payload["best_metric_value"] = metric_value
    _write(best_payload, checkpoint_path, staged)
    return checkpoint_path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint dictionary from disk."""

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RunRefused("checkpoint must contain a dictionary")
    return cast(dict[str, Any], checkpoint)


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Return latest.pt when present, otherwise the highest numbered checkpoint."""

    checkpoint_dir = Path(output_dir) / "checkpoints"
    if not checkpoint_dir.is_dir():
        return None

    latest_path = get_latest_checkpoint_path(output_dir)
    if latest_path.is_file():
        return latest_path

    checkpoints: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("round_*.pt"):
        match = _CHECKPOINT_NAME_PATTERN.match(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))

    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def prune_old_checkpoints(output_dir: str | Path, keep_last: int | None) -> list[Path]:
    """Remove old numbered checkpoints while preserving latest.pt and best.pt."""

    if keep_last is None:
        return []
    if keep_last < 0:
        raise RunRefused("keep_last must be >= 0")

    checkpoint_dir = Path(output_dir) / "checkpoints"
    if not checkpoint_dir.is_dir():
        return []

    checkpoints: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("round_*.pt"):
        match = _CHECKPOINT_NAME_PATTERN.match(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))

    checkpoints.sort(key=lambda item: item[0])
    to_remove = checkpoints[: max(0, len(checkpoints) - keep_last)]
    removed: list[Path] = []
    for _, path in to_remove:
        path.unlink(missing_ok=True)
        removed.append(path)
    return removed


def refuse_a_reconfigured_resume(
    owner: str,
    state: Mapping[str, Any],
    configured: Mapping[str, Any],
) -> None:
    """Refuse a checkpoint whose hyperparameters disagree with this config.

    Every ``load_state`` on a client or a server writes the checkpoint's value
    over the one the config just built -- ``self.beta1 = float(state.get(
    "beta1", self.beta1))`` and a dozen more like it. That is right for state
    a run *learns* (SCAFFOLD's control variate, FedOpt's moments) and wrong for
    a value the config *sets*: the run continues at the old
    hyperparameter while `run.json` records the new one, so the run's own
    record of itself is false.

    The packed SLURM scripts pass ``--resume-latest`` whenever a ``latest.pt``
    exists under the arm's output dir, so editing a config and resubmitting
    into the same directory is one command, and nothing about the result says
    which value was used.

    Refused rather than resolved in either direction, because neither
    direction is supported: chapter 09 §5 and chapter 10 both say not to change
    config between a run and its resume. Picking the config would silently
    change an experiment mid-run; picking the checkpoint is what this fixes.
    It is the same call `runner._refuse_a_foreign_seed` already makes for
    `experiment.seed`, whose comment gives the reason -- a directory whose
    contents cannot be attributed to either run.

    Args:
        owner: What is being restored, as the message should name it --
            ``"fedopt server"``, ``"scaffold client"``.
        state: The checkpointed state mapping.
        configured: The values this run was configured with, keyed as `state`
            keys them. A key absent from `state` is not compared, so a
            checkpoint written before a setting existed still loads.

    Raises:
        RunRefused: If any key present in both disagrees.
    """

    changed = [
        (key, state[key], value)
        for key, value in configured.items()
        if key in state and state[key] != value
    ]
    if not changed:
        return

    lines = "\n".join(
        f"  {key}: checkpoint {checkpointed!r}, config {value!r}"
        for key, checkpointed, value in sorted(changed)
    )
    raise RunRefused(
        f"cannot resume this {owner}: the checkpoint and the config disagree "
        f"about {len(changed)} "
        f"{'hyperparameter' if len(changed) == 1 else 'hyperparameters'}.\n{lines}\n"
        "A run resumed at different settings is a different experiment, and "
        "nothing downstream would say so: run.json records the config either "
        "way. Two ways forward.\n"
        "  Continue THIS experiment: restore the config to the checkpoint's "
        "values and resume as before. To continue it in a new directory, copy "
        "the whole run directory first and resume inside the copy -- a resume "
        "replays round_metrics.csv, so pointing --resume-from at a checkpoint "
        "beside an empty output_dir restarts from round 1 rather than "
        "continuing.\n"
        "  Run the NEW settings: start from round 1 with the new config in "
        "its own output_dir. There is no warm start, and that is deliberate: "
        "a curve whose hyperparameters change partway is not a measurement of "
        "either setting."
    )


#: Client-state keys a checkpoint written before a rename carries, mapped to the
#: key that replaced them. `refuse_a_reconfigured_resume` compares only keys
#: the checkpoint has, so a renamed key would otherwise go uncompared: the
#: checkpoint lacks the new name, the old name is compared against nothing, and
#: a resume across an edited value is taken without a word.
RENAMED_CLIENT_STATE_KEYS: dict[str, str] = {"local_epochs": "local_iterations"}


def refuse_a_pre_rename_client_state(owner: str, state: Mapping[str, Any]) -> None:
    """Refuse a client state that carries a key from before a rename.

    Raises:
        RunRefused: If ``state`` holds any key in `RENAMED_CLIENT_STATE_KEYS`.
    """

    stale = sorted(key for key in RENAMED_CLIENT_STATE_KEYS if key in state)
    if not stale:
        return
    renames = ", ".join(f"{key} -> {RENAMED_CLIENT_STATE_KEYS[key]}" for key in stale)
    raise RunRefused(
        f"cannot resume this {owner}: its checkpoint predates the rename "
        f"{renames}. The setting is the same one under its new name, but a "
        "checkpoint from before the rename is refused rather than loaded: "
        "nothing would compare its value with this config's, so a resume "
        "across an edited value would be taken silently. Two ways forward.\n"
        "  Finish THIS run with the code that wrote the checkpoint, from before "
        "the rename.\n"
        "  Or start from round 1 with this code, in its own output_dir."
    )


def refuse_a_pre_rename_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Refuse a whole checkpoint if any client state in it predates a rename.

    Checked once, up front, before anything is restored. `load_state` checks
    each client too, but the lazy client pool calls it only when a client is
    first built, which can be rounds into the resumed run.
    """

    client_states = checkpoint.get("client_states")
    if not isinstance(client_states, Mapping):
        return
    for client_id, state in client_states.items():
        if isinstance(state, Mapping):
            refuse_a_pre_rename_client_state(f"checkpoint (client {client_id})", state)


def get_checkpoint_round_id(checkpoint: Mapping[str, Any]) -> int:
    """Read and validate a checkpoint round id."""

    round_id = checkpoint.get("round_id")
    if not isinstance(round_id, int) or isinstance(round_id, bool) or round_id <= 0:
        raise RunRefused("checkpoint round_id must be a positive int")
    return round_id
