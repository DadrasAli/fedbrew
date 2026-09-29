"""The client splits a run evaluates, against the splits its data carries.

``evaluation.val`` and ``evaluation.test`` have defaults (every 5 and every 10
rounds) that do not look at the data. A generated dataset need not carry
either split -- no causal-LM generator writes a client test split, and the
``synthetic_classification`` generator writes no val split -- and a run that
evaluated one it did not have trained its first round and then stopped on
"no client reported a non-empty val split", after a preflight that had said
the run was ready.

So the split schedules are resolved against the data when the config is
loaded, before anything trains:

- a split the data does not carry and the config does not name takes
  ``every: never``, and ``EvaluationConfig.splits_without_data`` records it,
  which the plan header prints as "not evaluated (no <split> data)";
- a split the data does not carry and the config asks for is refused, naming
  the split and the data.

What a dataset carries is read from its manifest's client records, without
loading a shard: each record's ``num_train_examples``, ``num_eval_examples``
and ``num_test_examples``, the three counts ``ManifestDataset`` serves. A
split is judged absent only when every client's count of it is known and 0.
A count a record does not state is known when the record states the total
and the other two -- the causal-LM generators write no test count, and their
totals are train plus eval -- and unknown otherwise; one unknown count leaves
the split as configured. The in-memory synthetic backend carries every split;
an extension backend's splits are not known here and are left as configured.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fedbrew.core.refusal import RunRefused

#: The evaluation split name, and the key its client count is written under
#: (val is the manifest's "eval" split).
SPLIT_COUNT_KEYS = {
    "train": "num_train_examples",
    "val": "num_eval_examples",
    "test": "num_test_examples",
}


def resolve_evaluated_splits(config: Any, stated: Mapping[str, Any]) -> None:
    """Set a data-less split the config does not name to never; refuse one it does.

    ``stated`` is the config file's ``evaluation`` mapping as written, which
    says whether a split's ``every`` was the config's or the default's.
    Changes ``config.evaluation`` in place.
    """

    absent = splits_without_data(config)
    if not absent:
        return
    evaluation = config.evaluation
    for split in absent:
        schedule = getattr(evaluation, split)
        written = stated.get(split)
        if isinstance(written, Mapping) and "every" in written and written["every"] != "never":
            raise RunRefused(
                f"evaluation.{split}.every is {written['every']!r}, but the dataset "
                f"{config.data.path} carries no client {split} split (every client's "
                f"{SPLIT_COUNT_KEYS[split]} is 0), so no {split} metric can be measured. "
                f"Set evaluation.{split}.every: never, or regenerate the data with a "
                f"{split} split."
            )
        schedule.every = "never"
    evaluation.splits_without_data = tuple(absent)


def splits_without_data(config: Any) -> list[str]:
    """The client splits the run's data is known to carry no rows of, in split order."""

    if config.data.name != "manifest_dataset" or not config.data.path:
        return []
    counts = _client_split_counts(config.data.path)
    if counts is None:
        return []
    return [split for split in ("val", "test") if all(client.get(split) == 0 for client in counts)]


def _client_split_counts(path: str) -> list[dict[str, int | None]] | None:
    """Each client's count per split, None where unknown; None if the records cannot be read.

    Nothing here refuses: an unreadable manifest is the dataset's to report,
    where it is built, and preflight's data check.
    """

    from fedbrew.core.inferred import read_manifest
    from fedbrew.core.paths import resolve_data_path

    manifest = read_manifest(path)
    if manifest is None:
        return None
    try:
        clients_path = Path(resolve_data_path(path)).parent / str(manifest["clients_file"])
        records = [
            json.loads(line)
            for line in clients_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not records:
        return None
    return [_counts(record) for record in records]


def _counts(record: Mapping[str, Any]) -> dict[str, int | None]:
    stated = {
        split: record[key]
        for split, key in SPLIT_COUNT_KEYS.items()
        if isinstance(record.get(key), int) and not isinstance(record.get(key), bool)
    }
    total = record.get("num_examples")
    missing = [split for split in SPLIT_COUNT_KEYS if split not in stated]
    if len(missing) == 1 and isinstance(total, int) and not isinstance(total, bool):
        remainder = total - sum(stated.values())
        if remainder >= 0:
            stated[missing[0]] = remainder
    return {split: stated.get(split) for split in SPLIT_COUNT_KEYS}
