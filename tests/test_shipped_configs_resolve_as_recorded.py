"""Every shipped run config resolves to what it did before the config was reshaped.

The config restructure (keys made optional where they do nothing, values
inferred from the data and the path, arm files folded onto family bases, the
numerics keys in a block of their own) claims to change no run. This holds it
to that: every shipped run config is loaded, and its resolved ``FullConfig``
and planned column list are compared with ``shipped_resolved_configs.json``,
recorded before the first of those changes (``tests/shipped_resolved_configs.py``
says how, and on what data).

Two differences are allowed, both declared here rather than taken on trust:

- a key the loader marks as inferred (``FullConfig.inferred``), whose value
  comes from the data or the path now, not the file -- and of those only the
  keys in ``INFERRED_AND_CHANGED`` may hold another value than recorded. That
  is ``experiment.name`` of the 26 configs that never stated one, which were
  named after their file (``fedavg``) and are named after their directory and
  file now (``femnist-fedavg``), as the 69 example arms that did state one
  already were. Every other inferred value, the 79 inferred output
  directories and the 60 strategies inferred from the update rule among
  them, must equal the one the file used to state;
- a key a step moved, under ``MOVED``: compared at its new place against the
  value recorded at its old one -- or, where the record has none because the
  config left it out, against the value its old reader took then
  (``MOVED_FROM_ABSENT``). The numerics block's six keys moved out of
  ``runtime`` and ``runtime.performance``; every shipped config states all
  six now, where some left ``deterministic_warn_only`` to its default. The
  reporting block took ``client_statistics`` whole, and
  ``reporting.fit_metrics`` is the recorded ``server.metrics``: the two fit
  filters became one list, and the list that decided which fit columns
  reached the round record was the server's, so the merged list keeps every
  shipped config's columns -- which the planned column list, compared as
  recorded, confirms. ``client.metrics`` is ``MERGED`` into it and not
  compared on its own.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

import pytest

from tests.shipped_resolved_configs import RECORD, resolved, shipped_run_configs

#: Resolved keys that moved, new place -> the place the record has them at.
MOVED: dict[str, str] = {
    "numerics.deterministic": "runtime.extra.deterministic",
    "numerics.deterministic_warn_only": "runtime.extra.deterministic_warn_only",
    "numerics.matmul_precision": "runtime.extra.performance.matmul_precision",
    "numerics.cudnn_benchmark": "runtime.extra.performance.cudnn_benchmark",
    "numerics.precision": "runtime.extra.performance.precision",
    "numerics.use_amp": "runtime.use_amp",
    # The new block's own record of unknown keys, empty in every config.
    "numerics.extra": "numerics.extra",
    "reporting.fit_metrics": "server.metrics",
    "reporting.per_client_csv": "client_statistics.per_client_csv",
    "reporting.statistics.std": "client_statistics.std",
    "reporting.statistics.variance": "client_statistics.variance",
    "reporting.statistics.min": "client_statistics.min",
    "reporting.statistics.max": "client_statistics.max",
    "reporting.statistics.worst_percent": "client_statistics.worst_percent",
    "reporting.statistics.extra": "client_statistics.extra",
    "reporting.extra": "reporting.extra",
}

#: Recorded keys merged into a moved one, not compared on their own.
MERGED: dict[str, str] = {"client.metrics": "reporting.fit_metrics"}

#: A moved key's old place -> what its reader took when a config left it out.
MOVED_FROM_ABSENT: dict[str, Any] = {
    "runtime.extra.deterministic": False,
    "runtime.extra.deterministic_warn_only": False,
    "runtime.extra.performance.matmul_precision": None,
    "runtime.extra.performance.cudnn_benchmark": None,
    "runtime.extra.performance.precision": "reference",
    "numerics.extra": {},
    "reporting.extra": {},
}

#: Inferred keys whose inferred value may differ from the recorded one.
INFERRED_AND_CHANGED = frozenset({"experiment.name"})


def _differences(before: dict[str, Any], now: dict[str, Any]) -> list[str]:
    inferred = {key[len("inferred.") :] for key in now if key.startswith("inferred.")}
    moved_back = {MOVED.get(key, key): value for key, value in now.items()}
    differences = []
    for key in sorted(set(before) | set(moved_back)):
        if key == "inferred" or key.startswith("inferred."):
            continue
        if key in inferred and key in INFERRED_AND_CHANGED:
            continue
        if key not in moved_back:
            if key in MERGED:
                continue
            differences.append(f"{key}: {before[key]!r} -> (gone)")
        elif key not in before:
            if key in MOVED_FROM_ABSENT and moved_back[key] == MOVED_FROM_ABSENT[key]:
                continue
            differences.append(f"{key}: (new) -> {moved_back[key]!r}")
        elif before[key] != moved_back[key]:
            differences.append(f"{key}: {before[key]!r} -> {moved_back[key]!r}")
    return differences


class EveryShippedConfigResolvesAsRecordedTest(unittest.TestCase):
    """Not fast: loading the example arms runs their extensions' reference solves."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.record = json.loads(RECORD.read_text(encoding="utf-8"))

    @pytest.mark.fast
    def test_the_record_covers_every_shipped_config(self) -> None:
        self.assertEqual(sorted(self.record), sorted(str(path) for path in shipped_run_configs()))
        self.assertEqual(len(self.record), 97)

    def test_each_resolves_to_its_record(self) -> None:
        for path in shipped_run_configs():
            with self.subTest(config=str(path)):
                before = self.record[str(path)]
                now = resolved(path)
                self.assertEqual(_differences(before["config"], now["config"]), [])
                self.assertEqual(now["planned"], before["planned"])

    @pytest.mark.fast
    def test_the_allowance_is_only_what_was_inferred(self) -> None:
        """A difference on a key nobody marked inferred is reported, not waved through."""

        before = {
            "experiment.name": "fedavg",
            "experiment.output_dir": "outputs/femnist/fedavg",
            "client.batch_size": 32,
        }
        now = {
            "experiment.name": "femnist-fedavg",
            "inferred.experiment.name": "the config path",
            "experiment.output_dir": "outputs/femnist/other",
            "inferred.experiment.output_dir": "the config path",
            "client.batch_size": 64,
        }
        self.assertEqual(
            _differences(before, now),
            [
                "client.batch_size: 32 -> 64",
                "experiment.output_dir: 'outputs/femnist/fedavg' -> 'outputs/femnist/other'",
            ],
        )


if __name__ == "__main__":
    unittest.main()
