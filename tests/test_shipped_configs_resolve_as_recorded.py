"""Every shipped run config resolves to what it did before the config was reshaped.

The config restructure (keys made optional where they do nothing, values
inferred from the data and the path, arm files folded onto family bases, the
numerics keys in a block of their own) claims to change no run. This holds it
to that: every shipped run config is loaded, and its resolved ``FullConfig``
and planned column list are compared with ``shipped_resolved_configs.json``,
recorded before the first of those changes (``tests/shipped_resolved_configs.py``
says how, and on what data).

Two differences are allowed, both declared here rather than taken on trust:

- a key the loader marks as inferred (``FullConfig.inferred``). Its value
  comes from the data or the path now, not the file; the one that differs is
  ``experiment.name`` of the 26 configs that never stated one, which were
  named after their file (``fedavg``) and are named after their directory and
  file now (``femnist-fedavg``), as the 71 that did state one already were;
- a key a step moved, under ``MOVED``: compared at its new place against the
  value recorded at its old one.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

import pytest

from tests.shipped_resolved_configs import RECORD, resolved, shipped_run_configs

#: Resolved keys that moved, new place -> the place the record has them at.
MOVED: dict[str, str] = {}


def _differences(before: dict[str, Any], now: dict[str, Any]) -> list[str]:
    inferred = {key[len("inferred.") :] for key in now if key.startswith("inferred.")}
    moved_back = {MOVED.get(key, key): value for key, value in now.items()}
    differences = []
    for key in sorted(set(before) | set(moved_back)):
        if key == "inferred" or key.startswith("inferred.") or key in inferred:
            continue
        if key not in moved_back:
            differences.append(f"{key}: {before[key]!r} -> (gone)")
        elif key not in before:
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

        before = {"experiment.name": "fedavg", "client.batch_size": 32}
        now = {
            "experiment.name": "femnist-fedavg",
            "inferred.experiment.name": "the config path",
            "client.batch_size": 64,
        }
        self.assertEqual(_differences(before, now), ["client.batch_size: 32 -> 64"])


if __name__ == "__main__":
    unittest.main()
