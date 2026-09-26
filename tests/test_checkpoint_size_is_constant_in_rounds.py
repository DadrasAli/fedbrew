"""A checkpoint's size does not grow with the round count.

Every server kept a list of each round's aggregated metrics and wrote the
whole list into its save_state(), so latest.pt -- rewritten and fsynced every
round under save_last -- carried the run's full metric history. One save
cost about 1.6 + 0.0104 N ms at history length N, 106 ms and 1.6 MB at
N = 10,000 (perf/report.txt). Nothing restored from the list ever read it:
the run's history is round_metrics.csv, which a resume already replays. The
list is gone. What is pinned here:

- latest.pt is the same number of bytes after 3 rounds and after 30, for every
  server family (FedAvg, SCAFFOLD, FedOpt, FedLALR);
- a checkpoint written before the change, which still carries the history
  under server_state["round_metrics"], resumes to the uninterrupted run.
"""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch
import yaml

from fedbrew.core.runner import run
from tests.test_reproducibility import _config, _digest

_SAVE_LAST = {
    "enabled": True,
    "save_last": True,
    "save_every_round": False,
    "interval": 100000,
    "keep_last": 0,
}

#: The fixture's SGD options, which the SCAFFOLD and FedLALR rules refuse.
_SGD_ONLY = dict.fromkeys(
    ("momentum", "weight_decay", "nesterov", "learning_rate_schedule", "min_learning_rate")
)

#: The server and client blocks that turn the fixture into each family; a
#: client key set to None is removed.
_FAMILIES: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "fedavg": ({}, {}),
    "scaffold": ({"strategy": "scaffold"}, {"update_rule": "scaffold", **_SGD_ONLY}),
    "fedadam": (
        {
            "strategy": "fedadam",
            "server_learning_rate": 0.01,
            "beta1": 0.9,
            "beta2": 0.99,
            "tau": 1.0e-3,
        },
        {},
    ),
    "fedlalr": ({"strategy": "fedlalr"}, {"update_rule": "fedlalr", **_SGD_ONLY}),
}


def _write(root: Path, name: str, family: str, rounds: int) -> Path:
    config = _config(root / name, rounds=rounds)
    server, client = _FAMILIES[family]
    config["server"].update(server)
    config["client"].update(client)
    config["client"] = {key: value for key, value in config["client"].items() if value is not None}
    config["runtime"]["checkpointing"] = dict(_SAVE_LAST)
    path = root / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _run(path: Path, *, resume_latest: bool = False) -> Path:
    args = argparse.Namespace(resume_latest=resume_latest) if resume_latest else None
    run(path, args=args)
    return Path(yaml.safe_load(path.read_text())["experiment"]["output_dir"])


class _TempRoot(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)


class LatestCheckpointSizeTest(_TempRoot):
    def test_three_rounds_and_thirty_write_the_same_bytes(self) -> None:
        for family in _FAMILIES:
            with self.subTest(family=family):
                sizes = [
                    (
                        _run(_write(self.root, f"{family}-{rounds}", family, rounds))
                        / "checkpoints"
                        / "latest.pt"
                    )
                    .stat()
                    .st_size
                    for rounds in (3, 30)
                ]
                self.assertEqual(sizes[0], sizes[1])

    def test_no_server_state_carries_a_history(self) -> None:
        for family in _FAMILIES:
            with self.subTest(family=family):
                output_dir = _run(_write(self.root, f"{family}-keys", family, 3))
                checkpoint = torch.load(
                    output_dir / "checkpoints" / "latest.pt", weights_only=False
                )
                self.assertNotIn("round_metrics", checkpoint["server_state"])


class AnOldCheckpointStillResumesTest(_TempRoot):
    def test_a_carried_history_is_ignored(self) -> None:
        full = _run(_write(self.root, "full", "fedavg", 4))
        partial = _run(_write(self.root, "part", "fedavg", 2))
        latest = partial / "checkpoints" / "latest.pt"
        checkpoint = torch.load(latest, weights_only=False)
        checkpoint["server_state"]["round_metrics"] = [{"fit_loss": 9.0}, {"fit_loss": 8.0}]
        torch.save(checkpoint, latest)

        resumed = _run(_write(self.root, "part", "fedavg", 4), resume_latest=True)
        self.assertEqual(_digest(full), _digest(resumed))


if __name__ == "__main__":
    unittest.main()
