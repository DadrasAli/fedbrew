"""A write killed before its rename leaves the previous file readable. POST-F23.

save_latest_checkpoint called torch.save on latest.pt itself, which truncates
the file before writing it. On 2026-09-20 a SLURM time limit's SIGTERM reached
a SCAFFOLD point inside that write: latest.pt was left at zero bytes, and a run
stopped at round 7235 of 10000 could only restart from round 1. save_checkpoint
(round_NNN.pt), save_best_checkpoint (best.pt) and save_run_json (run.json,
which a resume reads for its attempt count and the time already spent) wrote
the same way.

Each now writes a sibling ".tmp" and os.replace()s it over the target. Pinned
three ways:

- in-process, the rename is made to fail -- the moment after the new bytes are
  on disk and before they are visible -- and the old file must still load with
  its old contents; a write that fails part-way must also clean up its temp;
- in a child process, the write of latest.pt is stopped halfway and the process
  SIGKILLed, which is the 2026-09-20 failure itself;
- a temp file a kill leaves behind is swept at the next start and is never
  mistaken for a checkpoint in the meantime.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
from pathlib import Path
from unittest import mock

import pytest
import torch

from fedbrew.core import artifacts, checkpointing
from fedbrew.core.artifacts import clear_stale_temp_files, save_run_json
from fedbrew.core.checkpointing import (
    find_latest_checkpoint,
    get_checkpoint_path,
    get_latest_checkpoint_path,
    load_checkpoint,
    save_best_checkpoint,
    save_checkpoint,
    save_latest_checkpoint,
)
from fedbrew.core.config import load_config
from fedbrew.core.state import MetricRecord

# Not the fast gate: the checkpoint tests call torch.save and torch.load, which
# chapter 13 section 2.2 keeps out of it. The run.json test qualifies and is
# marked on its own.

REPO_ROOT = Path(__file__).resolve().parent.parent


class _Killed(Exception):
    """Stands in for the signal: raised where the process would have died."""


def _state(round_id: int) -> dict:
    return {"round_id": round_id, "weights": torch.full((64,), float(round_id))}


def _history(last: int) -> list[MetricRecord]:
    return [
        MetricRecord(round_id=r, metrics={"fit_loss": 1.0}, num_clients=2, num_examples=8)
        for r in range(1, last + 1)
    ]


def _half_then_raise(obj, handle, *args, **kwargs):
    """A torch.save that dies halfway through its bytes."""

    import io

    buffer = io.BytesIO()
    _REAL_SAVE(obj, buffer, *args, **kwargs)
    data = buffer.getvalue()
    handle.write(data[: len(data) // 2])
    raise _Killed("killed mid-write")


_REAL_SAVE = torch.save


class TheOldFileSurvivesAnInterruptedWriteTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)

    def _rename_fails(self):
        return mock.patch.object(checkpointing.os, "replace", side_effect=_Killed)

    def _leftovers(self) -> list[str]:
        return sorted(p.name for p in (self.out / "checkpoints").glob("*.tmp"))

    def test_latest_pt_keeps_the_previous_round(self) -> None:
        save_latest_checkpoint(_state(1), self.out)
        with self._rename_fails(), self.assertRaises(_Killed):
            save_latest_checkpoint(_state(2), self.out)

        self.assertEqual(load_checkpoint(get_latest_checkpoint_path(self.out))["round_id"], 1)

    def test_a_numbered_checkpoint_is_never_half_visible(self) -> None:
        """find_latest_checkpoint falls back to the highest round_NNN.pt, so a
        partial one would be the file a resume loads."""

        save_checkpoint(_state(1), self.out, 1)
        with self._rename_fails(), self.assertRaises(_Killed):
            save_checkpoint(_state(2), self.out, 2)

        self.assertFalse(get_checkpoint_path(self.out, 2).exists())
        latest = find_latest_checkpoint(self.out)
        self.assertEqual(latest, get_checkpoint_path(self.out, 1))
        self.assertEqual(load_checkpoint(latest)["round_id"], 1)

    def test_best_pt_keeps_the_previous_best(self) -> None:
        save_best_checkpoint(_state(1), self.out, "val_loss_avg", 0.5)
        with self._rename_fails(), self.assertRaises(_Killed):
            save_best_checkpoint(_state(2), self.out, "val_loss_avg", 0.25)

        best = load_checkpoint(self.out / "checkpoints" / "best.pt")
        self.assertEqual((best["round_id"], best["best_metric_value"]), (1, 0.5))

    def test_a_write_that_fails_part_way_removes_its_temp(self) -> None:
        save_latest_checkpoint(_state(1), self.out)
        with (
            mock.patch.object(checkpointing.torch, "save", _half_then_raise),
            self.assertRaises(_Killed),
        ):
            save_latest_checkpoint(_state(2), self.out)

        self.assertEqual(load_checkpoint(get_latest_checkpoint_path(self.out))["round_id"], 1)
        self.assertEqual(self._leftovers(), [])


@pytest.mark.fast
class RunJsonSurvivesAnInterruptedWriteTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)

    def test_run_json_keeps_the_previous_round(self) -> None:
        """A resume reads attempts and duration_sec from it."""

        config = load_config("configs/dev/smoke.yaml")
        metadata = {"run_id": "r", "status": "running"}
        save_run_json(_history(3), self.out, config, run_metadata=metadata)
        with (
            mock.patch.object(artifacts.os, "replace", side_effect=_Killed),
            self.assertRaises(_Killed),
        ):
            save_run_json(_history(5), self.out, config, run_metadata=metadata)

        with (self.out / "run.json").open(encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["final_round"], 3)


#: The child: write round 1 whole, then write round 2 halfway and wait to be
#: killed. Under the old in-place torch.save, the halfway bytes are latest.pt.
_CHILD = textwrap.dedent(
    """
    import io, os, sys, time
    import torch
    from fedbrew.core import checkpointing

    out = sys.argv[1]
    checkpointing.save_latest_checkpoint(
        {"round_id": 1, "weights": torch.full((4096,), 1.0)}, out
    )
    real_save = torch.save

    def stalled(obj, f, *args, **kwargs):
        buffer = io.BytesIO()
        real_save(obj, buffer, *args, **kwargs)
        data = buffer.getvalue()
        handle = open(f, "wb") if isinstance(f, (str, os.PathLike)) else f
        handle.write(data[: len(data) // 2])
        handle.flush()
        os.fsync(handle.fileno())
        print("mid-write", flush=True)
        time.sleep(120)

    torch.save = stalled
    checkpointing.save_latest_checkpoint(
        {"round_id": 2, "weights": torch.full((4096,), 2.0)}, out
    )
    """
)


class ASigkillMidWriteLeavesLatestReadableTest(unittest.TestCase):
    """The 2026-09-20 failure, reproduced with a real process and a real kill."""

    def test_latest_pt_still_loads_after_the_kill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
            child = subprocess.Popen(
                [sys.executable, "-c", _CHILD, str(out)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            watchdog = threading.Timer(120, child.kill)
            watchdog.start()
            try:
                line = child.stdout.readline().strip()
                self.assertEqual(line, "mid-write", child.stderr.read() if not line else line)
                os.kill(child.pid, signal.SIGKILL)
                child.wait()
            finally:
                watchdog.cancel()
                child.stdout.close()
                child.stderr.close()

            self.assertEqual(child.returncode, -signal.SIGKILL)
            latest = get_latest_checkpoint_path(out)
            self.assertEqual(load_checkpoint(latest)["round_id"], 1)

            # The halfway bytes are in the temp file, which the next start sweeps.
            leftover = latest.with_name("latest.pt.tmp")
            self.assertTrue(leftover.is_file())
            clear_stale_temp_files(out)
            self.assertFalse(leftover.exists())
            self.assertEqual(load_checkpoint(latest)["round_id"], 1)


class AKilledWritesTempIsNeverReadAsACheckpointTest(unittest.TestCase):
    def test_it_is_ignored_and_then_swept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            save_checkpoint(_state(3), out, 3)
            directory = out / "checkpoints"
            for name in ("latest.pt.tmp", "round_004.pt.tmp", "best.pt.tmp"):
                (directory / name).write_bytes(b"half a checkpoint")

            self.assertEqual(find_latest_checkpoint(out), get_checkpoint_path(out, 3))
            self.assertEqual(sorted(p.name for p in directory.glob("*.pt")), ["round_003.pt"])

            clear_stale_temp_files(out)
            self.assertEqual(sorted(p.name for p in directory.iterdir()), ["round_003.pt"])


if __name__ == "__main__":
    unittest.main()
