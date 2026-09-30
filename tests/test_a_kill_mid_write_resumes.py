"""A run killed while its writer writes a flush resumes to the uninterrupted run, bit for bit.

Every run's writes -- each round's staged checkpoints, and at a flush the CSV
rows, run.json and the commit -- run on one thread behind the loop
(``fedbrew/core/resident_flush.py``). Here the process is killed outright,
with SIGKILL from inside that thread, at three places in round 4's flush:

- between round_metrics.csv and the per-client CSVs;
- halfway through writing round 4's checkpoint to its temporary file;
- after the CSVs and run.json, just before the commit makes the checkpoints
  visible.

Each time the checkpoint on disk is round 2's (the last flush the writer
finished), the history reaches it, and ``--resume-latest`` completes the run:
every CSV cell but the timings, every checkpoint and how the run ended are
the uninterrupted run's. For the sequential executor, the batched executor's
per-round path and the resident round.
"""

from __future__ import annotations

import argparse
import copy
import signal
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import yaml

from fedbrew.core.checkpointing import find_latest_checkpoint, get_checkpoint_round_id
from fedbrew.core.checkpointing import load_checkpoint as load
from tests.test_batched_executor_tolerance import classification_config, set_performance
from tests.test_resident_round import FEDAVG, ResidentRuns, _clean, _executor

REPO = Path(__file__).resolve().parent.parent

#: Run in a child process: the config, where to die, and the round whose flush it dies in.
DRIVER = textwrap.dedent(
    """
    import os, signal, sys
    from pathlib import Path
    from fedbrew.core import artifacts, checkpointing, resident, runner

    path, point, round_id, per_round = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]

    def die():
        os.kill(os.getpid(), signal.SIGKILL)

    if per_round == "1":
        resident.resident_unsupported = lambda context: "per round"
    if point == "csv":
        real_clients = artifacts.flush_client_csvs

        def flush_client_csvs(evaluations, updates, *args, **kwargs):
            if updates and updates[-1].round_id == round_id:
                die()
            return real_clients(evaluations, updates, *args, **kwargs)

        artifacts.flush_client_csvs = flush_client_csvs
    elif point == "staging":
        real_stage = checkpointing._stage

        def stage(payload, target):
            if Path(target).name == f"round_{round_id:03d}.pt":
                with Path(target).with_name(Path(target).name + ".tmp").open("wb") as handle:
                    handle.write(b"PK\\x03\\x04 half a checkpoint")
                die()
            return real_stage(payload, target)

        checkpointing._stage = stage
    elif point == "commit":
        real_commit = checkpointing.StagedCheckpoints.commit

        def commit(self):
            if any(target.name == f"round_{round_id:03d}.pt" for _, target in self._pending):
                die()
            return real_commit(self)

        checkpointing.StagedCheckpoints.commit = commit
    runner.run(path, args=None)
    """
)


class AKillInsideAFlushResumesToTheRunTest(ResidentRuns):
    def _config(self) -> dict[str, Any]:
        config = classification_config(**FEDAVG, update_mode="single_batch")
        config["schedule"]["rounds"] = 6
        config["runtime"]["flush_every"] = 2
        config["runtime"]["checkpointing"].update(save_every_round=True, save_last=True)
        return _clean(config)

    def test_at_each_point_of_the_write_on_each_path(self) -> None:
        for executor, per_round, used in (
            ("sequential", False, None),
            ("batched", True, "per_round"),
            ("batched", False, "resident"),
        ):
            config = self._config()
            with _per_round(per_round):
                whole = self.run_config(config, executor)
            if used is not None:
                self.assertEqual(_executor(whole)["rounds"]["used"], used)
            for point in ("csv", "staging", "commit"):
                with self.subTest(executor=executor, per_round=per_round, point=point):
                    output = self._killed(config, executor, per_round, point)
                    latest = find_latest_checkpoint(output)
                    assert latest is not None
                    self.assertEqual(get_checkpoint_round_id(load(latest)), 2)
                    with _per_round(per_round):
                        self._resume(output)
                    self.assertSameRun(output, whole)

    def _killed(self, config: dict[str, Any], executor: str, per_round: bool, point: str) -> Path:
        name = f"killed-{executor}-{int(per_round)}-{point}"
        config = copy.deepcopy(config)
        config["experiment"]["output_dir"] = str(self.root / name)
        set_performance(config, executor=executor)
        path = self.root / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        done = subprocess.run(
            [sys.executable, "-c", DRIVER, str(path), point, "4", str(int(per_round))],
            cwd=REPO,
            env={**_environment(), "PYTHONPATH": str(REPO)},
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(done.returncode, -signal.SIGKILL, done.stderr[-2000:])
        self._path = path
        return self.root / name

    def _resume(self, output: Path) -> None:
        from fedbrew.core import runner

        runner.run(self._path, args=argparse.Namespace(resume_latest=True))


def _environment() -> dict[str, str]:
    import os

    return {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}


class _per_round:  # noqa: N801 -- used as a context manager
    """The resident round refused, when asked, as ``tests/test_resident_round.py`` refuses it."""

    def __init__(self, active: bool) -> None:
        self.active = active
        self._context: Any = None

    def __enter__(self) -> None:
        if self.active:
            from tests.test_resident_round import per_round

            self._context = per_round()
            self._context.__enter__()

    def __exit__(self, *exc: Any) -> None:
        if self._context is not None:
            self._context.__exit__(*exc)
