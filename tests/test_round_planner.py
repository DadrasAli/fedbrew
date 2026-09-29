"""A round's orders planned ahead from the roster are the round's own, tensor for tensor.

``fedbrew/core/round_planner.py`` plans each round's sampled clients and batch
orders from the roster alone, in worker processes ahead of the loop, with the
functions the round itself calls. What holds, and is checked here:

- ``plan_roster_round`` is ``plan_round`` on the round's own plans -- every
  tensor of both orders and every structure equal -- for FedAvg's and the
  own-loop rules' update modes, at full participation, a fixed rate and a
  probability, shuffled or not, with ``drop_last``, and over clients whose
  splits differ in size;
- workers hand out exactly what this process computes, and a worker that dies
  leaves the planning to this process, recorded;
- a round asked for before a worker has started is planned in this process,
  without waiting for one, and is the same round; the workers take over from
  the round after the first one asked for once one has started;
- a batched run planned ahead, in this process or by workers, is bit-identical
  to one planned in the round.
"""

from __future__ import annotations

import copy
import queue
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch
import yaml

from fedbrew.clients.batched_update import plan_round
from fedbrew.core import batched_executor
from fedbrew.core.batched_executor import _plans
from fedbrew.core.config import load_config
from fedbrew.core.factory import build_components
from fedbrew.core.loop import _build_client_infos
from fedbrew.core.protocol import RoundInfo
from fedbrew.core.round_planner import (
    RoundPlanner,
    plan_roster_round,
    planned_for,
    roster_plan,
)
from tests.test_batched_executor_tolerance import ExecutorRuns, classification_config

ROUNDS = 4  # the tolerance suite's assertAgree expects its four checkpoints
FIELDS = ("indices", "lengths", "starts", "steps", "contiguous")


def planner_arms() -> Iterator[tuple[str, dict[str, Any], dict[str, Any]]]:
    """(label, client settings, server settings) over rules, modes, loaders and sampling."""

    fedavg = {"update_rule": "fedavg", "frozen_gradient_weighting": "examples"}
    for mode in ("single_batch", "sequential_epoch", "frozen_batch_gradients", "full_gradient"):
        yield f"fedavg/{mode}", {**fedavg, "update_mode": mode}, {}
    yield "local_sgd/sequential_epoch", {"update_mode": "sequential_epoch"}, {}
    yield "local_sgd/full_gradient", {"update_mode": "full_gradient"}, {}
    single = {**fedavg, "update_mode": "single_batch", "local_iterations": 7}
    yield "rate 0.5", single, {"participation_rate": 0.5}
    yield (
        "probability 0.4",
        single,
        {"participation_rate": None, "participation_probability": 0.4},
    )
    yield "unshuffled", {**single, "train_shuffle": False}, {}
    yield "drop_last", {**fedavg, "update_mode": "sequential_epoch", "drop_last": True}, {}


@contextmanager
def ragged() -> Iterator[None]:
    """Train splits of 20, 17, 14, ... rows by client number: batches of differing lengths."""

    from fedbrew.data.synthetic_classification import SyntheticClassificationDataset as Data

    real = Data.get_client_data

    def truncated(self: Any, client_id: str) -> dict[str, Any]:
        data = real(self, client_id)
        keep = 20 - 3 * (int(client_id.rsplit("_", 1)[-1]) % 5)
        train = {key: value[:keep] for key, value in data["train"].items()}
        return {**data, "train": train}

    with mock.patch.object(Data, "get_client_data", truncated):
        yield


def _config(client: dict[str, Any], server: dict[str, Any]) -> dict[str, Any]:
    client = dict(client)
    iterations = client.pop("local_iterations", None)
    config = classification_config(**client)
    config["client"]["batch_size"] = 3
    if iterations is not None:
        config["defaults"]["local_iterations"] = iterations
    config["server"].update(server)
    config["server"] = {key: value for key, value in config["server"].items() if value is not None}
    config["defaults"]["global_rounds"] = ROUNDS
    return config


def _components(config: dict[str, Any], directory: Path) -> Any:
    path = directory / "config.yaml"
    config = copy.deepcopy(config)
    config["experiment"]["output_dir"] = str(directory / "out")
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return build_components(load_config(path))


@pytest.mark.fast
class ThePlannedOrdersAreTheRoundsTest(unittest.TestCase):
    def test_every_arm_every_round(self) -> None:
        for label, client, server in planner_arms():
            with self.subTest(arm=label), ragged(), tempfile.TemporaryDirectory() as directory:
                components = _components(_config(client, server), Path(directory))
                roster, reason = roster_plan(components)
                self.assertIsNone(reason)
                assert roster is not None
                self._each_round(components, roster)

    def _each_round(self, components: Any, roster: Any) -> None:
        server, clients = components.server, components.clients
        infos = _build_client_infos(components.dataset)
        sampled = set()
        for round_id in range(1, ROUNDS + 1):
            requests = list(
                server.configure_round(RoundInfo(round_id=round_id, total_rounds=ROUNDS), infos)
            )
            for request in requests:
                request.post_fit_evaluation = True
            members = [clients[request.client_id] for request in requests]
            template = members[0].task.build_model(members[0].model_config)
            plans = _plans(members, requests, template)
            planned = plan_roster_round(roster, round_id)
            self.assertTrue(planned_for(roster, planned, plans))
            expected = plan_round(plans, round_id)
            for got, want in zip((planned.train, planned.evaluation), expected, strict=True):
                for name in FIELDS:
                    self.assertTrue(torch.equal(getattr(got, name), getattr(want, name)), name)
                self.assertEqual(got.structure, want.structure)
            sampled.add(tuple(planned.positions))
        # Partial participation drew different clients in different rounds.
        self.assertGreaterEqual(len(sampled), 1)


class WorkersPlanWhatThisProcessPlansTest(unittest.TestCase):
    def _roster(self, directory: Path) -> Any:
        client = {
            "update_rule": "fedavg",
            "update_mode": "single_batch",
            "frozen_gradient_weighting": "examples",
            "local_iterations": 4,
        }
        components = _components(
            _config(client, {"participation_rate": None, "participation_probability": 0.5}),
            directory,
        )
        roster, _ = roster_plan(components)
        return roster

    def test_rounds_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            roster = self._roster(Path(directory))
            planner = RoundPlanner(roster, 8, workers=2, ahead=3)
            try:
                self.assertEqual(planner.record["workers"], 2)
                self.assertTrue(planner.workers_ready(timeout=120))
                for round_id in range(1, 9):
                    got, want = planner.plan(round_id), plan_roster_round(roster, round_id)
                    self.assertEqual(got.positions, want.positions)
                    self.assertEqual(got.train.structure, want.train.structure)
                    self.assertEqual(got.evaluation.structure, want.evaluation.structure)
                    for name in FIELDS:
                        self.assertTrue(
                            torch.equal(getattr(got.train, name), getattr(want.train, name))
                        )
                        self.assertTrue(
                            torch.equal(
                                getattr(got.evaluation, name), getattr(want.evaluation, name)
                            )
                        )
            finally:
                planner.close()
            self.assertNotIn("fallback", planner.record)
            # Round 1 was planned here; the workers planned 2 to 8.
            self.assertEqual(planner.record["in_process"], 1)

    def test_a_dead_worker_leaves_the_planning_here(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            roster = self._roster(Path(directory))
            planner = RoundPlanner(roster, 8, workers=1, ahead=1)
            try:
                self.assertTrue(planner.workers_ready(timeout=120))
                first = planner.plan(1)
                for process in planner._processes:
                    process.kill()
                    process.join()
                with mock.patch("fedbrew.core.round_planner.WORKER_TIMEOUT_SEC", 2.0):
                    later = [planner.plan(round_id) for round_id in range(2, 6)]
            finally:
                planner.close()
            self.assertIn("fallback", planner.record)
            for planned in [first, *later]:
                want = plan_roster_round(roster, planned.round_id)
                self.assertEqual(planned.positions, want.positions)
                self.assertTrue(torch.equal(planned.train.indices, want.train.indices))


class _Held:
    """A planner's results queue whose messages the loop does not see until released."""

    def __init__(self, real: Any) -> None:
        self.real = real
        self.held = True

    def get_nowait(self) -> Any:
        if self.held:
            raise queue.Empty
        return self.real.get_nowait()

    def get(self, timeout: float | None = None) -> Any:
        if self.held:
            raise queue.Empty
        return self.real.get(timeout=timeout)


class TheLoopDoesNotWaitForAWorkerToStartTest(WorkersPlanWhatThisProcessPlansTest):
    """Rounds asked for before a worker has started are planned here, at once, and are the same."""

    def test_rounds_before_and_after_a_worker_has_started(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            roster = self._roster(Path(directory))
            planner = RoundPlanner(roster, 8, workers=1, ahead=2)
            held = planner._results = _Held(planner._results)
            try:
                before = [planner.plan(round_id) for round_id in (1, 2, 3)]
                self.assertEqual(planner.record["in_process"], 3)
                self.assertEqual(planner.record["waited_sec"], 0.0)
                # No round was handed to a worker that had not started.
                self.assertIsNone(planner._next_task)
                held.held = False
                self.assertTrue(planner.workers_ready(timeout=120))
                after = [planner.plan(round_id) for round_id in range(4, 9)]
            finally:
                planner.close()
            self.assertNotIn("fallback", planner.record)
            # Round 4, the first asked for once the worker had started, was
            # planned here too; the worker planned 5 to 8.
            self.assertEqual(planner.record["in_process"], 4)
            for planned in [*before, *after]:
                want = plan_roster_round(roster, planned.round_id)
                self.assertEqual(planned.positions, want.positions)
                for got, expected in (
                    (planned.train, want.train),
                    (planned.evaluation, want.evaluation),
                ):
                    self.assertEqual(got.structure, expected.structure)
                    for name in FIELDS:
                        self.assertTrue(torch.equal(getattr(got, name), getattr(expected, name)))


class AScriptWithoutAMainGuardIsNotRunAgainTest(unittest.TestCase):
    """A worker is spawned, and spawning imports the parent's main module: not this one's."""

    def test_the_script_runs_once(self) -> None:
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "ran.txt"
            script = root / "driver.py"
            script.write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).open('a').write('ran\\n')\n"
                "from tests.test_round_planner import WorkersPlanWhatThisProcessPlansTest\n"
                "from fedbrew.core.round_planner import RoundPlanner\n"
                "import tempfile\n"
                "case = WorkersPlanWhatThisProcessPlansTest()\n"
                "with tempfile.TemporaryDirectory() as inner:\n"
                "    planner = RoundPlanner(case._roster(Path(inner)), 3, workers=1)\n"
                "    assert planner.workers_ready(timeout=120)\n"
                "    rounds = [planner.plan(r).round_id for r in (1, 2, 3)]\n"
                "    print('planned', rounds, planner.record)\n"
                "    planner.close()\n",
                encoding="utf-8",
            )
            done = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=Path(__file__).resolve().parent.parent,
            )
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
            self.assertIn("planned [1, 2, 3]", done.stdout)
            self.assertNotIn("fallback", done.stdout)
            self.assertEqual(marker.read_text(encoding="utf-8").splitlines(), ["ran"])


def _unplanned() -> Any:
    return mock.patch.object(
        batched_executor, "roster_plan", lambda components: (None, "switched off by the test")
    )


class APlannedRunIsTheRunTest(ExecutorRuns):
    """Batched runs planned ahead against the same runs planned in the round."""

    def _arms(self) -> Iterator[tuple[str, dict[str, Any]]]:
        fedavg = {"update_rule": "fedavg", "frozen_gradient_weighting": "examples"}
        yield "full", _config({**fedavg, "update_mode": "single_batch"}, {})
        yield (
            "probability",
            _config(
                {**fedavg, "update_mode": "sequential_epoch"},
                {"participation_rate": None, "participation_probability": 0.5},
            ),
        )

    def test_in_this_process_and_by_workers(self) -> None:
        for label, config in self._arms():
            with self.subTest(arm=label), ragged():
                with _unplanned():
                    reference = self.run_config(config, "batched")
                planned = self.run_config(config, "batched")
                with (
                    mock.patch.object(batched_executor, "planner_workers", lambda model: 2),
                    # The loop waits for a worker to start, so the workers
                    # plan every round after the first.
                    mock.patch("fedbrew.core.round_planner.READY_WAIT_SEC", 120.0),
                ):
                    by_workers = self.run_config(config, "batched")
                self.assertAgree(planned, reference, exact=True)
                self.assertAgree(by_workers, reference, exact=True)
                record = self._executor_record(by_workers)
                self.assertEqual(record["planner"]["used"], "on")
                self.assertEqual(record["planner"]["workers"], 2)
                self.assertEqual(record["planner"]["in_process"], 1)
                self.assertNotIn("fallback", record["planner"])
                self.assertEqual(self._executor_record(reference)["planner"]["used"], "off")

    def _executor_record(self, output: Path) -> dict[str, Any]:
        import json

        run = json.loads((output / "run.json").read_text(encoding="utf-8"))
        return run["reproducibility"]["executor"]


if __name__ == "__main__":
    unittest.main()
