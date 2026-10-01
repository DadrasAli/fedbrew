"""A FedOpt run's rounds are held on the device, and are the per-round path's, bit for bit.

The resident round (``fedbrew/core/resident.py``) hands each round's fold to the
server's own update on its device (``FedOptServer.update_from_fold``): the next
round starts from the updated model without a copy, the moments stay on the
device, and the flush reads each round's model and moments back to the server
(``adopt_update``). Every run here is run twice -- resident, and with the resident
round refused so the per-round path folds, updates and broadcasts through the host --
and everything either wrote is compared bit for bit: every non-timing CSV cell,
every checkpoint (model, the server's moments and update count, client states, RNG
state) and how the run ended.

The runs cover the four server optimizers; full participation (an MNIST-like round)
and a Bernoulli participation over clients of different sizes, where a round can
select none (the moments do not decay on it); uniform weighting; a flush every third
round with a post-fit pass on some; a float64 linear example; a stall verdict and a
non-finite aggregate inside a flush window; and a stop between two flushes, resumed
from ``latest.pt`` with the moments it holds.
"""

from __future__ import annotations

import argparse
import unittest
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch
import yaml

from fedbrew.core.checkpointing import load_checkpoint
from fedbrew.servers.fedopt import FedOptServer
from tests.test_batched_executor_tolerance import (
    classification_config,
    example_config,
    ragged_clients,
    set_performance,
)
from tests.test_resident_round import (
    FEDAVG,
    ResidentRuns,
    _clean,
    _executor,
    _run,
    _stopping,
    per_round,
    ragged,
)

#: The optimizers and the hyperparameters each reads (UNREAD_FEDOPT_HYPERPARAMETERS).
SERVERS = {
    "fedavgm": {"server_learning_rate": 0.5, "beta1": 0.9},
    "fedadagrad": {"server_learning_rate": 0.1, "beta1": 0.9, "tau": 1e-3},
    "fedadam": {"server_learning_rate": 0.05, "beta1": 0.9, "beta2": 0.99, "tau": 1e-3},
    "fedyogi": {"server_learning_rate": 0.05, "beta1": 0.9, "beta2": 0.99, "tau": 1e-3},
}


def fedopt(optimizer: str, **edits: Any) -> dict[str, Any]:
    """The synthetic classification run under one server optimizer, every round checkpointed."""

    config = classification_config(**FEDAVG, update_mode="single_batch")
    config["runtime"]["checkpointing"].update(save_every_round=True)
    config["server"].update(strategy=optimizer, **SERVERS[optimizer])
    for section, values in edits.items():
        config.setdefault(section, {}).update(values)
    return config


class FedOptIsHeldOnTheDeviceTest(ResidentRuns):
    def test_every_optimizer_at_full_participation(self) -> None:
        for optimizer in SERVERS:
            with self.subTest(optimizer=optimizer):
                self.assertSameRun(*self.pair(fedopt(optimizer)))

    def test_every_optimizer_at_partial_participation_over_ragged_clients(self) -> None:
        for optimizer in SERVERS:
            with self.subTest(optimizer=optimizer):
                config = fedopt(
                    optimizer, server={"participation_rate": None, "participation_probability": 0.4}
                )
                config["schedule"]["rounds"] = 8
                self.assertSameRun(*self.pair(config, ragged))

    def test_uniform_weighting_and_a_window(self) -> None:
        config = fedopt(
            "fedadam",
            server={"aggregation_weighting": "uniform"},
            runtime={"flush_every": 3},
            evaluation={"fit": {"every": 2}},
        )
        self.assertSameRun(*self.pair(config, ragged))

    def test_a_float64_linear_example(self) -> None:
        for arm in ("fedadam", "fedyogi", "fedadagrad", "fedavgm"):
            with self.subTest(arm=arm):
                config = example_config("fed-lasso", arm)
                self.assertSameRun(*self.pair(config, ragged_clients))

    def test_the_moments_are_what_a_checkpoint_holds(self) -> None:
        held, reference = self.pair(fedopt("fedyogi"))
        for run in (held, reference):
            state = load_checkpoint(run / "checkpoints" / "latest.pt")["server_state"]
            self.assertEqual(state["update_step"], 4)
            self.assertTrue(all(torch.isfinite(t).all() for t in state["m"].values()))
            self.assertTrue(any(bool(t.abs().sum() > 0) for t in state["v"].values()))
        self.assertSameRun(held, reference)

    def test_a_round_that_selects_no_client_leaves_the_moments_as_they_were(self) -> None:
        config = fedopt(
            "fedadam", server={"participation_rate": None, "participation_probability": 0.1}
        )
        config["schedule"]["rounds"] = 10
        held, reference = self.pair(config)
        steps = {
            path.name: load_checkpoint(path)["server_state"]["update_step"]
            for path in sorted((held / "checkpoints").glob("round_*.pt"))
        }
        self.assertLess(max(steps.values()), 10, steps)
        self.assertSameRun(held, reference)


class StopsInsideAWindowTest(ResidentRuns):
    def _config(self, **edits: Any) -> dict[str, Any]:
        config = fedopt("fedadam", runtime={"flush_every": 3})
        for section, values in edits.items():
            config.setdefault(section, {}).update(values)
        return config

    def test_a_stall_verdict(self) -> None:
        config = self._config(divergence={"metric": "fit_loss", "patience": 1, "min_delta": 0.9})
        held, reference = self.pair(config)
        self.assertEqual(_run(held)["status"], "stalled")
        self.assertSameRun(held, reference)

    def test_an_aggregate_that_is_not_finite(self) -> None:
        config = self._config(client={"learning_rate": 1e38})
        held, reference = self.pair(config)
        self.assertEqual(_run(held)["status"], "diverged")
        self.assertSameRun(held, reference)


class AResumeKeepsTheMomentsTest(ResidentRuns):
    def test_stopped_between_two_flushes_and_resumed(self) -> None:
        from fedbrew.core import runner

        for optimizer in ("fedadam", "fedavgm"):
            with self.subTest(optimizer=optimizer):
                config = _clean(fedopt(optimizer, runtime={"flush_every": 2}))
                config["schedule"]["rounds"] = 6
                whole = self.run_config(config, "batched")
                path = self.root / f"stopped-{optimizer}.yaml"
                config["experiment"]["output_dir"] = str(self.root / f"stopped-{optimizer}")
                set_performance(config, executor="batched")
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                with mock.patch.object(runner, "_round_progress_reporter", _stopping(5)):
                    with self.assertRaises(KeyboardInterrupt):
                        runner.run(path, args=None)
                output = Path(config["experiment"]["output_dir"])
                self.assertEqual(
                    load_checkpoint(output / "checkpoints" / "latest.pt")["round_id"], 4
                )
                runner.run(path, args=argparse.Namespace(resume_latest=True))
                self.assertSameRun(output, whole)


class WhoTakesItTest(ResidentRuns):
    def test_fedopt_is_resident(self) -> None:
        output = self.run_config(_clean(fedopt("fedadam")), "batched")
        self.assertEqual(_executor(output)["rounds"], {"used": "resident"})

    def test_a_server_that_updates_its_own_way_is_not(self) -> None:
        class Own(FedOptServer):
            def update_from_fold(self, model: Any, fold: Any, carried: Any) -> Any:
                return super().update_from_fold(model, fold, carried)

        with mock.patch("fedbrew.core.registry._build_fedopt_server", lambda *a, **k: Own(*a, **k)):
            output = self.run_config(_clean(fedopt("fedadam")), "batched")
        rounds = _executor(output)["rounds"]
        self.assertEqual(rounds["used"], "per_round")
        self.assertIn("folds its results its own way", rounds["reason"])

    def test_the_other_servers_are_as_they_were(self) -> None:
        output = self.run_config(
            _clean(classification_config(**FEDAVG, update_mode="single_batch")), "batched"
        )
        self.assertEqual(_executor(output)["rounds"], {"used": "resident"})


def _arms() -> Iterator[tuple[str, dict[str, Any], Any]]:
    yield "mnist-like", fedopt("fedadam"), nullcontext
    yield (
        "several buckets",
        fedopt("fedyogi", client={"update_mode": "sequential_epoch"}),
        ragged,
    )


def _cuda_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
    except RuntimeError:
        return False
    return True


@pytest.mark.cuda
@unittest.skipUnless(_cuda_usable(), "needs a usable CUDA device")
class OnCudaTest(ResidentRuns):
    """On CUDA the update runs on the device: eager and replayed, the per-round path's."""

    def test_eager_and_replayed(self) -> None:
        for label, config, data in _arms():
            with self.subTest(arm=label):
                config = _clean(config)
                config["schedule"]["rounds"] = 6
                config["runtime"]["device"] = "cuda"
                with data():
                    eager = self.run_config(config, "batched")
                    graphed = self.run_config(config, "batched", cuda_graphs="on")
                    with per_round():
                        reference = self.run_config(config, "batched")
                self.assertEqual(_executor(eager)["rounds"], {"used": "resident"})
                self.assertSameRun(eager, reference)
                self.assertSameRun(graphed, reference)
                self.assertEqual(_executor(graphed)["cuda_graphs"]["used"], "on")


if __name__ == "__main__":
    unittest.main()
