"""Each worker builds its optimizer once and resets it for every local update.

Every local update constructed its own torch.optim optimizer: 86 us per
client per round (measured on 2026-09-26). reused_optimizer keeps one per class per
worker, bound to each update's parameters with no state, and rebuilds it only
when the hyperparameters change. What is pinned here:

- the trajectory is bit-identical to one where every update builds its own,
  for every rule that steps a torch optimizer -- local SGD with momentum,
  Nesterov and weight decay under a cosine rate, AdamW, FedAvg's two batch
  modes, SCAFFOLD and FedProx -- which is where a momentum buffer or an Adam
  moment carried from another client would show;
- a run at a constant rate constructs one optimizer, and a cosine-scheduled
  one one per round;
- a handed-back optimizer holds no state and no parameter, and one still out
  is never handed out twice.
"""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import torch
import yaml
from torch import optim

from fedbrew.clients import (
    local_update_modes,
    torch_adamw_client,
    torch_fedprox_client,
    torch_scaffold_client,
    torch_sgd_client,
)
from fedbrew.clients.local_update_modes import release_optimizer, reused_optimizer
from fedbrew.core.runner import run
from tests.test_evaluation_cadence import _trajectory
from tests.test_reproducibility import TIMING, _config

ROUNDS = 5

_SGD_ONLY = dict.fromkeys(
    ("momentum", "weight_decay", "nesterov", "learning_rate_schedule", "min_learning_rate")
)

#: Every rule that steps a torch optimizer, as client-block overrides; None removes a key.
RULES: dict[str, dict[str, Any]] = {
    "local_sgd": {
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 0.01,
        "learning_rate_schedule": "cosine",
    },
    "local_adamw": {
        "update_rule": "local_adamw",
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1.0e-8,
        "weight_decay": 0.01,
        "momentum": None,
        "nesterov": None,
    },
    "fedavg_single_batch": {
        "update_rule": "fedavg",
        "update_mode": "single_batch",
        "frozen_gradient_weighting": "examples",
    },
    "fedavg_sequential_epoch": {
        "update_rule": "fedavg",
        "update_mode": "sequential_epoch",
        "frozen_gradient_weighting": "examples",
    },
    "scaffold": {"update_rule": "scaffold", **_SGD_ONLY},
    "fedprox": {"update_rule": "fedprox", "proximal_mu": 0.1, **_SGD_ONLY},
}


def _write(root: Path, name: str, client: dict[str, Any]) -> Path:
    config = _config(root / name, rounds=ROUNDS, checkpoint=True)
    config["runtime"]["checkpointing"]["keep_last"] = None
    config.setdefault("reporting", {})["per_client_csv"] = True
    config["client"].update(client)
    config["client"] = {key: value for key, value in config["client"].items() if value is not None}
    if config["client"]["update_rule"] == "scaffold":
        config["server"]["strategy"] = "scaffold"
    path = root / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _run(path: Path) -> Path:
    run(path, args=None)
    return Path(yaml.safe_load(path.read_text())["experiment"]["output_dir"])


def _fresh(optimizer_class: type[optim.Optimizer], parameters: Any, **hyperparameters: Any):
    return optimizer_class(list(parameters), **hyperparameters)


@contextmanager
def _one_per_update() -> Iterator[None]:
    """What every update did before: build its own optimizer."""

    with (
        mock.patch.object(local_update_modes, "reused_optimizer", _fresh),
        mock.patch.object(torch_sgd_client, "reused_optimizer", _fresh),
        mock.patch.object(torch_adamw_client, "reused_optimizer", _fresh),
        mock.patch.object(torch_scaffold_client, "reused_optimizer", _fresh),
        mock.patch.object(torch_fedprox_client, "reused_optimizer", _fresh),
    ):
        yield


@contextmanager
def _empty_workshop() -> Iterator[None]:
    """This thread's reused optimizers forgotten, before and after."""

    local_update_modes._REUSED.__init__()
    try:
        yield
    finally:
        local_update_modes._REUSED.__init__()


class _TempRoot(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)


class TheTrajectoryIsTheOneEveryUpdateBuiltItsOwnForTest(_TempRoot):
    def test_every_rule(self) -> None:
        for rule, client in RULES.items():
            with self.subTest(rule=rule):
                with _empty_workshop():
                    reused = _trajectory(_run(_write(self.root, f"{rule}-reused", client)))
                with _one_per_update():
                    fresh = _trajectory(_run(_write(self.root, f"{rule}-fresh", client)))
                self.assertEqual(len(reused["checkpoints"]), ROUNDS)
                self.assertEqual(reused, fresh)
                self.assertTrue(any(key not in TIMING for key in reused["rounds"][0]))


class OneOptimizerPerWorkerTest(_TempRoot):
    def _constructions(self, optimizer_class: type[optim.Optimizer], client: dict) -> int:
        built = 0
        real_init = optimizer_class.__init__

        def counted(self: Any, *args: Any, **kwargs: Any) -> None:
            nonlocal built
            built += 1
            real_init(self, *args, **kwargs)

        with _empty_workshop(), mock.patch.object(optimizer_class, "__init__", counted):
            _run(_write(self.root, f"count-{len(list(self.root.iterdir()))}", client))
        return built

    def test_one_at_a_constant_rate(self) -> None:
        self.assertEqual(self._constructions(optim.SGD, RULES["fedavg_single_batch"]), 1)
        self.assertEqual(self._constructions(optim.AdamW, RULES["local_adamw"]), 1)

    def test_one_a_round_under_a_cosine_rate(self) -> None:
        self.assertEqual(self._constructions(optim.SGD, RULES["local_sgd"]), ROUNDS)


class HandedBackTest(unittest.TestCase):
    def test_it_keeps_no_state_and_no_parameter(self) -> None:
        with _empty_workshop():
            weight = torch.nn.Parameter(torch.ones(3))
            optimizer = reused_optimizer(optim.SGD, [weight], lr=0.1, momentum=0.9)
            weight.grad = torch.ones(3)
            optimizer.step()
            self.assertTrue(optimizer.state)
            release_optimizer(optimizer)
            self.assertEqual(len(optimizer.state), 0)
            self.assertEqual(optimizer.param_groups[0]["params"], [])

            other = torch.nn.Parameter(torch.zeros(2))
            again = reused_optimizer(optim.SGD, [other], lr=0.1, momentum=0.9)
            self.assertIs(again, optimizer)
            self.assertEqual(again.param_groups[0]["params"], [other])
            release_optimizer(again)

    def test_one_still_out_is_not_handed_out_twice(self) -> None:
        with _empty_workshop():
            first = reused_optimizer(optim.SGD, [torch.nn.Parameter(torch.ones(1))], lr=0.1)
            second = reused_optimizer(optim.SGD, [torch.nn.Parameter(torch.ones(1))], lr=0.1)
            self.assertIsNot(first, second)
            release_optimizer(second)
            release_optimizer(first)

    def test_an_empty_parameter_list_is_refused_as_torch_refuses_it(self) -> None:
        with _empty_workshop():
            release_optimizer(
                reused_optimizer(optim.SGD, [torch.nn.Parameter(torch.ones(1))], lr=0.1)
            )
            with self.assertRaises(ValueError):
                reused_optimizer(optim.SGD, [], lr=0.1)


if __name__ == "__main__":
    unittest.main()
