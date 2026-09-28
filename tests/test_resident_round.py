"""A run's rounds held on its device compute what the per-round batched path computes, bit for bit.

``fedbrew/core/resident.py`` trains a batched run's rounds from the planner's
orders on rows stacked once for the run, keeps the model on the device, and
records each round at the flush from its outputs read back then. Every run
here is run twice -- resident, and with the resident round refused so the
per-round batched path runs it -- and everything either wrote is compared bit
for bit: every non-timing cell of the three CSVs, every checkpoint the run
wrote (each round's, the latest and the best: model, server and client
states, metrics, RNG state), and what run.json says of how the run ended.

The runs cover the synthetic classification task at full participation (an
MNIST-like round: one bucket), at a Bernoulli participation over clients of
different sizes under ``sequential_epoch`` (a FEMNIST-like round: several
buckets, some of one client, whose fold is the CPU's), the own-loop rules,
uniform weighting, a post-fit pass on some rounds only, a flush every third
round, and stops inside a flush window: a stall verdict and an aggregate that
is not finite.
"""

from __future__ import annotations

import json
import unittest
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any
from unittest import mock

import torch

from fedbrew.core import resident
from fedbrew.core.checkpointing import load_checkpoint
from tests.test_batched_executor_tolerance import (
    CSVS,
    ExecutorRuns,
    _rows,
    classification_config,
    classification_rule_config,
    example_config,
)
from tests.test_reproducibility import TIMING
from tests.test_round_planner import ragged

FEDAVG = {"update_rule": "fedavg", "frozen_gradient_weighting": "examples"}


def arms() -> Iterator[tuple[str, dict[str, Any], Any]]:
    """(label, config, data context) for every run the resident round must match."""

    def config(client: dict[str, Any], **edits: Any) -> dict[str, Any]:
        built = classification_rule_config(client)
        built["runtime"]["checkpointing"].update(save_every_round=True)
        for section, values in edits.items():
            built.setdefault(section, {}).update(values)
        return built

    yield "mnist-like", config({**FEDAVG, "update_mode": "single_batch"}), nullcontext
    yield (
        "femnist-like",
        config(
            {**FEDAVG, "update_mode": "sequential_epoch"},
            server={"participation_rate": None, "participation_probability": 0.6},
        ),
        ragged,
    )
    yield (
        "local_sgd",
        config({"momentum": 0.9, "nesterov": True, "weight_decay": 0.01}),
        ragged,
    )
    yield (
        "local_adamw",
        config(
            {
                "update_rule": "local_adamw",
                "learning_rate": 0.01,
                "weight_decay": 0.01,
                "beta1": 0.9,
                "beta2": 0.99,
                "epsilon": 1e-8,
            }
        ),
        nullcontext,
    )
    yield (
        "uniform",
        config(
            {**FEDAVG, "update_mode": "single_batch"}, server={"aggregation_weighting": "uniform"}
        ),
        ragged,
    )
    yield (
        "flush every 3, fit every 2",
        config(
            {**FEDAVG, "update_mode": "frozen_batch_gradients"},
            runtime={"flush_every": 3},
            evaluation={"fit": {"every": 2}},
        ),
        ragged,
    )


def _clean(config: dict[str, Any]) -> dict[str, Any]:
    config["server"] = {key: value for key, value in config["server"].items() if value is not None}
    return config


@contextmanager
def per_round() -> Iterator[None]:
    """The resident round refused, so the per-round batched path runs the same config."""

    with mock.patch.object(resident, "resident_unsupported", lambda context: "per round"):
        yield


class ResidentRuns(ExecutorRuns):
    """Runs one configuration resident and per round, and compares everything they wrote."""

    def pair(self, config: dict[str, Any], data: Any = nullcontext) -> tuple[Path, Path]:
        config = _clean(config)
        with data():
            held = self.run_config(config, "batched")
            with per_round():
                reference = self.run_config(config, "batched")
        self.assertEqual(_executor(held)["rounds"], {"used": "resident"})
        self.assertEqual(_executor(reference)["rounds"]["used"], "per_round")
        return held, reference

    def assertSameRun(self, held: Path, reference: Path) -> None:
        for name in CSVS:
            self._same_csv(held / name, reference / name)
        files = sorted(path.name for path in (reference / "checkpoints").glob("*.pt"))
        self.assertEqual(sorted(path.name for path in (held / "checkpoints").glob("*.pt")), files)
        for name in files:
            _same(
                self,
                load_checkpoint(held / "checkpoints" / name),
                load_checkpoint(reference / "checkpoints" / name),
                name,
            )
        run_held, run_reference = (_run(path) for path in (held, reference))
        for key in ("status", "termination", "stopped_round", "checkpointing"):
            self.assertEqual(run_held.get(key), run_reference.get(key), key)

    def _same_csv(self, held: Path, reference: Path) -> None:
        if not reference.exists():
            self.assertFalse(held.exists(), held.name)
            return
        rows_held, rows_reference = _rows(held), _rows(reference)
        self.assertEqual(len(rows_held), len(rows_reference), held.name)
        for row_held, row_reference in zip(rows_held, rows_reference, strict=True):
            self.assertEqual(
                {key: value for key, value in row_held.items() if key not in TIMING},
                {key: value for key, value in row_reference.items() if key not in TIMING},
                held.name,
            )


def _executor(output: Path) -> dict[str, Any]:
    return _run(output)["reproducibility"]["executor"]


def _run(output: Path) -> dict[str, Any]:
    return json.loads((output / "run.json").read_text(encoding="utf-8"))


def _same(case: unittest.TestCase, held: Any, reference: Any, where: str) -> None:
    """Equal all the way down: tensors bit for bit, in the same dtype and shape."""

    if isinstance(reference, torch.Tensor):
        case.assertIsInstance(held, torch.Tensor, where)
        case.assertEqual(held.dtype, reference.dtype, where)
        case.assertTrue(torch.equal(held, reference), where)
    elif isinstance(reference, dict):
        case.assertEqual(list(held), list(reference), where)
        for key, value in reference.items():
            _same(case, held[key], value, f"{where}.{key}")
    elif isinstance(reference, list | tuple):
        case.assertEqual(len(held), len(reference), where)
        for index, (value_held, value) in enumerate(zip(held, reference, strict=True)):
            _same(case, value_held, value, f"{where}[{index}]")
    elif hasattr(reference, "shape") and hasattr(reference, "dtype"):  # a numpy array
        case.assertTrue((held == reference).all(), where)
    else:
        case.assertEqual(held, reference, where)


class TheResidentRoundIsThePerRoundPathTest(ResidentRuns):
    def test_every_arm(self) -> None:
        for label, config, data in arms():
            with self.subTest(arm=label):
                self.assertSameRun(*self.pair(config, data))

    def test_both_folds_are_taken(self) -> None:
        """Buckets of several clients fold on the device; a round with one of one, on the CPU."""

        for label, host in (("mnist-like", False), ("femnist-like", True)):
            with self.subTest(arm=label):
                seen = _fold_sites()
                _, config, data = next(arm for arm in arms() if arm[0] == label)
                with seen:
                    held, reference = self.pair(config, data)
                self.assertIn(host, seen.hosts)
                self.assertSameRun(held, reference)


class _fold_sites:  # noqa: N801 -- used as a context manager, named for what it records
    """Records where each resident round folded: on the CPU (True) or the device (False)."""

    def __init__(self) -> None:
        self.hosts: list[bool] = []
        self._patch: Any = None

    def __enter__(self) -> _fold_sites:
        real = resident._Fold.__init__
        hosts = self.hosts

        def spy(fold: Any, rounds: Any, device_round: Any, host: bool) -> None:
            hosts.append(host)
            real(fold, rounds, device_round, host)

        self._patch = mock.patch.object(resident._Fold, "__init__", spy)
        self._patch.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._patch.stop()


class AManifestDatasetTest(ResidentRuns):
    """A manifest dataset's clients are built on demand, and stay built while cached."""

    def test_the_linear_examples(self) -> None:
        for name in ("fed-lasso-l2", "pl-1d"):
            with self.subTest(example=name):
                config = example_config(name)
                self.assertSameRun(*self.pair(config))

    def test_a_cache_too_small_for_every_shard_runs_per_round(self) -> None:
        config = example_config("fed-lasso-l2")
        config["runtime"].setdefault("performance", {})["shard_cache_bytes"] = 1
        output = self.run_config(_clean(config), "batched")
        rounds = _executor(output)["rounds"]
        self.assertEqual(rounds["used"], "per_round")
        self.assertIn("shard cache", rounds["reason"])


class StopsInsideAWindowTest(ResidentRuns):
    def _config(self, **edits: Any) -> dict[str, Any]:
        config = classification_config(**FEDAVG, update_mode="single_batch")
        config["runtime"]["checkpointing"].update(save_every_round=True)
        config["runtime"]["flush_every"] = 3
        for section, values in edits.items():
            config.setdefault(section, {}).update(values)
        return config

    def test_a_stall_verdict(self) -> None:
        config = self._config(divergence={"metric": "fit_loss", "patience": 1, "min_delta": 0.9})
        held, reference = self.pair(config)
        self.assertEqual(_run(held)["status"], "stalled")
        self.assertEqual(_run(held)["termination"]["round_id"], 2)
        self.assertSameRun(held, reference)

    def test_an_aggregate_that_is_not_finite(self) -> None:
        config = self._config(client={"learning_rate": 1e38})
        held, reference = self.pair(config)
        self.assertEqual(_run(held)["status"], "diverged")
        self.assertEqual(_run(held)["termination"]["detector"], "non_finite_client_state")
        self.assertSameRun(held, reference)


class WhoTakesItTest(ResidentRuns):
    def test_a_rule_with_per_client_state_runs_per_round_and_says_why(self) -> None:
        config = classification_rule_config({"update_rule": "fedprox", "proximal_mu": 0.01})
        output = self.run_config(_clean(config), "batched")
        rounds = _executor(output)["rounds"]
        self.assertEqual(rounds["used"], "per_round")
        self.assertIn("keeps per-client state", rounds["reason"])

    def test_the_sequential_executor_records_nothing_of_it(self) -> None:
        config = classification_config(**FEDAVG, update_mode="single_batch")
        output = self.run_config(_clean(config), "sequential")
        self.assertNotIn("rounds", _executor(output))


if __name__ == "__main__":
    unittest.main()
