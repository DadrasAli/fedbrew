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

import pytest
import torch
import yaml

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


@contextmanager
def without(split: str, every: int = 3) -> Iterator[None]:
    """Every ``every``-th client (by number) holds no ``split``: an empty one of its tensors."""

    from fedbrew.data.synthetic_classification import SyntheticClassificationDataset as Data

    real = Data.get_client_data

    def fewer(self: Any, client_id: str) -> dict[str, Any]:
        data = real(self, client_id)
        if int(client_id.rsplit("_", 1)[-1]) % every == 0:
            data[split] = {key: value[:0] for key, value in data[split].items()}
        return data

    with mock.patch.object(Data, "get_client_data", fewer):
        yield


class TheEvaluationIsMeasuredOnTheDeviceTest(ResidentRuns):
    """Client splits and the central pass: measured at the round, recorded at the flush."""

    def _config(self, **evaluation: Any) -> dict[str, Any]:
        config = classification_config(**FEDAVG, update_mode="single_batch")
        config["runtime"]["checkpointing"].update(save_every_round=True)
        config["runtime"]["flush_every"] = 2
        config["evaluation"].update(evaluation)
        return config

    def test_its_stages_are_taken(self) -> None:
        from fedbrew.core import resident_evaluation

        taken: dict[str, int] = {"clients": 0, "central": 0}
        real = (
            resident_evaluation.ResidentEvaluation.enqueue,
            (resident_evaluation.ResidentEvaluation.enqueue_central),
        )

        def clients(self: Any, *args: Any) -> Any:
            stage = real[0](self, *args)
            taken["clients"] += stage is not None
            return stage

        def central(self: Any, *args: Any) -> Any:
            stage = real[1](self, *args)
            taken["central"] += stage is not None
            return stage

        with (
            mock.patch.object(resident_evaluation.ResidentEvaluation, "enqueue", clients),
            mock.patch.object(resident_evaluation.ResidentEvaluation, "enqueue_central", central),
        ):
            held, reference = self.pair(self._config())
        self.assertEqual(taken, {"clients": 4, "central": 4})
        self.assertSameRun(held, reference)

    def test_schedules_and_client_scopes(self) -> None:
        config = self._config(
            train={"every": 2, "clients": "participating"},
            val={"every": 3, "clients": "sample:3"},
            test={"every": 1, "clients": "resample:5"},
            central_test={"every": 2},
        )
        config["server"] = {
            **config["server"],
            "participation_rate": None,
            "participation_probability": 0.5,
        }
        self.assertSameRun(*self.pair(config, ragged))

    def test_a_shuffled_evaluation_loader(self) -> None:
        config = self._config()
        config["client"].update(eval_shuffle=True, eval_batch_size=2)
        self.assertSameRun(*self.pair(config, ragged))

    def test_a_client_without_a_val_split(self) -> None:
        self.assertSameRun(*self.pair(self._config(), lambda: without("eval")))

    def test_a_client_without_a_test_split_is_refused_in_the_same_words(self) -> None:
        messages = []
        for context in (nullcontext, per_round):
            with self.subTest(path=context.__name__), without("test"), context():
                with self.assertRaises(ValueError) as caught:
                    self.run_config(_clean(self._config()), "batched")
                messages.append(str(caught.exception))
        self.assertEqual(messages[0], messages[1])
        self.assertIn("has no non-empty test split", messages[0])


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


class AResumeFromAnAsynchronousFlushIsTheRunTest(ResidentRuns):
    """Stopped at a flush or between two, and resumed from latest.pt: the uninterrupted run.

    The flush's writes run on the writer thread; a stop hands the loop's
    finally the writer to finish, as a kill leaves what it had committed.
    """

    def _config(self) -> dict[str, Any]:
        config = classification_config(**FEDAVG, update_mode="single_batch")
        config["defaults"]["global_rounds"] = 6
        config["runtime"]["flush_every"] = 2
        config["runtime"]["checkpointing"].update(save_every_round=True)
        return _clean(config)

    def test_stopped_at_a_flush_and_between_two(self) -> None:
        import argparse

        from fedbrew.core import runner

        config = self._config()
        whole = self.run_config(config, "batched")
        for stop_at in (4, 5):
            with self.subTest(stop_at=stop_at):
                path = self.root / f"stopped-{stop_at}.yaml"
                config["experiment"]["output_dir"] = str(self.root / f"stopped-{stop_at}")
                config["runtime"].setdefault("performance", {})["executor"] = "batched"
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                with mock.patch.object(runner, "_round_progress_reporter", _stopping(stop_at)):
                    with self.assertRaises(KeyboardInterrupt):
                        runner.run(path, args=None)
                output = Path(config["experiment"]["output_dir"])
                latest = load_checkpoint(output / "checkpoints" / "latest.pt")["round_id"]
                self.assertEqual(latest, 4)
                self.assertEqual(
                    max(int(row["round_id"]) for row in _rows(output / "round_metrics.csv")), 4
                )
                runner.run(path, args=argparse.Namespace(resume_latest=True))
                self.assertSameRun(output, whole)


def _stopping(round_id: int) -> Any:
    """A round reporter that interrupts the run as round ``round_id`` is reported."""

    def reporter(config: Any, progress: Any) -> Any:
        def on_round_end(record: Any) -> None:
            if record.round_id == round_id:
                raise KeyboardInterrupt

        return on_round_end

    return reporter


@pytest.mark.fast
class TheFlushsWritesTest(unittest.TestCase):
    """The writer runs a flush's writes in order, and hands back what one raised."""

    def test_a_path_staged_again_keeps_its_place_and_takes_the_later_payload(self) -> None:
        from fedbrew.core.resident_flush import DeferredStaged

        staged = DeferredStaged()
        staged.stage({"round_id": 1}, Path("a/best.pt"))
        staged.stage({"round_id": 1}, Path("a/latest.pt"))
        staged.stage({"round_id": 2}, Path("a/best.pt"))
        self.assertEqual(list(staged.pending), [Path("a/best.pt"), Path("a/latest.pt")])
        self.assertEqual(staged.pending[Path("a/best.pt")], {"round_id": 2})

    def test_writes_run_in_order_and_an_error_reaches_the_loop(self) -> None:
        from fedbrew.core.resident_flush import FlushWriter

        writer = FlushWriter()
        done: list[int] = []
        try:
            writer.submit(lambda: done.append(1))
            writer.submit(lambda: done.append(2))
            writer.wait()
            self.assertEqual(done, [1, 2])

            def fails() -> None:
                raise OSError("disk full")

            writer.submit(fails)
            writer.submit(lambda: done.append(3))
            with self.assertRaisesRegex(OSError, "disk full"):
                writer.wait()
            # Nothing after a failed flush is written.
            self.assertEqual(done, [1, 2])
        finally:
            writer.close()


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
