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
is not finite. SCAFFOLD, whose controls the round holds on the device, is
held to the same: one bucket and several, runs of one client folded on the
CPU, both modes, its norms reported, a float64 example, a control that stops
being finite, and a resume from inside a flush window.
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
from torch import nn

from fedbrew.core import resident
from fedbrew.core.checkpointing import load_checkpoint
from fedbrew.core.torch_utils import set_training
from tests.test_batched_executor_tolerance import (
    CSVS,
    ExecutorRuns,
    _rows,
    classification_config,
    classification_rule_config,
    example_config,
    ragged_clients,
    set_performance,
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


SCAFFOLD = {"update_rule": "scaffold"}

#: Every column SCAFFOLD adds, client's and server's, and the divergence metric.
SCAFFOLD_METRICS = [
    "fit_loss",
    "control_delta_norm",
    "client_control_norm",
    "local_steps",
    "communicated_parameters",
    "communicated_bytes",
    "server_control_norm",
    "mean_client_control_delta_norm",
]


def scaffold_arms() -> Iterator[tuple[str, dict[str, Any], Any]]:
    """(label, config, data context) for SCAFFOLD's runs."""

    def config(client: dict[str, Any], **edits: Any) -> dict[str, Any]:
        built = classification_rule_config({**SCAFFOLD, **client})
        built["runtime"]["checkpointing"].update(save_every_round=True)
        for section, values in edits.items():
            built.setdefault(section, {}).update(values)
        return built

    yield "one bucket", config({}), nullcontext
    yield (
        "several buckets, runs of one",
        config({}, server={"participation_rate": None, "participation_probability": 0.6}),
        ragged,
    )
    yield (
        "full_gradient, flush every 3, fit every 2, its norms",
        config(
            {"update_mode": "full_gradient"},
            runtime={"flush_every": 3},
            evaluation={"fit": {"every": 2}},
            reporting={"fit_metrics": SCAFFOLD_METRICS},
        ),
        ragged,
    )
    lasso = example_config("fed-lasso-l2", "scaffold")
    lasso["reporting"]["fit_metrics"] = SCAFFOLD_METRICS
    yield "fed-lasso-l2, float64", lasso, ragged_clients


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


class ABucketsStepsAreKeptWhileItsOrdersAreTest(ResidentRuns):
    """Orders that draw nothing are every round's, so a bucket's steps are made once for the run.

    Shuffled training orders are drawn each round, so its training steps are
    made each round, and its post-fit steps, over an unshuffled loader, once.
    Either way the run is the per-round path's.
    """

    def test_unshuffled_and_shuffled(self) -> None:
        real = resident._Steps
        for shuffle, made_for_training in ((False, 1), (True, 4)):
            with self.subTest(shuffle=shuffle):
                config = classification_rule_config(
                    {
                        **FEDAVG,
                        "update_mode": "full_gradient",
                        "train_shuffle": shuffle,
                        "eval_shuffle": False,
                    }
                )
                made: list[Any] = []

                def counted(*args: Any, _made: list[Any] = made, **kwargs: Any) -> Any:
                    _made.append(args[1])
                    return real(*args, **kwargs)

                with mock.patch.object(resident, "_Steps", side_effect=counted):
                    held, reference = self.pair(config)
                rounds = len(_rows(held / "round_metrics.csv"))
                self.assertEqual(rounds, 4)
                # Training's one full-gradient update takes a client's 7 batches of 3
                # rows; the post-fit pass's epoch, 7 updates of one.
                training = [orders for orders in made if orders.structure[0] == (7,)]
                self.assertEqual(len(training), made_for_training)
                self.assertEqual(len(made) - len(training), 1)
                self.assertSameRun(held, reference)


class ARepeatedRoundKeepsItsHostHalfTest(ResidentRuns):
    """A round that repeats the last keeps its plan, and its buckets are stepped again.

    Unshuffled, every round's orders are the first's: one plan for the run,
    its buckets made once. Shuffled training orders are another round's each
    time: a plan and buckets a round. A post-fit pass every other round keeps
    one plan for the rounds that run it and one for those that do not. Either
    way the run is the per-round path's.
    """

    def test_kept_and_made_again(self) -> None:
        real_bucket, real_chunks = resident._Bucket, resident.cut_chunks
        per_plan: list[int] = []
        for shuffle, fit_every, plans in ((False, 1, 1), (True, 1, 4), (False, 2, 2)):
            with self.subTest(shuffle=shuffle, fit_every=fit_every):
                config = classification_rule_config(
                    {
                        **FEDAVG,
                        "update_mode": "full_gradient",
                        "train_shuffle": shuffle,
                        "eval_shuffle": False,
                    }
                )
                config.setdefault("evaluation", {})["fit"] = {"every": fit_every}
                made: list[Any] = []
                planned: list[Any] = []

                def bucket(*args: Any, _made: list[Any] = made, **kwargs: Any) -> Any:
                    _made.append(args[3])
                    return real_bucket(*args, **kwargs)

                def chunks(*args: Any, _planned: list[Any] = planned, **kwargs: Any) -> Any:
                    _planned.append(args)
                    return real_chunks(*args, **kwargs)

                with (
                    mock.patch.object(resident, "_Bucket", side_effect=bucket),
                    mock.patch.object(resident, "cut_chunks", side_effect=chunks),
                ):
                    held, reference = self.pair(config)
                self.assertEqual(len(_rows(held / "round_metrics.csv")), 4)
                self.assertEqual(len(planned), plans)
                # Each plan's buckets, made by the first round that runs it: as
                # many as the one plan of the unshuffled run has.
                per_plan = per_plan or [len(made)]
                self.assertGreater(per_plan[0], 0)
                self.assertEqual(len(made), plans * per_plan[0])
                self.assertSameRun(held, reference)


class _fold_sites:  # noqa: N801 -- used as a context manager, named for what it records
    """Records where each resident round folded: on the CPU (True) or the device (False)."""

    def __init__(self) -> None:
        self.hosts: list[bool] = []
        self._patch: Any = None

    def __enter__(self) -> _fold_sites:
        real = resident._Fold.__init__
        hosts = self.hosts

        def spy(fold: Any, rounds: Any, plan: Any, inputs: Any) -> None:
            hosts.append(plan.host_fold)
            real(fold, rounds, plan, inputs)

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

    def test_a_split_evaluated_again_beside_other_work(self) -> None:
        """The same sampled clients' val split, alone one round and beside train the next.

        The two rounds hold those clients at other places in their work, so
        a split's kept plan is one round's alone. Kept by whose split it was
        and not where, the second round read the first's places: past the end
        of its work it raised (configs/reference_evaluation.yaml did), and
        inside it measured other clients' splits in their stead, which
        recorded other numbers and raised nothing -- the second case here.
        """

        for rate, test, sample in ((0.5, "sample:3", 3), (0.5, "all", 2)):
            with self.subTest(participation=rate, sample=sample):
                config = self._config(
                    train={"every": 2, "clients": "participating"},
                    val={"every": 1, "clients": f"sample:{sample}"},
                    test={"every": 2 if test != "all" else 3, "clients": test},
                )
                config["server"] = {**config["server"], "participation_rate": rate}
                self.assertSameRun(*self.pair(config))

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
        config["schedule"]["rounds"] = 6
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
                set_performance(config, executor="batched")
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


class SCAFFOLDIsHeldOnTheDeviceTest(ResidentRuns):
    """SCAFFOLD's controls on the device: the per-round path's run, bit for bit."""

    def test_every_arm(self) -> None:
        for label, config, data in scaffold_arms():
            with self.subTest(arm=label):
                seen = _fold_sites()
                with seen:
                    held, reference = self.pair(config, data)
                self.assertSameRun(held, reference)
                if label.startswith("several"):
                    self.assertIn(True, seen.hosts)

    def test_a_control_that_is_not_finite(self) -> None:
        config = classification_rule_config({**SCAFFOLD, "learning_rate": 1e38})
        config["runtime"]["flush_every"] = 3
        config["runtime"]["checkpointing"].update(save_every_round=True)
        held, reference = self.pair(config)
        self.assertEqual(_run(held)["status"], "diverged")
        self.assertSameRun(held, reference)

    def test_resumed_from_inside_a_flush_window(self) -> None:
        import argparse

        from fedbrew.core import runner

        config = _clean(classification_rule_config(dict(SCAFFOLD)))
        config["schedule"]["rounds"] = 6
        config["runtime"]["flush_every"] = 4
        config["runtime"]["checkpointing"].update(save_every_round=True)
        whole = self.run_config(config, "batched")
        self.assertEqual(_executor(whole)["rounds"], {"used": "resident"})
        path = self.root / "stopped.yaml"
        config["experiment"]["output_dir"] = str(self.root / "stopped")
        set_performance(config, executor="batched")
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        # Stopped at round 5: latest.pt is round 4's flush, with every c_i as round 4 left it.
        with mock.patch.object(runner, "_round_progress_reporter", _stopping(5)):
            with self.assertRaises(KeyboardInterrupt):
                runner.run(path, args=None)
        output = Path(config["experiment"]["output_dir"])
        self.assertEqual(load_checkpoint(output / "checkpoints" / "latest.pt")["round_id"], 4)
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
class AProgramsKeyIsKeptWhileItsValuesAreTest(unittest.TestCase):
    """``_program_key``: the program's repr, without the server's rate, kept for the same values.

    The same values are the same fields, each the same float to its sign
    (``_same_values``): -0.0 is not 0.0, NaN is NaN, and a mutable dataclass is
    never taken for the same, not even itself; the same values are the same
    reprs.
    """

    def test_the_key_and_the_values_it_reads(self) -> None:
        import dataclasses as dc
        from types import SimpleNamespace

        from fedbrew.clients.batched_update import LocalProgram, OptimizerSpec
        from fedbrew.core.resident import ResidentRounds, _same_values

        def program(lr: float, momentum: float = 0.0) -> LocalProgram:
            return LocalProgram(optimizer=OptimizerSpec("sgd", lr=lr, momentum=momentum))

        self.assertTrue(_same_values(program(0.5), program(0.5)))
        self.assertFalse(_same_values(program(0.5), program(0.25)))
        self.assertFalse(_same_values(program(0.0), program(-0.0)))
        self.assertTrue(_same_values(program(float("nan")), program(float("nan"))))
        self.assertFalse(_same_values(program(1.0), LocalProgram(OptimizerSpec("sgd", lr=1))))

        @dc.dataclass
        class Mutable:
            lr: float

        self.assertFalse(_same_values(Mutable(1.0), Mutable(1.0)))
        mutable, nan = Mutable(1.0), float("nan")
        # One object is its own value only where it is an immutable scalar.
        self.assertFalse(_same_values(mutable, mutable))
        self.assertTrue(_same_values(nan, nan))
        self.assertTrue(_same_values(program(nan), program(nan)))
        # Whatever the order of first use (each type's fields are listed once),
        # the same values are the same reprs.
        values = (0.5, 0.25, 0.0, -0.0, float("nan"), float("inf"))
        for a, b in [(a, b) for a in values for b in values]:
            ours, theirs = program(a, momentum=b), program(b, momentum=a)
            same = repr(ours) == repr(theirs)
            self.assertEqual(_same_values(ours, theirs), same, (a, b))
        for rated in (False, True):
            rounds = SimpleNamespace(rated=rated, _held_program=None)
            with self.subTest(rated=rated):
                keys = []
                for value in (0.5, 0.5, -0.0, 0.0, 0.0):
                    made = program(value)
                    keys.append(ResidentRounds._program_key(rounds, made))  # type: ignore[arg-type]
                    shape = (
                        dc.replace(made, optimizer=dc.replace(made.optimizer, lr=0.0))
                        if rated
                        else made
                    )
                    self.assertEqual(keys[-1], repr(shape))
                self.assertIs(keys[1], keys[0])
                self.assertIs(keys[4], keys[3])
                self.assertEqual(keys[2] == keys[3], rated)


def _two_level_model() -> nn.Module:
    return nn.Sequential(nn.Linear(2, 3), nn.Sequential(nn.Dropout(0.5), nn.Linear(3, 1)))


@pytest.mark.fast
class SetTrainingTest(unittest.TestCase):
    """``set_training``: ``model.train(mode)``, skipped only where it would change nothing.

    Every module ends in the mode ``train`` leaves it in, from any mix; the
    walk is skipped where every module holds it, never where one has a
    ``train`` of its own.
    """

    def test_every_module_ends_in_the_mode(self) -> None:
        for mode in (True, False):
            for mixed in range(4):
                with self.subTest(mode=mode, mixed=mixed):
                    model = _two_level_model()
                    for index, module in enumerate(model.modules()):
                        module.training = bool((mixed >> (index % 2)) & 1)
                    set_training(model, mode)
                    self.assertEqual({m.training for m in model.modules()}, {mode})

    def test_the_walk_is_skipped_only_where_nothing_changes(self) -> None:
        model = _two_level_model()
        model.eval()
        with mock.patch.object(nn.Module, "train", autospec=True) as train:
            set_training(model, False)
            train.assert_not_called()
            set_training(model, True)
            train.assert_called_once_with(model, True)
        model[1][0].training = True
        with mock.patch.object(nn.Module, "train", autospec=True) as train:
            set_training(model, False)
            train.assert_called_once_with(model, False)

    def test_a_module_with_its_own_train_is_always_called(self) -> None:
        calls = []

        class Own(nn.Linear):
            def train(self, mode: bool = True) -> Own:
                calls.append(mode)
                return super().train(mode)

        model = nn.Sequential(Own(2, 2))
        model.eval()
        calls.clear()
        set_training(model, False)
        self.assertEqual(calls, [False])


@pytest.mark.fast
class AOneChunkRoundsSumsAreKeptTest(unittest.TestCase):
    """``_KeptSums``: the server's metric sums, kept while a round's columns are the last's.

    Every round's result is a fresh accumulator's over the same columns:
    kept for equal columns, counts and masks, summed again for any other,
    a NaN or a 0.0 where -0.0 was, and never put into an accumulator that
    already holds sums.
    """

    def test_kept_only_for_the_same_columns(self) -> None:
        from unittest import mock

        from fedbrew.core.resident import _KeptSums
        from fedbrew.servers.fedavg import WeightedMetricAccumulator

        def fresh(columns: Any, counts: Any, reported: Any) -> dict[str, float]:
            accumulator = WeightedMetricAccumulator()
            accumulator.add_columns(columns, counts, reported)
            return accumulator.result()

        kept = _KeptSums()
        rounds = [
            ({"a": [1.0, 2.0], "b": [0.5, 0.25]}, [3, 4], {}),
            ({"a": [1.0, 2.0], "b": [0.5, 0.25]}, [3, 4], {}),
            ({"a": [1.0, 2.5], "b": [0.5, 0.25]}, [3, 4], {}),
            ({"a": [1.0, 2.5], "b": [0.5, 0.25]}, [3, 5], {}),
            ({"a": [1.0, 2.5], "b": [0.5, 0.25]}, [3, 5], {"b": [True, False]}),
            ({"a": [-0.0, 0.0], "b": [0.5, 0.25]}, [3, 5], {}),
            ({"a": [0.0, 0.0], "b": [0.5, 0.25]}, [3, 5], {}),
            ({"a": [float("nan"), 1.0], "b": [0.5, 0.25]}, [3, 5], {}),
            ({"a": [float("nan"), 1.0], "b": [0.5, 0.25]}, [3, 5], {}),
        ]
        summed = mock.patch.object(
            WeightedMetricAccumulator,
            "add_columns",
            autospec=True,
            side_effect=WeightedMetricAccumulator.add_columns,
        )
        with summed as add_columns:
            for columns, counts, reported in rounds:
                accumulator = WeightedMetricAccumulator()
                kept.add_columns(accumulator, columns, counts, reported)
                self.assertEqual(repr(accumulator.result()), repr(fresh(columns, counts, reported)))
        # Summed again for every round but the second and the seventh (0.0 sums as -0.0).
        self.assertEqual(add_columns.call_count, len(rounds) + len(rounds) - 2)
        busy = WeightedMetricAccumulator()
        busy.add_columns({"a": [7.0]}, [1], {})
        kept.add_columns(busy, *rounds[0])
        expected = WeightedMetricAccumulator()
        expected.add_columns({"a": [7.0]}, [1], {})
        expected.add_columns(*rounds[0])
        self.assertEqual(repr(busy.result()), repr(expected.result()))


class TheWriterStagesCheckpointsTest(unittest.TestCase):
    """Each staged checkpoint is written to its temporary file by the writer, in order."""

    def test_a_path_staged_again_keeps_its_place_and_takes_the_later_payload(self) -> None:
        import tempfile

        from fedbrew.core.checkpointing import load_checkpoint
        from fedbrew.core.resident_flush import FlushWriter, WriterStaged

        writer = FlushWriter()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                staged = WriterStaged(writer)
                staged.stage({"round_id": 1}, root / "best.pt")
                staged.stage({"round_id": 1}, root / "latest.pt")
                staged.stage({"round_id": 2}, root / "best.pt")
                writer.wait()
                names = sorted(p.name for p in root.iterdir())
                self.assertEqual(names, ["best.pt.tmp", "latest.pt.tmp"])
                staged.written().commit()
                self.assertEqual(sorted(p.name for p in root.iterdir()), ["best.pt", "latest.pt"])
                self.assertEqual(load_checkpoint(root / "best.pt")["round_id"], 2)
        finally:
            writer.close()


@pytest.mark.fast
class TheFlushsWritesTest(unittest.TestCase):
    """The writer runs a flush's writes in order, and hands back what one raised."""

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

    def test_a_flush_is_waited_for_and_not_what_was_staged_after_it(self) -> None:
        """``wait_flush``: the last flush and what came before it, not the writes queued after."""

        import threading

        from fedbrew.core.resident_flush import FlushWriter

        writer = FlushWriter()
        release = threading.Event()
        done: list[str] = []
        try:
            writer.submit(lambda: done.append("staged before"))
            writer.submit(lambda: done.append("flush"), flush=True)
            writer.submit(lambda: (release.wait(10), done.append("staged after")))
            writer.wait_flush()
            self.assertEqual(done, ["staged before", "flush"])
            release.set()
            writer.wait()
            self.assertEqual(done, ["staged before", "flush", "staged after"])

            def fails() -> None:
                raise OSError("disk full")

            writer.submit(fails)
            writer.submit(lambda: done.append("skipped"), flush=True)
            with self.assertRaisesRegex(OSError, "disk full"):
                writer.wait_flush()
            self.assertNotIn("skipped", done)
        finally:
            release.set()
            writer.close()


def _cuda_usable() -> bool:
    """A CUDA device this process can allocate on."""

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
    """On CUDA: the resident round, eager and replayed from CUDA graphs, is the per-round path."""

    def _arms(self) -> Iterator[tuple[str, dict[str, Any], Any]]:
        from tests.test_batched_executor_tolerance import cnn_config, images

        mnist = classification_config(**FEDAVG, update_mode="single_batch")
        yield "mnist-like", mnist, nullcontext
        femnist = cnn_config(**FEDAVG, update_mode="single_batch")
        femnist["server"].update(participation_rate=None, participation_probability=0.6)

        @contextmanager
        def data() -> Iterator[None]:
            with images(), ragged():
                yield

        yield "femnist-like", femnist, data
        several = classification_config(**FEDAVG, update_mode="sequential_epoch")
        yield "several buckets", several, ragged

    def test_eager_and_replayed(self) -> None:
        for label, config, data in self._arms():
            with self.subTest(arm=label):
                config = _clean(config)
                config["schedule"]["rounds"] = 6
                config["runtime"]["device"] = "cuda"
                config["runtime"]["checkpointing"].update(save_every_round=True)
                with data():
                    eager = self.run_config(config, "batched")
                    graphed = self.run_config(config, "batched", cuda_graphs="on")
                    with per_round():
                        reference = self.run_config(config, "batched")
                self.assertEqual(_executor(eager)["rounds"], {"used": "resident"})
                self.assertSameRun(eager, reference)
                self.assertSameRun(graphed, reference)
                graphs = _executor(graphed)["cuda_graphs"]
                self.assertEqual(graphs["used"], "on", graphs)
                if label == "mnist-like":
                    # One shape every round: recorded at round 2, replayed from then on.
                    self.assertEqual((graphs["captured"], graphs["replayed"]), (1, 5))


class WhoTakesItTest(ResidentRuns):
    def test_a_rule_with_per_client_state_runs_per_round_and_says_why(self) -> None:
        config = classification_rule_config({"update_rule": "fedprox", "proximal_mu": 0.01})
        output = self.run_config(_clean(config), "batched")
        rounds = _executor(output)["rounds"]
        self.assertEqual(rounds["used"], "per_round")
        self.assertIn("keeps per-client state", rounds["reason"])

    def test_cuda_graphs_on_the_cpu_are_recorded_off(self) -> None:
        config = classification_config(**FEDAVG, update_mode="single_batch")
        config["runtime"].setdefault("performance", {})["cuda_graphs"] = "on"
        output = self.run_config(_clean(config), "batched")
        graphs = _executor(output)["cuda_graphs"]
        self.assertEqual(graphs["used"], "off")
        self.assertIn("need a CUDA device", graphs["fallback"])

    def test_cuda_graphs_without_the_batched_executor_are_refused(self) -> None:
        from fedbrew.core.refusal import RunRefused

        config = classification_config(**FEDAVG, update_mode="single_batch")
        config["runtime"].setdefault("performance", {})["cuda_graphs"] = "on"
        with self.assertRaisesRegex(RunRefused, "cuda_graphs is a mode of the batched"):
            self.run_config(_clean(config), "sequential")

    def test_the_sequential_executor_records_nothing_of_it(self) -> None:
        config = classification_config(**FEDAVG, update_mode="single_batch")
        output = self.run_config(_clean(config), "sequential")
        self.assertNotIn("rounds", _executor(output))


if __name__ == "__main__":
    unittest.main()
