"""Evaluation on a cadence, the post-fit fit_ pass included, moves nothing it does not measure.

The client splits and the central test set already had schedules
(evaluation.<split>.every); the pass each training client makes over its own
train split after its local update, which is where fit_loss and fit_accuracy
come from, ran every round, and was 18% of an MNIST MLP round
(measured on 2026-09-26). evaluation.fit.every schedules it too. On a skipped round
the client still takes the example count the pass would have taken -- its
aggregation weight -- by iterating the same loader and reading each batch's
count off the batch (TaskAdapter.evaluation_total). What is pinned here, with
every schedule at c against every schedule at 1:

- the training trajectory is bit-identical: every checkpoint's model, server
  and client state and RNG position, and every round column no evaluation
  produces;
- the evaluated rows are identical: each evaluated cell is the every-round
  run's, on exactly the rounds the schedule names;
- the count a skipped round takes is the pass's count, without a forward pass,
  for the classification task and for every example task;
- an injected blow-up stops the run within c rounds of where every-round
  evaluation stops it, and a non-finite client state is still refused on the
  round it arrives;
- evaluation.fit.every takes the split schedules' values, and 'never' is
  refused while the divergence monitor watches a fit_ metric.
"""

from __future__ import annotations

import copy
import csv
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import torch
import yaml

from fedbrew.clients.torch_sgd_client import TorchSGDClient
from fedbrew.core import loop
from fedbrew.core.config import evaluates_round, standalone_config_mapping
from fedbrew.core.refusal import RunRefused
from fedbrew.core.runner import run
from fedbrew.servers.fedavg import FedAvgServer
from tests.test_reproducibility import TIMING, _config

ROUNDS = 9
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Round columns an evaluation pass produces; every other one comes from training.
_EVALUATED = ("train_", "val_", "test_", "central_test_", "fit_", "personal_")

#: Which schedule each evaluated column follows.
_SCHEDULES = {"fit_": "fit", "central_test_": "central_test"}


def _write(root: Path, name: str, every: int | str, **extra: Any) -> Path:
    config = _config(root / name, rounds=ROUNDS, checkpoint=True)
    config["runtime"]["checkpointing"]["keep_last"] = None
    config.setdefault("reporting", {})["per_client_csv"] = True
    config["evaluation"] = {
        "train": {"every": every, "clients": "all"},
        "val": {"every": every, "clients": "all"},
        "test": {"every": every, "clients": "all"},
        "central_test": {"every": every},
        "fit": {"every": every},
    }
    config.update(extra)
    path = root / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _run(path: Path) -> Path:
    run(path, args=None)
    return Path(yaml.safe_load(path.read_text())["experiment"]["output_dir"])


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _bits(value: Any) -> Any:
    """A checkpoint value with every tensor and array as its bytes, for ==."""

    if isinstance(value, torch.Tensor):
        return (str(value.dtype), tuple(value.shape), bytes(value.reshape(-1).view(torch.uint8)))
    if hasattr(value, "tobytes") and hasattr(value, "dtype"):
        return (str(value.dtype), value.tobytes())
    if isinstance(value, dict):
        return {key: _bits(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_bits(item) for item in value)
    return value


def _trajectory(output_dir: Path) -> dict[str, Any]:
    """Every checkpoint but the round's metrics, and every column no evaluation writes."""

    checkpoints = {}
    for path in sorted((output_dir / "checkpoints").glob("round_*.pt")):
        checkpoint = torch.load(path, weights_only=False)
        checkpoint.pop("metrics")
        checkpoints[path.name] = _bits(checkpoint)
    rounds = [
        {
            key: value
            for key, value in row.items()
            if key not in TIMING and not key.startswith(_EVALUATED)
        }
        for row in _rows(output_dir / "round_metrics.csv")
    ]
    updates = [
        {key: value for key, value in row.items() if not key.startswith("fit_")}
        for row in _rows(output_dir / "client_update_metrics.csv")
    ]
    return {"checkpoints": checkpoints, "rounds": rounds, "updates": updates}


def _schedule_of(column: str) -> str:
    for prefix, schedule in _SCHEDULES.items():
        if column.startswith(prefix):
            return schedule
    return "split"


class _TempRoot(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)


class TheCadenceMovesNothingItDoesNotMeasureTest(_TempRoot):
    @classmethod
    def setUpClass(cls) -> None:
        cls._class_tmp = tempfile.TemporaryDirectory()
        root = Path(cls._class_tmp.name)
        cls.outputs = {every: _run(_write(root, f"every{every}", every)) for every in (1, 3, 4)}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._class_tmp.cleanup()

    def test_the_training_trajectory_is_bit_identical(self) -> None:
        reference = _trajectory(self.outputs[1])
        self.assertEqual(len(reference["checkpoints"]), ROUNDS)
        for every in (3, 4):
            with self.subTest(every=every):
                self.assertEqual(_trajectory(self.outputs[every]), reference)

    def test_the_evaluated_rows_are_the_every_round_runs(self) -> None:
        reference = _rows(self.outputs[1] / "round_metrics.csv")
        evaluated = [
            key for key in reference[0] if key.startswith(_EVALUATED) and key not in TIMING
        ]
        self.assertTrue(any(key.startswith("fit_") for key in evaluated))
        self.assertTrue(any(key.startswith("central_test_") for key in evaluated))
        for every in (3, 4):
            rows = _rows(self.outputs[every] / "round_metrics.csv")
            for round_id, (row, full) in enumerate(zip(rows, reference, strict=True), start=1):
                due = evaluates_round(every, round_id, ROUNDS)
                for key in evaluated:
                    with self.subTest(every=every, round=round_id, column=key):
                        self.assertEqual(row.get(key, ""), full[key] if due else "")

    def test_the_per_client_rows_are_the_every_round_runs(self) -> None:
        reference = _rows(self.outputs[1] / "client_metrics.csv")
        updates = _rows(self.outputs[1] / "client_update_metrics.csv")
        for every in (3, 4):
            with self.subTest(every=every):
                due = {r for r in range(1, ROUNDS + 1) if evaluates_round(every, r, ROUNDS)}
                self.assertEqual(
                    _rows(self.outputs[every] / "client_metrics.csv"),
                    [row for row in reference if int(row["round_id"]) in due],
                )
                for row, full in zip(
                    _rows(self.outputs[every] / "client_update_metrics.csv"), updates, strict=True
                ):
                    for key in (key for key in full if key.startswith("fit_")):
                        expected = full[key] if int(full["round_id"]) in due else ""
                        self.assertEqual(row[key], expected)


@contextmanager
def _counts_checked(test: unittest.TestCase, seen: list[int]):
    """Check each skipped round's count against the pass's, and that no forward ran."""

    real = TorchSGDClient._evaluated_example_count

    def checked(self: TorchSGDClient, model: Any, data: Any, round_id: int) -> int:
        refused = AssertionError("a skipped post-fit pass ran eval_step")
        with mock.patch.object(type(self.task), "eval_step", side_effect=refused):
            count = real(self, model, data, round_id)
        test.assertEqual(count, self._evaluate_model(model, data, round_id=round_id)[1])
        seen.append(count)
        return count

    with mock.patch.object(TorchSGDClient, "_evaluated_example_count", checked):
        yield


class ASkippedRoundCountsWithoutEvaluatingTest(_TempRoot):
    def test_the_classification_task(self) -> None:
        seen: list[int] = []
        with _counts_checked(self, seen):
            _run(_write(self.root, "classification", "final"))
        self.assertEqual(len(seen), 4 * (ROUNDS - 1))

    def test_every_example_task(self) -> None:
        from fedbrew.data.generate import generate_from_config

        for name in ("fed-lasso", "drift-quad", "simplex-lsq", "nonconvex-simplex", "pl-1d"):
            with self.subTest(example=name):
                generator = yaml.safe_load(
                    (REPO_ROOT / "data" / "configs" / "examples" / f"{name}.yaml").read_text()
                )
                generator["dataset"]["output_dir"] = str(self.root / "data" / name)
                generator_path = self.root / f"{name}-data.yaml"
                generator_path.write_text(yaml.safe_dump(generator), encoding="utf-8")
                manifest = generate_from_config(generator_path)

                config = standalone_config_mapping(
                    REPO_ROOT / "configs" / "examples" / name / "fedavg.yaml"
                )
                config["experiment"]["output_dir"] = str(self.root / "runs" / name)
                config["data"]["path"] = str(manifest)
                config["schedule"]["rounds"] = 3
                config.setdefault("evaluation", {})["fit"] = {"every": "final"}
                # The count is TorchSGDClient's, which the sequential executor runs.
                config["runtime"].setdefault("performance", {})["executor"] = "sequential"
                path = self.root / f"{name}.yaml"
                path.write_text(yaml.safe_dump(config), encoding="utf-8")

                seen: list[int] = []
                with _counts_checked(self, seen):
                    _run(path)
                self.assertTrue(seen, "no round skipped the pass")


def _blow_up_from(round_id: int):
    """FedAvg's aggregate, with the global model scaled up a hundredfold at round_id."""

    real = FedAvgServer.aggregate_stream

    def aggregate(self: FedAvgServer, round_info: Any, results: Any) -> Any:
        payload = real(self, round_info, results)
        if round_info.round_id == round_id:
            self._model_state = {
                key: value * 100.0 if value.is_floating_point() else value
                for key, value in self._model_state.items()
            }
            payload = dict(payload, model_state=self._model_state)
        return payload

    return aggregate


class ABlowUpIsCaughtWithinTheCadenceTest(_TempRoot):
    DIVERGENCE = {"metric": "fit_loss", "non_finite": True, "blowup_absolute": 50.0}

    def _stopped_at(self, every: int) -> dict[str, Any]:
        path = _write(self.root, f"blown{every}", every, divergence=dict(self.DIVERGENCE))
        with mock.patch.object(FedAvgServer, "aggregate_stream", _blow_up_from(4)):
            output_dir = _run(path)
        return yaml.safe_load((output_dir / "run.json").read_text())["termination"]

    def test_within_c_rounds(self) -> None:
        reference = self._stopped_at(1)
        self.assertIn(reference["detector"], {"blowup_absolute", "non_finite"})
        for every in (3, 4):
            with self.subTest(every=every):
                stopped = self._stopped_at(every)
                self.assertIn(stopped["detector"], {"blowup_absolute", "non_finite"})
                self.assertGreaterEqual(stopped["round_id"], reference["round_id"])
                self.assertLess(stopped["round_id"], reference["round_id"] + every)

    def test_a_non_finite_client_state_is_refused_on_its_round(self) -> None:
        real = loop._fit_client

        def poisoned(client: Any, request: Any) -> Any:
            result = real(client, request)
            if request.round_id == 2:
                state = copy.copy(result.payload["model_state"])
                key = next(iter(state))
                state[key] = torch.full_like(state[key], float("nan"))
                result.payload = dict(result.payload, model_state=state)
            return result

        with mock.patch.object(loop, "_fit_client", poisoned):
            output_dir = _run(_write(self.root, "poisoned", 4))
        termination = yaml.safe_load((output_dir / "run.json").read_text())["termination"]
        self.assertEqual(termination["detector"], "non_finite_client_state")
        self.assertEqual(termination["round_id"], 2)


class TheSettingTest(_TempRoot):
    def _path(self, every: Any, divergence: dict[str, Any]) -> Path:
        path = _write(self.root, "setting", 1, divergence=divergence)
        raw = yaml.safe_load(path.read_text())
        raw["evaluation"]["fit"] = {"every": every}
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return path

    def test_the_default_is_every_round(self) -> None:
        from fedbrew.core.config import EvaluationConfig

        self.assertEqual(EvaluationConfig().fit.every, 1)

    def test_the_split_schedules_values(self) -> None:
        from fedbrew.core.config import load_config

        off = {"non_finite": False, "blowup_factor": None}
        for good in (1, 5, "final", "never"):
            with self.subTest(every=good):
                self.assertEqual(load_config(self._path(good, off)).evaluation.fit.every, good)
        for bad in (0, -2, True, "sometimes"):
            with self.subTest(every=bad), self.assertRaises(RunRefused) as refused:
                load_config(self._path(bad, off))
            self.assertIn("evaluation.fit", str(refused.exception))

    def test_never_is_refused_while_the_monitor_watches_a_fit_metric(self) -> None:
        from fedbrew.core.config import load_config

        watching = {"metric": "fit_loss", "non_finite": True}
        with self.assertRaises(RunRefused) as refused:
            load_config(self._path("never", watching))
        self.assertIn("divergence.metric", str(refused.exception))


if __name__ == "__main__":
    unittest.main()
