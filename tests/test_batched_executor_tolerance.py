"""The batched executor computes what the sequential one computes, to summation order.

``runtime.performance.executor: batched`` trains a round's sampled clients
together (``fedbrew/core/batched_executor.py``). The same configuration is run
through both executors here, and what is compared is everything a run
computes: every round's global model (a checkpoint per round), every
persistent client state those checkpoints hold, and every non-timing cell of
the three CSVs. The design's tolerance (chapter 11 §9):

- ``|a - b| / max(|b|, 1e-300) <= 1e-12`` per CSV cell, and
  ``max |a - b| / max(max |b|, 1e-300) <= 1e-12`` per tensor, in float64;
  the identity and count columns -- ``round_id``, ``client_id``,
  ``num_clients``, ``num_examples``, ``optimizer_steps`` -- equal. A tensor
  is measured against its own scale rather than element by element because
  an element whose exact value is 0 holds rounding residue: on fed-lasso's
  first ``full_gradient`` round one coordinate is 1.08e-19 sequentially and
  9.5e-20 batched, beside a largest coordinate of 1.2e-2 (measured
  2026-09-27) -- 1.2e-18 of the tensor, and 0.13 of the element;
- **bit-identical** when every chunk holds one client
  (``executor_chunk_bytes: 1``) or a round samples one: nothing is then
  vmapped, and the batched arithmetic is the sequential arithmetic;
- the same clients selected at participation below 1;
- local iterations above 1, over shuffled batches, on every run below.

The matrix, per update rule, is the five linear examples' shipped arms, every
FedAvg update mode (each frozen weighting, with and without ``max_grad_norm``),
ragged clients whose split sizes differ (padded, masked batches and buckets of
different shapes), and partial participation.
"""

from __future__ import annotations

import copy
import csv
import math
import tempfile
import unittest
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest import mock

import torch
import yaml

from fedbrew.core.runner import run
from fedbrew.data.manifest_dataset import ManifestFederatedDataset
from tests.test_reproducibility import TIMING

REPO_ROOT = Path(__file__).resolve().parent.parent
ROUNDS = 4
TOLERANCE = 1e-12
EXAMPLES = ("fed-lasso", "drift-quad", "simplex-lsq", "nonconvex-simplex", "pl-1d")
#: Columns that are identities or counts, compared for equality.
EXACT_COLUMNS = {"round_id", "client_id", "phase", "num_clients", "num_examples", "optimizer_steps"}
CSVS = ("round_metrics.csv", "client_update_metrics.csv", "client_metrics.csv")

_generated: dict[str, Path] = {}
_root = tempfile.TemporaryDirectory()


def example_manifest(name: str) -> Path:
    """The example's data, generated once per process from its shipped generator config."""

    if name not in _generated:
        from fedbrew.data.generate import generate_from_config

        generator = yaml.safe_load(
            (REPO_ROOT / "data" / "configs" / "examples" / f"{name}.yaml").read_text()
        )
        generator["dataset"]["output_dir"] = str(Path(_root.name) / "data" / name)
        path = Path(_root.name) / f"{name}-data.yaml"
        path.write_text(yaml.safe_dump(generator), encoding="utf-8")
        _generated[name] = Path(generate_from_config(path))
    return _generated[name]


def example_config(name: str, arm: str = "fedavg") -> dict[str, Any]:
    """A shipped arm of a linear example, on this process's data, checkpointing every round."""

    config = yaml.safe_load((REPO_ROOT / "configs" / "examples" / name / f"{arm}.yaml").read_text())
    config["data"]["path"] = str(example_manifest(name))
    config["defaults"]["global_rounds"] = ROUNDS
    config["runtime"]["quiet"] = True
    config["runtime"]["checkpointing"].update(
        enabled=True, save_last=True, save_every_round=True, keep_last=None
    )
    config["client_statistics"] = {**config.get("client_statistics", {}), "per_client_csv": True}
    return config


def ragged_clients() -> Any:
    """Train splits of 16, 13, 10, 7 and 4 rows, by client number: ragged batches and buckets."""

    real = ManifestFederatedDataset.get_client_data

    def truncated(self: ManifestFederatedDataset, client_id: str) -> dict[str, Any]:
        shard = real(self, client_id)
        keep = 16 - 3 * (int(client_id.rsplit("_", 1)[-1]) % 5)
        train = {**shard["train"]}
        for key in ("x", "y"):
            if key in train:
                train[key] = train[key][:keep]
        return {**shard, "train": train}

    return mock.patch.object(ManifestFederatedDataset, "get_client_data", truncated)


class ExecutorRuns(unittest.TestCase):
    """Runs one configuration through both executors and compares what they wrote."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self._count = 0

    def tearDown(self) -> None:
        self._directory.cleanup()

    def run_config(self, config: dict[str, Any], executor: str, **performance: Any) -> Path:
        self._count += 1
        config = copy.deepcopy(config)
        output = self.root / f"run{self._count}-{executor}"
        config["experiment"]["output_dir"] = str(output)
        config["runtime"].setdefault("performance", {}).update(executor=executor, **performance)
        path = self.root / f"run{self._count}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        run(path, args=None)
        return output

    def both(
        self,
        config: dict[str, Any],
        *,
        data: Callable[[], Any] = nullcontext,
        **performance: Any,
    ) -> tuple[Path, Path]:
        """(batched, sequential) output directories for one configuration."""

        with data():
            sequential = self.run_config(config, "sequential")
            batched = self.run_config(config, "batched", **performance)
        return batched, sequential

    def assertAgree(self, batched: Path, sequential: Path, *, exact: bool = False) -> None:
        """Every non-timing cell and every checkpoint agree, to TOLERANCE or bit for bit."""

        for name in CSVS:
            self._compare_csv(batched / name, sequential / name, exact)
        checkpoints = sorted((sequential / "checkpoints").glob("round_*.pt"))
        self.assertEqual(len(checkpoints), ROUNDS)
        for path in checkpoints:
            self._compare_checkpoint(batched / "checkpoints" / path.name, path, exact)

    def _compare_csv(self, batched: Path, sequential: Path, exact: bool) -> None:
        if not sequential.exists():
            self.assertFalse(batched.exists(), batched.name)
            return
        rows_b, rows_s = _rows(batched), _rows(sequential)
        self.assertEqual(len(rows_b), len(rows_s), batched.name)
        for number, (row_b, row_s) in enumerate(zip(rows_b, rows_s, strict=True)):
            self.assertEqual(list(row_b), list(row_s), batched.name)
            for column, value_s in row_s.items():
                if column in TIMING or value_s == row_b[column]:
                    continue
                where = f"{batched.name} row {number} {column}: {row_b[column]} vs {value_s}"
                self.assertFalse(exact or column in EXACT_COLUMNS, where)
                self.assertLessEqual(
                    _relative(float(row_b[column]), float(value_s)), TOLERANCE, where
                )

    def _compare_checkpoint(self, batched: Path, sequential: Path, exact: bool) -> None:
        loaded_b = torch.load(batched, weights_only=False)
        loaded_s = torch.load(sequential, weights_only=False)
        pairs = [("model_state", loaded_b["model_state"], loaded_s["model_state"])]
        for client, state in loaded_s["client_states"].items():
            for key, value in state.items():
                if isinstance(value, dict):
                    pairs.append((f"{client}.{key}", loaded_b["client_states"][client][key], value))
        for where, states_b, states_s in pairs:
            self.assertEqual(list(states_b), list(states_s), where)
            for key, tensor_s in states_s.items():
                tensor_b = states_b[key]
                label = f"{sequential.name} {where}.{key}"
                if exact:
                    self.assertTrue(torch.equal(tensor_b, tensor_s), label)
                else:
                    scale = max(float(tensor_s.abs().max()), 1e-300)
                    error = float((tensor_b - tensor_s).abs().max()) / scale
                    self.assertLessEqual(error, TOLERANCE, label)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _relative(a: float, b: float) -> float:
    if math.isnan(a) and math.isnan(b):
        return 0.0
    return abs(a - b) / max(abs(b), 1e-300)


def fedavg_modes() -> Iterator[tuple[str, dict[str, Any]]]:
    """Every FedAvg update mode, each frozen weighting, with and without clipping."""

    for mode in ("single_batch", "sequential_epoch", "frozen_batch_gradients", "full_gradient"):
        weightings = (
            ("examples", "uniform", "sum") if mode == "frozen_batch_gradients" else ("examples",)
        )
        for weighting in weightings:
            for clip in (None, 0.05):
                client = {
                    "update_mode": mode,
                    "frozen_gradient_weighting": weighting,
                    "batch_size": 3,
                }
                if clip is not None:
                    client["max_grad_norm"] = clip
                yield f"{mode}/{weighting}/clip={clip}", client


def with_client(config: dict[str, Any], **client: Any) -> dict[str, Any]:
    edited = copy.deepcopy(config)
    edited["client"].update(client)
    return edited


class EveryLinearExampleTest(ExecutorRuns):
    """The shipped FedAvg arm of each: three shuffled epochs a round, eight clients."""

    def test_each_example(self) -> None:
        for name in EXAMPLES:
            with self.subTest(example=name):
                self.assertAgree(*self.both(example_config(name)))

    def test_each_example_one_client_per_chunk_is_bit_identical(self) -> None:
        for name in EXAMPLES:
            with self.subTest(example=name):
                batched, sequential = self.both(example_config(name), executor_chunk_bytes=1)
                self.assertAgree(batched, sequential, exact=True)


class EveryFedAvgModeTest(ExecutorRuns):
    """fed-lasso at batch size 3: batches of 3, 3, 3, 3, 3 and 1."""

    def test_every_mode(self) -> None:
        for label, client in fedavg_modes():
            with self.subTest(mode=label):
                self.assertAgree(*self.both(with_client(example_config("fed-lasso"), **client)))

    def test_every_mode_one_client_per_chunk_is_bit_identical(self) -> None:
        for label, client in fedavg_modes():
            with self.subTest(mode=label):
                config = with_client(example_config("fed-lasso"), **client)
                self.assertAgree(*self.both(config, executor_chunk_bytes=1), exact=True)


class RaggedClientsTest(ExecutorRuns):
    """Clients of 16, 13, 10, 7 and 4 rows: padded batches, masks, and several buckets."""

    def test_every_mode(self) -> None:
        for label, client in fedavg_modes():
            with self.subTest(mode=label):
                config = with_client(example_config("fed-lasso"), **client)
                self.assertAgree(*self.both(config, data=ragged_clients))

    def test_one_client_per_chunk_is_bit_identical(self) -> None:
        for label, client in fedavg_modes():
            with self.subTest(mode=label):
                config = with_client(example_config("fed-lasso"), **client)
                batched, sequential = self.both(config, data=ragged_clients, executor_chunk_bytes=1)
                self.assertAgree(batched, sequential, exact=True)


class PartialParticipationTest(ExecutorRuns):
    def test_the_same_clients_are_selected(self) -> None:
        config = example_config("fed-lasso")
        config["server"]["participation_rate"] = 0.5
        batched, sequential = self.both(config)
        # client_id is an exact column, so the selection is compared row by row.
        self.assertAgree(batched, sequential)
        per_round: dict[str, int] = {}
        for row in _rows(batched / "client_update_metrics.csv"):
            per_round[row["round_id"]] = per_round.get(row["round_id"], 0) + 1
        self.assertEqual(list(per_round.values()), [4] * ROUNDS)

    def test_one_sampled_client_is_bit_identical(self) -> None:
        config = example_config("fed-lasso")
        config["server"]["participation_rate"] = 0.125
        self.assertAgree(*self.both(config), exact=True)


class BatchedRunsAreDeterministicTest(ExecutorRuns):
    def test_two_batched_runs_are_identical(self) -> None:
        config = with_client(example_config("fed-lasso"), update_mode="single_batch")
        with ragged_clients():
            first = self.run_config(config, "batched")
            second = self.run_config(config, "batched")
        self.assertAgree(first, second, exact=True)


if __name__ == "__main__":
    unittest.main()
