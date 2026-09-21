"""Tests for configurable post-aggregation global-model evaluation."""

from __future__ import annotations

import csv
import statistics
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch

from fedbrew.clients.base import ClientUpdate
from fedbrew.clients.lazy_pool import LazyClientPool
from fedbrew.clients.torch_sgd_client import _get_evaluation_split
from fedbrew.core.artifacts import (
    load_client_metrics_csv,
    save_client_metrics_csv,
)
from fedbrew.core.config import (
    CentralTestConfig,
    ClientStatisticsConfig,
    EvaluationConfig,
    SplitEvaluationConfig,
)
from fedbrew.core.loop import (
    _aggregate_client_split_metrics,
    _evaluate_central_test_set,
    run_fl_loop,
)
from fedbrew.core.protocol import (
    ClientInfo,
    EvalRequest,
    EvalResult,
    FitRequest,
    FitResult,
    RoundInfo,
)
from fedbrew.core.state import ClientEvaluationRecord
from fedbrew.data.dataset import FederatedDataset
from fedbrew.data.synthetic_classification import SyntheticClassificationDataset
from fedbrew.servers.base import ServerStrategy

pytestmark = pytest.mark.fast

_CLIENT_METRICS = {
    "client_0": {
        "train_loss": 1.0,
        "test_loss": 2.0,
        "train_accuracy": 0.5,
        "test_accuracy": 0.25,
    },
    "client_1": {
        "train_loss": 3.0,
        "test_loss": 4.0,
        "train_accuracy": 1.0,
        "test_accuracy": 0.5,
    },
    "client_2": {
        "train_loss": 5.0,
        "test_loss": 6.0,
        "train_accuracy": 0.0,
        "test_accuracy": 1.0,
    },
}

_SPLIT_COUNTS = {
    "client_0": {"train": 2, "test": 4},
    "client_1": {"train": 6, "test": 2},
    "client_2": {"train": 2, "test": 4},
}


class _EvaluationDataset(FederatedDataset):
    def list_clients(self) -> list[str]:
        return list(_CLIENT_METRICS)

    def get_client_data(self, client_id: str) -> dict[str, Any]:
        raise AssertionError("the fake clients already own their data")

    def get_client_metadata(self, client_id: str) -> dict[str, Any]:
        counts = _SPLIT_COUNTS[client_id]
        return {
            "client_id": client_id,
            "num_examples": counts["train"] + counts["test"],
            "num_train_examples": counts["train"],
            "num_eval_examples": counts["test"],
        }

    def get_global_data(self) -> Any:
        raise AssertionError("central global data must not be used")

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "evaluation-test"}


class _GlobalEvaluationDataset(_EvaluationDataset):
    def __init__(self) -> None:
        self.global_data_requests = 0
        self.global_test_data = {"scope": "central-global-test"}

    def get_global_data(self) -> dict[str, str]:
        self.global_data_requests += 1
        return self.global_test_data


class _MissingGlobalTestDataset(_EvaluationDataset):
    def get_global_data(self) -> None:
        return None


class _EvaluationClient(ClientUpdate):
    def __init__(self, client_id: str) -> None:
        self.client_id = client_id
        self.evaluated_model_versions: list[int] = []
        self.evaluated_splits: list[list[str]] = []

    def setup(self, client_info: ClientInfo) -> None:
        self.client_id = client_info.client_id

    def fit(self, request: FitRequest) -> FitResult:
        return FitResult(
            round_id=request.round_id,
            client_id=self.client_id,
            num_examples=_SPLIT_COUNTS[self.client_id]["train"],
            payload={"model_state": {"version": request.round_id}},
            metrics={"fit_loss": 99.0, "fit_accuracy": 0.0, "local_steps": 7.0},
        )

    def evaluate(self, request: EvalRequest) -> EvalResult:
        splits = self.assert_global_request(request)
        model_state = request.payload["model_state"]
        version = int(model_state["version"])
        self.evaluated_model_versions.append(version)
        self.evaluated_splits.append(list(splits))
        counts = {split: _SPLIT_COUNTS[self.client_id][split] for split in splits}
        metrics = {
            name: value
            for name, value in _CLIENT_METRICS[self.client_id].items()
            if any(name.startswith(f"{split}_") for split in splits)
        }
        return EvalResult(
            round_id=request.round_id,
            client_id=self.client_id,
            num_examples=sum(counts.values()),
            metrics=metrics,
            payload={
                "model_scope": "global",
                "num_examples_by_split": counts,
            },
        )

    def assert_global_request(self, request: EvalRequest) -> list[str]:
        if request.payload.get("model_scope") != "global":
            raise AssertionError("expected global model evaluation")
        splits = request.payload.get("splits")
        if splits not in (["train"], ["train", "test"]):
            raise AssertionError("expected strict client evaluation splits")
        if request.payload.get("metrics") != ["loss", "accuracy"]:
            raise AssertionError("expected loss and accuracy")
        model_state = request.payload.get("model_state")
        if not isinstance(model_state, dict):
            raise AssertionError("expected aggregated model state")
        if model_state.get("version") != request.round_id:
            raise AssertionError("evaluation did not use the newly aggregated model")
        return list(splits)

    def get_state(self) -> dict[str, Any]:
        return {"client_id": self.client_id}

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.client_id = str(state.get("client_id", self.client_id))


class _MissingClientTestClient(_EvaluationClient):
    def evaluate(self, request: EvalRequest) -> EvalResult:
        splits = request.payload.get("splits", [])
        if "test" in splits:
            raise ValueError(f"client {self.client_id!r} has no non-empty test split")
        return super().evaluate(request)


class _NoAccuracyClient(_EvaluationClient):
    """A task with no notion of correct/incorrect: loss only, every round.

    The loop still asks for ``["loss", "accuracy"]`` -- that request shape
    is unconditional -- but nothing requires the client to answer with both.
    """

    def evaluate(self, request: EvalRequest) -> EvalResult:
        result = super().evaluate(request)
        metrics = {name: value for name, value in result.metrics.items() if "accuracy" not in name}
        return EvalResult(
            round_id=result.round_id,
            client_id=result.client_id,
            num_examples=result.num_examples,
            metrics=metrics,
            payload=result.payload,
        )


class _EvaluationServer(ServerStrategy):
    def initialize(self) -> dict[str, Any]:
        return {"model_state": {"version": 0}}

    def configure_round(
        self,
        round_info: RoundInfo,
        clients: Sequence[ClientInfo],
    ) -> Sequence[FitRequest]:
        return [
            FitRequest(
                round_id=round_info.round_id,
                client_id=clients[0].client_id,
                payload={"model_state": {"version": round_info.round_id - 1}},
            )
        ]

    def aggregate(
        self,
        round_info: RoundInfo,
        results: Sequence[FitResult],
    ) -> dict[str, Any]:
        if len(results) != 1:
            raise AssertionError("expected partial participation")
        round_info.metrics.update({"fit_loss": 99.0, "fit_accuracy": 0.0, "local_steps": 7.0})
        return {
            "model_state": {"version": round_info.round_id},
            "metrics": dict(round_info.metrics),
        }

    def evaluate(
        self,
        round_info: RoundInfo,
        results: Sequence[EvalResult],
    ) -> dict[str, float]:
        return {}

    def save_state(self) -> dict[str, Any]:
        return {}

    def load_state(self, state: Mapping[str, Any]) -> None:
        return None


class _GlobalEvaluationServer(_EvaluationServer):
    def __init__(self) -> None:
        self.evaluated_global_data: list[Any] = []

    def evaluate_global(self, global_data: Any) -> dict[str, float]:
        self.evaluated_global_data.append(global_data)
        return {"global_loss": 1.25, "global_accuracy": 0.75}


class GlobalEvaluationTests(unittest.TestCase):
    def test_round_metrics_use_all_clients_and_split_specific_weights(self) -> None:
        clients = {client_id: _EvaluationClient(client_id) for client_id in _CLIENT_METRICS}

        state = run_fl_loop(
            _EvaluationServer(),
            clients,
            _EvaluationDataset(),
            global_rounds=1,
            checkpointing={"enabled": False},
            evaluation=EvaluationConfig(
                train=SplitEvaluationConfig(every=1, clients="all"),
                val=SplitEvaluationConfig(every="never", clients="all"),
                test=SplitEvaluationConfig(every=1, clients="all"),
                central_test=CentralTestConfig(every="never"),
            ),
        )

        metrics = state.metrics_history[0].metrics
        self.assertAlmostEqual(metrics["train_loss_sample_weighted_avg"], 3.0)
        self.assertAlmostEqual(metrics["train_accuracy_sample_weighted_avg"], 0.7)
        self.assertAlmostEqual(metrics["test_loss_sample_weighted_avg"], 4.0)
        self.assertAlmostEqual(metrics["test_loss_avg"], 4.0)
        self.assertAlmostEqual(metrics["test_accuracy_sample_weighted_avg"], 0.6)
        self.assertAlmostEqual(metrics["test_accuracy_avg"], 7.0 / 12.0)
        self.assertAlmostEqual(
            metrics["test_accuracy_std"],
            statistics.pstdev([0.25, 0.5, 1.0]),
        )
        self.assertAlmostEqual(metrics["test_accuracy_min"], 0.25)
        self.assertAlmostEqual(metrics["test_accuracy_worst10"], 0.25)
        self.assertEqual(metrics["fit_loss"], 99.0)
        self.assertEqual(metrics["fit_accuracy"], 0.0)
        self.assertEqual(metrics["local_steps"], 7.0)
        for client in clients.values():
            self.assertEqual(client.evaluated_model_versions, [1])
            self.assertEqual(client.evaluated_splits, [["train", "test"]])

    def test_global_test_only_keeps_client_train_evaluation(self) -> None:
        clients = {client_id: _EvaluationClient(client_id) for client_id in _CLIENT_METRICS}
        dataset = _GlobalEvaluationDataset()
        server = _GlobalEvaluationServer()

        state = run_fl_loop(
            server,
            clients,
            dataset,
            global_rounds=1,
            checkpointing={"enabled": False},
            evaluation=EvaluationConfig(
                train=SplitEvaluationConfig(every=1, clients="all"),
                val=SplitEvaluationConfig(every="never", clients="all"),
                test=SplitEvaluationConfig(every="never", clients="all"),
                central_test=CentralTestConfig(every=1),
            ),
        )

        metrics = state.metrics_history[0].metrics
        self.assertEqual(
            set(metrics),
            {
                *(
                    f"train_{metric}_{statistic}"
                    for metric in ("loss", "accuracy")
                    for statistic in (
                        "sample_weighted_avg",
                        "avg",
                        "std",
                        "min",
                        "max",
                        "worst10",
                    )
                ),
                # Beside the twelve aggregates, how many clients they are
                # over -- three here, all reporting a non-empty train split.
                # P07-F06.
                "train_num_clients",
                "central_test_loss",
                "central_test_accuracy",
                "fit_loss",
                "fit_accuracy",
                "local_steps",
            },
        )
        self.assertEqual(metrics["train_num_clients"], 3.0)
        self.assertAlmostEqual(metrics["train_loss_sample_weighted_avg"], 3.0)
        self.assertAlmostEqual(metrics["train_accuracy_sample_weighted_avg"], 0.7)
        self.assertAlmostEqual(metrics["central_test_loss"], 1.25)
        self.assertAlmostEqual(metrics["central_test_accuracy"], 0.75)
        self.assertEqual(server.evaluated_global_data, [dataset.global_test_data])
        self.assertEqual(dataset.global_data_requests, 1)
        self.assertEqual(len(state.client_metrics_history), 3)
        records = {record.client_id: record for record in state.client_metrics_history}
        self.assertEqual(records["client_1"].train_num_examples, 6)
        self.assertEqual(records["client_1"].global_model_train_loss, 3.0)
        self.assertEqual(records["client_1"].global_model_train_accuracy, 1.0)
        for record in records.values():
            self.assertEqual(record.test_num_examples, 0)
            self.assertIsNone(record.global_model_test_loss)
            self.assertIsNone(record.global_model_test_accuracy)
        for client in clients.values():
            self.assertEqual(client.evaluated_model_versions, [1])
            self.assertEqual(client.evaluated_splits, [["train"]])

    def test_one_run_reports_global_and_client_test_metrics(self) -> None:
        clients = {client_id: _EvaluationClient(client_id) for client_id in _CLIENT_METRICS}
        dataset = _GlobalEvaluationDataset()
        server = _GlobalEvaluationServer()

        state = run_fl_loop(
            server,
            clients,
            dataset,
            global_rounds=1,
            checkpointing={"enabled": False},
            evaluation=EvaluationConfig(
                train=SplitEvaluationConfig(every=1, clients="all"),
                val=SplitEvaluationConfig(every="never", clients="all"),
                test=SplitEvaluationConfig(every=1, clients="all"),
                central_test=CentralTestConfig(every=1),
            ),
        )

        metrics = state.metrics_history[0].metrics
        self.assertAlmostEqual(metrics["central_test_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["central_test_loss"], 1.25)
        self.assertAlmostEqual(metrics["test_accuracy_sample_weighted_avg"], 0.6)
        self.assertAlmostEqual(metrics["test_accuracy_avg"], 7.0 / 12.0)
        self.assertEqual(server.evaluated_global_data, [dataset.global_test_data])
        for evaluated_client in clients.values():
            self.assertEqual(
                evaluated_client.evaluated_splits,
                [["train", "test"]],
            )

    def test_central_pass_widens_beyond_loss_and_accuracy(self) -> None:
        """Every finite numeric key `evaluate_global` reports reaches
        round_metrics.csv as `central_test_<name>`, not just loss/accuracy --
        a task's own central-pass diagnostic (`optimality_gap`) now survives
        the trip instead of being discarded for being neither."""

        class _WideEvaluator:
            def evaluate_global(self, global_data: Any) -> dict[str, Any]:
                del global_data
                return {
                    "global_loss": 1.25,
                    "global_accuracy": 0.75,
                    "global_optimality_gap": 0.5,
                    "distance_to_optimum": 0.1,  # no known prefix at all
                    "test_extra": 9.0,  # a different known prefix
                    "non_finite": float("nan"),
                    "also_diverged": float("inf"),
                    "not_a_number": "nope",
                    "a_flag": True,  # bool is not a metric, even though it is an int
                }

        metrics = _evaluate_central_test_set(_WideEvaluator(), _GlobalEvaluationDataset())
        self.assertEqual(
            metrics,
            {
                "central_test_loss": 1.25,
                "central_test_accuracy": 0.75,
                "central_test_optimality_gap": 0.5,
                "central_test_distance_to_optimum": 0.1,
                "central_test_extra": 9.0,
            },
        )

    def test_a_duplicate_bare_name_keeps_the_first_key_seen(self) -> None:
        class _DuplicateEvaluator:
            def evaluate_global(self, global_data: Any) -> dict[str, Any]:
                del global_data
                return {
                    "global_loss": 1.0,
                    "global_accuracy": 1.0,
                    "global_gap": 1.0,
                    "test_gap": 2.0,
                }

        metrics = _evaluate_central_test_set(_DuplicateEvaluator(), _GlobalEvaluationDataset())
        self.assertEqual(metrics["central_test_gap"], 1.0)

    def test_central_pass_accuracy_is_optional_but_loss_is_not(self) -> None:
        """The spike's hit-indicator shim existed only because this refused a
        task with no notion of correct/incorrect. A central pass reporting
        loss alone now succeeds; one reporting accuracy alone still fails,
        exactly as a central pass reporting neither always has."""

        class _LossOnlyEvaluator:
            def evaluate_global(self, global_data: Any) -> dict[str, Any]:
                del global_data
                return {"global_loss": 1.25}

        metrics = _evaluate_central_test_set(_LossOnlyEvaluator(), _GlobalEvaluationDataset())
        self.assertEqual(metrics, {"central_test_loss": 1.25})

        class _AccuracyOnlyEvaluator:
            def evaluate_global(self, global_data: Any) -> dict[str, Any]:
                del global_data
                return {"global_accuracy": 0.9}

        with self.assertRaisesRegex(ValueError, "numeric loss metric"):
            _evaluate_central_test_set(_AccuracyOnlyEvaluator(), _GlobalEvaluationDataset())

    def test_accuracy_is_optional_end_to_end(self) -> None:
        """A task with no notion of correct/incorrect reports loss only, at
        every one of the three places that used to demand a number it does
        not have: client-evaluation validation, the per-split aggregate, and
        the per-client CSV record. The round completes; no *_accuracy_*
        column appears anywhere, and no accuracy is fabricated."""

        clients = {client_id: _NoAccuracyClient(client_id) for client_id in _CLIENT_METRICS}

        state = run_fl_loop(
            _EvaluationServer(),
            clients,
            _EvaluationDataset(),
            global_rounds=1,
            checkpointing={"enabled": False},
            evaluation=EvaluationConfig(
                train=SplitEvaluationConfig(every=1, clients="all"),
                val=SplitEvaluationConfig(every="never", clients="all"),
                test=SplitEvaluationConfig(every=1, clients="all"),
                central_test=CentralTestConfig(every="never"),
            ),
        )

        metrics = state.metrics_history[0].metrics
        self.assertAlmostEqual(metrics["train_loss_sample_weighted_avg"], 3.0)
        self.assertAlmostEqual(metrics["test_loss_avg"], 4.0)
        # fit_accuracy is the fit path's own number, free-form and untouched
        # by this change; the evaluation-derived columns are what must be
        # absent, since nothing reported them.
        for name in (
            "train_accuracy_avg",
            "train_accuracy_sample_weighted_avg",
            "test_accuracy_avg",
            "test_accuracy_sample_weighted_avg",
        ):
            self.assertNotIn(name, metrics)

        self.assertEqual(len(state.client_metrics_history), 3)
        records = {record.client_id: record for record in state.client_metrics_history}
        self.assertEqual(records["client_1"].global_model_train_loss, 3.0)
        self.assertIsNone(records["client_1"].global_model_train_accuracy)
        self.assertEqual(records["client_1"].global_model_test_loss, 4.0)
        self.assertIsNone(records["client_1"].global_model_test_accuracy)

    def test_a_metric_missing_from_only_some_clients_is_skipped_not_partial(self) -> None:
        """One client's rule forgetting to report accuracy is not the same
        defect as the task having none -- but the aggregate cannot tell them
        apart here, so both get the same safe answer: no accuracy column for
        this round, rather than an average over whoever happened to report
        one."""

        results = [
            EvalResult(
                round_id=1,
                client_id="client_0",
                num_examples=2,
                metrics={"test_loss": 1.0, "test_accuracy": 0.5},
                payload={"num_examples_by_split": {"test": 2}},
            ),
            EvalResult(
                round_id=1,
                client_id="client_1",
                num_examples=4,
                metrics={"test_loss": 2.0},
                payload={"num_examples_by_split": {"test": 4}},
            ),
        ]

        aggregated = _aggregate_client_split_metrics(results, "test", ClientStatisticsConfig())
        self.assertAlmostEqual(aggregated["test_loss_sample_weighted_avg"], 5.0 / 3.0)
        self.assertEqual([name for name in aggregated if "accuracy" in name], [])

    def test_requested_global_test_must_be_available(self) -> None:
        clients = {client_id: _EvaluationClient(client_id) for client_id in _CLIENT_METRICS}

        with self.assertRaisesRegex(ValueError, "does not provide global_test data"):
            run_fl_loop(
                _GlobalEvaluationServer(),
                clients,
                _MissingGlobalTestDataset(),
                global_rounds=1,
                checkpointing={"enabled": False},
                evaluation=EvaluationConfig(
                    train=SplitEvaluationConfig(every=1, clients="all"),
                    val=SplitEvaluationConfig(every="never", clients="all"),
                    test=SplitEvaluationConfig(every="never", clients="all"),
                    central_test=CentralTestConfig(every=1),
                ),
            )

    def test_requested_client_test_must_be_available(self) -> None:
        clients = {client_id: _MissingClientTestClient(client_id) for client_id in _CLIENT_METRICS}

        with self.assertRaisesRegex(ValueError, "no non-empty test split"):
            run_fl_loop(
                _EvaluationServer(),
                clients,
                _EvaluationDataset(),
                global_rounds=1,
                checkpointing={"enabled": False},
                evaluation=EvaluationConfig(
                    train=SplitEvaluationConfig(every=1, clients="all"),
                    val=SplitEvaluationConfig(every="never", clients="all"),
                    test=SplitEvaluationConfig(every=1, clients="all"),
                    central_test=CentralTestConfig(every="never"),
                ),
            )

    def test_an_invalid_schedule_is_rejected_before_training(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a positive integer"):
            run_fl_loop(
                _EvaluationServer(),
                {client_id: _EvaluationClient(client_id) for client_id in _CLIENT_METRICS},
                _EvaluationDataset(),
                global_rounds=1,
                checkpointing={"enabled": False},
                evaluation=EvaluationConfig(
                    test=SplitEvaluationConfig(every="sometimes", clients="all"),
                ),
            )

    def test_client_records_include_every_client_and_participation(self) -> None:
        clients = {client_id: _EvaluationClient(client_id) for client_id in _CLIENT_METRICS}

        state = run_fl_loop(
            _EvaluationServer(),
            clients,
            _EvaluationDataset(),
            global_rounds=1,
            checkpointing={"enabled": False},
            evaluation=EvaluationConfig(
                train=SplitEvaluationConfig(every=1, clients="all"),
                val=SplitEvaluationConfig(every="never", clients="all"),
                test=SplitEvaluationConfig(every=1, clients="all"),
                central_test=CentralTestConfig(every="never"),
            ),
        )

        self.assertEqual(len(state.client_metrics_history), 3)
        records = {record.client_id: record for record in state.client_metrics_history}
        self.assertTrue(records["client_0"].participated)
        self.assertFalse(records["client_1"].participated)
        self.assertFalse(records["client_2"].participated)
        self.assertEqual(records["client_1"].train_num_examples, 6)
        self.assertEqual(records["client_1"].test_num_examples, 2)
        self.assertEqual(records["client_1"].global_model_test_loss, 4.0)
        self.assertEqual(len(state.client_update_metrics_history), 1)
        self.assertEqual(state.client_update_metrics_history[0].client_id, "client_0")

    def test_strict_test_split_never_falls_back_to_train(self) -> None:
        train_only = {
            "train": {"X": torch.zeros(2, 1), "y": torch.zeros(2)},
            "eval": {"X": torch.ones(3, 1), "y": torch.ones(3)},
        }

        self.assertIsNone(_get_evaluation_split(train_only, "test"))

    def test_synthetic_client_tests_partition_global_test_data(self) -> None:
        dataset = SyntheticClassificationDataset(
            num_clients=3,
            samples_per_client=4,
            input_dim=2,
            num_classes=2,
            seed=9,
        )

        client_tests = [
            dataset.get_client_data(client_id)["test"] for client_id in dataset.list_clients()
        ]
        global_test = dataset.get_global_data("test")
        self.assertTrue(
            torch.equal(
                torch.cat([split["X"] for split in client_tests]),
                global_test["X"],
            )
        )
        self.assertTrue(
            torch.equal(
                torch.cat([split["y"] for split in client_tests]),
                global_test["y"],
            )
        )

    def test_client_csv_has_one_clear_fixed_schema_row(self) -> None:
        record = ClientEvaluationRecord(
            round_id=3,
            client_id="client_7",
            participated=False,
            train_num_examples=11,
            test_num_examples=5,
            global_model_train_loss=0.8,
            global_model_test_loss=0.9,
            global_model_train_accuracy=0.6,
            global_model_test_accuracy=0.4,
            val_num_examples=3,
            global_model_val_loss=0.85,
            global_model_val_accuracy=0.5,
        )
        expected_fields = [
            "round_id",
            "client_id",
            "participated",
            "train_num_examples",
            "val_num_examples",
            "test_num_examples",
            "global_model_train_loss",
            "global_model_val_loss",
            "global_model_test_loss",
            "global_model_train_accuracy",
            "global_model_val_accuracy",
            "global_model_test_accuracy",
            # Appended when evaluation.model_scope reached this file: the
            # global_model_* names hold the personalized model's numbers under
            # model_scope: personal, and this is what says so.
            "model_scope",
        ]

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            csv_path = save_client_metrics_csv([record], output_dir)
            with csv_path.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(rows[0].keys(), dict.fromkeys(expected_fields).keys())
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["participated"], "False")
            self.assertEqual(load_client_metrics_csv(output_dir), [record])

    def test_lazy_pool_can_release_evaluated_clients(self) -> None:
        pool = LazyClientPool(
            ["client_0"],
            lambda client_id: _EvaluationClient(client_id),
        )
        pool.setup_client_infos([ClientInfo("client_0", 6)])

        first = pool["client_0"]
        pool.release_client("client_0")
        self.assertEqual(pool.materialized_client_ids, [])
        second = pool["client_0"]
        self.assertIsNot(first, second)
        self.assertEqual(second.client_id, "client_0")


if __name__ == "__main__":
    unittest.main()
