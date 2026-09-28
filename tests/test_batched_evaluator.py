"""The batched evaluator measures what the sequential one measures, to summation order.

Beside the batched executor, a round's due clients are evaluated together
(``fedbrew/core/batched_evaluator.py``): every requested split of every due
client held as rows, and batch ``k`` of each measured in one vmapped
``functional_eval`` at the broadcast model. The central pass keeps one model
and copies the server's state into it, and reads the global test shard once.

``tests/test_batched_executor_tolerance.py`` already runs every client split
every round through both executors; what it does not vary is here: splits of
different lengths within one split name, a shuffled evaluation loader, an
evaluation batch size of its own, a client with no val split, one with no
test split -- refused in the sequential words, for the same client -- and
what the central pass keeps between rounds.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import torch

from fedbrew.data.synthetic_classification import SyntheticClassificationDataset as Data
from tests.test_batched_executor_tolerance import (
    ExecutorRuns,
    classification_config,
    example_config,
    float64_classification,
)
from tests.test_resident_round import per_round


def ragged_evaluation_splits() -> Any:
    """Val and test splits of 1 to 5 rows by client number, and client 3 with no val."""

    real = Data.get_client_data

    def cut(self: Any, client_id: str) -> dict[str, Any]:
        data = real(self, client_id)
        number = int(client_id.rsplit("_", 1)[-1])
        keep = 1 + number % 5
        cut_data = dict(data)
        for split in ("eval", "test"):
            if isinstance(data.get(split), dict):
                cut_data[split] = {key: value[:keep] for key, value in data[split].items()}
        if number == 3:
            cut_data["eval"] = {key: value[:0] for key, value in data["eval"].items()}
        return cut_data

    return mock.patch.object(Data, "get_client_data", cut)


def evaluation_config(**client: Any) -> dict[str, Any]:
    config = classification_config(**client)
    config["client"].update(client)
    return config


class ClientEvaluationTest(ExecutorRuns):
    def test_ragged_splits_and_a_missing_val_split(self) -> None:
        from fedbrew.core import batched_evaluator

        measured = mock.Mock(wraps=batched_evaluator.measure_splits)
        with (
            float64_classification(),
            mock.patch.object(batched_evaluator, "measure_splits", measured),
        ):
            batched, sequential = self.both(
                evaluation_config(eval_batch_size=2), data=ragged_evaluation_splits
            )
        self.assertAgree(batched, sequential)
        # Three splits a round, each one chunk: the clients were measured together.
        self.assertEqual(measured.call_count, 3 * 4)

    def test_a_shuffled_evaluation_loader(self) -> None:
        with float64_classification():
            batched, sequential = self.both(
                evaluation_config(eval_batch_size=3, eval_shuffle=True),
                data=ragged_evaluation_splits,
            )
        self.assertAgree(batched, sequential)

    def test_one_client_per_chunk_is_bit_identical(self) -> None:
        batched, sequential = self.both(
            evaluation_config(eval_batch_size=2, eval_shuffle=True),
            data=ragged_evaluation_splits,
            executor_chunk_bytes=1,
        )
        self.assertAgree(batched, sequential, exact=True)

    def test_a_client_without_a_test_split_is_refused_in_the_same_words(self) -> None:
        real = Data.get_client_data

        def no_test(self: Any, client_id: str) -> dict[str, Any]:
            data = dict(real(self, client_id))
            if client_id.endswith("_2"):
                data["test"] = {key: value[:0] for key, value in data["test"].items()}
            return data

        messages = []
        for executor in ("sequential", "batched"):
            with mock.patch.object(Data, "get_client_data", no_test):
                with self.assertRaises(ValueError) as caught:
                    self.run_config(evaluation_config(), executor)
            messages.append(str(caught.exception))
        self.assertIn("has no non-empty test split", messages[0])
        self.assertEqual(messages[0], messages[1])


class CentralEvaluationTest(ExecutorRuns):
    def test_one_model_is_kept_and_the_shard_read_once(self) -> None:
        from fedbrew.data.manifest_dataset import ManifestFederatedDataset
        from fedbrew.servers.fedavg import FedAvgServer

        config = example_config("fed-lasso")
        real_evaluate = FedAvgServer.evaluate_global
        real_read = ManifestFederatedDataset.get_global_data
        models: list[int] = []

        def evaluate(server: Any, data: Any, model: Any = None) -> Any:
            models.append(id(model))
            return real_evaluate(server, data, model=model)

        with (
            mock.patch.object(FedAvgServer, "evaluate_global", evaluate),
            mock.patch.object(
                ManifestFederatedDataset, "get_global_data", autospec=True, side_effect=real_read
            ) as reads,
        ):
            self.run_config(config, "batched")
        self.assertEqual(len(models), 4)
        self.assertEqual(len(set(models)), 1)
        self.assertNotEqual(models[0], id(None))
        self.assertEqual(reads.call_count, 1)

    def test_an_edited_global_shard_is_refused(self) -> None:
        from fedbrew.core.batched_evaluator import BatchedEvaluator

        evaluator = BatchedEvaluator()
        shard = {"x": torch.zeros(3, 2), "y": torch.zeros(3)}
        dataset = mock.Mock()
        dataset.get_global_data.return_value = shard
        evaluator._global_data(dataset)["x"].add_(1.0)
        with self.assertRaisesRegex(RuntimeError, "edited in place"):
            evaluator._global_data(dataset)


class KeptRowsTest(ExecutorRuns):
    def test_a_split_not_due_keeps_its_rows(self) -> None:
        """Test is due at rounds 1 and 4, val every round: nothing is stacked twice."""

        from fedbrew.core import batched_executor

        config = evaluation_config()
        config["evaluation"]["train"] = {"every": "never"}
        config["evaluation"]["test"] = {"every": 3, "clients": "all"}
        real = batched_executor._Rows.__init__
        built: list[int] = []

        def counted(self: Any, task: Any, sources: list[Any]) -> None:
            built.append(len(sources))
            real(self, task, sources)

        # The per-round path, whose evaluator reads the executor's kept rows;
        # a resident run stacks its training rows once for the run instead.
        with mock.patch.object(batched_executor._Rows, "__init__", counted), per_round():
            self.run_config(config, "batched")
        # Round 1 stacks the training rows, val's and test's, once each.
        self.assertEqual(built, [8, 8, 8])
