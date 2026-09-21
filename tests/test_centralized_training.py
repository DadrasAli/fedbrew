"""Tests for the centralized training baseline."""

from __future__ import annotations

import math
import tempfile
import textwrap
import unittest
from pathlib import Path

import pytest
import torch

from fedbrew.core import runner
from fedbrew.core.config import load_config
from fedbrew.core.factory import build_components, is_centralized
from fedbrew.core.registry import (
    client_updates,
    register_builtin_components,
    server_strategies,
)
from fedbrew.data.centralized_dataset import (
    CENTRALIZED_CLIENT_ID,
    CentralizedFederatedDataset,
)
from fedbrew.data.synthetic_classification import SyntheticClassificationDataset

_SOURCE_CLIENTS = 3
_SAMPLES_PER_CLIENT = 4


def _synthetic_dataset() -> SyntheticClassificationDataset:
    return SyntheticClassificationDataset(
        num_clients=_SOURCE_CLIENTS,
        samples_per_client=_SAMPLES_PER_CLIENT,
        input_dim=4,
        num_classes=2,
        seed=42,
    )


def _write_config(
    directory: Path,
    name: str,
    *,
    strategy: str,
    update_rule: str,
    update_mode: str = "single_batch",
    num_clients: int = _SOURCE_CLIENTS,
    batch_size: int = 5,
    local_iterations: int = 1,
    global_rounds: int = 1,
) -> Path:
    config_path = directory / f"{name}.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            experiment:
              seed: 42
              output_dir: {directory / name}

            server:
              strategy: {strategy}
              participation_rate: 1
              metrics:
                - fit_loss
                - fit_accuracy

            client:
              update_rule: {update_rule}
              batch_size: {batch_size}
              learning_rate: 0.05
              learning_rate_schedule: constant
              min_learning_rate: 0.0
              update_mode: {update_mode}
              frozen_gradient_weighting: examples
              momentum: 0.0
              weight_decay: 0.0
              nesterov: false
              train_shuffle: false
              metrics:
                - fit_loss
                - fit_accuracy
                - optimizer_steps

            data:
              num_clients: {num_clients}
              samples_per_client: {_SAMPLES_PER_CLIENT}
              input_dim: 4
              num_classes: 2

            model:
              name: mlp
              input_dim: 4
              hidden_dim: 4
              num_classes: 2

            runtime:
              deterministic: true
              deterministic_warn_only: true
              device: cpu
              use_amp: false

            evaluation:
              train:
                every: 1
                clients: all
              val:
                every: 1
                clients: all
              test:
                every: 1
                clients: all
              central_test:
                every: 1

            client_statistics:
              # This test asserts client_metrics.csv is written, so it has to
              # be the thing that asks for it -- it is opt-in now.
              per_client_csv: true

            defaults:
              global_rounds: {global_rounds}
              local_iterations: {local_iterations}
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return config_path


@pytest.mark.fast
class CentralizedDatasetTests(unittest.TestCase):
    def test_view_exposes_one_client_holding_every_source_client(self) -> None:
        source = _synthetic_dataset()
        view = CentralizedFederatedDataset(source)

        self.assertEqual(view.list_clients(), [CENTRALIZED_CLIENT_ID])
        self.assertEqual(view.source_client_ids, source.list_clients())

    def test_splits_concatenate_source_shards_in_client_order(self) -> None:
        source = _synthetic_dataset()
        view = CentralizedFederatedDataset(source)

        pooled = view.get_client_data(CENTRALIZED_CLIENT_ID)
        for split in ("train", "eval", "test"):
            expected_features = torch.cat(
                [
                    source.get_client_data(client_id)[split]["X"]
                    for client_id in source.list_clients()
                ],
                dim=0,
            )
            expected_targets = torch.cat(
                [
                    source.get_client_data(client_id)[split]["y"]
                    for client_id in source.list_clients()
                ],
                dim=0,
            )
            self.assertTrue(torch.equal(pooled[split]["x"], expected_features), split)
            self.assertTrue(torch.equal(pooled[split]["y"], expected_targets), split)

    def test_split_counts_sum_the_source_clients(self) -> None:
        view = CentralizedFederatedDataset(_synthetic_dataset())

        metadata = view.get_client_metadata(CENTRALIZED_CLIENT_ID)
        pooled_examples = _SOURCE_CLIENTS * _SAMPLES_PER_CLIENT
        self.assertEqual(metadata["num_train_examples"], pooled_examples)
        self.assertEqual(metadata["num_eval_examples"], pooled_examples)
        self.assertEqual(metadata["num_test_examples"], pooled_examples)
        self.assertEqual(metadata["num_examples"], 2 * pooled_examples)

    def test_global_data_and_metadata_come_from_the_source_dataset(self) -> None:
        source = _synthetic_dataset()
        view = CentralizedFederatedDataset(source)

        global_data = view.get_global_data()
        self.assertTrue(torch.equal(global_data["X"], source.get_global_data()["X"]))
        metadata = view.get_metadata()
        self.assertTrue(metadata["centralized"])
        self.assertEqual(metadata["num_source_clients"], _SOURCE_CLIENTS)

    def test_unknown_client_id_is_rejected(self) -> None:
        view = CentralizedFederatedDataset(_synthetic_dataset())

        with self.assertRaises(KeyError):
            view.get_client_data("client_0")

    def test_source_shards_are_read_once(self) -> None:
        source = _synthetic_dataset()
        view = CentralizedFederatedDataset(source)
        reads = 0
        original = source.get_client_data

        def counting_get_client_data(client_id: str) -> dict[str, object]:
            nonlocal reads
            reads += 1
            return original(client_id)

        source.get_client_data = counting_get_client_data  # type: ignore[method-assign]
        view.get_client_data(CENTRALIZED_CLIENT_ID)
        view.get_client_data(CENTRALIZED_CLIENT_ID)
        view.get_client_metadata(CENTRALIZED_CLIENT_ID)

        self.assertEqual(reads, _SOURCE_CLIENTS)


@pytest.mark.fast
class CentralizedConfigurationTests(unittest.TestCase):
    def test_strategy_and_update_rule_are_registered(self) -> None:
        register_builtin_components()

        self.assertTrue(server_strategies.exists("centralized"))
        self.assertTrue(client_updates.exists("centralized"))

    def test_half_configured_runs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server_only = _write_config(
                root, "server_only", strategy="centralized", update_rule="fedavg"
            )
            client_only = _write_config(
                root, "client_only", strategy="fedavg", update_rule="centralized"
            )

            with self.assertRaises(ValueError):
                load_config(server_only)
            with self.assertRaises(ValueError):
                load_config(client_only)

    def test_components_expose_a_single_pooled_client(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = _write_config(
                Path(directory),
                "centralized",
                strategy="centralized",
                update_rule="centralized",
            )
            config = load_config(config_path)

            self.assertTrue(is_centralized(config))
            components = build_components(config)
            self.assertEqual(list(components.clients), [CENTRALIZED_CLIENT_ID])
            self.assertEqual(components.dataset.list_clients(), [CENTRALIZED_CLIENT_ID])


class CentralizedRunTests(unittest.TestCase):
    def _run(self, config_path: Path) -> object:
        args = runner.parse_args(["--quiet"])
        return runner.run(config_path, args)

    def test_matches_fedavg_when_the_source_has_a_single_client(self) -> None:
        # Averaging one client's update is the identity, so with shuffling off
        # the pooled run must reproduce the federated run metric for metric.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            federated = _write_config(
                root,
                "federated",
                strategy="fedavg",
                update_rule="fedavg",
                update_mode="sequential_epoch",
                num_clients=1,
                global_rounds=3,
            )
            centralized = _write_config(
                root,
                "centralized",
                strategy="centralized",
                update_rule="centralized",
                update_mode="sequential_epoch",
                num_clients=1,
                global_rounds=3,
            )

            federated_state = self._run(federated)
            centralized_state = self._run(centralized)

            federated_metrics = [record.metrics for record in federated_state.metrics_history]
            centralized_metrics = [record.metrics for record in centralized_state.metrics_history]
            self.assertEqual(len(centralized_metrics), 3)
            for federated_round, centralized_round in zip(
                federated_metrics, centralized_metrics, strict=True
            ):
                self.assertEqual(set(federated_round), set(centralized_round))
                for name, value in federated_round.items():
                    self.assertAlmostEqual(value, centralized_round[name], places=10, msg=name)

    def test_run_reports_the_pooled_client_and_finite_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = _write_config(
                root, "centralized", strategy="centralized", update_rule="centralized"
            )

            state = self._run(config_path)

            record = state.metrics_history[0]
            self.assertEqual(record.num_clients, 1)
            self.assertTrue(all(math.isfinite(v) for v in record.metrics.values()))
            self.assertIn("central_test_accuracy", record.metrics)
            self.assertIn("test_accuracy_sample_weighted_avg", record.metrics)
            self.assertEqual(
                [update.client_id for update in state.client_update_metrics_history],
                [CENTRALIZED_CLIENT_ID],
            )
            for name in ("round_metrics.csv", "client_metrics.csv", "run.json"):
                self.assertTrue((root / "centralized" / name).is_file(), name)

    def test_update_modes_apply_their_documented_step_counts(self) -> None:
        batch_size = 5
        local_iterations = 2
        pooled_examples = _SOURCE_CLIENTS * _SAMPLES_PER_CLIENT
        batches_per_epoch = math.ceil(pooled_examples / batch_size)
        expected_steps = {
            "single_batch": local_iterations,
            "sequential_epoch": local_iterations * batches_per_epoch,
            "frozen_batch_gradients": local_iterations,
        }

        for update_mode, expected in expected_steps.items():
            with self.subTest(update_mode=update_mode), tempfile.TemporaryDirectory() as directory:
                config_path = _write_config(
                    Path(directory),
                    update_mode,
                    strategy="centralized",
                    update_rule="centralized",
                    update_mode=update_mode,
                    batch_size=batch_size,
                    local_iterations=local_iterations,
                )

                state = self._run(config_path)

                update = state.client_update_metrics_history[0]
                self.assertEqual(update.metrics["optimizer_steps"], float(expected))


if __name__ == "__main__":
    unittest.main()
