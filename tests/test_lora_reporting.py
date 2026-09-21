"""Focused tests for the run.json results block."""

from __future__ import annotations

import json
import tempfile
import unittest

import pytest

from fedbrew.core.artifacts import save_run_json
from fedbrew.core.state import MetricRecord

pytestmark = pytest.mark.fast


def _minimal_config():
    """Smallest FullConfig save_run_json will accept, for artifact-shape tests."""

    from fedbrew.core.config import (
        ClientConfig,
        DataConfig,
        ExperimentConfig,
        FullConfig,
        ModelConfig,
        RuntimeConfig,
        ServerConfig,
        TaskConfig,
    )

    return FullConfig(
        experiment=ExperimentConfig(seed=0, output_dir="outputs/test"),
        server=ServerConfig(strategy="fedavg", global_rounds=1, participation_rate=1.0, metrics=[]),
        client=ClientConfig(update_rule="fedavg", local_iterations=1, batch_size=1, metrics=[]),
        task=TaskConfig(name="classification"),
        data=DataConfig(),
        model=ModelConfig(name="mlp"),
        runtime=RuntimeConfig(device="cpu", use_amp=False),
    )


class RunJsonResultsTests(unittest.TestCase):
    def test_final_metrics_are_the_last_round_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = save_run_json(
                [
                    MetricRecord(
                        round_id=1,
                        metrics={"central_test_loss": 2.0},
                        num_clients=1,
                        num_examples=3,
                    ),
                    MetricRecord(
                        round_id=2,
                        metrics={"central_test_loss": 1.5, "central_test_accuracy": 0.4},
                        num_clients=1,
                        num_examples=3,
                    ),
                ],
                directory,
                _minimal_config(),
            )
            run = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(run["final_round"], 2)
        self.assertEqual(
            run["results"]["final_metrics"],
            {"central_test_accuracy": 0.4, "central_test_loss": 1.5},
        )
        # No derived metric is synthesised into the results block: what the run
        # measured is what gets reported.
        self.assertEqual(len(run["results"]), 1)

    def test_a_diverged_run_still_writes_valid_json(self) -> None:
        # json.dumps emits a bare Infinity for a non-finite float, which is not
        # valid JSON. A diverged run must still produce a loadable run.json.
        with tempfile.TemporaryDirectory() as directory:
            path = save_run_json(
                [
                    MetricRecord(
                        round_id=1,
                        metrics={
                            "central_test_loss": float("inf"),
                            "central_test_accuracy": float("nan"),
                            "fit_loss": 1.25,
                        },
                        num_clients=1,
                        num_examples=3,
                    )
                ],
                directory,
                _minimal_config(),
            )
            serialized = path.read_text(encoding="utf-8")
            run = json.loads(serialized)

        self.assertNotIn("Infinity", serialized)
        self.assertNotIn("NaN", serialized)
        self.assertIsNone(run["results"]["final_metrics"]["central_test_loss"])
        self.assertIsNone(run["results"]["final_metrics"]["central_test_accuracy"])
        self.assertAlmostEqual(run["results"]["final_metrics"]["fit_loss"], 1.25)

    def test_scale_counters_carry_no_duplicated_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = save_run_json(
                [
                    MetricRecord(
                        round_id=1,
                        metrics={"fit_loss": 1.0},
                        num_clients=2,
                        num_examples=6,
                    )
                ],
                directory,
                _minimal_config(),
            )
            scale = json.loads(path.read_text(encoding="utf-8"))["scale"]

        self.assertNotIn("total_client_updates", scale)
        self.assertNotIn("total_client_evaluation_records", scale)
        self.assertNotIn("personalized_model_evaluation_records", scale)


if __name__ == "__main__":
    unittest.main()
