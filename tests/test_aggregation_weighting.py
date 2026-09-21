"""Unit tests for server.aggregation_weighting.

FedAvg as published weights each client's model state by its example count.
The papers behind the newer client rules (Delta-SGD, FedLALR) instead
average uniformly over the sampled clients, 1/|S_t|. These tests pin both
modes, and pin that "examples" stays the default so every config written
before the knob existed aggregates exactly as it did.
"""

from __future__ import annotations

import unittest

import pytest
import torch

from fedbrew.core.protocol import FitResult, RoundInfo
from fedbrew.servers.fedavg import FedAvgServer
from fedbrew.servers.scaffold import ScaffoldServer

pytestmark = pytest.mark.fast

#: Deliberately lopsided: client "a" holds 3x the data of client "b", so the
#: two weighting modes cannot coincide.
NUM_EXAMPLES = {"a": 30, "b": 10}
VALUES = {"a": 4.0, "b": 0.0}

EXAMPLE_WEIGHTED_MEAN = (30 * 4.0 + 10 * 0.0) / 40  # 3.0
UNIFORM_MEAN = (4.0 + 0.0) / 2  # 2.0


def _fit_results() -> list[FitResult]:
    return [
        FitResult(
            round_id=1,
            client_id=client_id,
            num_examples=NUM_EXAMPLES[client_id],
            payload={
                "model_state": {"w": torch.full((1,), VALUES[client_id])},
                "model_state_scope": "full",
                "model_state_metadata": {"model_state_scope": "full"},
            },
            metrics={"fit_loss": VALUES[client_id]},
        )
        for client_id in ("a", "b")
    ]


def _server(weighting: str | None = None) -> FedAvgServer:
    kwargs = {} if weighting is None else {"aggregation_weighting": weighting}
    server = FedAvgServer(participation_rate=1.0, seed=0, **kwargs)
    server._model_state = {"w": torch.zeros(1)}
    server._model_state_scope = "full"
    server._model_state_metadata = {"model_state_scope": "full"}
    return server


class AggregationWeightingTest(unittest.TestCase):
    """The knob changes the model aggregate and nothing else."""

    def test_default_is_example_weighted(self) -> None:
        server = _server()
        self.assertEqual(server.aggregation_weighting, "examples")
        server.aggregate(RoundInfo(round_id=1), _fit_results())
        self.assertAlmostEqual(float(server._model_state["w"].item()), EXAMPLE_WEIGHTED_MEAN)

    def test_uniform_weights_clients_equally(self) -> None:
        server = _server("uniform")
        server.aggregate(RoundInfo(round_id=1), _fit_results())
        self.assertAlmostEqual(float(server._model_state["w"].item()), UNIFORM_MEAN)

    def test_metrics_stay_example_weighted_under_uniform(self) -> None:
        """How much a client's parameters count is an algorithm choice; how much
        its reported loss counts is what makes the number a population mean."""

        server = _server("uniform")
        server.metrics = ["fit_loss"]
        round_info = RoundInfo(round_id=1)
        server.aggregate(round_info, _fit_results())
        self.assertAlmostEqual(round_info.metrics["fit_loss"], EXAMPLE_WEIGHTED_MEAN)

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FedAvgServer(participation_rate=1.0, seed=0, aggregation_weighting="sqrt_examples")

    def test_mode_survives_a_checkpoint_round_trip(self) -> None:
        """A resume keeps the mode -- when the config still says the same thing.

        This test used to load a `uniform` checkpoint into a server configured
        `examples` and assert that `uniform` won, which is P10-F14 written as a
        guarantee: the run would have averaged uniformly while `run.json` said
        `examples`. The round trip is still the property worth pinning; a
        resume across a *changed* config is the case below.
        """

        server = _server("uniform")
        restored = _server("uniform")
        restored.load_state(server.save_state())
        self.assertEqual(restored.aggregation_weighting, "uniform")

    def test_a_checkpoint_that_disagrees_with_the_config_is_refused(self) -> None:
        """P10-F14. `tests/test_resume_refuses_a_changed_hyperparameter.py`
        owns the general case; this is the one that used to pass here."""

        server = _server("uniform")
        restored = _server("examples")
        with self.assertRaises(ValueError) as caught:
            restored.load_state(server.save_state())
        self.assertIn("aggregation_weighting", str(caught.exception))
        self.assertEqual(restored.aggregation_weighting, "examples")

    def test_state_without_the_key_keeps_the_current_mode(self) -> None:
        """Checkpoints written before this knob existed must still load."""

        server = _server("uniform")
        legacy_state = server.save_state()
        del legacy_state["aggregation_weighting"]
        restored = _server("uniform")
        restored.load_state(legacy_state)
        self.assertEqual(restored.aggregation_weighting, "uniform")


class ScaffoldAggregationWeightingTest(unittest.TestCase):
    """SCAFFOLD keeps its own accumulator loop, so it needs its own check."""

    def _scaffold(self, weighting: str) -> ScaffoldServer:
        server = ScaffoldServer(participation_rate=1.0, seed=0, aggregation_weighting=weighting)
        server._model_state = {"w": torch.zeros(1)}
        server._model_state_scope = "full"
        server._model_state_metadata = {"model_state_scope": "full"}
        server._server_control = {"w": torch.zeros(1)}
        server._num_clients = 2
        return server

    def _results_with_control(self) -> list[FitResult]:
        results = _fit_results()
        for result in results:
            result.payload["control_delta"] = {"w": torch.zeros(1)}
        return results

    def test_examples_default_matches_fedavg(self) -> None:
        server = self._scaffold("examples")
        server.aggregate(RoundInfo(round_id=1), self._results_with_control())
        self.assertAlmostEqual(float(server._model_state["w"].item()), EXAMPLE_WEIGHTED_MEAN)

    def test_uniform_applies_to_the_model_average(self) -> None:
        server = self._scaffold("uniform")
        server.aggregate(RoundInfo(round_id=1), self._results_with_control())
        self.assertAlmostEqual(float(server._model_state["w"].item()), UNIFORM_MEAN)


if __name__ == "__main__":
    unittest.main()
