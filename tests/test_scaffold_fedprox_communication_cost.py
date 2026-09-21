"""SCAFFOLD's rounds cost double, and nothing measured them.

Every other client reports communicated_parameters/communicated_bytes for the
volume it puts on the wire. FedProx and SCAFFOLD computed neither, so a cost
table joining client_update_metrics.csv across arms read their absence as no
cost at all -- and SCAFFOLD's real per-round volume is twice FedAvg's, because
it uploads the control-variate delta beside the model and the server sends its
own control variate back down beside the model.

The trap in fixing it is that the one-line pattern used everywhere else,
model_state_size(model_state), produces a plausible non-crashing number that
is exactly half the truth. FedLALR already had to solve this -- it adds its
momentum and second-moment states explicitly -- and these tests pin that
SCAFFOLD counts both of its states rather than inheriting the halved answer.
"""

from __future__ import annotations

import copy
import unittest
from typing import Any

import pytest
import torch

from fedbrew.clients.torch_fedprox_client import TorchFedProxClient
from fedbrew.clients.torch_scaffold_client import TorchScaffoldClient
from fedbrew.core.config import load_config
from fedbrew.core.federated_state import model_state_size
from fedbrew.core.protocol import FitRequest
from fedbrew.core.validation import validate_full_config
from tests.test_empty_training_batches import _data, _Task

COST_METRICS = ("communicated_parameters", "communicated_bytes")


def _request() -> FitRequest:
    torch.manual_seed(0)
    model = _Task().build_model()
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    return FitRequest(
        round_id=1,
        client_id="c0",
        payload={
            "model_state": state,
            "server_control": {name: torch.zeros_like(value) for name, value in state.items()},
        },
    )


def _kwargs(metrics: list[str]) -> dict[str, Any]:
    return {
        "client_id": "c0",
        "task": _Task(),
        "model_config": {},
        "client_data": copy.deepcopy(_data(8)),
        "local_iterations": 1,
        "batch_size": 4,
        "learning_rate": 0.1,
        "train_shuffle": False,
        "metrics": metrics,
    }


class ScaffoldCostTest(unittest.TestCase):
    def test_the_cost_counts_the_control_delta_as_well_as_the_model(self) -> None:
        result = TorchScaffoldClient(**_kwargs(list(COST_METRICS))).fit(_request())

        model_parameters, model_bytes = model_state_size(result.payload["model_state"])
        delta_parameters, delta_bytes = model_state_size(result.payload["control_delta"])
        self.assertEqual(
            result.metrics["communicated_parameters"],
            float(model_parameters + delta_parameters),
        )
        self.assertEqual(result.metrics["communicated_bytes"], float(model_bytes + delta_bytes))

    def test_it_is_not_the_model_alone(self) -> None:
        """The halved answer the one-line pattern would have produced."""

        result = TorchScaffoldClient(**_kwargs(list(COST_METRICS))).fit(_request())
        _, model_bytes = model_state_size(result.payload["model_state"])
        self.assertGreater(result.metrics["communicated_bytes"], float(model_bytes))

    def test_a_round_is_twice_a_fedavg_arm_at_the_same_model_size(self) -> None:
        """The control variate is one tensor per parameter, so exactly 2x."""

        result = TorchScaffoldClient(**_kwargs(list(COST_METRICS))).fit(_request())
        _, model_bytes = model_state_size(result.payload["model_state"])
        self.assertEqual(result.metrics["communicated_bytes"], 2.0 * model_bytes)


class FedProxCostTest(unittest.TestCase):
    def test_the_cost_is_one_model_and_is_reported(self) -> None:
        """The proximal term is local, so nothing extra crosses the wire --
        but a missing column reads as no cost when arms are joined."""

        client = TorchFedProxClient(**(_kwargs(list(COST_METRICS)) | {"proximal_mu": 0.01}))
        result = client.fit(_request())
        model_parameters, model_bytes = model_state_size(result.payload["model_state"])
        self.assertEqual(result.metrics["communicated_parameters"], float(model_parameters))
        self.assertEqual(result.metrics["communicated_bytes"], float(model_bytes))


class ShippedConfigTest(unittest.TestCase):
    @pytest.mark.fast
    def test_both_configs_ask_for_the_metrics(self) -> None:
        for path in ("configs/femnist/scaffold.yaml", "configs/femnist/fedprox.yaml"):
            with self.subTest(path=path):
                config = load_config(path)
                for name in COST_METRICS:
                    self.assertIn(name, config.client.metrics)

    def test_the_client_still_reports_them_through_the_shipped_list(self) -> None:
        """filter_metrics is what dropped them for delta_sgd."""

        metrics = list(load_config("configs/femnist/scaffold.yaml").client.metrics)
        result = TorchScaffoldClient(**_kwargs(metrics)).fit(_request())
        for name in COST_METRICS:
            self.assertIn(name, result.metrics)


class ValidationNoticeTest(unittest.TestCase):
    """FedLALR's 3x notice has existed since it landed; SCAFFOLD had none."""

    def test_preflight_says_a_scaffold_round_costs_double(self) -> None:
        report = validate_full_config(
            load_config("configs/femnist/scaffold.yaml"),
            config_path="configs/femnist/scaffold.yaml",
        )
        codes = {issue.code for issue in report.issues}
        self.assertIn("algorithm.scaffold_communication_cost", codes)
        notice = next(
            issue
            for issue in report.issues
            if issue.code == "algorithm.scaffold_communication_cost"
        )
        self.assertEqual(notice.severity, "info")
        self.assertIn("2x", notice.message)

    def test_a_fedavg_config_gets_no_such_notice(self) -> None:
        report = validate_full_config(
            load_config("configs/femnist/fedavg.yaml"),
            config_path="configs/femnist/fedavg.yaml",
        )
        self.assertNotIn(
            "algorithm.scaffold_communication_cost",
            {issue.code for issue in report.issues},
        )


if __name__ == "__main__":
    unittest.main()
