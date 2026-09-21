"""FedLALR: local AMSGrad with a synchronized second moment (arXiv:2309.09719).

Algorithm 1. Global initialization is m_{-1} = 0 and v_hat_{-1} = epsilon^2.
At the start of round t each client sets

    m_{t,0,i} = m_{t-1},   v_{t,0,i} = v_hat_{t,0,i} = v_hat_{t-1}

and then, per local step,

    m     = beta1 * m + (1 - beta1) * g
    v     = beta2 * v + (1 - beta2) * g^2
    v_hat = max(v_hat, v)
    x     = x - alpha * m / sqrt(v_hat)

The server averages x, m and v_hat across the sampled clients and broadcasts
all three back. v itself is never communicated.

The paper's Corollary 2 (do not send momenta) and Corollary 3 (aggregate
v_hat by maximum) are variants; neither is implemented, and their existence is
what settles that Algorithm 1 does send and does average.
"""

from __future__ import annotations

import copy
import math
import unittest
from collections.abc import Mapping
from typing import Any

import pytest
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, TensorDataset

from fedbrew.clients.torch_fedlalr_client import (
    DEFAULT_BETA1,
    DEFAULT_BETA2,
    DEFAULT_EPSILON,
    TorchFedLALRClient,
    _amsgrad_step,
)
from fedbrew.core.config import load_config, validate_config
from fedbrew.core.factory import build_components
from fedbrew.core.protocol import ClientInfo, FitRequest, FitResult, RoundInfo
from fedbrew.core.validation import validate_full_config
from fedbrew.servers.fedlalr import FedLALRServer
from fedbrew.tasks.base import TaskAdapter

ALPHA = 0.01
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8

_BASE_CONFIG = load_config("configs/dev/synthetic.yaml")


@pytest.mark.fast
class AMSGradStepTest(unittest.TestCase):
    """One local step, against arithmetic written out in full."""

    def _one_parameter_model(self, value: float, gradient: float) -> nn.Module:
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(value)
        model.weight.grad = torch.full((1, 1), gradient)
        return model

    def test_first_step_matches_the_closed_form(self) -> None:
        gradient = 2.0
        model = self._one_parameter_model(1.0, gradient)
        momentum = {"weight": torch.zeros(1, 1)}
        second_moment = {"weight": torch.full((1, 1), EPSILON**2)}
        second_moment_hat = {"weight": torch.full((1, 1), EPSILON**2)}

        _amsgrad_step(
            model,
            momentum,
            second_moment,
            second_moment_hat,
            learning_rate=ALPHA,
            beta1=BETA1,
            beta2=BETA2,
        )

        expected_m = (1.0 - BETA1) * gradient
        expected_v = BETA2 * EPSILON**2 + (1.0 - BETA2) * gradient**2
        expected_v_hat = max(EPSILON**2, expected_v)
        expected_w = 1.0 - ALPHA * expected_m / math.sqrt(expected_v_hat)

        # places=6: the buffers are float32, so the reference arithmetic in
        # float64 agrees only to single precision.
        self.assertAlmostEqual(float(momentum["weight"]), expected_m, places=6)
        self.assertAlmostEqual(float(second_moment["weight"]), expected_v, places=6)
        self.assertAlmostEqual(float(second_moment_hat["weight"]), expected_v_hat, places=6)
        self.assertAlmostEqual(float(model.weight), expected_w, places=6)

    def test_v_hat_is_a_running_maximum(self) -> None:
        """A shrinking gradient must not shrink v_hat -- that is the AMSGrad
        part, and it is what keeps the learning rate monotone within a round."""

        model = self._one_parameter_model(0.0, 10.0)
        momentum = {"weight": torch.zeros(1, 1)}
        second_moment = {"weight": torch.zeros(1, 1)}
        second_moment_hat = {"weight": torch.full((1, 1), EPSILON**2)}

        seen: list[float] = []
        for gradient in (10.0, 1.0, 0.1, 0.0):
            model.weight.grad = torch.full((1, 1), gradient)
            _amsgrad_step(
                model,
                momentum,
                second_moment,
                second_moment_hat,
                learning_rate=ALPHA,
                beta1=BETA1,
                beta2=BETA2,
            )
            seen.append(float(second_moment_hat["weight"]))

        self.assertEqual(seen, sorted(seen))
        # v, unlike v_hat, is free to decay.
        self.assertLess(float(second_moment["weight"]), seen[-1])

    def test_epsilon_squared_floors_the_learning_rate(self) -> None:
        """No epsilon appears in the denominator; the floor comes from
        v_hat_{-1} = epsilon^2 surviving the running maximum."""

        model = self._one_parameter_model(0.0, 0.0)
        momentum = {"weight": torch.zeros(1, 1)}
        second_moment = {"weight": torch.full((1, 1), EPSILON**2)}
        second_moment_hat = {"weight": torch.full((1, 1), EPSILON**2)}

        for _ in range(5):
            model.weight.grad = torch.zeros(1, 1)
            _amsgrad_step(
                model,
                momentum,
                second_moment,
                second_moment_hat,
                learning_rate=ALPHA,
                beta1=BETA1,
                beta2=BETA2,
            )

        self.assertGreaterEqual(float(second_moment_hat["weight"]), EPSILON**2)
        effective = ALPHA / math.sqrt(float(second_moment_hat["weight"]))
        self.assertLessEqual(effective, ALPHA / EPSILON)
        self.assertTrue(torch.isfinite(model.weight).all())

    def test_v_is_seeded_from_v_hat_not_from_zero(self) -> None:
        """Algorithm 1 line 3: v_{t,0,i} = v_hat_{t,0,i} = v_hat_{t-1}.

        Under a sustained gradient the two candidate seeds diverge: starting v
        at v_hat lets it keep climbing past the broadcast value, whereas
        starting it at zero leaves v below v_hat for the whole round, so v_hat
        never moves. Seeding from zero would therefore freeze the learning rate
        at whatever the server last broadcast.
        """

        def run(v_seed: float) -> float:
            model = self._one_parameter_model(0.0, 2.0)
            momentum = {"weight": torch.zeros(1, 1)}
            second_moment = {"weight": torch.full((1, 1), v_seed)}
            second_moment_hat = {"weight": torch.ones(1, 1)}
            for _ in range(4):
                model.weight.grad = torch.full((1, 1), 2.0)
                _amsgrad_step(
                    model,
                    momentum,
                    second_moment,
                    second_moment_hat,
                    learning_rate=ALPHA,
                    beta1=BETA1,
                    beta2=BETA2,
                )
            return float(second_moment_hat["weight"])

        seeded_from_v_hat = run(1.0)
        seeded_from_zero = run(0.0)
        self.assertGreater(seeded_from_v_hat, seeded_from_zero)
        self.assertAlmostEqual(seeded_from_zero, 1.0, places=6)

    def test_frozen_parameters_are_skipped(self) -> None:
        model = nn.Linear(1, 1, bias=False)
        model.weight.requires_grad_(False)
        before = model.weight.detach().clone()
        _amsgrad_step(
            model,
            {},
            {},
            {},
            learning_rate=ALPHA,
            beta1=BETA1,
            beta2=BETA2,
        )
        self.assertTrue(torch.equal(model.weight.detach(), before))


def _server(**overrides: Any) -> FedLALRServer:
    kwargs: dict[str, Any] = {
        "epsilon": EPSILON,
        "participation_rate": 1.0,
        "seed": 0,
        "aggregation_weighting": "uniform",
    }
    kwargs.update(overrides)
    server = FedLALRServer(**kwargs)
    server._model_state = {"w": torch.zeros(2)}
    server._model_state_scope = "full"
    server._model_state_metadata = {"model_state_scope": "full"}
    return server


def _fit_result(client_id: str, *, model: float, momentum: float, v_hat: float) -> FitResult:
    return FitResult(
        round_id=1,
        client_id=client_id,
        num_examples=10,
        payload={
            "model_state": {"w": torch.full((2,), model)},
            "model_state_scope": "full",
            "model_state_metadata": {"model_state_scope": "full"},
            "momentum_state": {"w": torch.full((2,), momentum)},
            "second_moment_state": {"w": torch.full((2,), v_hat)},
        },
        metrics={"effective_learning_rate_coordinate_mean": model},
    )


@pytest.mark.fast
class ServerTest(unittest.TestCase):
    """Algorithm 1's synchronization block."""

    def test_initialization_is_zero_momentum_and_epsilon_squared(self) -> None:
        server = _server()
        server._ensure_optimizer_state()
        self.assertEqual(float(server._momentum["w"][0]), 0.0)
        self.assertAlmostEqual(float(server._second_moment["w"][0]), EPSILON**2)

    def test_all_three_states_are_averaged(self) -> None:
        server = _server()
        server._ensure_optimizer_state()
        server.aggregate(
            RoundInfo(round_id=1),
            [
                _fit_result("a", model=2.0, momentum=4.0, v_hat=6.0),
                _fit_result("b", model=4.0, momentum=8.0, v_hat=10.0),
            ],
        )
        self.assertAlmostEqual(float(server._model_state["w"][0]), 3.0)
        self.assertAlmostEqual(float(server._momentum["w"][0]), 6.0)
        self.assertAlmostEqual(float(server._second_moment["w"][0]), 8.0)

    def test_broadcast_carries_all_three_plus_the_state_scope(self) -> None:
        server = _server()
        requests = server.configure_round(
            RoundInfo(round_id=1),
            [ClientInfo(client_id="a", num_examples=10)],
        )
        payload = requests[0].payload
        for key in (
            "model_state",
            "model_state_scope",
            "model_state_metadata",
            "momentum_state",
            "second_moment_state",
        ):
            self.assertIn(key, payload)

    def test_a_client_missing_its_optimizer_state_is_rejected(self) -> None:
        server = _server()
        server._ensure_optimizer_state()
        result = _fit_result("a", model=1.0, momentum=1.0, v_hat=1.0)
        del result.payload["second_moment_state"]
        with self.assertRaises(ValueError):
            server.aggregate(RoundInfo(round_id=1), [result])

    def test_dispersion_metrics_report_the_spread_across_clients(self) -> None:
        server = _server(metrics=["effective_learning_rate_across_clients_std"])
        server._ensure_optimizer_state()
        round_info = RoundInfo(round_id=1)
        server.aggregate(
            round_info,
            [
                _fit_result("a", model=2.0, momentum=0.0, v_hat=1.0),
                _fit_result("b", model=4.0, momentum=0.0, v_hat=1.0),
            ],
        )
        self.assertAlmostEqual(
            round_info.metrics["effective_learning_rate_across_clients_std"], 1.0
        )

    def test_optimizer_state_survives_a_checkpoint_round_trip(self) -> None:
        server = _server()
        server._ensure_optimizer_state()
        server._momentum["w"].fill_(3.0)
        restored = _server()
        restored.load_state(server.save_state())
        self.assertAlmostEqual(float(restored._momentum["w"][0]), 3.0)
        self.assertAlmostEqual(float(restored._second_moment["w"][0]), EPSILON**2)
        self.assertEqual(restored.epsilon, EPSILON)

    def test_integer_buffers_are_left_out_of_the_optimizer_state(self) -> None:
        """epsilon**2 truncates to 0 in an integer dtype, and the client would
        then divide by sqrt(0)."""

        server = _server()
        server._model_state = {
            "w": torch.zeros(2),
            "num_batches_tracked": torch.zeros(1, dtype=torch.long),
        }
        server._ensure_optimizer_state()
        self.assertEqual(set(server._second_moment), {"w"})


class _LinearTask(TaskAdapter):
    """Deterministic least-squares task over three parameters."""

    #: The clients move broadcast tensors onto the task's device, as the
    #: SCAFFOLD client does.
    device = torch.device("cpu")

    def build_model(self, config: Mapping[str, Any] | None = None) -> nn.Module:
        model = nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[0.5, -0.25, 1.0]]))
        return model

    def build_dataloader(self, data: Any, config: Mapping[str, Any]) -> DataLoader[Any]:
        return DataLoader(data, batch_size=int(config.get("batch_size", 2)), shuffle=False)

    def train_step(
        self,
        model: nn.Module,
        batch: Any,
        optimizer: optim.Optimizer | None = None,
    ) -> dict[str, float]:
        if optimizer is None:
            raise ValueError("optimizer is required")
        features, targets = batch
        optimizer.zero_grad()
        loss = ((model(features).squeeze(-1) - targets) ** 2).mean()
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.detach()), "correct": 0.0, "total": float(targets.numel())}

    def eval_step(self, model: nn.Module, batch: Any) -> dict[str, float]:
        features, targets = batch
        with torch.no_grad():
            loss = ((model(features).squeeze(-1) - targets) ** 2).mean()
        return {"loss": float(loss), "correct": 0.0, "total": float(targets.numel())}

    def compute_metrics(self, outputs: Any, targets: Tensor | None = None) -> dict[str, float]:
        total = sum(float(output["total"]) for output in outputs) or 1.0
        loss = sum(float(output["loss"]) * float(output["total"]) for output in outputs)
        return {"loss": loss / total, "accuracy": 0.0}


def _client_data() -> TensorDataset:
    """A bare dataset, as _get_train_data expects when the shard is not a
    mapping of split -> tensor dict."""

    torch.manual_seed(0)
    return TensorDataset(torch.randn(8, 3), torch.randn(8))


def _client(client_id: str = "c0") -> TorchFedLALRClient:
    return TorchFedLALRClient(
        client_id=client_id,
        task=_LinearTask(),
        model_config={},
        client_data=_client_data(),
        local_iterations=1,
        batch_size=2,
        learning_rate=ALPHA,
        beta1=BETA1,
        beta2=BETA2,
        epsilon=EPSILON,
        train_shuffle=False,
    )


def _broadcast(momentum: float = 0.0, v_hat: float = EPSILON**2) -> dict[str, Any]:
    shape = (1, 3)
    return {
        "model_state": {"weight": torch.tensor([[0.5, -0.25, 1.0]])},
        "model_state_scope": "full",
        "model_state_metadata": {"model_state_scope": "full"},
        "momentum_state": {"weight": torch.full(shape, momentum)},
        "second_moment_state": {"weight": torch.full(shape, v_hat)},
    }


class ClientTest(unittest.TestCase):
    def test_fit_returns_all_three_states(self) -> None:
        result = _client().fit(FitRequest(round_id=1, client_id="c0", payload=_broadcast()))
        for key in ("model_state", "momentum_state", "second_moment_state"):
            self.assertIn(key, result.payload)
        self.assertEqual(result.metrics["local_steps"], 4.0)

    def test_the_shared_broadcast_payload_is_never_mutated(self) -> None:
        """The server hands ONE payload dict to every client in a round. A
        client writing its optimizer state in place would corrupt what the
        next client starts from."""

        payload = _broadcast(momentum=0.25, v_hat=0.5)
        before = {
            name: {key: tensor.clone() for key, tensor in state.items()}
            for name, state in payload.items()
            if name.endswith("_state") and name != "model_state_metadata"
        }

        for client_id in ("c0", "c1"):
            _client(client_id).fit(FitRequest(round_id=1, client_id=client_id, payload=payload))

        for name, state in before.items():
            for key, tensor in state.items():
                with self.subTest(state=name, tensor=key):
                    self.assertTrue(torch.equal(payload[name][key], tensor))

    def test_the_returned_v_hat_never_falls_below_the_broadcast(self) -> None:
        """v_hat is a running maximum seeded from the broadcast, so a round can
        only ever raise it."""

        broadcast_v_hat = 0.5
        result = _client().fit(
            FitRequest(round_id=1, client_id="c0", payload=_broadcast(v_hat=broadcast_v_hat))
        )
        returned = result.payload["second_moment_state"]["weight"]
        self.assertTrue(bool((returned >= broadcast_v_hat).all()))

    def test_communicated_volume_counts_all_three_states(self) -> None:
        result = _client().fit(FitRequest(round_id=1, client_id="c0", payload=_broadcast()))
        model_parameters = 3
        self.assertEqual(result.metrics["communicated_parameters"], float(3 * model_parameters))

    @pytest.mark.fast
    def test_a_missing_broadcast_state_is_rejected(self) -> None:
        payload = _broadcast()
        del payload["momentum_state"]
        with self.assertRaises(ValueError):
            _client().fit(FitRequest(round_id=1, client_id="c0", payload=payload))

    @pytest.mark.fast
    def test_options_the_amsgrad_update_would_ignore_are_rejected(self) -> None:
        for option in ({"momentum": 0.9}, {"weight_decay": 0.1}, {"max_local_steps": 2}):
            with self.subTest(option=option), self.assertRaises(ValueError):
                TorchFedLALRClient(
                    client_id="c0",
                    task=_LinearTask(),
                    model_config={},
                    client_data=_client_data(),
                    local_iterations=1,
                    batch_size=2,
                    learning_rate=ALPHA,
                    beta1=BETA1,
                    beta2=BETA2,
                    epsilon=EPSILON,
                    **option,
                )

    @pytest.mark.fast
    def test_beta_outside_the_unit_interval_is_rejected(self) -> None:
        for field in ("beta1", "beta2"):
            for value in (-0.1, 1.0):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    _client_kwargs = {
                        "client_id": "c0",
                        "task": _LinearTask(),
                        "model_config": {},
                        "client_data": _client_data(),
                        "local_iterations": 1,
                        "batch_size": 2,
                        "learning_rate": ALPHA,
                        "beta1": BETA1,
                        "beta2": BETA2,
                        "epsilon": EPSILON,
                    }
                    _client_kwargs[field] = value
                    TorchFedLALRClient(**_client_kwargs)


class RoundTripTest(unittest.TestCase):
    """Client and server against each other, over two rounds."""

    def test_two_rounds_through_the_real_server(self) -> None:
        server = _server()
        server._model_state = {"weight": torch.tensor([[0.5, -0.25, 1.0]])}
        server._ensure_optimizer_state()
        clients = [_client("c0"), _client("c1")]

        first_v_hat = None
        for round_id in (1, 2):
            requests = server.configure_round(
                RoundInfo(round_id=round_id),
                [ClientInfo(client_id=c.client_id, num_examples=8) for c in clients],
            )
            results = [
                client.fit(request) for client, request in zip(clients, requests, strict=True)
            ]
            server.aggregate(RoundInfo(round_id=round_id), results)
            if first_v_hat is None:
                first_v_hat = float(server._second_moment["weight"].max())

        # v_hat is a running maximum on every client and is then averaged, so
        # the synchronized value cannot fall between rounds.
        self.assertGreaterEqual(float(server._second_moment["weight"].max()), first_v_hat)
        self.assertTrue(torch.isfinite(server._model_state["weight"]).all())

    def test_the_client_carries_no_state_across_rounds(self) -> None:
        """m, v and v_hat are all reseeded from the broadcast, so a checkpoint
        has nothing model-sized to store per client."""

        client = _client()
        client.fit(FitRequest(round_id=1, client_id="c0", payload=_broadcast()))
        for value in client.get_state().values():
            self.assertNotIsInstance(value, torch.Tensor)


@pytest.mark.fast
class ConfigurationTest(unittest.TestCase):
    def _config(self, **client_extra: Any) -> Any:
        config = copy.deepcopy(_BASE_CONFIG)
        config.server.strategy = "fedlalr"
        config.server.extra["aggregation_weighting"] = "uniform"
        config.client.update_rule = "fedlalr"
        config.client.learning_rate = ALPHA
        config.client.extra = dict(client_extra)
        return config

    def _preflight_errors(self, config: Any) -> list[str]:
        return [
            issue.code for issue in validate_full_config(config).issues if issue.severity == "error"
        ]

    def test_the_paper_defaults_need_no_configuration(self) -> None:
        config = self._config()
        validate_config(config)
        self.assertEqual(self._preflight_errors(config), [])

    def test_server_and_client_must_be_paired(self) -> None:
        server_only = self._config()
        server_only.client.update_rule = "local_sgd"
        with self.assertRaises(ValueError):
            validate_config(server_only)
        self.assertIn("algorithm.fedlalr_client_incompatible", self._preflight_errors(server_only))

        client_only = self._config()
        client_only.server.strategy = "fedavg"
        with self.assertRaises(ValueError):
            validate_config(client_only)
        self.assertIn("algorithm.fedlalr_server_incompatible", self._preflight_errors(client_only))

    def test_alpha_is_required(self) -> None:
        config = self._config()
        config.client.learning_rate = None
        with self.assertRaises(ValueError):
            validate_config(config)
        self.assertIn("algorithm.fedlalr_learning_rate_invalid", self._preflight_errors(config))

    def test_out_of_range_hyperparameters_are_rejected(self) -> None:
        for field, value, code in (
            ("beta1", 1.0, "algorithm.fedlalr_beta1_invalid"),
            ("beta2", -0.1, "algorithm.fedlalr_beta2_invalid"),
            ("epsilon", 0.0, "algorithm.fedlalr_epsilon_invalid"),
        ):
            with self.subTest(field=field):
                config = self._config(**{field: value})
                with self.assertRaises(ValueError):
                    validate_config(config)
                self.assertIn(code, self._preflight_errors(config))

    def test_amp_is_rejected(self) -> None:
        config = self._config()
        config.runtime.use_amp = True
        with self.assertRaises(ValueError):
            validate_config(config)
        self.assertIn("algorithm.fedlalr_amp_unsupported", self._preflight_errors(config))

    def test_the_3x_communication_cost_is_surfaced(self) -> None:
        """Sending x, m and v_hat is three times a FedAvg round in both
        directions. That has to be visible before a sweep is launched."""

        codes = {
            issue.code: issue.severity for issue in validate_full_config(self._config()).issues
        }
        self.assertEqual(codes.get("algorithm.fedlalr_communication_cost"), "info")

    def test_the_documented_defaults_stay_pinned(self) -> None:
        """beta1 and epsilon are the paper's CIFAR settings; beta2 is not (0.995
        there), and the client's comment says so. A change here is a change to
        what every fedlalr config without these keys runs."""

        self.assertEqual(DEFAULT_BETA1, 0.9)
        self.assertEqual(DEFAULT_BETA2, 0.999)
        self.assertEqual(DEFAULT_EPSILON, 1e-8)

    def test_the_factory_builds_a_matched_pair(self) -> None:
        components = build_components(self._config())
        self.assertIsInstance(components.server, FedLALRServer)
        client = components.clients[next(iter(components.clients))]
        self.assertIsInstance(client, TorchFedLALRClient)
        # The server's epsilon comes from client.epsilon: one number, one knob.
        self.assertEqual(components.server.epsilon, client.epsilon)


if __name__ == "__main__":
    unittest.main()
