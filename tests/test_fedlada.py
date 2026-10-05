"""FedLADA (arXiv:2308.00522), as a reference port reads it: the server and the client.

- the server's fold, worked out by hand: ``x + eta_g (x_bar - x)``, ``v`` the
  mean of the ``v_hat_i``, ``g_a`` the mean of the clients' terms, and a refused
  round leaves the server as it was;
- the client's round against the same updates written out here, step by step,
  from autograd gradients of the same task: ``m`` from zero, ``v`` and ``v_hat``
  from the server's ``v``, the step amended by ``g_a``, and the uploaded term
  ``(x_t - x_i) / (alpha_l K)``;
- two identities: at ``lada_alpha`` 0 with ``g_a`` 0 the model never moves; at
  ``lada_alpha`` 1 and ``beta1`` 0 a client round is FedLALR's with
  ``beta1`` 0 (the same ``beta2`` and ``epsilon``), within float32 round-off;
- the config: every setting required, its range checked.
"""

from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from fedbrew.clients.torch_fedlada_client import TorchFedLADAClient
from fedbrew.clients.torch_fedlalr_client import TorchFedLALRClient
from fedbrew.core.config import load_config
from fedbrew.core.protocol import FitRequest, FitResult, RoundInfo
from fedbrew.core.refusal import RunRefused
from fedbrew.core.torch_utils import NonFiniteStateError
from fedbrew.servers.fedlada import FedLADAServer
from tests.test_client_communication_cost import _kwargs

REPO = Path(__file__).resolve().parent.parent


def _state(*values: float) -> dict[str, torch.Tensor]:
    return {"w": torch.tensor(values, dtype=torch.float64)}


def _server(eta_g: float = 0.5) -> FedLADAServer:
    server = FedLADAServer(
        epsilon=1e-3,
        server_learning_rate=eta_g,
        participation_rate=1.0,
        seed=0,
        aggregation_weighting="uniform",
    )
    server._model_state = _state(1.0, -2.0)
    server._model_state_scope = "full"
    server._model_state_metadata = {"model_state_scope": "full"}
    server._ensure_state()
    return server


def _result(client: str, x: tuple[float, ...], v: tuple[float, ...], g: tuple[float, ...]) -> Any:
    return FitResult(
        round_id=1,
        client_id=client,
        num_examples=10,
        payload={
            "model_state": _state(*x),
            "model_state_scope": "full",
            "model_state_metadata": {"model_state_scope": "full"},
            "second_moment_state": _state(*v),
            "amended_direction_state": _state(*g),
        },
        metrics={},
    )


@pytest.mark.fast
class TheServerFoldTest(unittest.TestCase):
    def test_the_state_it_starts_from(self) -> None:
        server = _server()
        self.assertTrue(
            torch.equal(server._second_moment["w"], torch.full((2,), 1e-6, dtype=torch.float64))
        )
        self.assertTrue(
            torch.equal(server._amended_direction["w"], torch.zeros(2, dtype=torch.float64))
        )

    def test_one_round_by_hand(self) -> None:
        server = _server(eta_g=0.5)
        payload = server.aggregate(
            RoundInfo(round_id=1),
            [
                _result("a", (3.0, 0.0), (4.0, 1.0), (0.5, -1.0)),
                _result("b", (5.0, -4.0), (2.0, 3.0), (1.5, 1.0)),
            ],
        )
        # x_bar = (4, -2); x + 0.5 (x_bar - x) = (1 + 1.5, -2 + 0) = (2.5, -2).
        self.assertTrue(
            torch.equal(payload["model_state"]["w"], torch.tensor([2.5, -2.0], dtype=torch.float64))
        )
        self.assertTrue(
            torch.equal(
                payload["second_moment_state"]["w"], torch.tensor([3.0, 2.0], dtype=torch.float64)
            )
        )
        self.assertTrue(
            torch.equal(
                payload["amended_direction_state"]["w"],
                torch.tensor([1.0, 0.0], dtype=torch.float64),
            )
        )

    def test_a_round_that_is_not_finite_is_refused_and_changes_nothing(self) -> None:
        server = _server()
        before = copy.deepcopy(
            (server._model_state, server._second_moment, server._amended_direction)
        )
        with self.assertRaises(NonFiniteStateError):
            server.aggregate(
                RoundInfo(round_id=1), [_result("a", (math.inf, 0.0), (1.0, 1.0), (0.0, 0.0))]
            )
        after = (server._model_state, server._second_moment, server._amended_direction)
        for held, now in zip(before, after, strict=True):
            self.assertTrue(torch.equal(held["w"], now["w"]))

    def test_the_state_survives_a_checkpoint(self) -> None:
        server = _server()
        server.aggregate(RoundInfo(round_id=1), [_result("a", (3.0, 0.0), (4.0, 1.0), (0.5, -1.0))])
        restored = _server()
        restored.load_state(server.save_state())
        self.assertTrue(torch.equal(restored._second_moment["w"], server._second_moment["w"]))
        self.assertTrue(
            torch.equal(restored._amended_direction["w"], server._amended_direction["w"])
        )
        with self.assertRaisesRegex(ValueError, "server_learning_rate"):
            _server(eta_g=0.25).load_state(server.save_state())


def _client(**settings: Any) -> TorchFedLADAClient:
    values = {"beta1": 0.9, "beta2": 0.99, "epsilon": 1e-3, "lada_alpha": 0.3, **settings}
    return TorchFedLADAClient(**_kwargs(local_iterations=3, batch_size=8, metrics=[]), **values)


def _request(client: Any, v: float, g_a: float) -> FitRequest:
    model = client.task.build_model()
    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    return FitRequest(
        round_id=1,
        client_id=client.client_id,
        payload={
            "model_state": state,
            "second_moment_state": {k: torch.full_like(t, v) for k, t in state.items()},
            "amended_direction_state": {k: torch.full_like(t, g_a) for k, t in state.items()},
            "momentum_state": {k: torch.zeros_like(t) for k, t in state.items()},
        },
        total_rounds=5,
    )


def _gradient(task: Any, params: dict[str, torch.Tensor], batch: Any) -> dict[str, torch.Tensor]:
    model = task.build_model()
    model.load_state_dict(params)
    model.zero_grad()
    x, y = batch
    torch.nn.functional.cross_entropy(model(x), y).backward()
    return {name: p.grad.detach().clone() for name, p in model.named_parameters()}


class TheClientRoundTest(unittest.TestCase):
    def test_three_steps_against_the_updates_written_out(self) -> None:
        client = _client()
        request = _request(client, v=0.04, g_a=0.2)
        result = client.fit(request)

        x = {k: t.clone() for k, t in request.payload["model_state"].items()}
        start = {k: t.clone() for k, t in x.items()}
        m = {k: torch.zeros_like(t) for k, t in x.items()}
        v = {k: torch.full_like(t, 0.04) for k, t in x.items()}
        v_hat = {k: t.clone() for k, t in v.items()}
        batch = (client.client_data["train"]["x"], client.client_data["train"]["y"])
        for _ in range(3):
            g = _gradient(client.task, x, batch)
            for k in x:
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v[k] = 0.99 * v[k] + 0.01 * g[k] ** 2
                v_hat[k] = torch.maximum(v_hat[k], v[k])
                x[k] = x[k] - 0.1 * (0.3 * m[k] / v_hat[k].sqrt() + 0.7 * 0.2)
        for k in x:
            torch.testing.assert_close(result.payload["model_state"][k], x[k], rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(
                result.payload["second_moment_state"][k], v_hat[k], rtol=1e-6, atol=1e-9
            )
            torch.testing.assert_close(
                result.payload["amended_direction_state"][k],
                (start[k] - x[k]) / (0.1 * 3),
                rtol=1e-5,
                atol=1e-6,
            )
        self.assertEqual(result.metrics["local_steps"], 3.0)

    def test_at_alpha_zero_with_no_amended_direction_the_model_never_moves(self) -> None:
        client = _client(lada_alpha=0.0)
        request = _request(client, v=0.04, g_a=0.0)
        result = client.fit(request)
        for k, tensor in request.payload["model_state"].items():
            self.assertTrue(torch.equal(result.payload["model_state"][k], tensor))

    def test_at_alpha_one_and_beta1_zero_its_client_is_fedlalrs(self) -> None:
        lada = _client(lada_alpha=1.0, beta1=0.0)
        lalr = TorchFedLALRClient(
            **_kwargs(local_iterations=3, batch_size=8), beta1=0.0, beta2=0.99, epsilon=1e-3
        )
        request = _request(lada, v=0.04, g_a=0.7)
        mine, theirs = lada.fit(request), lalr.fit(request)
        for k, tensor in theirs.payload["model_state"].items():
            torch.testing.assert_close(mine.payload["model_state"][k], tensor, rtol=1e-6, atol=1e-7)
        for k, tensor in theirs.payload["second_moment_state"].items():
            torch.testing.assert_close(
                mine.payload["second_moment_state"][k], tensor, rtol=1e-6, atol=1e-9
            )


@pytest.mark.fast
class TheConfigTest(unittest.TestCase):
    def _load(
        self, client: dict[str, Any] | None = None, server: dict[str, Any] | None = None
    ) -> Any:
        config = yaml.safe_load((REPO / "configs" / "dev" / "fedlada.yaml").read_text())
        config["client"].update(client or {})
        for key, value in (server or {}).items():
            if value is None:
                config["server"].pop(key, None)
            else:
                config["server"][key] = value
        for key, value in (client or {}).items():
            if value is None:
                config["client"].pop(key, None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fedlada.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            return load_config(path)

    def test_the_shipped_config_loads_with_its_strategy_inferred(self) -> None:
        config = self._load()
        self.assertEqual(config.server.strategy, "fedlada")

    def test_every_setting_is_required(self) -> None:
        for name in ("beta1", "beta2", "epsilon", "lada_alpha"):
            with self.subTest(name=name), self.assertRaisesRegex(RunRefused, name):
                self._load(client={name: None})
        with self.assertRaisesRegex(RunRefused, "server_learning_rate"):
            self._load(server={"server_learning_rate": None})

    def test_ranges(self) -> None:
        for client, needle in (
            ({"beta1": 1.0}, "beta1"),
            ({"beta2": -0.1}, "beta2"),
            ({"epsilon": 0.0}, "epsilon"),
            ({"lada_alpha": 1.5}, "lada_alpha"),
        ):
            with self.subTest(client=client), self.assertRaisesRegex(RunRefused, needle):
                self._load(client=client)
        with self.assertRaisesRegex(RunRefused, "server_learning_rate"):
            self._load(server={"server_learning_rate": 0.0})
