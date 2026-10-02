"""FAFED (AAAI 2023, arXiv:2212.00974), as AdaFed's port reads it: the server, the client, the loop.

- the server: the initial moments are the mean and mean square of the clients'
  initial gradients, a round's fold is ``x_bar - eta m / (sqrt(v) + rho)`` with
  the averaged moments, round 1's request says it is the first and no later one
  does, partial participation is refused, and the state survives a checkpoint;
- the client: its initial answer is ``g_0`` at ``x_0`` and keeps ``x_0``; a round
  is the updates written out here, step by step, from autograd gradients of the
  same task, round 1's extra step included and the last step left to the server;
- the loop: the initial pass runs once, before round 1 and only then; a run
  stopped and resumed from its latest checkpoint is the run uninterrupted, the
  clients' previous iterates carried by the checkpoint;
- the config: every setting required, full participation.
"""

from __future__ import annotations

import argparse
import copy
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch
import yaml

from fedbrew.clients.torch_fafed_client import TorchFAFEDClient
from fedbrew.core import runner
from fedbrew.core.checkpointing import load_checkpoint
from fedbrew.core.config import load_config
from fedbrew.core.protocol import ClientInfo, FitRequest, FitResult, RoundInfo
from fedbrew.core.refusal import RunRefused
from fedbrew.servers.fafed import FAFEDServer
from tests.test_batched_executor_tolerance import _rows
from tests.test_client_communication_cost import _kwargs
from tests.test_resident_round import _stopping

REPO = Path(__file__).resolve().parent.parent


def _state(*values: float) -> dict[str, torch.Tensor]:
    return {"w": torch.tensor(values, dtype=torch.float64)}


def _server() -> FAFEDServer:
    server = FAFEDServer(
        learning_rate=0.5,
        fafed_rho=0.25,
        participation_rate=1.0,
        seed=0,
        aggregation_weighting="uniform",
    )
    server._model_state = _state(1.0, -2.0)
    server._model_state_scope = "full"
    server._model_state_metadata = {"model_state_scope": "full"}
    return server


def _initial(client: str, *g: float) -> FitResult:
    return FitResult(
        round_id=0,
        client_id=client,
        num_examples=10,
        payload={"initial_gradient_state": _state(*g)},
    )


def _result(client: str, x: tuple[float, ...], m: tuple[float, ...], v: tuple[float, ...]) -> Any:
    return FitResult(
        round_id=1,
        client_id=client,
        num_examples=10,
        payload={
            "model_state": _state(*x),
            "model_state_scope": "full",
            "model_state_metadata": {"model_state_scope": "full"},
            "momentum_state": _state(*m),
            "second_moment_state": _state(*v),
        },
        metrics={},
    )


@pytest.mark.fast
class TheServerTest(unittest.TestCase):
    def test_the_initial_moments_are_the_mean_and_mean_square(self) -> None:
        server = _server()
        requests = server.initial_requests([ClientInfo("a", 10), ClientInfo("b", 10)])
        self.assertEqual([r.client_id for r in requests], ["a", "b"])
        self.assertTrue(all(r.payload["fafed_initial"] for r in requests))
        server.absorb_initial([_initial("a", 1.0, -3.0), _initial("b", 3.0, 1.0)])
        self.assertTrue(
            torch.equal(server._momentum["w"], torch.tensor([2.0, -1.0], dtype=torch.float64))
        )
        self.assertTrue(
            torch.equal(server._second_moment["w"], torch.tensor([5.0, 5.0], dtype=torch.float64))
        )

    def test_one_round_by_hand(self) -> None:
        server = _server()
        server.absorb_initial([_initial("a", 1.0, 1.0)])
        payload = server.aggregate(
            RoundInfo(round_id=1),
            [
                _result("a", (3.0, 0.0), (1.0, -1.0), (0.04, 1.0)),
                _result("b", (5.0, -4.0), (3.0, 1.0), (0.04, 3.0)),
            ],
        )
        # x_bar (4, -2), m (2, 0), v (0.04, 2), den (0.2 + 0.25, sqrt 2 + 0.25).
        expected = torch.tensor([4.0 - 2.0 * 0.5 / 0.45, -2.0], dtype=torch.float64)
        torch.testing.assert_close(payload["model_state"]["w"], expected, rtol=0, atol=1e-15)
        self.assertTrue(
            torch.equal(
                payload["momentum_state"]["w"], torch.tensor([2.0, 0.0], dtype=torch.float64)
            )
        )

    def test_only_round_one_is_the_first(self) -> None:
        server = _server()
        server.absorb_initial([_initial("a", 1.0, 1.0)])
        infos = [ClientInfo("a", 10)]
        self.assertTrue(
            server.configure_round(RoundInfo(round_id=1), infos)[0].payload["fafed_first_round"]
        )
        server.aggregate(RoundInfo(round_id=1), [_result("a", (1.0, 1.0), (1.0, 1.0), (1.0, 1.0))])
        self.assertFalse(
            server.configure_round(RoundInfo(round_id=2), infos)[0].payload["fafed_first_round"]
        )

    def test_a_round_before_the_initial_pass_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "initial pass"):
            _server().configure_round(RoundInfo(round_id=1), [ClientInfo("a", 10)])

    def test_partial_participation_is_refused(self) -> None:
        for kwargs in (
            {"participation_rate": 0.5},
            {"participation_rate": None, "participation_probability": 0.9},
        ):
            with (
                self.subTest(**kwargs),
                self.assertRaisesRegex(ValueError, "participation must be 1"),
            ):
                FAFEDServer(
                    learning_rate=0.5,
                    fafed_rho=0.25,
                    seed=0,
                    **{"participation_rate": None, **kwargs},
                )

    def test_the_state_survives_a_checkpoint(self) -> None:
        server = _server()
        server.absorb_initial([_initial("a", 1.0, 1.0)])
        server.aggregate(
            RoundInfo(round_id=1), [_result("a", (3.0, 0.0), (1.0, -1.0), (0.04, 1.0))]
        )
        restored = _server()
        restored.load_state(server.save_state())
        self.assertEqual(restored._rounds, 1)
        self.assertTrue(torch.equal(restored._momentum["w"], server._momentum["w"]))
        self.assertTrue(torch.equal(restored._second_moment["w"], server._second_moment["w"]))


def _client(**settings: Any) -> TorchFAFEDClient:
    values = {"beta2": 0.9, "fafed_alpha": 0.3, "fafed_rho": 0.05, **settings}
    return TorchFAFEDClient(**_kwargs(local_iterations=3, batch_size=8, metrics=[]), **values)


def _gradient(task: Any, params: dict[str, torch.Tensor], batch: Any) -> dict[str, torch.Tensor]:
    model = task.build_model()
    model.load_state_dict(params)
    model.zero_grad()
    x, y = batch
    torch.nn.functional.cross_entropy(model(x), y).backward()
    return {name: p.grad.detach().clone() for name, p in model.named_parameters()}


class TheClientTest(unittest.TestCase):
    def _start(self, client: TorchFAFEDClient) -> tuple[dict[str, torch.Tensor], Any]:
        x0 = {k: t.detach().clone() for k, t in client.task.build_model().state_dict().items()}
        initial = client.fit(
            FitRequest(
                round_id=0, client_id="c0", payload={"model_state": x0, "fafed_initial": True}
            )
        )
        batch = (client.client_data["train"]["x"], client.client_data["train"]["y"])
        for k, g in _gradient(client.task, x0, batch).items():
            torch.testing.assert_close(initial.payload["initial_gradient_state"][k], g)
        return x0, batch

    def test_round_one_against_the_updates_written_out(self) -> None:
        client = _client()
        x0, batch = self._start(client)
        m0 = {k: torch.full_like(t, 0.2) for k, t in x0.items()}
        v0 = {k: torch.full_like(t, 0.01) for k, t in x0.items()}
        payload = {
            "model_state": x0,
            "momentum_state": m0,
            "second_moment_state": v0,
            "fafed_first_round": True,
        }
        result = client.fit(FitRequest(round_id=1, client_id="c0", payload=payload, total_rounds=5))

        den = {k: v.sqrt() + 0.05 for k, v in v0.items()}
        m = {k: t.clone() for k, t in m0.items()}
        v = {k: t.clone() for k, t in v0.items()}
        prev = {k: t.clone() for k, t in x0.items()}
        x = {k: x0[k] - m0[k] * 0.1 for k in x0}
        for step in range(3):
            g, gp = _gradient(client.task, x, batch), _gradient(client.task, prev, batch)
            for k in x:
                m[k] = g[k] + 0.7 * (m[k] - gp[k])
                v[k] = 0.9 * v[k] + 0.1 * g[k] ** 2
                prev[k] = x[k].clone()
                if step < 2:
                    x[k] = x[k] - m[k] * 0.1 / den[k]
        for k in x:
            torch.testing.assert_close(result.payload["model_state"][k], x[k], rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(
                result.payload["momentum_state"][k], m[k], rtol=1e-6, atol=1e-7
            )
            torch.testing.assert_close(
                result.payload["second_moment_state"][k], v[k], rtol=1e-6, atol=1e-9
            )
            torch.testing.assert_close(
                client.get_state()["previous_iterate"][k], x[k], rtol=1e-6, atol=1e-7
            )
        self.assertEqual(result.metrics["local_steps"], 3.0)

    def test_a_round_without_a_previous_iterate_is_refused(self) -> None:
        client = _client()
        x0 = {k: t.detach().clone() for k, t in client.task.build_model().state_dict().items()}
        zeros = {k: torch.zeros_like(t) for k, t in x0.items()}
        payload = {"model_state": x0, "momentum_state": zeros, "second_moment_state": zeros}
        with self.assertRaisesRegex(ValueError, "no previous iterate"):
            client.fit(FitRequest(round_id=2, client_id="c0", payload=payload))


class TheLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def _config(self, name: str) -> tuple[Path, Path]:
        config = yaml.safe_load((REPO / "configs" / "dev" / "fafed.yaml").read_text())
        config["experiment"]["output_dir"] = str(self.root / name)
        config["schedule"]["rounds"] = 5
        config["runtime"]["flush_every"] = 2
        config["runtime"]["checkpointing"] = {
            "enabled": True,
            "save_last": True,
            "save_best": False,
        }
        path = self.root / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path, Path(config["experiment"]["output_dir"])

    def test_the_initial_pass_runs_once_before_round_one(self) -> None:
        path, _ = self._config("once")
        with mock.patch.object(
            FAFEDServer, "absorb_initial", autospec=True, side_effect=FAFEDServer.absorb_initial
        ) as absorbed:
            runner.run(path, args=None)
        self.assertEqual(absorbed.call_count, 1)

    def test_stopped_and_resumed_is_the_run_uninterrupted(self) -> None:
        whole, whole_out = self._config("whole")
        runner.run(whole, args=None)
        stopped, out = self._config("stopped")
        with mock.patch.object(runner, "_round_progress_reporter", _stopping(3)):
            with self.assertRaises(KeyboardInterrupt):
                runner.run(stopped, args=None)
        self.assertEqual(load_checkpoint(out / "checkpoints" / "latest.pt")["round_id"], 2)
        with mock.patch.object(FAFEDServer, "absorb_initial") as absorbed:
            runner.run(stopped, args=argparse.Namespace(resume_latest=True))
        absorbed.assert_not_called()
        from tests.test_reproducibility import TIMING

        for row_a, row_b in zip(
            _rows(out / "round_metrics.csv"), _rows(whole_out / "round_metrics.csv"), strict=True
        ):
            self.assertEqual(
                {k: v for k, v in row_a.items() if k not in TIMING},
                {k: v for k, v in row_b.items() if k not in TIMING},
            )


@pytest.mark.fast
class TheConfigTest(unittest.TestCase):
    def _load(
        self, client: dict[str, Any] | None = None, server: dict[str, Any] | None = None
    ) -> Any:
        config = yaml.safe_load((REPO / "configs" / "dev" / "fafed.yaml").read_text())
        for section, changes in (("client", client or {}), ("server", server or {})):
            for key, value in changes.items():
                if value is None:
                    config[section].pop(key, None)
                else:
                    config[section][key] = value
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fafed.yaml"
            path.write_text(yaml.safe_dump(copy.deepcopy(config)), encoding="utf-8")
            return load_config(path)

    def test_the_shipped_config_loads_with_its_strategy_inferred(self) -> None:
        self.assertEqual(self._load().server.strategy, "fafed")

    def test_every_setting_is_required(self) -> None:
        for name in ("beta2", "fafed_alpha", "fafed_rho"):
            with self.subTest(name=name), self.assertRaisesRegex(RunRefused, name):
                self._load(client={name: None})

    def test_full_participation(self) -> None:
        with self.assertRaisesRegex(RunRefused, "every client every round"):
            self._load(server={"participation_rate": 0.5})
        with self.assertRaisesRegex(RunRefused, "every client every round"):
            self._load(server={"participation_rate": None, "participation_probability": 0.9})

    def test_ranges(self) -> None:
        for client, needle in (
            ({"beta2": 1.0}, "beta2"),
            ({"fafed_alpha": -0.1}, "fafed_alpha"),
            ({"fafed_rho": 0.0}, "fafed_rho"),
        ):
            with self.subTest(client=client), self.assertRaisesRegex(RunRefused, needle):
                self._load(client=client)
