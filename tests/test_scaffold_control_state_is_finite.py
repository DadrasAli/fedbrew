"""A non-finite SCAFFOLD control delta is refused before it reaches ``c``.

The model state every client sends is checked by `WeightedStateAccumulator`,
which refuses NaN and Inf. SCAFFOLD's control deltas are summed beside it, not
through it, and nothing checked them: a finite model beside a NaN
``control_delta`` aggregated normally and left NaN in ``server_control``, which
the control update ``c <- c + (1/N) sum(dc_i)`` then keeps for the rest of the
run, and which the next checkpoint writes to disk. FINDINGS.csv POST-F27.

The contract held here is all-or-nothing: a refused round leaves the model and
``c`` bit-for-bit as they were, and the loop records it as the same
``non_finite_client_state`` divergence a non-finite model state produces, so
the last checkpoint on disk is the last healthy one and a resume from it runs.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch
import yaml

from fedbrew.clients.torch_scaffold_client import TorchScaffoldClient
from fedbrew.core import runner
from fedbrew.core.checkpointing import load_checkpoint
from fedbrew.core.protocol import FitResult, RoundInfo
from fedbrew.core.torch_utils import NonFiniteStateError
from fedbrew.servers.scaffold import ScaffoldServer

NON_FINITE = {"nan": float("nan"), "+inf": float("inf"), "-inf": float("-inf")}
FLOAT32_MAX = torch.finfo(torch.float32).max


def _server(num_clients: int = 2) -> ScaffoldServer:
    server = ScaffoldServer(participation_rate=1.0, seed=0)
    server._model_state = {"w": torch.tensor([0.25, -0.5])}
    server._model_state_scope = "full"
    server._model_state_metadata = {"model_state_scope": "full"}
    server._server_control = {"w": torch.tensor([0.125, 0.75])}
    server._num_clients = num_clients
    return server


def _result(client_id: str, model: list[float], delta: list[float]) -> FitResult:
    return FitResult(
        round_id=1,
        client_id=client_id,
        num_examples=4,
        payload={
            "model_state": {"w": torch.tensor(model)},
            "control_delta": {"w": torch.tensor(delta)},
            "model_state_scope": "full",
            "model_state_metadata": {"model_state_scope": "full"},
        },
        metrics={"fit_loss": 1.0},
    )


def _bits(tensor: torch.Tensor) -> bytes:
    """A tensor's bytes, bit for bit, without numpy: a core install has none."""

    return bytes(tensor.detach().reshape(-1).view(torch.uint8).tolist())


def _snapshot(server: ScaffoldServer) -> tuple[bytes, bytes]:
    return (_bits(server._model_state["w"]), _bits(server._server_control["w"]))


@pytest.mark.fast
class TheServerRefusesBeforeItChangesAnythingTest(unittest.TestCase):
    def _refused(self, server: ScaffoldServer, results: list[FitResult]) -> str:
        before = _snapshot(server)
        round_info = RoundInfo(round_id=1, total_rounds=1)
        with self.assertRaises(NonFiniteStateError) as caught:
            server.aggregate_stream(round_info, results)
        self.assertEqual(_snapshot(server), before, "a refused round changed the model or c")
        self.assertEqual(round_info.metrics, {})
        self.assertEqual(server._round_metrics, [])
        return str(caught.exception)

    def test_a_non_finite_control_delta_is_refused_and_both_states_are_kept(self) -> None:
        for name, value in NON_FINITE.items():
            for position in (0, 1):
                with self.subTest(value=name, bad_client=position):
                    results = [
                        _result("a", [1.0, 1.0], [0.5, 0.5]),
                        _result("b", [2.0, 2.0], [0.25, 0.25]),
                    ]
                    results[position].payload["control_delta"]["w"][1] = value
                    message = self._refused(_server(), results)
                    client = results[position].client_id
                    self.assertIn(f"control_delta from client {client!r}", message)

    def test_finite_deltas_whose_sum_leaves_the_float_range_are_refused(self) -> None:
        message = self._refused(
            _server(),
            [
                _result("a", [1.0, 1.0], [FLOAT32_MAX, 0.0]),
                _result("b", [2.0, 2.0], [FLOAT32_MAX, 0.0]),
            ],
        )
        self.assertIn("summed control_delta", message)

    def test_a_finite_sum_that_takes_c_out_of_range_is_refused(self) -> None:
        server = _server(num_clients=1)
        server._server_control = {"w": torch.tensor([FLOAT32_MAX, 0.0])}
        message = self._refused(server, [_result("a", [1.0, 1.0], [FLOAT32_MAX, 0.0])])
        self.assertIn("updated server_control", message)

    def test_a_non_finite_model_state_leaves_c_alone_too(self) -> None:
        self._refused(_server(), [_result("a", [float("nan"), 1.0], [0.5, 0.5])])

    def test_finite_deltas_update_both_states(self) -> None:
        server = _server(num_clients=4)
        server.aggregate_stream(
            RoundInfo(round_id=1, total_rounds=1),
            [_result("a", [1.0, 1.0], [0.5, 1.0]), _result("b", [3.0, 2.0], [0.25, 1.0])],
        )
        self.assertTrue(torch.equal(server._model_state["w"], torch.tensor([2.0, 1.5])))
        # c + (1/N) sum(dc_i), N = 4, the whole roster.
        self.assertTrue(torch.equal(server._server_control["w"], torch.tensor([0.3125, 1.25])))


def _scaffold_config(directory: Path) -> Path:
    raw = yaml.safe_load(Path("configs/dev/smoke.yaml").read_text(encoding="utf-8"))
    raw["experiment"]["output_dir"] = str(directory / "run")
    raw["server"]["strategy"] = "scaffold"
    client = raw["client"]
    client["update_rule"] = "scaffold"
    for key in ("momentum", "weight_decay", "nesterov", "learning_rate_schedule"):
        client.pop(key)
    client.pop("min_learning_rate")
    raw["runtime"]["checkpointing"] = {
        "enabled": True,
        "save_last": True,
        "save_best": False,
        "save_every_round": False,
        "keep_last": 0,
    }
    raw["defaults"]["global_rounds"] = 4
    path = directory / "scaffold.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


class TheLoopRecordsItAndTheLastCheckpointResumesTest(unittest.TestCase):
    """Through `fedbrew run`'s own path, with one client's round-3 delta poisoned."""

    BAD_ROUND = 3

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls._directory.name)
        cls.config = _scaffold_config(cls.root)
        cls.output = cls.root / "run"
        original = TorchScaffoldClient.fit

        def poisoned(client: TorchScaffoldClient, request: Any) -> FitResult:
            result = original(client, request)
            if request.round_id == cls.BAD_ROUND:
                delta = result.payload["control_delta"]
                key = next(iter(delta))
                delta[key] = torch.full_like(delta[key], float("nan"))
            return result

        with mock.patch.object(TorchScaffoldClient, "fit", poisoned):
            cls.state = runner.run(cls.config, runner.parse_args(["--quiet"]))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def test_the_run_is_recorded_as_diverged_like_a_non_finite_model(self) -> None:
        self.assertEqual(self.state.status, "diverged")
        self.assertEqual(self.state.termination["detector"], "non_finite_client_state")
        self.assertEqual(self.state.termination["round_id"], self.BAD_ROUND)
        self.assertIn("control_delta", self.state.termination["reason"])
        record = json.loads((self.output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "diverged")

    def test_the_last_checkpoint_is_the_last_healthy_round_and_finite(self) -> None:
        checkpoint = load_checkpoint(self.output / "checkpoints" / "latest.pt")
        self.assertEqual(checkpoint["round_id"], self.BAD_ROUND - 1)
        control = checkpoint["server_state"]["server_control"]
        self.assertTrue(control)
        for key, tensor in control.items():
            self.assertTrue(torch.isfinite(tensor).all(), key)

    def test_a_resume_from_it_runs_to_the_end(self) -> None:
        resumed_directory = self.root / "resumed"
        # A copy, so the class's own artifacts stay as the poisoned run left them.
        shutil.copytree(self.output, resumed_directory)
        latest = resumed_directory / "checkpoints" / "latest.pt"
        state = runner.run(
            self.config,
            runner.parse_args(
                ["--quiet", "--output-dir", str(resumed_directory), "--resume-from", str(latest)]
            ),
        )
        self.assertEqual(state.status, "completed")
        self.assertTrue(state.resumed)
        self.assertEqual([r.round_id for r in state.metrics_history], [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
