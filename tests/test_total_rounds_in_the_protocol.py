"""T reaches the client through the protocol, not through its constructor.

A client whose local step depends on T, the rounds in the run, needs it at every
local step, so a fit request has to say how many rounds the run has. The loop knows T;
it sets it on RoundInfo, and each server copies it into every FitRequest it
builds. Pinned here: every built-in server copies it faithfully, and in a real
run every request a client fits carries the run's T. An object built outside a
run carries None, which is what a client that needs T refuses.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import torch

from fedbrew.core import loop, runner
from fedbrew.core.protocol import ClientInfo, FitRequest, RoundInfo
from fedbrew.servers.fedavg import FedAvgServer
from fedbrew.servers.fedlalr import FedLALRServer
from fedbrew.servers.fedopt import FedOptServer
from fedbrew.servers.scaffold import ScaffoldServer
from tests.test_bernoulli_participation import _write_config


def _ready(server: FedAvgServer) -> FedAvgServer:
    server._model_state = {"w": torch.zeros(2)}
    server._model_state_scope = "full"
    server._model_state_metadata = {"model_state_scope": "full"}
    return server


def _servers() -> dict[str, FedAvgServer]:
    """Every built-in server class; `centralized` is built as a FedAvgServer."""

    common: dict[str, Any] = {"participation_rate": 1.0, "seed": 0}
    scaffold = _ready(ScaffoldServer(**common))
    scaffold._server_control = {"w": torch.zeros(2)}
    return {
        "fedavg": _ready(FedAvgServer(**common)),
        "scaffold": scaffold,
        "fedlalr": _ready(FedLALRServer(epsilon=1e-8, **common)),
        "fedopt": _ready(
            FedOptServer(
                server_optimizer="fedadam",
                server_learning_rate=0.01,
                beta1=0.9,
                beta2=0.99,
                tau=1e-3,
                **common,
            )
        ),
    }


class EveryBuiltInServerCopiesItTest(unittest.TestCase):
    ROSTER = [ClientInfo(client_id=f"c{index}", num_examples=4) for index in range(3)]

    def test_each_request_carries_the_rounds_its_round_info_does(self) -> None:
        for name, server in _servers().items():
            with self.subTest(strategy=name):
                requests = server.configure_round(
                    RoundInfo(round_id=2, total_rounds=7), self.ROSTER
                )
                self.assertEqual(len(requests), len(self.ROSTER))
                self.assertEqual([request.total_rounds for request in requests], [7, 7, 7])

    def test_a_round_info_without_it_is_copied_as_without_it(self) -> None:
        """Copied, not supplied: a server that invented a T would hide a loop
        that forgot to set one."""

        for name, server in _servers().items():
            with self.subTest(strategy=name):
                requests = server.configure_round(RoundInfo(round_id=2), self.ROSTER)
                self.assertEqual({request.total_rounds for request in requests}, {None})


class OutsideARunTest(unittest.TestCase):
    def test_a_hand_built_request_or_round_carries_none(self) -> None:
        self.assertIsNone(FitRequest(round_id=1, client_id="c0").total_rounds)
        self.assertIsNone(RoundInfo(round_id=1).total_rounds)


class ARealRunDeliversItTest(unittest.TestCase):
    def test_every_request_a_client_fits_carries_the_runs_rounds(self) -> None:
        seen: list[tuple[int, int | None]] = []
        fit_client = loop._fit_client

        def recording(client: Any, request: FitRequest) -> Any:
            seen.append((request.round_id, request.total_rounds))
            return fit_client(client, request)

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(loop, "_fit_client", recording),
        ):
            path = _write_config(Path(directory), rounds=3, participation_rate=1.0)
            runner.run(path, runner.parse_args(["--quiet"]))

        # Two clients at rate 1.0, three rounds.
        self.assertEqual(seen, [(round_id, 3) for round_id in (1, 2, 3) for _ in range(2)])


if __name__ == "__main__":
    unittest.main()
