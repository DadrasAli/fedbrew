"""Every server refuses a fit result whose federated state cannot join its own.

FedAvg checked each result's state scope and, for adapters, the base-model and
adapter identity (`validate_federated_state_metadata`) before folding it.
SCAFFOLD and FedLALR override the fold and did not: with identical tensor
shapes, an adapter-scoped result sent to a full-state server was refused by
FedAvg and averaged in by both of the others. All three now run the one check,
`FedAvgServer._compatible_model_state`, which also holds a result's keys and
shapes to the server's own state. FINDINGS.csv POST-F28.

Each case runs on all three servers, and every refusal is checked to leave the
server's persistent state -- model, control variate or moments, and the round
record -- bit-for-bit as it was, including when the bad result arrives after a
good one has already been folded.
"""

from __future__ import annotations

import unittest
from typing import Any

import pytest
import torch

from fedbrew.core.protocol import FitResult, RoundInfo
from fedbrew.servers.fedavg import FedAvgServer
from fedbrew.servers.fedlalr import FedLALRServer
from fedbrew.servers.scaffold import ScaffoldServer

pytestmark = pytest.mark.fast

FULL = {"model_state_scope": "full"}
ADAPTER = {
    "model_state_scope": "adapter",
    "base_model_identifier": "org/base",
    "base_model_resolved_revision": "abc123",
    "adapter_name": "federated_sft",
    "lora_config": {"r": 8, "lora_alpha": 16},
}
SERVERS = ("fedavg", "scaffold", "fedlalr")


def _server(kind: str, metadata: dict[str, Any]) -> FedAvgServer:
    if kind == "fedavg":
        server: FedAvgServer = FedAvgServer(participation_rate=1.0, seed=0)
    elif kind == "scaffold":
        server = ScaffoldServer(participation_rate=1.0, seed=0)
    else:
        server = FedLALRServer(epsilon=1e-8, participation_rate=1.0, seed=0)
    server._model_state = {"w": torch.tensor([0.5, -0.25])}
    server._model_state_scope = metadata["model_state_scope"]
    server._model_state_metadata = dict(metadata)
    server._state_validated = True
    if isinstance(server, ScaffoldServer):
        server._server_control = {"w": torch.tensor([0.125, 0.0625])}
        server._num_clients = 2
    if isinstance(server, FedLALRServer):
        server._ensure_optimizer_state()
    return server


def _result(
    kind: str,
    metadata: dict[str, Any] | None,
    *,
    client_id: str = "a",
    state: dict[str, torch.Tensor] | None = None,
    scope: str | None = None,
) -> FitResult:
    state = state if state is not None else {"w": torch.tensor([1.0, 2.0])}
    payload: dict[str, Any] = {"model_state": state}
    if metadata is not None:
        payload["model_state_metadata"] = dict(metadata)
        payload["model_state_scope"] = scope or metadata["model_state_scope"]
    elif scope is not None:
        payload["model_state_scope"] = scope
    if kind == "scaffold":
        payload["control_delta"] = {key: torch.full_like(v, 0.5) for key, v in state.items()}
    if kind == "fedlalr":
        payload["momentum_state"] = {key: torch.zeros_like(v) for key, v in state.items()}
        payload["second_moment_state"] = {key: torch.ones_like(v) for key, v in state.items()}
    return FitResult(round_id=1, client_id=client_id, num_examples=3, payload=payload)


def _persistent(server: FedAvgServer) -> dict[str, Any]:
    """Everything a round may change, as bytes and plain values."""

    def raw(state: dict[str, torch.Tensor] | None) -> dict[str, bytes] | None:
        if state is None:
            return None
        # Bit for bit, and torch only: a core install has no numpy for .numpy().
        return {
            key: bytes(value.detach().reshape(-1).view(torch.uint8).tolist())
            for key, value in state.items()
        }

    snapshot: dict[str, Any] = {
        "model": raw(server._model_state),
        "scope": server._model_state_scope,
        "metadata": dict(server._model_state_metadata or {}),
        "round_metrics": list(server._round_metrics),
    }
    if isinstance(server, ScaffoldServer):
        snapshot["control"] = raw(server._server_control)
    if isinstance(server, FedLALRServer):
        snapshot["momentum"] = raw(server._momentum)
        snapshot["second_moment"] = raw(server._second_moment)
    return snapshot


class EveryServerRunsTheOneCheckTest(unittest.TestCase):
    def _accepted(self, kind: str, server_metadata: dict, result_metadata: dict) -> None:
        server = _server(kind, server_metadata)
        before = _persistent(server)
        server.aggregate_stream(
            RoundInfo(round_id=1, total_rounds=1),
            [_result(kind, result_metadata), _result(kind, result_metadata, client_id="b")],
        )
        self.assertTrue(torch.equal(server._model_state["w"], torch.tensor([1.0, 2.0])))
        self.assertNotEqual(_persistent(server)["model"], before["model"])

    def _refused(self, kind: str, server: FedAvgServer, *results: FitResult) -> str:
        before = _persistent(server)
        round_info = RoundInfo(round_id=1, total_rounds=1)
        with self.assertRaises(ValueError) as caught:
            server.aggregate_stream(round_info, list(results))
        self.assertEqual(_persistent(server), before, f"{kind}: a refused round changed state")
        self.assertEqual(round_info.metrics, {})
        return str(caught.exception)

    def test_full_to_full_is_accepted(self) -> None:
        for kind in SERVERS:
            with self.subTest(server=kind):
                self._accepted(kind, FULL, FULL)

    def test_adapter_to_adapter_is_accepted(self) -> None:
        """Server-side only: which rules may train adapter state at all is
        decided at config load (chapter 07), and this is the backstop behind
        that, so all three servers take a matching adapter result."""

        for kind in SERVERS:
            with self.subTest(server=kind):
                self._accepted(kind, ADAPTER, ADAPTER)

    def test_an_adapter_result_is_refused_by_a_full_server(self) -> None:
        for kind in SERVERS:
            for position in (0, 1):
                with self.subTest(server=kind, bad_client=position):
                    results = [_result(kind, FULL), _result(kind, FULL, client_id="b")]
                    results[position] = _result(kind, ADAPTER, client_id="x")
                    message = self._refused(kind, _server(kind, FULL), *results)
                    self.assertIn("client 'x' has incompatible model state scopes", message)
                    self.assertIn("expected 'full', received 'adapter'", message)

    def test_a_full_result_is_refused_by_an_adapter_server(self) -> None:
        for kind in SERVERS:
            for result_metadata in (FULL, None):
                with self.subTest(server=kind, metadata=result_metadata):
                    message = self._refused(
                        kind, _server(kind, ADAPTER), _result(kind, result_metadata)
                    )
                    self.assertIn("expected 'adapter', received 'full'", message)

    def test_each_adapter_identity_field_must_match(self) -> None:
        for kind in SERVERS:
            for field in (
                "base_model_identifier",
                "base_model_resolved_revision",
                "adapter_name",
                "lora_config",
            ):
                with self.subTest(server=kind, field=field):
                    mismatched = dict(ADAPTER, **{field: "something else"})
                    message = self._refused(
                        kind,
                        _server(kind, ADAPTER),
                        _result(kind, ADAPTER),
                        _result(kind, mismatched, client_id="b"),
                    )
                    self.assertIn(f"incompatible adapter metadata for {field}", message)
                with self.subTest(server=kind, missing=field):
                    incomplete = {k: v for k, v in ADAPTER.items() if k != field}
                    message = self._refused(kind, _server(kind, ADAPTER), _result(kind, incomplete))
                    self.assertIn(f"missing compatibility field {field!r}", message)

    def test_adapter_state_without_metadata_is_refused(self) -> None:
        for kind in SERVERS:
            with self.subTest(server=kind):
                message = self._refused(
                    kind, _server(kind, ADAPTER), _result(kind, None, scope="adapter")
                )
                self.assertIn("adapter state requires model_state_metadata", message)

    def test_conflicting_scopes_in_one_payload_are_refused(self) -> None:
        for kind in SERVERS:
            with self.subTest(server=kind):
                message = self._refused(
                    kind, _server(kind, FULL), _result(kind, FULL, scope="adapter")
                )
                self.assertIn("conflicting model state scopes", message)

    def test_a_state_with_other_keys_or_shapes_is_refused(self) -> None:
        for kind in SERVERS:
            with self.subTest(server=kind, case="keys"):
                message = self._refused(
                    kind,
                    _server(kind, FULL),
                    _result(kind, FULL, state={"v": torch.tensor([1.0, 2.0])}),
                )
                self.assertIn("missing ['w'], unexpected ['v']", message)
            with self.subTest(server=kind, case="shape"):
                message = self._refused(
                    kind,
                    _server(kind, FULL),
                    _result(kind, FULL, state={"w": torch.tensor([1.0, 2.0, 3.0])}),
                )
                self.assertIn("has shape (3,), the server's has (2,)", message)


if __name__ == "__main__":
    unittest.main()
