"""Finiteness is checked once, on the round's aggregate, and still names the client.

WeightedStateAccumulator checked every client state as it arrived,
``isfinite(t).all()`` per tensor: 14% of an MNIST MLP round at 1000 clients
(perf/report.txt). It now checks the averaged result once, and names the
offending client and tensor from the per-state minima and maxima it records.
What is pinned here, through the real runner:

- NaN, +Inf or -Inf in one tensor of one client at round r is refused at
  round r with the status, detector and tensor the per-client check gave, and
  the client named; with two bad clients the first to arrive is named, as the
  per-client check named it;
- finite client states whose weighted mean overflows are refused too, and
  named as an overflow -- which the per-client check let through (POST-F32);
- a refused round leaves FedLALR's model and both moments as they were.
"""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import torch
import yaml

from fedbrew.core import loop, runner
from fedbrew.core.torch_utils import (
    NonFiniteStateError,
    WeightedStateAccumulator,
    refuse_non_finite_state,
)
from tests.test_checkpoint_size_is_constant_in_rounds import _FAMILIES
from tests.test_reproducibility import _config

ROUNDS = 5
BAD_ROUND = 3
KEY = "net.3.weight"


def _write(root: Path, name: str, family: str) -> Path:
    config = _config(root / name, rounds=ROUNDS, checkpoint=True)
    server, client = _FAMILIES[family]
    config["server"].update(server)
    config["client"].update(client)
    config["client"] = {key: value for key, value in config["client"].items() if value is not None}
    path = root / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _poisoning(where: dict[str, float], field: str = "model_state", key: str = KEY):
    """loop._fit_client, with ``where``'s clients sending ``value`` in one tensor at BAD_ROUND."""

    real = loop._fit_client

    def fit(client: Any, request: Any) -> Any:
        result = real(client, request)
        if request.round_id == BAD_ROUND and request.client_id in where:
            state = dict(result.payload[field])
            state[key] = state[key].clone()
            state[key].view(-1)[0] = where[request.client_id]
            result.payload = dict(result.payload, **{field: state})
        return result

    return fit


@contextmanager
def _checked_per_client() -> Iterator[None]:
    """WeightedStateAccumulator as it was: every client state checked as it arrives,
    and the averaged result not checked at all."""

    real = WeightedStateAccumulator.add

    def add(self: Any, state: Any, weight: float, source: str | None = None) -> None:
        refuse_non_finite_state(
            {k: v for k, v in state.items() if v.is_floating_point()}, "client state"
        )
        real(self, state, weight, source)

    with (
        mock.patch.object(WeightedStateAccumulator, "add", add),
        mock.patch.object(WeightedStateAccumulator, "_refuse", lambda self, key: None),
    ):
        yield


def _terminated(path: Path) -> dict[str, Any]:
    state = runner.run(path, args=None)
    return {"status": state.status, **state.termination}


class _TempRoot(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)


class OneBadClientTest(_TempRoot):
    def test_nan_and_both_infinities_in_every_family(self) -> None:
        values = {"nan": float("nan"), "+inf": float("inf"), "-inf": float("-inf")}
        for family in _FAMILIES:
            for label, value in values.items():
                with self.subTest(family=family, value=label):
                    poison = _poisoning({"client_2": value})
                    with mock.patch.object(loop, "_fit_client", poison):
                        now = _terminated(_write(self.root, f"{family}{label}", family))
                    with mock.patch.object(loop, "_fit_client", poison), _checked_per_client():
                        before = _terminated(_write(self.root, f"{family}{label}-b", family))
                    for verdict in (now, before):
                        self.assertEqual(verdict["status"], "diverged")
                        self.assertEqual(verdict["detector"], "non_finite_client_state")
                        self.assertEqual(verdict["round_id"], BAD_ROUND)
                        self.assertIn(f"tensor {KEY!r}", verdict["reason"])
                    self.assertIn("from client 'client_2'", now["reason"])

    def test_the_first_to_arrive_is_named(self) -> None:
        """client_1's bad tensor comes later in the state than client_3's; client_1 is named."""

        first = _poisoning({"client_1": float("nan")}, key="net.3.bias")
        real_first = first

        def both(client: Any, request: Any) -> Any:
            result = real_first(client, request)
            if request.round_id == BAD_ROUND and request.client_id == "client_3":
                state = dict(result.payload["model_state"])
                state["net.0.weight"] = torch.full_like(state["net.0.weight"], float("inf"))
                result.payload = dict(result.payload, model_state=state)
            return result

        with mock.patch.object(loop, "_fit_client", both):
            verdict = _terminated(_write(self.root, "two", "fedavg"))
        self.assertIn("from client 'client_1' tensor 'net.3.bias'", verdict["reason"])


class AnOverflowingMeanTest(_TempRoot):
    def test_it_is_refused_and_named_as_an_overflow(self) -> None:
        big = dict.fromkeys(("client_0", "client_1", "client_2", "client_3"), 3.0e38)
        with mock.patch.object(loop, "_fit_client", _poisoning(big)):
            now = _terminated(_write(self.root, "overflow", "fedavg"))
        self.assertEqual(now["round_id"], BAD_ROUND)
        self.assertEqual(now["detector"], "non_finite_client_state")
        self.assertIn(f"weighted mean of tensor {KEY!r} overflows torch.float32", now["reason"])
        self.assertNotIn("from client", now["reason"])

    def test_the_per_client_check_let_it_through(self) -> None:
        big = dict.fromkeys(("client_0", "client_1", "client_2", "client_3"), 3.0e38)
        with mock.patch.object(loop, "_fit_client", _poisoning(big)), _checked_per_client():
            before = _terminated(_write(self.root, "overflow-b", "fedavg"))
        self.assertNotEqual(
            (before.get("detector"), before.get("round_id")),
            ("non_finite_client_state", BAD_ROUND),
        )

    def test_the_accumulator_names_it(self) -> None:
        accumulator = WeightedStateAccumulator()
        accumulator.add({"w": torch.tensor([3.0e38])}, 1.0, source="a")
        accumulator.add({"w": torch.tensor([3.0e38])}, 1.0, source="b")
        with self.assertRaises(NonFiniteStateError) as caught:
            accumulator.result()
        self.assertIn("weighted mean of tensor 'w' overflows", str(caught.exception))


class ARefusedRoundChangesNothingTest(_TempRoot):
    def test_fedlalr_keeps_its_model_and_both_moments(self) -> None:
        captured: dict[str, Any] = {}
        real_build = runner.build_components

        def build(config: Any) -> Any:
            captured["components"] = real_build(config)
            return captured["components"]

        path = _write(self.root, "fedlalr", "fedlalr")
        poison = _poisoning({"client_2": float("nan")}, field="second_moment_state")
        with (
            mock.patch.object(loop, "_fit_client", poison),
            mock.patch.object(runner, "build_components", build),
        ):
            verdict = _terminated(path)
        self.assertEqual(verdict["round_id"], BAD_ROUND)
        server = captured["components"].server
        output_dir = Path(yaml.safe_load(path.read_text())["experiment"]["output_dir"])
        saved = torch.load(output_dir / "checkpoints" / "latest.pt", weights_only=False)
        self.assertEqual(saved["round_id"], BAD_ROUND - 1)
        for mine, theirs in (
            (server._model_state, saved["model_state"]),
            (server._momentum, saved["server_state"]["momentum_state"]),
            (server._second_moment, saved["server_state"]["second_moment_state"]),
        ):
            self.assertEqual(sorted(mine), sorted(theirs))
            for key in mine:
                self.assertTrue(torch.equal(mine[key], theirs[key]), key)


if __name__ == "__main__":
    unittest.main()
