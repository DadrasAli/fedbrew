"""Unit tests for the FedOpt family of server strategies.

Covers the four server optimizers in ``fedbrew/servers/fedopt.py`` against Algorithm 2 of
Reddi et al., "Adaptive Federated Optimization" (arXiv:2003.00295):

    delta_t = mean_i(x_i^K) - x_t
    m_t     = beta1 * m_{t-1} + (1 - beta1) * delta_t
    v_t     = v_{t-1} + delta_t^2                                     (FedAdagrad)
    v_t     = v_{t-1} - (1 - beta2) * delta_t^2 * sign(v_{t-1} - delta_t^2)  (FedYogi)
    v_t     = beta2 * v_{t-1} + (1 - beta2) * delta_t^2               (FedAdam)
    x_{t+1} = x_t + eta * m_t / (sqrt(v_t) + tau)

The tests drive ``_apply_fedopt_update`` directly with hand-computable deltas so the
expected values can be written out in closed form.
"""

from __future__ import annotations

import unittest

import pytest
import torch

from fedbrew.servers.fedopt import (
    FedOptServer,
    _normalize_server_optimizer,
    unread_fedopt_hyperparameters,
)

pytestmark = pytest.mark.fast

TAU = 0.1
ETA = 2.0
BETA1 = 0.9
BETA2 = 0.99


def _server(optimizer: str, *, beta1: float = BETA1, tau: float = TAU) -> FedOptServer:
    """Build a FedOpt server with a one-parameter model state already in place.

    ``beta2`` and ``tau`` are passed only to the optimizers that read them --
    fedavgm keeps no second moment and fedadagrad never decays the one it
    keeps, and the constructor refuses a value either would ignore. See
    ``UNREAD_FEDOPT_HYPERPARAMETERS`` and
    ``tests/test_fedopt_unread_hyperparameters.py``.
    """

    _, unread = unread_fedopt_hyperparameters(optimizer)
    server = FedOptServer(
        server_optimizer=optimizer,
        server_learning_rate=ETA,
        beta1=beta1,
        beta2=None if "beta2" in unread else BETA2,
        tau=None if "tau" in unread else tau,
        participation_rate=1.0,
        seed=0,
    )
    server._model_state = {"w": torch.zeros(1)}
    return server


def _delta(value: float) -> dict[str, torch.Tensor]:
    return {"w": torch.tensor([value])}


def _value(state: dict[str, torch.Tensor]) -> float:
    return float(state["w"].item())


class FedOptInitializationTest(unittest.TestCase):
    """The second-moment accumulator must start at tau^2, not zero."""

    def test_v_starts_at_tau_squared(self) -> None:
        # With delta = 0, Adagrad (v + 0) and Yogi (v - 0) hold v at its initial
        # value, while Adam's EMA still decays it by beta2. Either way the value
        # observed after one zero update pins down what v was initialized to.
        expected = {
            "fedadagrad": TAU**2,
            "fedyogi": TAU**2,
            "fedadam": BETA2 * TAU**2,
        }
        for optimizer, expected_v in expected.items():
            with self.subTest(optimizer=optimizer):
                server = _server(optimizer)
                server._apply_fedopt_update(_delta(0.0))
                self.assertAlmostEqual(_value(server._v), expected_v, places=7)

    def test_fedavgm_does_not_allocate_v(self) -> None:
        server = _server("fedavgm")
        server._apply_fedopt_update(_delta(1.0))
        self.assertIsNone(server._v)


class FedOptFirstStepTest(unittest.TestCase):
    """Closed-form checks of a single update for each optimizer."""

    def test_fedadagrad_first_step(self) -> None:
        server = _server("fedadagrad", beta1=0.0)
        new_state = server._apply_fedopt_update(_delta(1.0))
        # beta1=0 -> m = delta = 1;  v = tau^2 + 1;  x = 0 + eta*1/(sqrt(v)+tau)
        expected_v = TAU**2 + 1.0
        self.assertAlmostEqual(_value(server._v), expected_v, places=6)
        self.assertAlmostEqual(_value(server._m), 1.0, places=6)
        expected = ETA * 1.0 / (expected_v**0.5 + TAU)
        self.assertAlmostEqual(_value(new_state), expected, places=6)

    def test_fedadam_first_step(self) -> None:
        server = _server("fedadam")
        new_state = server._apply_fedopt_update(_delta(1.0))
        expected_m = (1.0 - BETA1) * 1.0
        expected_v = BETA2 * TAU**2 + (1.0 - BETA2) * 1.0
        self.assertAlmostEqual(_value(server._m), expected_m, places=6)
        self.assertAlmostEqual(_value(server._v), expected_v, places=6)
        expected = ETA * expected_m / (expected_v**0.5 + TAU)
        self.assertAlmostEqual(_value(new_state), expected, places=6)

    def test_fedyogi_first_step(self) -> None:
        server = _server("fedyogi")
        new_state = server._apply_fedopt_update(_delta(1.0))
        expected_m = (1.0 - BETA1) * 1.0
        # v0 = tau^2 < delta^2 = 1, so sign(v0 - delta^2) = -1 and v grows.
        expected_v = TAU**2 + (1.0 - BETA2) * 1.0
        self.assertAlmostEqual(_value(server._m), expected_m, places=6)
        self.assertAlmostEqual(_value(server._v), expected_v, places=6)
        expected = ETA * expected_m / (expected_v**0.5 + TAU)
        self.assertAlmostEqual(_value(new_state), expected, places=6)

    def test_fedavgm_first_step(self) -> None:
        server = _server("fedavgm")
        new_state = server._apply_fedopt_update(_delta(1.0))
        # FedAvgM has no v and no (1-beta1) scaling: m = beta1*0 + delta.
        self.assertAlmostEqual(_value(server._m), 1.0, places=6)
        self.assertAlmostEqual(_value(new_state), ETA * 1.0, places=6)


class FedOptAccumulatorBehaviourTest(unittest.TestCase):
    """Properties that distinguish the three adaptive rules from each other."""

    def test_fedadagrad_v_is_monotone_non_decreasing(self) -> None:
        server = _server("fedadagrad")
        previous = TAU**2
        for value in (1.0, -0.5, 0.25, -2.0):
            server._apply_fedopt_update(_delta(value))
            current = _value(server._v)
            self.assertGreaterEqual(current, previous)
            previous = current

    def test_fedadagrad_v_never_decays_but_fedadam_does(self) -> None:
        """With delta -> 0, Adagrad holds its accumulator and Adam forgets."""

        adagrad = _server("fedadagrad")
        adam = _server("fedadam")
        for server in (adagrad, adam):
            server._apply_fedopt_update(_delta(1.0))
        adagrad_after_spike = _value(adagrad._v)
        adam_after_spike = _value(adam._v)

        for _ in range(5):
            adagrad._apply_fedopt_update(_delta(0.0))
            adam._apply_fedopt_update(_delta(0.0))

        self.assertAlmostEqual(_value(adagrad._v), adagrad_after_spike, places=6)
        self.assertLess(_value(adam._v), adam_after_spike)

    def test_fedyogi_v_stays_positive_across_both_sign_branches(self) -> None:
        server = _server("fedyogi")
        # Large delta first (v < delta^2 -> v grows), then small ones
        # (v > delta^2 -> v shrinks). v must stay positive throughout, or
        # sqrt(v) would produce NaN.
        for value in (3.0, 0.01, 0.01, 0.01, 2.0, 0.0):
            server._apply_fedopt_update(_delta(value))
            self.assertGreater(_value(server._v), 0.0)

    def test_update_step_counts_every_update(self) -> None:
        server = _server("fedadagrad")
        for _ in range(4):
            server._apply_fedopt_update(_delta(0.5))
        self.assertEqual(server._update_step, 4)


class FedOptCheckpointTest(unittest.TestCase):
    """save_state/load_state must make a resumed server bit-identical."""

    def test_round_trip_reproduces_next_update(self) -> None:
        for optimizer in ("fedadagrad", "fedadam", "fedyogi", "fedavgm"):
            with self.subTest(optimizer=optimizer):
                original = _server(optimizer)
                for value in (1.0, -0.5, 0.75):
                    original._apply_fedopt_update(_delta(value))

                snapshot = original.save_state()
                restored = _server(optimizer)
                restored.load_state(snapshot)

                self.assertEqual(restored._update_step, original._update_step)
                self.assertAlmostEqual(_value(restored._m), _value(original._m), places=7)
                if original._v is not None:
                    self.assertAlmostEqual(_value(restored._v), _value(original._v), places=7)

                # The real guarantee: the next update matches on both servers.
                expected = _value(original._apply_fedopt_update(_delta(0.3)))
                actual = _value(restored._apply_fedopt_update(_delta(0.3)))
                self.assertAlmostEqual(actual, expected, places=7)

    def test_save_state_snapshot_is_not_aliased(self) -> None:
        server = _server("fedadam")
        server._apply_fedopt_update(_delta(1.0))
        snapshot = server.save_state()
        before = float(snapshot["v"]["w"].item())
        server._apply_fedopt_update(_delta(1.0))
        self.assertAlmostEqual(float(snapshot["v"]["w"].item()), before, places=7)


class FedOptOptimizerNameTest(unittest.TestCase):
    def test_normalizes_case_and_whitespace(self) -> None:
        self.assertEqual(_normalize_server_optimizer("  FedAdaGrad "), "fedadagrad")

    def test_rejects_unknown_optimizer(self) -> None:
        with self.assertRaises(ValueError):
            _normalize_server_optimizer("fedsgd")

    def test_fedadagrad_is_supported(self) -> None:
        server = _server("fedadagrad")
        self.assertEqual(server.server_optimizer, "fedadagrad")


if __name__ == "__main__":
    unittest.main()
