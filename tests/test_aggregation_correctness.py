"""The weighted mean the server computes, checked against arithmetic by hand.

tests/test_aggregation_weighting.py pins which weighting mode is selected.
These tests pin the number that comes out of the mode: a two-client mean
computed on paper, the identity that must hold when every client agrees, and
-- the part no other test covers -- that the denominator is the sum over the
clients that actually reported, never over the roster the server was given.

The roster case is the one worth stating explicitly. At FEMNIST's
participation_rate: 0.01 a roster of 3597 clients sends 36; dividing the
weighted sum by anything derived from 3597 would scale every round's update by
~1/100 and still produce a finite, plausible-looking model that simply never
learns. The denominator is the summed client weight, not anything derived from
the roster (torch_utils.py:163, 171-177); this is the test that keeps it so.
"""

from __future__ import annotations

import unittest

import pytest
import torch

from fedbrew.core.protocol import FitResult, RoundInfo
from fedbrew.core.torch_utils import WeightedStateAccumulator
from fedbrew.servers.fedavg import FedAvgServer

pytestmark = pytest.mark.fast


def _result(client_id: str, num_examples: int, value: float) -> FitResult:
    return FitResult(
        round_id=1,
        client_id=client_id,
        num_examples=num_examples,
        payload={
            "model_state": {"w": torch.full((2,), value)},
            "model_state_scope": "full",
            "model_state_metadata": {"model_state_scope": "full"},
        },
        metrics={},
    )


def _server(weighting: str = "examples") -> FedAvgServer:
    server = FedAvgServer(participation_rate=1.0, seed=0, aggregation_weighting=weighting)
    server._model_state = {"w": torch.zeros(2)}
    server._model_state_scope = "full"
    server._model_state_metadata = {"model_state_scope": "full"}
    return server


def _aggregate(server: FedAvgServer, results: list[FitResult]) -> torch.Tensor:
    server.aggregate(RoundInfo(round_id=1), results)
    return server._model_state["w"]


class IdenticalClientsTests(unittest.TestCase):
    """A mean over identical inputs is the input. Nothing may perturb it."""

    def test_identical_clients_return_the_client_model_unchanged(self) -> None:
        torch.manual_seed(0)
        shared = {
            "weight": torch.randn(3, 4),
            "bias": torch.randn(3),
        }
        results = [
            FitResult(
                round_id=1,
                client_id=f"c{index}",
                # Deliberately unequal: identical models must average to
                # themselves whatever the weights are.
                num_examples=(index + 1) * 7,
                payload={
                    "model_state": {key: value.clone() for key, value in shared.items()},
                    "model_state_scope": "full",
                    "model_state_metadata": {"model_state_scope": "full"},
                },
                metrics={},
            )
            for index in range(5)
        ]
        server = _server()
        server._model_state = {key: torch.zeros_like(value) for key, value in shared.items()}
        server.aggregate(RoundInfo(round_id=1), results)

        for key, expected in shared.items():
            # Not assertEqual on the tensors: the running sum divides by
            # sum(weights), so exact equality is not guaranteed in float32 and
            # demanding it would make this test about rounding, not about
            # aggregation. One ulp of headroom, no more.
            torch.testing.assert_close(server._model_state[key], expected, rtol=1e-6, atol=1e-7)

    def test_a_single_client_is_returned_unchanged_at_any_weight(self) -> None:
        for num_examples in (1, 17, 100000):
            with self.subTest(num_examples=num_examples):
                server = _server()
                result = _result("solo", num_examples=num_examples, value=2.5)
                torch.testing.assert_close(_aggregate(server, [result]), torch.full((2,), 2.5))


class HandComputedMeanTests(unittest.TestCase):
    """Small integers, so the expected value can be read off the source."""

    #: (num_examples, parameter value) per client. 3*1 + 5*3 + 2*8 = 34 over a
    #: denominator of 10, so the example-weighted mean is exactly 3.4 and the
    #: uniform mean is exactly 4.0. Neither is a value any single client holds,
    #: so a bug that returns some client's tensor cannot pass either check.
    CLIENTS = ((3, 1.0), (5, 3.0), (2, 8.0))
    EXAMPLE_WEIGHTED = (3 * 1.0 + 5 * 3.0 + 2 * 8.0) / (3 + 5 + 2)  # 3.4
    UNIFORM = (1.0 + 3.0 + 8.0) / 3  # 4.0

    def _results(self) -> list[FitResult]:
        return [
            _result(f"c{index}", num_examples=n, value=value)
            for index, (n, value) in enumerate(self.CLIENTS)
        ]

    def test_example_weighted_mean_matches_the_hand_computation(self) -> None:
        averaged = _aggregate(_server("examples"), self._results())
        torch.testing.assert_close(averaged, torch.full((2,), self.EXAMPLE_WEIGHTED))

    def test_uniform_mean_matches_the_hand_computation(self) -> None:
        averaged = _aggregate(_server("uniform"), self._results())
        torch.testing.assert_close(averaged, torch.full((2,), self.UNIFORM))

    def test_the_two_modes_disagree_on_this_input(self) -> None:
        # Guards the two tests above against a shared input that would let a
        # single wrong denominator satisfy both.
        self.assertNotAlmostEqual(self.EXAMPLE_WEIGHTED, self.UNIFORM)


class DenominatorIsTheParticipatingSetTests(unittest.TestCase):
    """Sum over the clients that reported -- not over the roster.

    The accumulator has no idea how many clients exist, which is exactly the
    property under test: it can only divide by what it was given. These tests
    pin that from the outside, through the server, where a roster IS in scope.
    """

    #: A roster far larger than the round, as in every cross-device config.
    ROSTER = 1000
    PARTICIPANTS = ((4, 2.0), (6, 7.0))
    #: 4*2 + 6*7 = 50 over 10 participating examples.
    EXPECTED = (4 * 2.0 + 6 * 7.0) / (4 + 6)  # 5.0

    def _participating_results(self) -> list[FitResult]:
        return [
            _result(f"c{index}", num_examples=n, value=value)
            for index, (n, value) in enumerate(self.PARTICIPANTS)
        ]

    def test_a_large_idle_roster_does_not_enter_the_denominator(self) -> None:
        server = _server("examples")
        # Everything the server could mistake for a denominator, set to the
        # roster rather than the round.
        server.participation_rate = len(self.PARTICIPANTS) / self.ROSTER
        averaged = _aggregate(server, self._participating_results())
        torch.testing.assert_close(averaged, torch.full((2,), self.EXPECTED))
        # Dividing by the roster's example count instead would land here.
        roster_examples = sum(n for n, _ in self.PARTICIPANTS) * self.ROSTER / 2
        wrong = sum(n * v for n, v in self.PARTICIPANTS) / roster_examples
        self.assertNotAlmostEqual(float(averaged[0]), wrong, places=6)

    def test_uniform_divides_by_the_round_size_not_the_roster(self) -> None:
        server = _server("uniform")
        server.participation_rate = len(self.PARTICIPANTS) / self.ROSTER
        averaged = _aggregate(server, self._participating_results())
        torch.testing.assert_close(averaged, torch.full((2,), (2.0 + 7.0) / 2))

    def test_the_accumulator_denominator_is_the_sum_of_added_weights(self) -> None:
        # Directly, one level below the server: the divisor is whatever was
        # handed to add(), so a client that never reported cannot dilute it.
        accumulator = WeightedStateAccumulator()
        accumulator.add({"w": torch.full((2,), 2.0)}, 4.0)
        accumulator.add({"w": torch.full((2,), 7.0)}, 6.0)
        torch.testing.assert_close(accumulator.result()["w"], torch.full((2,), self.EXPECTED))

    def test_a_dropped_client_changes_the_denominator_with_it(self) -> None:
        """Dropout must renormalize, not leave a hole weighted as zero."""

        full_round = self._participating_results()
        survivors = full_round[:1]
        averaged = _aggregate(_server("examples"), survivors)
        # The one survivor holds 2.0; a denominator that still counted the
        # absent client's 6 examples would give 4*2/10 = 0.8.
        torch.testing.assert_close(averaged, torch.full((2,), 2.0))
        self.assertNotAlmostEqual(float(averaged[0]), 0.8, places=6)


if __name__ == "__main__":
    unittest.main()
