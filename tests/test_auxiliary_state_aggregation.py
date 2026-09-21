"""How SCAFFOLD and FedLALR fold their auxiliary state, relative to the model.

Both strategies carry per-round state beside the model -- SCAFFOLD a control
variate, FedLALR a momentum and a second moment -- and in both cases that state
is only meaningful if it is combined over the same clients, with a stated
relationship to how the model itself was combined. The two make opposite
choices, and both choices are load-bearing:

  FedLALR    m and v_hat use the SAME weight as the model. Algorithm 1
             synchronizes all three as one object; a per-client rate divided by
             a differently-weighted second moment is not that object.

  SCAFFOLD   the model uses the aggregation weight, the control variate is
             ALWAYS 1/N over the full roster. The 1/N is what maintains
             c = (1/N) sum_i c_i across partial participation, and the paper's
             correction -c_i + c depends on that identity holding.

That makes SCAFFOLD's model and control weightings deliberately different
whenever aggregation_weighting is "examples" (the default). That is an
intentional simplification, not a bug: the model moves along
the example-weighted mean while c tracks the uniform one. It is pinned below
rather than asserted away, so that if it ever changes it changes visibly and
someone has to come here and say why.
"""

from __future__ import annotations

import unittest

import pytest
import torch

from fedbrew.core.protocol import ClientInfo, FitResult, RoundInfo
from fedbrew.servers.fedlalr import FedLALRServer
from fedbrew.servers.scaffold import ScaffoldServer

pytestmark = pytest.mark.fast

EPSILON = 1e-8

#: Deliberately unequal, so "examples" and "uniform" cannot agree by accident.
#: 2*1 + 8*6 = 50 over 10 examples = 5.0 example-weighted; (1+6)/2 = 3.5
#: uniform.
CLIENTS = (("a", 2, 1.0), ("b", 8, 6.0))
EXAMPLE_WEIGHTED = (2 * 1.0 + 8 * 6.0) / (2 + 8)
UNIFORM = (1.0 + 6.0) / 2


def _zeros() -> dict[str, torch.Tensor]:
    return {"w": torch.zeros(2)}


class FedLALRAuxiliaryStateTests(unittest.TestCase):
    """All three states, one weighting."""

    def _server(self, weighting: str) -> FedLALRServer:
        server = FedLALRServer(
            epsilon=EPSILON,
            participation_rate=1.0,
            seed=0,
            aggregation_weighting=weighting,
        )
        server._model_state = _zeros()
        server._model_state_scope = "full"
        server._model_state_metadata = {"model_state_scope": "full"}
        return server

    def _results(self) -> list[FitResult]:
        return [
            FitResult(
                round_id=1,
                client_id=client_id,
                num_examples=num_examples,
                payload={
                    "model_state": {"w": torch.full((2,), value)},
                    "model_state_scope": "full",
                    "model_state_metadata": {"model_state_scope": "full"},
                    # Distinct multiples of the model value, so a state folded
                    # with the wrong weight lands on a number no other state
                    # holds.
                    "momentum_state": {"w": torch.full((2,), value * 10)},
                    "second_moment_state": {"w": torch.full((2,), value * 100)},
                },
                metrics={},
            )
            for client_id, num_examples, value in CLIENTS
        ]

    def test_examples_weighting_applies_to_all_three_states(self) -> None:
        server = self._server("examples")
        server.aggregate(RoundInfo(round_id=1), self._results())

        self.assertAlmostEqual(float(server._model_state["w"][0]), EXAMPLE_WEIGHTED, 4)
        self.assertAlmostEqual(float(server._momentum["w"][0]), EXAMPLE_WEIGHTED * 10, 3)
        self.assertAlmostEqual(float(server._second_moment["w"][0]), EXAMPLE_WEIGHTED * 100, 2)

    def test_uniform_weighting_applies_to_all_three_states(self) -> None:
        server = self._server("uniform")
        server.aggregate(RoundInfo(round_id=1), self._results())

        self.assertAlmostEqual(float(server._model_state["w"][0]), UNIFORM, 4)
        self.assertAlmostEqual(float(server._momentum["w"][0]), UNIFORM * 10, 3)
        self.assertAlmostEqual(float(server._second_moment["w"][0]), UNIFORM * 100, 2)

    def test_the_auxiliary_states_track_the_model_exactly(self) -> None:
        """Stated as a ratio, so it holds whatever the weighting is.

        Each client's m is 10x its model value and its v_hat is 100x, so any
        weighted mean of them must preserve those ratios. A second accumulator
        using a different denominator would not.
        """

        for weighting in ("examples", "uniform"):
            with self.subTest(weighting=weighting):
                server = self._server(weighting)
                server.aggregate(RoundInfo(round_id=1), self._results())
                model = float(server._model_state["w"][0])
                self.assertAlmostEqual(float(server._momentum["w"][0]) / model, 10.0, places=4)
                self.assertAlmostEqual(
                    float(server._second_moment["w"][0]) / model, 100.0, places=3
                )

    def test_the_two_weightings_disagree_on_this_input(self) -> None:
        self.assertNotAlmostEqual(EXAMPLE_WEIGHTED, UNIFORM)


class ScaffoldControlVariateTests(unittest.TestCase):
    """c moves by (1/N) sum of the deltas that arrived -- N the roster."""

    ROSTER = 10

    def _server(self, weighting: str = "examples") -> ScaffoldServer:
        server = ScaffoldServer(participation_rate=0.2, seed=0, aggregation_weighting=weighting)
        server._model_state = _zeros()
        server._model_state_scope = "full"
        server._model_state_metadata = {"model_state_scope": "full"}
        server._server_control = _zeros()
        server._num_clients = self.ROSTER
        return server

    def _results(self, deltas: tuple[float, ...] | None = None) -> list[FitResult]:
        values = deltas if deltas is not None else tuple(value for _, _, value in CLIENTS)
        return [
            FitResult(
                round_id=1,
                client_id=client_id,
                num_examples=num_examples,
                payload={
                    "model_state": {"w": torch.full((2,), value)},
                    "model_state_scope": "full",
                    "model_state_metadata": {"model_state_scope": "full"},
                    "control_delta": {"w": torch.full((2,), delta)},
                },
                metrics={},
            )
            for (client_id, num_examples, value), delta in zip(CLIENTS, values, strict=True)
        ]

    def test_the_control_update_is_divided_by_the_roster_not_the_round(self) -> None:
        server = self._server()
        server.aggregate(RoundInfo(round_id=1), self._results((3.0, 7.0)))
        # (3 + 7) / 10 clients on the roster. Dividing by the 2 that reported
        # would give 5.0 -- a 5x overshoot at this participation rate, and 50x
        # at FEMNIST's 0.01.
        self.assertAlmostEqual(float(server._server_control["w"][0]), 1.0, places=5)

    def test_the_control_update_ignores_the_aggregation_weighting(self) -> None:
        """1/N regardless: c is defined as the uniform mean of the c_i."""

        for weighting in ("examples", "uniform"):
            with self.subTest(weighting=weighting):
                server = self._server(weighting)
                server.aggregate(RoundInfo(round_id=1), self._results((3.0, 7.0)))
                self.assertAlmostEqual(float(server._server_control["w"][0]), 1.0, places=5)

    def test_c_stays_the_uniform_mean_of_the_client_controls(self) -> None:
        """The invariant the drift correction -c_i + c is built on.

        Simulated over several partial rounds: every c_i starts at zero, a
        client that participates adds its delta to its own c_i, and the server
        adds the same delta scaled by 1/N to c. If the two scalings ever
        disagree, c stops being the mean of the c_i and the correction adds a
        bias that decays only as clients are resampled.
        """

        server = self._server()
        client_controls = [0.0] * self.ROSTER
        rounds = ((0, 3.0), (1, 7.0), (0, -2.0), (4, 5.0))

        for round_id, (client_index, delta) in enumerate(rounds, start=1):
            result = FitResult(
                round_id=round_id,
                client_id=f"c{client_index}",
                num_examples=10,
                payload={
                    "model_state": {"w": torch.zeros(2)},
                    "model_state_scope": "full",
                    "model_state_metadata": {"model_state_scope": "full"},
                    "control_delta": {"w": torch.full((2,), delta)},
                },
                metrics={},
            )
            server.aggregate(RoundInfo(round_id=round_id), [result])
            client_controls[client_index] += delta

            expected = sum(client_controls) / self.ROSTER
            self.assertAlmostEqual(float(server._server_control["w"][0]), expected, places=5)

    def test_aggregating_without_a_roster_size_raises_instead_of_guessing(self) -> None:
        """The branch that used to guess N from the round's sample is gone.

        `if self._num_clients <= 0: self._num_clients = num_results` silently
        put |S| where the paper puts N. Unreachable through the loop, which
        calls configure_round first -- so the branch could only ever convert a
        caller's mistake into a wrong number, in the one place N appears.
        """

        server = self._server()
        server._num_clients = 0
        with self.assertRaisesRegex(ValueError, "does not know the client-roster size N"):
            server.aggregate(RoundInfo(round_id=1), self._results((3.0, 7.0)))
        # And nothing was folded in on the way to raising.
        self.assertAlmostEqual(float(server._server_control["w"][0]), 0.0, places=5)

    def test_what_the_guess_would_have_cost_at_this_participation_rate(self) -> None:
        """Why raising beats guessing, as a number rather than an assertion.

        The two servers differ only in N. The one told the roster produces the
        paper's update; the one that guessed from the sample overshoots by
        N/|S|, which is 5x here and 100x at FEMNIST's participation_rate 0.01.
        Both answers are finite, same units, same order of magnitude as each
        other's neighbours -- there is nothing in the output to tell them
        apart, which is what made the silent branch worth deleting.
        """

        results = self._results((3.0, 7.0))
        correct = self._server()
        correct.aggregate(RoundInfo(round_id=1), results)

        guessed = self._server()
        guessed._num_clients = len(results)  # what the deleted branch assigned
        guessed.aggregate(RoundInfo(round_id=1), self._results((3.0, 7.0)))

        ratio = float(guessed._server_control["w"][0]) / float(correct._server_control["w"][0])
        self.assertAlmostEqual(ratio, self.ROSTER / len(results), places=5)
        self.assertAlmostEqual(ratio, 5.0, places=5)

    def test_a_resumed_server_takes_n_from_the_checkpoint(self) -> None:
        """The other way N arrives, and the other half of the raise's message."""

        source = self._server()
        state = source.save_state()
        self.assertEqual(state["num_clients"], self.ROSTER)

        resumed = self._server()
        resumed._num_clients = 0
        resumed.load_state(state)
        self.assertEqual(resumed._num_clients, self.ROSTER)
        resumed.aggregate(RoundInfo(round_id=1), self._results((3.0, 7.0)))
        self.assertAlmostEqual(float(resumed._server_control["w"][0]), 1.0, places=5)

    def test_configure_round_takes_n_from_the_roster_not_the_sample(self) -> None:
        """N must be every client, not the ones this round happened to draw.

        The defect this pins: _num_clients falls back to the number of results
        when configure_round has not run, which silently substitutes |S| for N
        and scales the control update by N/|S| -- 100x at FEMNIST's
        participation_rate of 0.01.
        """

        server = self._server()
        server._num_clients = 0
        roster = [
            ClientInfo(client_id=f"c{index}", num_examples=10) for index in range(self.ROSTER)
        ]
        requests = server.configure_round(RoundInfo(round_id=1), roster)

        self.assertEqual(server._num_clients, self.ROSTER)
        self.assertLess(len(requests), self.ROSTER)
        # And the roster size is what the clients are told, since the client
        # correction is scaled by it too.
        self.assertEqual(requests[0].payload["total_num_clients"], self.ROSTER)


class ScaffoldModelAndControlWeightingsDivergeTests(unittest.TestCase):
    """A pinned deviation, not an assertion that it is right.

    SCAFFOLD's paper averages the model uniformly over the
    sampled clients; this implementation averages it by example count under the
    default aggregation_weighting, while the control variate stays uniform at
    1/N. So under "examples" the two halves of the SCAFFOLD update are combined
    along different directions.

    Classified there as an intentional simplification kept for comparability
    with the other strategies, and left unfixed. This test states the current
    behaviour exactly so the deviation cannot drift or be silently removed --
    if it changes, this test fails and the change has to be argued for here.
    """

    ROSTER = 10

    def _server(self, weighting: str) -> ScaffoldServer:
        server = ScaffoldServer(participation_rate=0.2, seed=0, aggregation_weighting=weighting)
        server._model_state = _zeros()
        server._model_state_scope = "full"
        server._model_state_metadata = {"model_state_scope": "full"}
        server._server_control = _zeros()
        server._num_clients = self.ROSTER
        return server

    def _results(self) -> list[FitResult]:
        return [
            FitResult(
                round_id=1,
                client_id=client_id,
                num_examples=num_examples,
                payload={
                    "model_state": {"w": torch.full((2,), value)},
                    "model_state_scope": "full",
                    "model_state_metadata": {"model_state_scope": "full"},
                    "control_delta": {"w": torch.full((2,), value)},
                },
                metrics={},
            )
            for client_id, num_examples, value in CLIENTS
        ]

    def test_the_default_averages_the_model_by_examples_and_c_uniformly(self) -> None:
        server = self._server("examples")
        server.aggregate(RoundInfo(round_id=1), self._results())

        # The model: the example-weighted mean, as FedAvg would compute it.
        self.assertAlmostEqual(float(server._model_state["w"][0]), EXAMPLE_WEIGHTED, places=4)
        # c: the same client values, summed and divided by the roster.
        self.assertAlmostEqual(
            float(server._server_control["w"][0]),
            sum(value for _, _, value in CLIENTS) / self.ROSTER,
            places=5,
        )
        # And those are not the same combination. This inequality IS the
        # finding; it is asserted so that the day it stops being true is the
        # day this test asks why.
        self.assertNotAlmostEqual(
            float(server._model_state["w"][0]),
            UNIFORM,
            places=4,
            msg="the model is now averaged uniformly -- the example-weighted "
            "model average has gone, or the default weighting changed; "
            "update this test",
        )

    def test_setting_uniform_makes_both_halves_agree(self) -> None:
        """The knob that removes the deviation, for anyone who wants it gone.

        Under uniform, the model mean and the control mean are the same
        combination over the same clients, differing only by the |S|/N factor
        that partial participation requires. This is what
        configs/femnist/scaffold.yaml would set to match the paper.
        """

        server = self._server("uniform")
        server.aggregate(RoundInfo(round_id=1), self._results())

        participants = len(CLIENTS)
        self.assertAlmostEqual(float(server._model_state["w"][0]), UNIFORM, places=4)
        self.assertAlmostEqual(
            float(server._server_control["w"][0]),
            UNIFORM * participants / self.ROSTER,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
