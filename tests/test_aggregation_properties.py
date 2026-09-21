"""Property-based tests for the invariances a weighted mean must satisfy.

The example-based tests next door pin specific numbers. These pin the shape of
the operation over many inputs hypothesis chooses: reordering the clients, or
rescaling every weight by a common factor, must not move the result, and a
round with one client must return that client.

Why these three in particular. Client order is whatever
`random.Random(seed + t).sample` produced, so an order-sensitive aggregate
would make the result depend on the sampler rather than on the data. Weight
scale is free by construction -- `n_k` and `2 n_k` describe the same
population -- so scale sensitivity would mean the mean silently tracks dataset
size. And the single-client case is the boundary every partial-participation
round can reach when only one client reports.

Accumulation is float32 in client-arrival order (torch_utils.py:146, 157), so
none of these hold bit-exactly; the order effect was measured at max abs
5.8e-7 over 3597 FEMNIST clients. The tolerances below are set to catch
a wrong denominator or a dropped term, not to re-litigate float32.
"""

from __future__ import annotations

import unittest

import pytest
import torch

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st
except ImportError as exc:  # pragma: no cover - exercised only without the dev extra
    raise unittest.SkipTest("hypothesis is not installed (pip install -e '.[dev]')") from exc

from fedbrew.core.torch_utils import WeightedStateAccumulator

pytestmark = pytest.mark.fast

#: Bounded away from zero and from the float32 ceiling. The lower bound on the
#: weight matters: weights are example counts, so 0 is not a legal client, and
#: allowing denormals here would test float32 rather than the accumulator.
WEIGHTS = st.floats(min_value=1e-2, max_value=1e4, allow_nan=False, allow_infinity=False)
VALUES = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)

#: Two keys of different rank, so a bug that only survives on flat tensors --
#: an accumulator that reshapes, or indexes the first element -- is visible.
SHAPES = {"weight": (2, 3), "bias": (3,)}

CLIENTS = st.lists(
    st.tuples(
        WEIGHTS,
        st.lists(VALUES, min_size=9, max_size=9),
    ),
    min_size=1,
    max_size=8,
)

SETTINGS = settings(
    max_examples=150,
    # torch's first allocation in a worker can blow a 200ms deadline without
    # anything being wrong with the code under test.
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _state(values: list[float]) -> dict[str, torch.Tensor]:
    flat = torch.tensor(values, dtype=torch.float32)
    offset = 0
    state = {}
    for key, shape in SHAPES.items():
        size = int(torch.tensor(shape).prod())
        state[key] = flat[offset : offset + size].reshape(shape).clone()
        offset += size
    return state


def _mean(clients: list[tuple[float, list[float]]]) -> dict[str, torch.Tensor]:
    accumulator = WeightedStateAccumulator()
    for weight, values in clients:
        accumulator.add(_state(values), weight)
    return accumulator.result()


def _assert_close(left: dict, right: dict, *, rtol: float, atol: float) -> None:
    assert set(left) == set(right)
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=rtol, atol=atol)


class AggregationInvarianceTests(unittest.TestCase):
    @SETTINGS
    @given(clients=CLIENTS, rotation=st.integers(min_value=0, max_value=7))
    def test_the_mean_does_not_depend_on_client_order(
        self, clients: list[tuple[float, list[float]]], rotation: int
    ) -> None:
        rotation %= len(clients)
        rotated = clients[rotation:] + clients[:rotation]
        # Reversal as well as rotation: reversal is the worst case for a
        # running sum, and it is the order that measurement used.
        for reordered in (rotated, list(reversed(clients))):
            _assert_close(_mean(clients), _mean(reordered), rtol=1e-4, atol=1e-4)

    @SETTINGS
    @given(
        clients=CLIENTS,
        scale=st.floats(min_value=1e-2, max_value=1e2, allow_nan=False, allow_infinity=False),
    )
    def test_scaling_every_weight_leaves_the_mean_unchanged(
        self, clients: list[tuple[float, list[float]]], scale: float
    ) -> None:
        scaled = [(weight * scale, values) for weight, values in clients]
        _assert_close(_mean(clients), _mean(scaled), rtol=1e-4, atol=1e-4)

    @SETTINGS
    @given(weight=WEIGHTS, values=st.lists(VALUES, min_size=9, max_size=9))
    def test_one_client_is_returned_unchanged(self, weight: float, values: list[float]) -> None:
        state = _state(values)
        # Not bit-exact, and deliberately tested at the tightest tolerance that
        # is: the accumulator computes (0 + x*w)/w, so x round-trips through
        # two float32 roundings and comes back within ~1 ulp. Measured worst
        # case over this strategy is 6.3e-8 relative, so 1e-6 leaves a decade
        # of headroom while still failing on any real arithmetic error.
        _assert_close(_mean([(weight, values)]), state, rtol=1e-6, atol=1e-6)

    @SETTINGS
    @given(clients=CLIENTS)
    def test_the_mean_is_a_convex_combination_of_its_inputs(
        self, clients: list[tuple[float, list[float]]]
    ) -> None:
        """Elementwise, the result must sit inside the clients' range.

        This is what fails when a weight is applied twice, when a term is
        dropped from the numerator but not the denominator, or when the
        denominator is a roster rather than the participating set -- all of
        which leave the result finite and none of which stay inside the hull.
        """

        result = _mean(clients)
        states = [_state(values) for _, values in clients]
        for key in SHAPES:
            stacked = torch.stack([state[key] for state in states])
            lower, upper = stacked.min(dim=0).values, stacked.max(dim=0).values
            span = (upper - lower).clamp(min=1.0)
            self.assertTrue(bool((result[key] >= lower - 1e-4 * span).all()))
            self.assertTrue(bool((result[key] <= upper + 1e-4 * span).all()))

    @SETTINGS
    @given(clients=CLIENTS, duplicates=st.integers(min_value=2, max_value=4))
    def test_repeating_one_client_is_the_same_as_weighting_it_up(
        self, clients: list[tuple[float, list[float]]], duplicates: int
    ) -> None:
        """n copies at weight w == one copy at weight n*w.

        The property that makes `num_examples` a count rather than a label: a
        client holding twice the data must count exactly as much as two
        clients holding half each.
        """

        weight, values = clients[0]
        repeated = [(weight, values)] * duplicates + clients[1:]
        collapsed = [(weight * duplicates, values)] + clients[1:]
        _assert_close(_mean(repeated), _mean(collapsed), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
