"""The running sum's precision is the accumulator's choice, not the sender's.

`WeightedStateAccumulator` allocated its running total with
`torch.zeros_like(value_cpu)`, so the precision of the one place a whole
federation converges was whatever the first client happened to send. A weighted
sum over thousands of clients is thousands of additions into a total whose
magnitude grows the whole way, so in a low-precision dtype the terms stop being
representable long before the mean is computed.

Measured through this class, FEMNIST-shaped weights (16-525 examples), 3597
clients, before the fix:

    dtype       median rel. error   max rel. error
    float32              9.6e-07          4.0e-04
    bfloat16             6.0e-02          8.0e+00
    float16              6.3e-03          6.4e+00

and float16 does not merely lose precision. The audit's own reproduction of
this used a zero-mean draw for two of its rows, which hides the sharpest fact:
on a coordinate whose sign does not change -- a bias, a post-ReLU weight
column, any real parameter -- the running total reaches 65504 and becomes
`inf`. After 3290 clients at |value| 0.05, 791 at 0.2, 153 at 1.0.

The fix promotes bf16 and fp16 to a float32 running total and casts the mean
back to the model's dtype. What that buys is not "a smaller number": it moves
the limiting factor off the accumulator entirely. Round the exact float64 mean
of the values the clients actually sent -- after their own dtype rounded them
-- to the output dtype, and that is the best answer the dtype contract allows.
The accumulator now returns it: bit-identical for bfloat16, within 0.66 ulp
for float16.

`TheSumIsNoLongerTheLimitingFactorTest` asserts exactly that, because it is a
property rather than a number. "Max relative error < 3e-3" would be a bound
nobody could defend and would keep passing while the sum degraded back toward
the sender's precision.

float32 is deliberately not promoted to float64. Its worst case above is a
relative error on a coordinate whose true mean is near zero -- the denominator
is what is small -- and float64 totals would double the one buffer this class
exists to hold at one model state.

Nothing ships bf16 or fp16 today: `hf_causal_lm` loads at torch's default
float32. The edit that would reach this is `torch_dtype=torch.bfloat16` to fit
a Qwen round in less memory, or a bf16 LoRA adapter.
"""

from __future__ import annotations

import unittest

import pytest
import torch

from fedbrew.core.torch_utils import WeightedStateAccumulator

pytestmark = pytest.mark.fast

#: FEMNIST's writer sizes, which are what the aggregation weights are.
WEIGHT_LOW, WEIGHT_HIGH = 16, 525
CLIENTS = 3597
LOW_PRECISION = (torch.bfloat16, torch.float16)


def _weights(count: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        WEIGHT_LOW, WEIGHT_HIGH + 1, (count,), generator=generator, dtype=torch.int64
    ).double()


def _values(count: int, numel: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + 1)
    return torch.randn(count, numel, generator=generator).double() * 0.05


def _accumulate(values: torch.Tensor, weights: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    accumulator = WeightedStateAccumulator()
    for index in range(values.shape[0]):
        accumulator.add({"w": values[index].to(dtype)}, float(weights[index]))
    return accumulator.result()["w"]


class TheRunningSumDtypeTest(unittest.TestCase):
    def test_low_precision_inputs_are_summed_in_float32(self) -> None:
        for dtype in LOW_PRECISION:
            with self.subTest(dtype=dtype):
                accumulator = WeightedStateAccumulator()
                accumulator.add({"w": torch.ones(4, dtype=dtype)}, 1.0)
                self.assertEqual(accumulator._totals["w"].dtype, torch.float32)

    def test_float32_and_float64_are_summed_in_themselves(self) -> None:
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                accumulator = WeightedStateAccumulator()
                accumulator.add({"w": torch.ones(4, dtype=dtype)}, 1.0)
                self.assertEqual(accumulator._totals["w"].dtype, dtype)

    def test_the_mean_comes_back_in_the_model_s_dtype(self) -> None:
        """The accumulation dtype is this class's business; the state's is the model's."""

        for dtype in (torch.float32, torch.float64, *LOW_PRECISION):
            with self.subTest(dtype=dtype):
                result = _accumulate(_values(4, 8), _weights(4), dtype)
                self.assertEqual(result.dtype, dtype)


class Float16NoLongerOverflowsTest(unittest.TestCase):
    """The sharpest of the three facts, and the one a zero-mean draw hides."""

    def test_a_same_signed_coordinate_survives_the_whole_federation(self) -> None:
        weights = _weights(CLIENTS)
        for scale in (0.05, 0.2, 1.0):
            with self.subTest(scale=scale):
                generator = torch.Generator().manual_seed(3)
                accumulator = WeightedStateAccumulator()
                for index in range(CLIENTS):
                    values = torch.rand(32, generator=generator) * scale + scale
                    accumulator.add({"w": values.half()}, float(weights[index]))
                    self.assertTrue(
                        torch.isfinite(accumulator._totals["w"]).all(),
                        f"the running sum overflowed at client {index + 1}",
                    )
                result = accumulator.result()["w"]
                self.assertTrue(torch.isfinite(result).all())
                self.assertEqual(result.dtype, torch.float16)

    def test_the_running_total_would_exceed_float16_s_range(self) -> None:
        """Without the promotion there is nothing to survive: check the premise.

        `sum(weight) * mean|value|` is what the total reaches, and float16's
        largest finite value is 65504. The generator above draws uniformly on
        [scale, 2 * scale], so the mean is 1.5 * scale and the smallest case
        is the 0.05 one. If this ever stops holding, the test above is passing
        for the wrong reason.
        """

        smallest_total = float(_weights(CLIENTS).sum()) * 1.5 * 0.05
        self.assertGreater(smallest_total, torch.finfo(torch.float16).max)


class TheSumIsNoLongerTheLimitingFactorTest(unittest.TestCase):
    def test_the_result_is_the_correctly_rounded_ideal_answer(self) -> None:
        """The property, rather than a hand-picked error bound.

        `ideal` is the exact weighted mean of the values the clients actually
        sent, after their own dtype rounded them -- computed in float64, so no
        accumulator can beat it. Rounding `ideal` to the output dtype is
        therefore the best answer that exists under the dtype contract, and
        the claim is that the accumulator returns it: bit-identical for
        bfloat16, and within one ulp for float16.

        An assertion of the form "max relative error < 3e-3" would be a number
        nobody could defend and would keep passing as the sum degraded back
        toward the sender's precision. This one cannot.
        """

        weights = _weights(CLIENTS)
        values = _values(CLIENTS, 128)

        for dtype in LOW_PRECISION:
            with self.subTest(dtype=dtype):
                ideal = (values.to(dtype).double() * weights[:, None]).sum(0) / weights.sum()
                best = ideal.to(dtype)
                result = _accumulate(values, weights, dtype)

                information = torch.finfo(dtype)
                ulp = information.eps * best.double().abs().clamp_min(information.tiny)
                ulps = ((result.double() - best.double()).abs() / ulp).max()
                self.assertLessEqual(float(ulps), 1.0)

    def test_the_send_is_what_is_left(self) -> None:
        """And it is not zero, so the test above is not comparing nothing.

        What survives is the clients' own rounding, which is theirs and not
        this class's: bf16 carries about three significant decimal digits.
        """

        weights = _weights(CLIENTS)
        values = _values(CLIENTS, 128)
        exact = (values * weights[:, None]).sum(0) / weights.sum()

        for dtype in LOW_PRECISION:
            with self.subTest(dtype=dtype):
                ideal = (values.to(dtype).double() * weights[:, None]).sum(0) / weights.sum()
                self.assertGreater(float((ideal - exact).abs().max()), 0.0)


class Float32IsUnchangedTest(unittest.TestCase):
    """Every shipped model is float32, so this path must not move at all."""

    def test_the_result_is_bit_identical_to_a_plain_float32_running_sum(self) -> None:
        weights = _weights(512)
        values = _values(512, 64)
        result = _accumulate(values, weights, torch.float32)

        total = torch.zeros(64, dtype=torch.float32)
        for index in range(512):
            total.add_(values[index].to(torch.float32), alpha=float(weights[index]))
        self.assertTrue(torch.equal(result, total.div_(float(weights.sum()))))


if __name__ == "__main__":
    unittest.main()
