"""Rows of a stack are folded together, and the mean and the refusals are the rows'.

The batched executor yields each client's state as a row of the chunk's
stack (``StackedRow``), and ``WeightedStateAccumulator`` folds a stack's rows
in one weighted reduction per tensor instead of one add per row. What is
pinned here:

- the mean is the one adding the rows as plain states gives, to summation
  order (1e-12 relative in float64), and a stack of one row is folded bit for
  bit as the state alone;
- a subset of a stack, stacks interleaved with plain states, and the fold at
  ``result`` all count every row once;
- a non-finite row is refused naming the client and tensor the plain adds
  name, and finite rows whose mean overflows are refused as plain states are;
- the accumulator keeps no row alive: its reference is dtype and shape.
"""

from __future__ import annotations

import unittest

import torch

from fedbrew.core.torch_utils import (
    NonFiniteStateError,
    StackedRow,
    StateStack,
    WeightedStateAccumulator,
)

TOLERANCE = 1e-12


def _stack(size: int, *, seed: int = 0, dtype: torch.dtype = torch.float64) -> StateStack:
    generator = torch.Generator().manual_seed(seed)
    return StateStack(
        {
            "weight": torch.randn(size, 5, 3, generator=generator, dtype=dtype),
            "bias": torch.randn(size, 3, generator=generator, dtype=dtype),
        }
    )


def _plain(row: StackedRow) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in row.items()}


def _mean(adds: list[tuple[dict[str, torch.Tensor], float, str]]) -> dict[str, torch.Tensor]:
    accumulator = WeightedStateAccumulator()
    for state, weight, source in adds:
        accumulator.add(state, weight, source=source)
    return accumulator.result()


def _assert_close(test: unittest.TestCase, a: dict, b: dict) -> None:
    test.assertEqual(list(a), list(b))
    for key in a:
        error = ((a[key] - b[key]).abs() / b[key].abs().clamp_min(1e-300)).max().item()
        test.assertLessEqual(error, TOLERANCE, key)


class TheMeanIsTheRowsMeanTest(unittest.TestCase):
    def test_a_stack_folds_to_its_rows_mean(self) -> None:
        stack = _stack(7)
        weights = [3.0, 11.0, 7.0, 1.0, 5.0, 13.0, 2.0]
        rows = [stack.row(index) for index in range(7)]
        adds = [(row, weights[i], f"c{i}") for i, row in enumerate(rows)]
        folded = _mean(adds)
        plain = _mean([(_plain(row), weight, source) for row, weight, source in adds])
        _assert_close(self, folded, plain)

    def test_one_row_is_folded_bit_for_bit(self) -> None:
        first, second = _stack(1, seed=1), _stack(1, seed=2)
        folded = _mean([(first.row(0), 13.0, "a"), (second.row(0), 7.0, "b")])
        plain = _mean([(_plain(first.row(0)), 13.0, "a"), (_plain(second.row(0)), 7.0, "b")])
        for key in folded:
            self.assertTrue(torch.equal(folded[key], plain[key]), key)

    def test_subsets_interleaving_and_result_count_every_row_once(self) -> None:
        stack, other = _stack(4, seed=3), _stack(3, seed=4)
        plain_state = {
            key: value[0].clone() * 2 for key, value in _stack(1, seed=5).tensors.items()
        }
        adds = [
            (stack.row(0), 2.0, "s0"),
            (stack.row(2), 5.0, "s2"),  # a subset: rows 1 and 3 never arrive
            (plain_state, 3.0, "p"),
            (other.row(0), 1.0, "o0"),
            (other.row(1), 4.0, "o1"),  # folded at result(), before the stack is complete
        ]
        folded = _mean(adds)
        plain = _mean(
            [
                (state if not isinstance(state, StackedRow) else _plain(state), w, s)
                for state, w, s in adds
            ]
        )
        _assert_close(self, folded, plain)

    def test_float32_rows_fold_in_float32(self) -> None:
        stack = _stack(5, dtype=torch.float32)
        folded = _mean([(stack.row(i), float(i + 1), str(i)) for i in range(5)])
        self.assertEqual(folded["weight"].dtype, torch.float32)


class TheRefusalsAreTheRowsRefusalsTest(unittest.TestCase):
    def _refusal(self, adds: list) -> str:
        with self.assertRaises(NonFiniteStateError) as caught:
            _mean(adds)
        return str(caught.exception)

    def test_a_non_finite_row_is_named(self) -> None:
        stack = _stack(4)
        stack.tensors["bias"][2, 1] = float("nan")
        stack.tensors["weight"][3, 0, 0] = float("inf")
        rows = [(stack.row(i), 1.0, f"client_{i}") for i in range(4)]
        plain = [(_plain(state), w, s) for state, w, s in rows]
        folded_message = self._refusal(rows)
        self.assertEqual(folded_message, self._refusal(plain))
        self.assertIn("client_2", folded_message)
        self.assertIn("'bias'", folded_message)

    def test_finite_rows_whose_mean_overflows_are_refused(self) -> None:
        stack = StateStack({"w": torch.full((2, 3), 3.0e38, dtype=torch.float32)})
        message = self._refusal([(stack.row(0), 1.0, "a"), (stack.row(1), 1.0, "b")])
        self.assertIn("overflows", message)


class NoRowIsKeptAliveTest(unittest.TestCase):
    def test_the_reference_is_dtype_and_shape(self) -> None:
        stack = _stack(2)
        accumulator = WeightedStateAccumulator()
        accumulator.add(stack.row(0), 1.0)
        accumulator.add(stack.row(1), 1.0)
        for reference in accumulator._reference.values():
            self.assertEqual(reference.device.type, "meta")
        self.assertIsNone(accumulator._pending)


if __name__ == "__main__":
    unittest.main()
