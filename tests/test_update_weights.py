"""A pass's batch weights and a program's per-client values, computed once, are the step's own.

``update_weights`` takes the whole weight table at once where it can -- under
``full`` the rows themselves, under ``uniform`` and ``sum`` one division per
column -- where it assigned an update's columns a slice at a time; and
``_per_client`` keeps the shapes it made of an eager tensor of values, which
every step of a round cast and reshaped again. Held here against the slice at
a time and a fresh shaping, bit for bit.
"""

from __future__ import annotations

import unittest

import pytest
import torch

from fedbrew.clients.batched_update import LocalProgram, OptimizerSpec, _per_client, update_weights

pytestmark = pytest.mark.fast


def _slice_at_a_time(lengths: torch.Tensor, structure: tuple[int, ...], program: LocalProgram):  # type: ignore[no-untyped-def]
    rows = lengths.to(torch.float64)
    weights = torch.zeros_like(rows)
    if program.combine == "batch":
        return weights
    start = 0
    for count in structure:
        part = rows[:, start : start + count]
        if program.combine == "full":
            weights[:, start : start + count] = part
        elif program.weighting == "uniform":
            weights[:, start : start + count] = 1.0 / float(count)
        elif program.weighting == "sum":
            weights[:, start : start + count] = 1.0
        else:
            epoch_examples = torch.clamp(part.sum(dim=1, keepdim=True), min=1.0)
            weights[:, start : start + count] = part / epoch_examples
        start += count
    return weights


class TheWeightsTest(unittest.TestCase):
    def test_every_combination(self) -> None:
        generator = torch.Generator().manual_seed(5)
        spec = OptimizerSpec("sgd", lr=0.1)
        for structure in ((1,) * 7, (3, 3, 1), (7,), (2, 5)):
            lengths = torch.randint(0, 9, (4, 7), generator=generator)
            for combine, weighting in (
                ("batch", None),
                ("full", None),
                ("frozen", "uniform"),
                ("frozen", "sum"),
                ("frozen", "examples"),
            ):
                with self.subTest(structure=structure, combine=combine, weighting=weighting):
                    program = LocalProgram(optimizer=spec, combine=combine, weighting=weighting)
                    self.assertTrue(
                        torch.equal(
                            update_weights(lengths, structure, program),
                            _slice_at_a_time(lengths, structure, program),
                        )
                    )


class TheValuesShapesTest(unittest.TestCase):
    def test_a_kept_shape_is_a_fresh_one(self) -> None:
        value = torch.tensor([0.5, -0.25, 3.0], dtype=torch.float64)
        for like in (torch.zeros(3, 4), torch.zeros(3, 2, 5, dtype=torch.float64), torch.zeros(3)):
            with self.subTest(dims=like.dim(), dtype=like.dtype):
                fresh = value.to(like.dtype).reshape(3, *(1,) * (like.dim() - 1))
                first, again = _per_client(value, like), _per_client(value, like)
                self.assertIs(first, again)
                self.assertTrue(torch.equal(first, fresh))
                self.assertEqual(first.dtype, like.dtype)


if __name__ == "__main__":
    unittest.main()
