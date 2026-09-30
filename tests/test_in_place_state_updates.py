"""SCAFFOLD's per-client state updates made in place are the helpers' own, bit for bit.

``TorchScaffoldClient._update_client_control`` writes Option II's scaled
correction, the new ``c_i`` and its delta into tensors it made itself, and
``ScaffoldServer`` sums the deltas into its own sum (``_added_into``), where a
new tensor per operation per client was most of their cost for a thousand
clients. Held here against the out-of-place helpers they replace, in float32
and float64, and with a delta whose dtype differs (the promoting path).
"""

from __future__ import annotations

import unittest

import pytest
import torch

from fedbrew.clients.torch_scaffold_client import TorchScaffoldClient
from fedbrew.core.torch_utils import (
    add_model_states,
    scale_model_state,
    subtract_model_states,
    zeros_like_model_state,
)
from fedbrew.servers.scaffold import _added_into

pytestmark = pytest.mark.fast


def _state(generator: torch.Generator, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        "w": torch.randn(7, 5, generator=generator, dtype=dtype),
        "b": torch.randn(5, generator=generator, dtype=dtype),
    }


class TheControlUpdateTest(unittest.TestCase):
    def test_in_place_is_the_helpers(self) -> None:
        generator = torch.Generator().manual_seed(3)
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                states = [_state(generator, dtype) for _ in range(4)]
                client = TorchScaffoldClient.__new__(TorchScaffoldClient)
                client.learning_rate = 0.03
                delta = client._update_client_control(*states, local_steps=7)
                correction = scale_model_state(
                    subtract_model_states(states[0], states[1]), 1.0 / (7 * 0.03)
                )
                expected = add_model_states(subtract_model_states(states[2], states[3]), correction)
                for key in expected:
                    self.assertTrue(torch.equal(client._client_control[key], expected[key]))
                    expected_delta = subtract_model_states(expected, states[2])[key]
                    self.assertTrue(torch.equal(delta[key], expected_delta))


class TheServersSumTest(unittest.TestCase):
    def test_in_place_is_the_helpers(self) -> None:
        generator = torch.Generator().manual_seed(4)
        for dtypes in ((torch.float32,) * 3, (torch.float64,) * 3, (torch.float32, torch.float64)):
            with self.subTest(dtypes=dtypes):
                deltas = [_state(generator, dtype) for dtype in dtypes]
                total = zeros_like_model_state(deltas[0])
                expected = zeros_like_model_state(deltas[0])
                for delta in deltas:
                    total = _added_into(total, delta)
                    expected = add_model_states(expected, delta)
                for key in expected:
                    self.assertEqual(total[key].dtype, expected[key].dtype)
                    self.assertTrue(torch.equal(total[key], expected[key]))


if __name__ == "__main__":
    unittest.main()
