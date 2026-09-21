"""Tied parameters must be federated once, not once per state-dict key."""

from __future__ import annotations

import unittest

import pytest
import torch
from torch import nn

from fedbrew.core.torch_utils import (
    get_untied_model_state,
    load_untied_model_state,
    tied_state_aliases,
)

pytestmark = pytest.mark.fast


class _TiedModel(nn.Module):
    """Two layers sharing one weight, as an LM ties its head to its embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(4, 4, bias=False)
        self.head = nn.Linear(4, 4, bias=False)
        self.head.weight = self.embed.weight
        self.other = nn.Linear(4, 2, bias=False)


class _UntiedModel(nn.Module):
    """Same shapes, no sharing, standing in for the vision models."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(4, 4, bias=False)
        self.head = nn.Linear(4, 4, bias=False)
        self.other = nn.Linear(4, 2, bias=False)


class TiedWeightFederationTest(unittest.TestCase):
    def test_the_duplicate_key_is_detected(self) -> None:
        self.assertEqual(tied_state_aliases(_TiedModel()), {"head.weight": "embed.weight"})

    def test_untied_models_are_unaffected(self) -> None:
        model = _UntiedModel()
        self.assertEqual(tied_state_aliases(model), {})
        self.assertEqual(set(get_untied_model_state(model)), set(model.state_dict()))

    def test_the_duplicate_is_not_communicated(self) -> None:
        model = _TiedModel()
        state = get_untied_model_state(model)
        self.assertNotIn("head.weight", state)
        self.assertIn("embed.weight", state)
        communicated = sum(int(tensor.numel()) for tensor in state.values())
        unique = sum(int(p.numel()) for p in dict(model.named_parameters()).values())
        self.assertEqual(communicated, unique)

    def test_loading_restores_the_tie_and_round_trips(self) -> None:
        source = _TiedModel()
        with torch.no_grad():
            source.embed.weight.fill_(0.5)
        state = get_untied_model_state(source)

        target = _TiedModel()
        load_untied_model_state(target, state)
        self.assertTrue(torch.equal(target.embed.weight, source.embed.weight))
        self.assertTrue(torch.equal(target.head.weight, source.embed.weight))
        # The tie itself must survive the load, not just the values.
        self.assertEqual(target.head.weight.data_ptr(), target.embed.weight.data_ptr())

    def test_a_state_that_still_carries_the_duplicate_is_accepted(self) -> None:
        # Checkpoints written before deduplication must remain loadable.
        model = _TiedModel()
        legacy = dict(model.state_dict())
        legacy["head.weight"] = torch.full((4, 4), 0.25)
        legacy["embed.weight"] = torch.full((4, 4), 0.25)
        load_untied_model_state(model, legacy)
        self.assertTrue(torch.equal(model.embed.weight, torch.full((4, 4), 0.25)))

    def test_a_state_missing_the_shared_source_is_rejected(self) -> None:
        model = _TiedModel()
        state = get_untied_model_state(model)
        del state["embed.weight"]
        with self.assertRaisesRegex(ValueError, "needed to restore the tied"):
            load_untied_model_state(model, state)


if __name__ == "__main__":
    unittest.main()
