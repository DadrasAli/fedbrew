"""Focused tests for the optional tiny GPT-2 model and causal-LM task."""

from __future__ import annotations

import importlib.util
import math
import unittest

import pytest
import torch

from fedbrew.models.tiny_gpt2 import build_tiny_gpt2
from fedbrew.tasks.causal_lm import TorchCausalLMTask


@unittest.skipUnless(
    importlib.util.find_spec("transformers") is not None,
    "transformers is an optional LLM dependency",
)
class TinyCausalLMTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model_config = {
            "name": "tiny_gpt2",
            "vocab_size": 258,
            "sequence_length": 4,
            "n_embd": 16,
            "n_layer": 1,
            "n_head": 2,
            "dropout": 0.0,
        }

    @pytest.mark.fast
    def test_factory_builds_random_gpt2_with_sufficient_context(self) -> None:
        model = build_tiny_gpt2({**self.model_config, "n_positions": 2})

        self.assertEqual(model.config.vocab_size, 258)
        self.assertEqual(model.config.n_positions, 4)
        self.assertEqual(model.config.n_embd, 16)
        self.assertEqual(model.config.n_layer, 1)

    # Eight target positions, the last of them token id 0. Whether that
    # position is padding or a real word is a property of the dataset, and the
    # two tests below are the two answers.
    BATCH = (
        torch.tensor([[2, 3, 4, 5], [6, 7, 8, 0]], dtype=torch.long),
        torch.tensor([[3, 4, 5, 1], [7, 8, 1, 0]], dtype=torch.long),
    )

    def _run_one_step(self, model_config: dict[str, object]) -> dict[str, float]:
        task = TorchCausalLMTask(model_config=model_config, batch_size=2, device="cpu")
        model = task.build_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = next(model.parameters()).detach().clone()

        output = task.train_step(model, self.BATCH, optimizer)

        self.assertTrue(math.isfinite(output["loss"]))
        self.assertGreaterEqual(output["correct"], 0.0)
        self.assertLessEqual(output["correct"], output["total"])
        self.assertFalse(torch.equal(before, next(model.parameters()).detach()))

        metrics = task.compute_metrics([output])
        self.assertEqual(metrics["accuracy"], output["correct"] / output["total"])
        self.assertEqual(metrics["loss"], output["loss"])
        return output

    def test_a_declared_padding_token_is_excluded_from_the_token_counts(self) -> None:
        # tiny_causal_lm declares padding_token_id 0, and factory copies it to
        # model.pad_token_id, so the trailing 0 is padding: seven active
        # targets out of eight.
        output = self._run_one_step({**self.model_config, "pad_token_id": 0})

        self.assertEqual(output["total"], 7.0)

    def test_token_zero_counts_when_the_dataset_declares_no_padding_token(self) -> None:
        # No padding_token_id in the manifest means no padding token, not
        # token 0. Both SFT generators write `padding_token_id: null`, and on
        # the shipped Qwen2.5 vocabulary id 0 is "!" -- a word, not padding.
        # All eight targets are active.
        output = self._run_one_step(dict(self.model_config))

        self.assertEqual(output["total"], 8.0)


if __name__ == "__main__":
    unittest.main()
