"""A causal-LM padding token that is also the EOS token is refused.

``TorchCausalLMTask`` masks targets by **value**: a position is inactive when
its target equals ``ignore_index`` or equals ``pad_token_id``. That is fine for
a padding id no real target can carry, and wrong for one that can. EOS is the
case that occurs: ``hf_causal_lm_text._padding_token_id`` falls back to the
tokenizer's EOS id when the tokenizer has no distinct padding token, which is
the norm for GPT-2- and Qwen-style tokenizers, and the manifest then hands that
id to the task as ``pad_token_id``.

Applied, the filter removes every genuine end-of-document target from the loss,
from the accuracy numerator and denominator, and from the active-token count
that becomes the SFT aggregation weight -- so the model is never trained to
emit EOS and nothing in the metrics disagrees, because all three are measured
over the same surviving positions. The measurement below is what the refusal
replaces: four target positions, one of them a real EOS, counted as three.

`dataset_eos_token_id` is written by `factory._add_causal_manifest_metadata`
from the manifest's `eos_token_id`; this guard is its first reader. A manifest
without that key -- `tiny_causal_lm` records `end_of_text_token_id` instead --
carries no EOS claim to check, so the task is built as before.
"""

from __future__ import annotations

import unittest

import pytest
import torch

from fedbrew.tasks.causal_lm import TorchCausalLMTask

pytestmark = pytest.mark.fast

EOS_TOKEN_ID = 50256
VOCAB_SIZE = 50260


class PadTokenIdIsNotEosTests(unittest.TestCase):
    def test_pad_token_id_equal_to_dataset_eos_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            TorchCausalLMTask(
                model_config={
                    "pad_token_id": EOS_TOKEN_ID,
                    "dataset_eos_token_id": EOS_TOKEN_ID,
                },
                device="cpu",
            )

        message = str(caught.exception)
        self.assertIn("eos_token_id", message)
        self.assertIn(str(EOS_TOKEN_ID), message)

    def test_distinct_padding_token_still_builds_and_still_filters(self) -> None:
        task = TorchCausalLMTask(
            model_config={
                "pad_token_id": 0,
                "dataset_eos_token_id": EOS_TOKEN_ID,
            },
            device="cpu",
        )

        self.assertEqual(task.pad_token_id, 0)
        targets = torch.tensor([[10, 11, EOS_TOKEN_ID, 0]], dtype=torch.long)
        _, _, total = task._loss_and_counts(torch.zeros(1, 4, VOCAB_SIZE), targets)
        # The EOS target survives; the padding target does not.
        self.assertEqual(total, 3)

    def test_manifest_without_an_eos_claim_is_not_refused(self) -> None:
        task = TorchCausalLMTask(
            model_config={"pad_token_id": EOS_TOKEN_ID},
            device="cpu",
        )

        self.assertEqual(task.pad_token_id, EOS_TOKEN_ID)

    def test_padding_and_eos_are_indistinguishable_by_value(self) -> None:
        # What the refusal above is refusing, measured on the same tensors.
        task = TorchCausalLMTask(model_config={"pad_token_id": None}, device="cpu")
        task.pad_token_id = EOS_TOKEN_ID

        targets = torch.tensor([[10, 11, EOS_TOKEN_ID, 12]], dtype=torch.long)
        logits = torch.zeros(1, 4, VOCAB_SIZE)
        _, _, filtered = task._loss_and_counts(logits, targets)

        task.pad_token_id = None
        _, _, unfiltered = task._loss_and_counts(logits, targets)

        self.assertEqual(unfiltered, 4)
        self.assertEqual(filtered, 3)


if __name__ == "__main__":
    unittest.main()
