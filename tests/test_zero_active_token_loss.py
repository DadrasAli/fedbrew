"""A causal-LM batch with no supervised token contributes zero, not NaN.

``_loss_and_counts`` needs a loss for the case where every target position is
inactive: the caller runs ``backward()`` and ``step()`` unconditionally, so the
value has to be 0.0 *and* carry a gradient path. It used to build one as
``flat_logits.sum() * 0.0``, which reads every logit to produce a constant.

That form is not zero-safe. Any non-finite logit in the block makes the sum
non-finite, and ``inf * 0.0`` and ``nan * 0.0`` are both ``nan`` -- so a batch
that should contribute nothing injects a NaN loss and NaN gradients instead.
It bites only on an already-diverging model, which is when the run most needs
an honest number, and from there the NaN is averaged, broadcast and
checkpointed like any other value.

Accumulation overflow is the other half of the same defect and the far rarer
one: an LLM logit block is ``batch x sequence_length x vocabulary``, which at
batch 8, 512 tokens and Qwen's 151665-token vocabulary is 621,219,840 float32
values, so a plain sum overflows once the mean magnitude passes 5.5e29.
Diverging logits reach ``inf`` individually long before they reach that, which
is why the tests below drive the reachable path rather than that one.

``flat_logits[:0].sum()`` sums no elements: 0.0 whatever the logits hold,
still grad-connected, and its backward scatters zeros over the full tensor.
"""

from __future__ import annotations

import math
import unittest

import pytest
import torch

from fedbrew.tasks.causal_lm import TorchCausalLMTask

IGNORE_INDEX = -100
VOCAB_SIZE = 32


def _task() -> TorchCausalLMTask:
    return TorchCausalLMTask(model_config={"pad_token_id": None}, device="cpu")


def _all_inactive_targets() -> torch.Tensor:
    return torch.full((2, 4), IGNORE_INDEX, dtype=torch.long)


class ZeroActiveTokenLossTests(unittest.TestCase):
    @pytest.mark.fast
    def test_zero_active_tokens_gives_a_finite_zero_loss(self) -> None:
        loss, correct, total = _task()._loss_and_counts(
            torch.zeros(2, 4, VOCAB_SIZE, requires_grad=True),
            _all_inactive_targets(),
        )

        self.assertEqual(total, 0)
        self.assertEqual(correct, 0)
        self.assertEqual(loss.item(), 0.0)

    @pytest.mark.fast
    def test_a_non_finite_logit_does_not_reach_the_loss(self) -> None:
        for bad in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(logit=bad):
                logits = torch.zeros(2, 4, VOCAB_SIZE)
                logits[0, 0, 0] = bad
                logits.requires_grad_(True)

                loss, _, total = _task()._loss_and_counts(logits, _all_inactive_targets())

                self.assertEqual(total, 0)
                self.assertTrue(math.isfinite(loss.item()), f"{bad} reached the loss")
                self.assertEqual(loss.item(), 0.0)

    def test_the_zero_loss_still_carries_a_gradient_path(self) -> None:
        logits = torch.zeros(2, 4, VOCAB_SIZE, requires_grad=True)

        loss, _, _ = _task()._loss_and_counts(logits, _all_inactive_targets())
        self.assertTrue(loss.requires_grad)
        loss.backward()

        assert logits.grad is not None
        self.assertEqual(tuple(logits.grad.shape), (2, 4, VOCAB_SIZE))
        self.assertTrue(bool((logits.grad == 0).all()))

    def test_a_whole_train_step_survives_a_diverged_forward(self) -> None:
        # The path the caller actually takes: forward, loss, backward, step.
        # With the old form this raised nothing and quietly wrote NaN into
        # every parameter the logits depend on.
        model = torch.nn.Linear(VOCAB_SIZE, VOCAB_SIZE)
        with torch.no_grad():
            model.bias.fill_(float("inf"))
        task = _task()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        logits = model(torch.zeros(2, 4, VOCAB_SIZE))
        loss, _, total = task._loss_and_counts(logits, _all_inactive_targets())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.assertEqual(total, 0)
        self.assertTrue(math.isfinite(loss.item()))
        self.assertTrue(bool(torch.isfinite(model.weight).all()))


if __name__ == "__main__":
    unittest.main()
