"""``TorchCausalLMTask.train_step`` refuses to invent an optimizer.

It used to build a fresh ``AdamW(model.parameters(), lr=1e-3)`` whenever a
caller omitted one. That is not a default, it is a different algorithm. An
optimizer built per call has no accumulated state, and Adam's first step is

    lr * m_hat / (sqrt(v_hat) + eps) = lr * g / |g| = lr * sign(g)

so every batch takes a fixed-size step: sign-SGD at a hard-coded 1e-3, with the
configured learning rate unreachable. `test_a_per_call_optimizer_is_a_different
_algorithm` below is that measurement, run against the optimizers directly
rather than against the removed branch.

It alternates two batches, because the defect is invisible without that. On one
batch repeated, a reused Adam converges to `sign(g)` as well and both forms
take the same step; the gap opens only when consecutive gradients differ, which
is what a client iterating a dataloader does.

Every rule in `fedbrew/clients/` passes an optimizer -- `torch_sgd_client`,
`torch_scaffold_client`, `torch_fedprox_client`, `torch_fedlalr_client` and the
five call sites in `local_update_modes` -- so nothing in the tree took the
fallback. The refusal is for the next caller, which would otherwise get a
plausible loss curve from the wrong algorithm and nothing to read that says so.

The stub model keeps this file runnable without `transformers`, which the
causal-LM model builders need and the default test environment does not have.
"""

from __future__ import annotations

import unittest

import pytest
import torch
from torch import Tensor, nn

from fedbrew.tasks.causal_lm import TorchCausalLMTask

VOCAB_SIZE = 16
SEQUENCE_LENGTH = 4


class _StubCausalLM(nn.Module):
    """Emits (N, T, V) logits from input ids, with no optional dependencies."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Embedding(VOCAB_SIZE, VOCAB_SIZE)
        nn.init.normal_(self.projection.weight, std=0.02)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        return self.projection(input_ids)


def _task() -> TorchCausalLMTask:
    return TorchCausalLMTask(model_config={"pad_token_id": None}, batch_size=1, device="cpu")


def _batch() -> tuple[Tensor, Tensor]:
    return _BATCHES[0]


#: Two batches, alternated. The moment reset is invisible on one batch
#: repeated: a reused Adam converges to sign(g) there too, and both forms take
#: the same step. It shows only when consecutive gradients differ, which is
#: what a client iterating a dataloader actually does.
_BATCHES = (
    (
        torch.tensor([[2, 3, 4, 5]], dtype=torch.long),
        torch.tensor([[3, 4, 5, 6]], dtype=torch.long),
    ),
    (
        torch.tensor([[9, 10, 11, 12]], dtype=torch.long),
        torch.tensor([[10, 11, 12, 13]], dtype=torch.long),
    ),
)


class TrainStepRequiresAnOptimizerTests(unittest.TestCase):
    @pytest.mark.fast
    def test_omitting_the_optimizer_is_refused(self) -> None:
        torch.manual_seed(0)
        model = _StubCausalLM()
        before = model.projection.weight.detach().clone()

        with self.assertRaises(ValueError) as caught:
            _task().train_step(model, _batch())

        message = str(caught.exception)
        self.assertIn("optimizer", message)
        self.assertIn("state", message)
        # Refused before anything moved, not after a partial step.
        self.assertTrue(torch.equal(before, model.projection.weight.detach()))

    def test_passing_an_optimizer_still_trains(self) -> None:
        torch.manual_seed(0)
        model = _StubCausalLM()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.projection.weight.detach().clone()

        output = _task().train_step(model, _batch(), optimizer)

        self.assertEqual(output["total"], float(SEQUENCE_LENGTH))
        self.assertFalse(torch.equal(before, model.projection.weight.detach()))

    def test_a_per_call_optimizer_is_a_different_algorithm(self) -> None:
        # What the refusal replaces, driven against the optimizers directly
        # rather than against the removed branch.
        def step_sizes(fresh_each_step: bool, learning_rate: float) -> list[float]:
            torch.manual_seed(0)
            task = _task()
            model = _StubCausalLM()
            weight = model.projection.weight
            reused = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            sizes = []
            for index in range(6):
                optimizer = (
                    torch.optim.AdamW(model.parameters(), lr=learning_rate)
                    if fresh_each_step
                    else reused
                )
                before = weight.detach().clone()
                task.train_step(model, _BATCHES[index % 2], optimizer)
                sizes.append(float((weight.detach() - before).abs().max()))
            return sizes

        per_call = step_sizes(fresh_each_step=True, learning_rate=1e-3)
        reused = step_sizes(fresh_each_step=False, learning_rate=1e-3)

        # Sign-SGD: with no accumulated moment, every step is the learning
        # rate, and it stays the learning rate however the batches vary.
        for size in per_call:
            self.assertAlmostEqual(size, 1e-3, places=5)
        self.assertLess(max(per_call) - min(per_call), 1e-6)

        # A reused optimizer has memory, so its steps do not all match.
        self.assertGreater(max(reused) - min(reused), 1e-5)

        # And 1e-3 was hard-coded, so no configured rate could reach the
        # fallback: a real caller's learning rate produces a different step.
        self.assertGreater(step_sizes(fresh_each_step=False, learning_rate=0.05)[0], 0.01)


if __name__ == "__main__":
    unittest.main()
