"""``TorchClassificationTask.train_step`` refuses to invent an optimizer.

The sibling of `tests/test_causal_lm_train_step_requires_an_optimizer.py`, and
the half of the same defect that happened not to be filed. The review that
found this named the causal-LM task only, because a per-call **AdamW** resets
Adam's moments every batch and degenerates the rule to sign-SGD. This task
built a per-call `SGD(lr=0.01)`, and plain SGD has no moments to reset -- so
the sharper half of that finding does not apply here.

What does apply is the rest of it. The rate is hard-coded, and so are
`momentum`, `weight_decay` and `nesterov`, every one of which the client config
carries and passes through its own optimizer. A caller that omitted the
optimizer trained at 0.01 with no momentum whatever the config said, and
nothing in the run recorded that it had.

Both tasks now refuse, so the contract is one contract rather than two
different fallbacks. Every rule in `fedbrew/clients/` passes an optimizer, so
nothing in the tree took either.
"""

from __future__ import annotations

import unittest

import pytest
import torch
from torch import Tensor, nn

from fedbrew.tasks.classification.torch_classification import TorchClassificationTask

FEATURES = 4
CLASSES = 3


def _batch() -> tuple[Tensor, Tensor]:
    torch.manual_seed(11)
    return torch.randn(8, FEATURES), torch.randint(0, CLASSES, (8,))


class ClassificationTrainStepRequiresAnOptimizerTests(unittest.TestCase):
    @pytest.mark.fast
    def test_omitting_the_optimizer_is_refused(self) -> None:
        torch.manual_seed(0)
        task = TorchClassificationTask(device="cpu")
        model = nn.Linear(FEATURES, CLASSES)
        before = model.weight.detach().clone()

        with self.assertRaises(ValueError) as caught:
            task.train_step(model, _batch())

        message = str(caught.exception)
        self.assertIn("optimizer", message)
        self.assertIn("configured", message)
        # Refused before anything moved, not after a partial step.
        self.assertTrue(torch.equal(before, model.weight.detach()))

    def test_passing_an_optimizer_still_trains(self) -> None:
        torch.manual_seed(0)
        task = TorchClassificationTask(device="cpu")
        model = nn.Linear(FEATURES, CLASSES)
        before = model.weight.detach().clone()

        output = task.train_step(model, _batch(), torch.optim.SGD(model.parameters(), lr=0.01))

        self.assertIn("loss", output)
        self.assertFalse(torch.equal(before, model.weight.detach()))

    def test_a_hard_coded_rate_is_not_the_configured_one(self) -> None:
        # What the refusal replaces: the fallback's 0.01 is reachable only by
        # coincidence, and every other configured rate produces a different
        # step that the omitted optimizer would have thrown away.
        def step_size(learning_rate: float) -> float:
            torch.manual_seed(0)
            task = TorchClassificationTask(device="cpu")
            model = nn.Linear(FEATURES, CLASSES)
            before = model.weight.detach().clone()
            task.train_step(model, _batch(), torch.optim.SGD(model.parameters(), lr=learning_rate))
            return float((model.weight.detach() - before).abs().max())

        at_the_fallback_rate = step_size(0.01)
        self.assertGreater(step_size(0.5) / at_the_fallback_rate, 10.0)

    @pytest.mark.fast
    def test_both_tasks_refuse_the_same_way(self) -> None:
        # The asymmetry this commit removes: one task refused and the other
        # invented a rule. A future fallback added to either fails here.
        #
        # Both models must own parameters. A parameterless one makes
        # torch.optim raise "empty parameter list" of its own accord, which is
        # also a ValueError -- so this test would pass against a restored
        # fallback and assert nothing. It did, until that was caught.
        from fedbrew.tasks.causal_lm import TorchCausalLMTask

        class _Logits(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.projection = nn.Embedding(CLASSES, CLASSES)

            def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
                return self.projection(input_ids)

        causal_lm = TorchCausalLMTask(model_config={"pad_token_id": None}, device="cpu")
        tokens = torch.zeros(1, 2, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "train_step requires an optimizer"):
            causal_lm.train_step(_Logits(), (tokens, tokens))
        with self.assertRaisesRegex(ValueError, "train_step requires an optimizer"):
            TorchClassificationTask(device="cpu").train_step(nn.Linear(FEATURES, CLASSES), _batch())


if __name__ == "__main__":
    unittest.main()
