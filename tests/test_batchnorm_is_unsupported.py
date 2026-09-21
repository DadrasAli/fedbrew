"""A BatchNorm model raises in aggregation, and chapter 06 has to say so.

Chapter 06 section 4 framed GroupNorm as a modelling preference: running
statistics averaged across non-IID clients describe no client. True, and an
understatement of what the code does. `BatchNorm*` also registers
`num_batches_tracked`, an int64 counter incremented once per training forward
pass, and `WeightedStateAccumulator` requires every non-floating tensor to be
bit-identical across clients -- there is no meaningful mean of a counter. Two
clients that took different numbers of optimizer steps therefore fail
aggregation outright, mid-round, after the local iterations are spent.

Different step counts are the normal case: `local_iterations` full passes over
clients of unequal size is unequal steps by construction. So BatchNorm is
unsupported rather than discouraged, and a reader told it was a preference
would find that out several GPU-minutes into a round.

The refusal is narrower than "BatchNorm does not work", and the tests below
pin both edges: equal step counts aggregate cleanly, and GroupNorm under the
same unequal steps aggregates cleanly. A guard that overstated the refusal
would be the same defect in the other direction.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import pytest
import torch
from torch import nn

from fedbrew.core.torch_utils import WeightedStateAccumulator

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER = REPO_ROOT / "docs" / "06-models-and-tasks.md"

#: The counter BatchNorm carries and GroupNorm does not. Not spelled by hand in
#: the assertions below -- it is read back off a real module's state_dict, so a
#: torch release that renames it fails here rather than silently passing.
COUNTER_SUFFIX = "num_batches_tracked"


def _chapter_text() -> str:
    return CHAPTER.read_text(encoding="utf-8")


def _trained_state(norm: nn.Module, steps: int) -> dict[str, torch.Tensor]:
    """One client's state after `steps` optimizer steps, buffers included."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(1, 4, 3), norm)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(steps):
        optimizer.zero_grad()
        model(torch.randn(2, 1, 8, 8)).sum().backward()
        optimizer.step()
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _batch_norm(steps: int) -> dict[str, torch.Tensor]:
    return _trained_state(nn.BatchNorm2d(4), steps)


def _group_norm(steps: int) -> dict[str, torch.Tensor]:
    return _trained_state(nn.GroupNorm(2, 4), steps)


def _aggregate(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    accumulator = WeightedStateAccumulator()
    for state in states:
        accumulator.add(state, 1.0)
    return accumulator.result()


class TheCounterIsWhatBreaksTest(unittest.TestCase):
    def test_batch_norm_carries_a_non_floating_counter(self) -> None:
        state = _batch_norm(3)
        counters = [name for name in state if name.endswith(COUNTER_SUFFIX)]
        self.assertEqual(len(counters), 1, f"BatchNorm2d's buffers changed: {sorted(state)}")
        self.assertFalse(state[counters[0]].is_floating_point())

    def test_it_counts_training_forward_passes(self) -> None:
        """Which is why unequal step counts make it differ between clients."""

        for steps in (1, 3, 5):
            with self.subTest(steps=steps):
                state = _batch_norm(steps)
                counter = next(v for k, v in state.items() if k.endswith(COUNTER_SUFFIX))
                self.assertEqual(int(counter), steps)

    def test_group_norm_carries_no_such_buffer(self) -> None:
        state = _group_norm(3)
        self.assertEqual([name for name in state if name.endswith(COUNTER_SUFFIX)], [])


class AggregationRefusesItTest(unittest.TestCase):
    def test_unequal_step_counts_raise_and_name_the_buffer(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _aggregate([_batch_norm(3), _batch_norm(5)])
        message = str(caught.exception)
        self.assertIn("non-floating state tensor", message)
        self.assertIn(COUNTER_SUFFIX, message)

    def test_equal_step_counts_aggregate_cleanly(self) -> None:
        """The qualification the chapter makes; the refusal is not universal."""

        result = _aggregate([_batch_norm(3), _batch_norm(3)])
        self.assertIn("1.running_mean", result)

    def test_group_norm_survives_the_same_unequal_steps(self) -> None:
        """The contrast that makes the substitution the chapter's answer."""

        result = _aggregate([_group_norm(3), _group_norm(5)])
        self.assertTrue(result)


@pytest.mark.fast
class NothingShippedUsesItTest(unittest.TestCase):
    def test_no_shipped_model_builder_constructs_a_batch_norm(self) -> None:
        offenders = []
        for path in sorted((REPO_ROOT / "fedbrew" / "models").glob("*.py")):
            source = path.read_text(encoding="utf-8")
            if re.search(r"\bnn\.BatchNorm\w*\(", source):
                offenders.append(path.name)
        self.assertEqual(
            offenders,
            [],
            "a shipped builder constructs BatchNorm, which aggregation refuses",
        )


class TheChapterSaysSoTest(unittest.TestCase):
    """It said "GroupNorm, not BatchNorm" and gave only the statistics reason."""

    def setUp(self) -> None:
        self.text = _chapter_text()

    @pytest.mark.fast
    def test_the_section_exists(self) -> None:
        self.assertIn("### 4.1 BatchNorm is refused, not merely discouraged", self.text)

    @pytest.mark.fast
    def test_it_names_the_buffer_and_the_error(self) -> None:
        for phrase in (COUNTER_SUFFIX, "non-floating state tensor"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.text)

    def test_it_quotes_the_message_the_accumulator_actually_raises(self) -> None:
        """Prose paraphrasing an error is how a chapter drifts from it."""

        with self.assertRaises(ValueError) as caught:
            _aggregate([_batch_norm(3), _batch_norm(5)])
        # The key is model-specific, so compare the invariant half.
        quoted = re.search(r"ValueError: (non-floating state tensor .+)$", self.text, re.MULTILINE)
        self.assertIsNotNone(quoted, "section 4.1 must quote the error, not describe it")
        assert quoted is not None
        raised = str(caught.exception)
        self.assertEqual(
            quoted.group(1).split("'")[0],
            raised.split("'")[0],
            "the quoted error and the raised one have diverged",
        )
        self.assertTrue(quoted.group(1).endswith("differs between clients"))

    @pytest.mark.fast
    def test_it_says_which_refusal_does_not_fire_on_equal_step_counts(self) -> None:
        """Overstating it would be the same defect pointing the other way.

        The chapter used to say "Equal step counts aggregate cleanly", which
        was true of the accumulator and, once `factory.build_components`
        started refusing federated buffers, no longer true of a run. Both
        facts have to survive: `test_equal_step_counts_aggregate_cleanly`
        above still pins the accumulator's half, and the chapter has to keep
        distinguishing the two rather than replacing one claim with the other.
        """

        self.assertIn("the same step count", self.text)
        self.assertIn("factory.build_components", self.text)
        self.assertIn("running_mean", self.text)

    @pytest.mark.fast
    def test_it_says_where_the_failure_lands(self) -> None:
        self.assertIn("mid-round", self.text)

    @pytest.mark.fast
    def test_the_invariant_no_longer_reads_as_a_preference_alone(self) -> None:
        invariants = self.text[self.text.index("### Invariants") :]
        self.assertIn(COUNTER_SUFFIX, invariants)


if __name__ == "__main__":
    unittest.main()
