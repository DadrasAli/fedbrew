"""The option-scoring accuracies divide by the questions they scored.

`fedbrew eval-medmcqa` runs two passes over each question. The letter pass
tokenizes the shared context plus one letter token; the option pass tokenizes
each complete candidate response, which is a strictly longer string. So a
question can clear the first length test and fail the second -- and when it
did, the `continue` left it contributing nothing to `option_sum_correct` or
`option_norm_correct` while `scored += len(usable)` still counted it. Both
option accuracies were biased low by exactly that fraction, and nothing in the
record said how many questions it was. `accuracy_letter` and
`skipped_too_long` were unaffected. P10-F27.

Measured on the real held-out MedMCQA sets at the shipped `--max-length 1024`,
replaying `_held_out_rows` and both length tests against the Qwen2.5-0.5B
tokenizer:

    split         held-out  letter skips  option skips  longest candidate
    client_eval     16,367             0             0          416
    global_test     16,242             0             0          417

Zero at that cap, with 2.5x headroom. At `--max-length 384` the mechanism fires
-- client_eval skips two questions in the letter pass and one more in the option
pass -- which is what these tests reproduce in miniature.
"""

from __future__ import annotations

import unittest
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
import torch

from fedbrew.cli.eval_medmcqa_choice import ChoiceTotals, accuracy_record, score_questions
from tests.test_medmcqa_choice_eval import _WordTokenizer

pytestmark = pytest.mark.fast

LABELS = ["A", "B", "C", "D"]
SYSTEM_PROMPT = "You are a medical expert."
ANSWER_PREFIX = "The correct answer is "


class _ConstantModel(torch.nn.Module):
    """Always prefers the first candidate, so every count is predictable."""

    def __init__(self, vocab: int = 4096) -> None:
        super().__init__()
        self.vocab = vocab
        self.calls = 0

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        self.calls += 1
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.vocab)
        # Row 0 of a candidate batch wins; in the letter pass the same bias
        # falls on whichever token id is lowest, which is stable per question.
        logits[0] += 1.0
        return type("Output", (), {"logits": logits})()


def _question(index: int, *, option_words: int, prompt_words: int = 4) -> dict[str, Any]:
    """A question whose prompt length and option length move independently.

    That separation is the finding. The letter pass tokenizes the prompt plus
    one letter token; the option pass tokenizes the prompt plus a whole
    option. So a short prompt with long options clears the first length test
    and fails the second. A real MedMCQA prompt restates its options, which is
    why the two lengths usually move together -- and why, on the real data at
    --max-length 1024, the second test never fires.
    """

    options = [" ".join([f"opt{index}x{choice}"] * option_words) for choice in range(4)]
    return {
        "id": f"row-{index}",
        "prompt": " ".join([f"q{index}word"] * prompt_words),
        "options": options,
        "answer_index": 0,
    }


def _score(questions: Sequence[Mapping[str, Any]], *, max_length: int, **kwargs: Any):
    return score_questions(
        questions,
        model=_ConstantModel(),
        tokenizer=_WordTokenizer(),
        labels=LABELS,
        system_prompt=SYSTEM_PROMPT,
        answer_prefix=ANSWER_PREFIX,
        device=torch.device("cpu"),
        batch_size=4,
        max_length=max_length,
        **kwargs,
    )


class TheTwoDenominatorsAreCountedApartTest(unittest.TestCase):
    def test_a_question_too_long_for_the_option_pass_leaves_its_denominator(self) -> None:
        """The defect: it stayed in `scored` and scored for no option metric."""

        short = [_question(index, option_words=1) for index in range(3)]
        long_options = [_question(99, option_words=200)]
        totals = _score([*short, *long_options], max_length=64)

        self.assertEqual(totals.skipped_too_long, 0, "the letter pass must accept all four")
        self.assertEqual(totals.scored, 4)
        self.assertEqual(totals.option_scored, 3)
        self.assertEqual(totals.skipped_option_too_long, 1)

    def test_the_option_rates_divide_by_the_option_count(self) -> None:
        totals = ChoiceTotals(
            scored=4,
            option_scored=3,
            skipped_option_too_long=1,
            letter_correct=4,
            option_sum_correct=3,
            option_norm_correct=3,
        )
        record = accuracy_record(totals, labels=LABELS, letter_only=False)
        self.assertEqual(record["accuracy_option_sum"], 1.0)
        self.assertEqual(record["accuracy_option_lengthnorm"], 1.0)
        # The pre-fix arithmetic, for contrast: 3/4.
        self.assertNotEqual(record["accuracy_option_sum"], 0.75)

    def test_the_letter_rate_still_divides_by_the_letter_count(self) -> None:
        """Unaffected by the finding, and must stay that way."""

        totals = ChoiceTotals(scored=4, option_scored=3, letter_correct=2)
        record = accuracy_record(totals, labels=LABELS, letter_only=False)
        self.assertEqual(record["accuracy_letter"], 0.5)

    def test_both_counts_are_in_the_record_beside_their_rates(self) -> None:
        """The bias was invisible; a reader can now check either denominator."""

        totals = ChoiceTotals(scored=4, option_scored=3, skipped_option_too_long=1)
        record = accuracy_record(totals, labels=LABELS, letter_only=False)
        self.assertEqual(record["evaluated"], 4)
        self.assertEqual(record["option_evaluated"], 3)
        self.assertEqual(record["skipped_option_too_long"], 1)

    def test_letter_only_reports_no_option_measurement(self) -> None:
        """A zero would read as a measurement; there was none."""

        record = accuracy_record(ChoiceTotals(scored=4), labels=LABELS, letter_only=True)
        for key in ("option_evaluated", "skipped_option_too_long", "accuracy_option_sum"):
            with self.subTest(key=key):
                self.assertNotIn(key, record)

    def test_letter_only_skips_the_option_pass_entirely(self) -> None:
        totals = _score(
            [_question(index, option_words=1) for index in range(3)],
            max_length=64,
            letter_only=True,
        )
        self.assertEqual(totals.scored, 3)
        self.assertEqual(totals.option_scored, 0)
        self.assertEqual(totals.skipped_option_too_long, 0)


class TheLetterPassSkipIsSeparateTest(unittest.TestCase):
    def test_a_question_too_long_for_the_letter_pass_enters_neither_count(self) -> None:
        """It never becomes `usable`, so it is in no denominator at all."""

        # A long *prompt* this time: the letter pass tokenizes that, so this
        # question is refused before it can be usable for either pass.
        totals = _score(
            [_question(0, option_words=1), _question(1, option_words=1, prompt_words=400)],
            max_length=48,
        )
        self.assertEqual(totals.skipped_too_long, 1)
        self.assertEqual(totals.scored, 1)
        self.assertEqual(totals.option_scored, 1)
        self.assertEqual(totals.skipped_option_too_long, 0)

    def test_nothing_is_skipped_when_everything_fits(self) -> None:
        """A guard whose cases all skip would pass on a loop that skips always."""

        totals = _score([_question(index, option_words=1) for index in range(5)], max_length=4096)
        self.assertEqual(totals.skipped_too_long, 0)
        self.assertEqual(totals.skipped_option_too_long, 0)
        self.assertEqual(totals.scored, 5)
        self.assertEqual(totals.option_scored, 5)


if __name__ == "__main__":
    unittest.main()
