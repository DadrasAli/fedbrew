"""The MC-accuracy evaluator must reproduce the training split and prompt exactly.

Every failure mode here is silent: a drifted filter or template still produces a
plausible accuracy, just for a different question set or a distribution the
model was never trained on.
"""

from __future__ import annotations

import unittest
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from fedbrew.cli.eval_medmcqa_choice import (
    _cross_check,
    _held_out_rows,
    _letter_positions,
    _subsample,
)
from fedbrew.data.oasst1_sft import assign_tree_split

pytestmark = pytest.mark.fast

PROMPT_TEMPLATE = "{question}\n\nA. {opa}\nB. {opb}\nC. {opc}\nD. {opd}"
RESPONSE_TEMPLATE = "The correct answer is {answer_label}. {answer_text}\n\n{exp}"
SYSTEM_PROMPT = "You are a medical expert answering board examination questions."

CHOICE = {
    "index_field": "cop",
    "option_fields": ["opa", "opb", "opc", "opd"],
    "labels": ["A", "B", "C", "D"],
}

SFT_CONFIG: dict[str, Any] = {
    "prompt_template": PROMPT_TEMPLATE,
    "response_template": RESPONSE_TEMPLATE,
    "system_prompt": SYSTEM_PROMPT,
    "group_field": "id",
    "client_field": "subject_name",
    "required_fields": ["exp", "question"],
    "choice": CHOICE,
}

MANIFEST: dict[str, Any] = {
    "prompt_template": PROMPT_TEMPLATE,
    "response_template": RESPONSE_TEMPLATE,
    "system_prompt": SYSTEM_PROMPT,
    "group_field": "id",
    "partition_field": "subject_name",
    "required_fields": ["exp", "question"],
    "seed": 42,
    "split_ratios": {"train": 0.8, "client_eval": 0.1, "global_test": 0.1},
}


def _row(row_id: str, **overrides: Any) -> dict[str, Any]:
    row = {
        "id": row_id,
        "subject_name": "Anatomy",
        "question": "Which structure is described?",
        "opa": "First",
        "opb": "Second",
        "opc": "Third",
        "opd": "Fourth",
        "cop": 2,
        "exp": "Because of the third one.",
    }
    row.update(overrides)
    return row


class _WordTokenizer:
    """Whitespace tokenizer with a chat template, standing in for the real one.

    Word-level rather than byte-level on purpose: the evaluator must not depend
    on how a particular tokenizer splits ``" A"``.
    """

    chat_template = "present"
    eos_token_id = 0

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def _id(self, token: str) -> int:
        return self._ids.setdefault(token, len(self._ids) + 1)

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        return_tensors: Any = None,
    ) -> list[int]:
        parts: list[str] = []
        for message in messages:
            parts.extend([f"<{message['role']}>", *str(message["content"]).split()])
        if add_generation_prompt:
            parts.append("<assistant>")
        return [self._id(token) for token in parts]


class CrossCheckTest(unittest.TestCase):
    def test_matching_config_and_manifest_pass(self) -> None:
        _cross_check(SFT_CONFIG, MANIFEST)

    def test_a_drifted_prompt_template_is_rejected(self) -> None:
        drifted = dict(SFT_CONFIG, prompt_template=PROMPT_TEMPLATE + "\nAnswer:")
        with self.assertRaisesRegex(ValueError, "prompt_template"):
            _cross_check(drifted, MANIFEST)

    def test_drifted_required_fields_are_rejected(self) -> None:
        # A looser filter silently evaluates on questions training never held out.
        drifted = dict(SFT_CONFIG, required_fields=["question"])
        with self.assertRaisesRegex(ValueError, "required_fields"):
            _cross_check(drifted, MANIFEST)


class HeldOutRecoveryTest(unittest.TestCase):
    def _rows(self) -> list[dict[str, Any]]:
        return [_row(f"row-{index}") for index in range(400)]

    def _expected(self, rows, split: str) -> set[str]:
        return {
            row["id"]
            for row in rows
            if assign_tree_split(
                row["id"], seed=42, train_ratio=0.8, client_eval_ratio=0.1, global_test_ratio=0.1
            )
            == split
        }

    def test_only_the_requested_split_is_kept(self) -> None:
        rows = self._rows()
        for split in ("client_eval", "global_test"):
            with self.subTest(split=split):
                kept, _ = _held_out_rows(
                    rows, sft_config=SFT_CONFIG, manifest=MANIFEST, split=split
                )
                expected = self._expected(rows, split)
                self.assertEqual({item["id"] for item in kept}, expected)
                self.assertTrue(expected, "the fixture produced no held-out rows")

    def test_the_default_split_is_validation_not_test(self) -> None:
        """Scoring only ever produced a global_test number,
        so arms were compared on test data and the winner's test number
        reported. client_eval is disjoint, already in the shards, and already
        evaluated every 5 rounds by the shipped configs."""

        rows = self._rows()
        kept, _ = _held_out_rows(rows, sft_config=SFT_CONFIG, manifest=MANIFEST)

        self.assertEqual({item["id"] for item in kept}, self._expected(rows, "client_eval"))

    def test_the_two_splits_share_no_questions(self) -> None:
        rows = self._rows()
        kept = {
            split: {
                item["id"]
                for item in _held_out_rows(
                    rows, sft_config=SFT_CONFIG, manifest=MANIFEST, split=split
                )[0]
            }
            for split in ("client_eval", "global_test")
        }

        self.assertFalse(kept["client_eval"] & kept["global_test"])
        self.assertTrue(kept["client_eval"])
        self.assertTrue(kept["global_test"])

    def test_rows_missing_a_required_field_are_dropped(self) -> None:
        rows = [*self._rows(), _row("no-exp", exp=None), _row("blank-exp", exp="   ")]
        _, counts = _held_out_rows(rows, sft_config=SFT_CONFIG, manifest=MANIFEST)
        self.assertEqual(counts["rows_missing_required_fields"], 2)

    def test_an_out_of_range_answer_index_is_dropped(self) -> None:
        rows = [_row("bad-cop", cop=7)]
        kept, counts = _held_out_rows(rows, sft_config=SFT_CONFIG, manifest=MANIFEST)
        self.assertEqual(kept, [])
        self.assertEqual(counts["rows_failing_template_rendering"], 1)

    def test_the_recovered_answer_index_and_options_survive(self) -> None:
        rows = self._rows()
        kept, _ = _held_out_rows(rows, sft_config=SFT_CONFIG, manifest=MANIFEST)
        self.assertEqual(kept[0]["answer_index"], 2)
        self.assertEqual(kept[0]["options"], ["First", "Second", "Third", "Fourth"])

    def test_the_split_is_stable_across_row_order(self) -> None:
        rows = self._rows()
        forward, _ = _held_out_rows(rows, sft_config=SFT_CONFIG, manifest=MANIFEST)
        backward, _ = _held_out_rows(list(reversed(rows)), sft_config=SFT_CONFIG, manifest=MANIFEST)
        self.assertEqual({i["id"] for i in forward}, {i["id"] for i in backward})


class LetterPositionTest(unittest.TestCase):
    def _question(self) -> dict[str, Any]:
        return {
            "prompt": "Which structure?\n\nA. First\nB. Second\nC. Third\nD. Fourth",
            "options": ["First", "Second", "Third", "Fourth"],
            "answer_index": 2,
        }

    def test_the_four_candidates_diverge_at_distinct_letter_tokens(self) -> None:
        context, letter_ids = _letter_positions(
            self._question(),
            labels=["A", "B", "C", "D"],
            system_prompt=SYSTEM_PROMPT,
            answer_prefix="The correct answer is ",
            tokenizer=_WordTokenizer(),
        )
        self.assertEqual(len(letter_ids), 4)
        self.assertEqual(len(set(letter_ids)), 4)
        self.assertNotIn(letter_ids[0], context)

    def test_the_context_ends_before_the_letter(self) -> None:
        tokenizer = _WordTokenizer()
        context, letter_ids = _letter_positions(
            self._question(),
            labels=["A", "B", "C", "D"],
            system_prompt=SYSTEM_PROMPT,
            answer_prefix="The correct answer is ",
            tokenizer=tokenizer,
        )
        # The scored position must predict the letter, so the shared context is
        # everything up to but excluding it.
        self.assertEqual(context[-1], tokenizer._id("is"))

    def test_a_prefix_that_never_diverges_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "never diverge"):
            _letter_positions(
                self._question(),
                labels=["A", "A", "A", "A"],
                system_prompt=SYSTEM_PROMPT,
                answer_prefix="The correct answer is ",
                tokenizer=_WordTokenizer(),
            )


class SubsampleTest(unittest.TestCase):
    def _questions(self) -> list[dict[str, Any]]:
        return [{"id": f"row-{index}"} for index in range(500)]

    def test_subsampling_is_deterministic(self) -> None:
        first = _subsample(self._questions(), 50)
        second = _subsample(list(reversed(self._questions())), 50)
        self.assertEqual([item["id"] for item in first], [item["id"] for item in second])

    def test_no_limit_returns_everything(self) -> None:
        self.assertEqual(len(_subsample(self._questions(), None)), 500)
        self.assertEqual(len(_subsample(self._questions(), 9999)), 500)

    def test_the_subsample_does_not_reuse_the_split_hash(self) -> None:
        # Salting with the split key would bias the subsample toward one end of
        # the hash space that defined global_test in the first place.
        chosen = [item["id"] for item in _subsample(self._questions(), 50)]
        by_split = sorted(
            self._questions(),
            key=lambda item: assign_tree_split(
                item["id"],
                seed=42,
                train_ratio=0.8,
                client_eval_ratio=0.1,
                global_test_ratio=0.1,
            ),
        )[:50]
        self.assertNotEqual(chosen, [item["id"] for item in by_split])


if __name__ == "__main__":
    unittest.main()
