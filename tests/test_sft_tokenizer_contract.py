"""A broken masking contract must not be counted as a few skipped rows.

tokenize_assistant_target's prompt-prefix check is the entire guarantee that
prompt tokens are excluded from the loss. generic_sft caught its ValueError,
incremented rows_failing_tokenization and moved on, so a tokenizer or chat
template that broke the prefix property produced a quietly smaller dataset --
visible only as a number in manifest.filtering_summary, with no threshold --
while every row it *did* accept was masked by the same broken template.

The distinction the fix draws: a row failure is a property of that row's own
text and skipping it is right; a contract failure is a property of the
tokenizer and applies to every row, so it must stop the generator.
"""

from __future__ import annotations

import unittest
from typing import Any

import pytest

from fedbrew.data.generic_sft import FieldMapping, _prepare_rows
from fedbrew.data.oasst1_sft import TokenizerContractError, tokenize_assistant_target

pytestmark = pytest.mark.fast

#: Content a real tokenizer would normalize away entirely -- a row property.
UNTOKENIZABLE = "<blank>"


class StubTokenizer:
    """A chat template with one word per token and a settable defect.

    Each message renders as a turn marker followed by one token per word, so
    the prompt rendering (every message but the last, plus the assistant turn
    marker) is a genuine prefix of the complete rendering. That is exactly the
    property tokenize_assistant_target relies on to place the label mask, and
    each ``defect`` breaks it the way a real tokenizer or template could.
    """

    def __init__(self, defect: str | None = None) -> None:
        self.defect = defect
        self._vocabulary: dict[str, int] = {}
        self.eos_token_id = 0

    def _id(self, token: str) -> int:
        return self._vocabulary.setdefault(token, len(self._vocabulary) + 1)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        return_tensors: Any = None,
    ) -> Any:
        rendered: list[str] = []
        for message in messages:
            rendered.append(f"<{message['role']}>")
            if message["content"].strip() == UNTOKENIZABLE:
                continue
            rendered.extend(f"{message['role']}:{word}" for word in message["content"].split())
        if add_generation_prompt:
            rendered.append("<assistant>")
        ids = [self._id(token) for token in rendered]

        if self.defect == "prefix" and not add_generation_prompt:
            # A template that prepends BOS only to the complete conversation:
            # the prompt is no longer a prefix, so the mask boundary is lost.
            ids = [999] + ids
        if self.defect == "batch":
            return [ids, ids]
        if self.defect == "not_a_sequence":
            return 17
        if self.defect == "non_integer":
            return [float(value) for value in ids]
        return ids


def _mapping() -> FieldMapping:
    return FieldMapping(
        prompt_template="{question}",
        response_template="{answer}",
        client_field="subject",
        group_field="subject",
        required_fields=("question", "answer", "subject"),
        choice=None,
        anonymize_client_ids=False,
        system_prompt=None,
    )


def _rows(count: int = 4) -> list[dict[str, str]]:
    return [
        {
            "question": f"question {index}",
            "answer": f"answer {index}",
            "subject": "anatomy",
        }
        for index in range(count)
    ]


def _prepare(rows: list[dict[str, str]], tokenizer: StubTokenizer) -> Any:
    return _prepare_rows(
        rows,
        mapping=_mapping(),
        tokenizer=tokenizer,
        seed=7,
        ratios=(0.8, 0.1, 0.1),
    )


class ContractFailureTest(unittest.TestCase):
    """Every property of the tokenizer stops the generator."""

    def test_a_broken_prefix_stops_the_generator(self) -> None:
        with self.assertRaises(TokenizerContractError) as caught:
            _prepare(_rows(), StubTokenizer(defect="prefix"))
        message = str(caught.exception)
        self.assertIn("not a prefix", message)
        # The first row already fails, so nothing was silently accepted.
        self.assertIn("source row 1 of 4", message)
        self.assertIn("0 rows tokenized cleanly", message)

    def test_the_message_says_earlier_rows_are_affected_too(self) -> None:
        """A defect that switches on mid-pass must not read as one bad row."""

        tokenizer = StubTokenizer()
        rows = _rows(5)
        real_template = tokenizer.apply_chat_template
        calls = {"n": 0}

        def failing_after_two(*args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            if calls["n"] > 4:  # two calls per row: prompt, then complete
                tokenizer.defect = "prefix"
            return real_template(*args, **kwargs)

        tokenizer.apply_chat_template = failing_after_two  # type: ignore[method-assign]
        with self.assertRaises(TokenizerContractError) as caught:
            _prepare(rows, tokenizer)
        message = str(caught.exception)
        self.assertIn("source row 3 of 5", message)
        self.assertIn("2 rows tokenized cleanly", message)
        self.assertIn("masked wrongly", message)

    def test_every_chat_template_defect_is_a_contract_error(self) -> None:
        for defect in ("batch", "not_a_sequence", "non_integer"):
            with self.subTest(defect=defect):
                with self.assertRaises(TokenizerContractError):
                    _prepare(_rows(), StubTokenizer(defect=defect))

    def test_a_contract_error_is_still_a_value_error(self) -> None:
        """Callers that guard on ValueError keep working; only the swallow site
        in generic_sft distinguishes the two."""

        self.assertTrue(issubclass(TokenizerContractError, ValueError))
        with self.assertRaises(ValueError):
            tokenize_assistant_target(
                [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ],
                StubTokenizer(defect="prefix"),
            )


class RowFailureTest(unittest.TestCase):
    """A row whose own text cannot be tokenized is still skipped and counted."""

    def test_an_empty_assistant_target_is_counted_not_raised(self) -> None:
        rows = _rows(3)
        # Content this tokenizer normalizes away entirely, so the complete ids
        # equal the prompt ids: no active target. That is this row's problem
        # alone, and every other row is masked correctly.
        rows[1]["answer"] = UNTOKENIZABLE
        prepared, counts = _prepare(rows, StubTokenizer())
        self.assertEqual(counts["rows_failing_tokenization"], 1)
        self.assertEqual(counts["qualifying_rows"], 2)
        self.assertEqual(len(prepared), 2)

    def test_a_clean_pass_counts_no_failures(self) -> None:
        prepared, counts = _prepare(_rows(4), StubTokenizer())
        # Counter, so an untouched key is absent rather than zero.
        self.assertEqual(counts.get("rows_failing_tokenization", 0), 0)
        self.assertEqual(counts["qualifying_rows"], 4)
        self.assertEqual(len(prepared), 4)


if __name__ == "__main__":
    unittest.main()
