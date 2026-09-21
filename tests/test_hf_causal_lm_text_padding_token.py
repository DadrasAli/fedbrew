"""A manifest declares a padding token only when the dataset contains padding.

`hf_causal_lm_text._padding_token_id` returned the tokenizer's own
`pad_token_id` whenever it had one, before looking at whether padding was
enabled. With `causal_lm.pad_incomplete_window: false` -- the default, and what
the shipped dev config sets -- every window is full and nothing is padded, yet
the manifest named a padding id anyway.

That id is not decoration. `TorchCausalLMTask` masks targets by **value**: a
position is inactive when its target equals `pad_token_id`. A padding id in the
manifest of a dataset with no padding is a live filter over real tokens, and
the tokenizer that makes it bite is the common one -- Qwen2.5 ships
`pad_token == eos_token`, so the manifest would have declared the EOS id and
every end-of-document target would have left the loss, the accuracy and the
aggregation weight. The task refuses such a manifest at build time now
(`test_pad_token_id_is_not_eos.py`), which turns the silent mask into a dataset
that generation calls a success and no run can load. Both halves are settled
here instead: padding off records nothing, and padding on with an EOS-valued
pad token is refused where the config can still be changed.

These cases need no tokenizer library -- `_padding_token_id` reads two
attributes -- so they run everywhere. The end-to-end manifest path needs the
`llm` extra and lives in `test_hf_causal_lm_text_generator_offline.py`.
"""

from __future__ import annotations

import unittest

import pytest

from fedbrew.data.hf_causal_lm_text import _padding_token_id, build_next_token_examples

pytestmark = pytest.mark.fast

#: Distinct ids, the case a padding filter is safe for.
PAD = 7
EOS = 1
#: Qwen2.5 and the GPT-2 family: one token doing both jobs.
SHARED = 151643


class _Tokenizer:
    """Everything `_padding_token_id` reads off a tokenizer."""

    def __init__(self, pad_token_id: int | None, eos_token_id: int | None) -> None:
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id


class PaddingOffRecordsNoPaddingTokenTests(unittest.TestCase):
    #: (label, tokenizer). Every shape of tokenizer, with padding off.
    TOKENIZERS = (
        ("a distinct padding token", _Tokenizer(PAD, EOS), EOS),
        ("a padding token that is also EOS", _Tokenizer(SHARED, SHARED), SHARED),
        ("no padding token at all", _Tokenizer(None, EOS), EOS),
    )

    def test_nothing_is_declared_when_nothing_is_padded(self) -> None:
        for label, tokenizer, eos in self.TOKENIZERS:
            with self.subTest(tokenizer=label):
                self.assertIsNone(
                    _padding_token_id(
                        tokenizer=tokenizer,
                        eos_token_id=eos,
                        pad_incomplete=False,
                    )
                )


class PaddingOnDeclaresTheIdItPadsWithTests(unittest.TestCase):
    def test_a_distinct_padding_token_is_recorded(self) -> None:
        self.assertEqual(
            _padding_token_id(
                tokenizer=_Tokenizer(PAD, EOS),
                eos_token_id=EOS,
                pad_incomplete=True,
            ),
            PAD,
        )

    def test_a_padding_token_equal_to_eos_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _padding_token_id(
                tokenizer=_Tokenizer(SHARED, SHARED),
                eos_token_id=SHARED,
                pad_incomplete=True,
            )
        message = str(caught.exception)
        self.assertIn(str(SHARED), message)
        # Both ways out, so the reader does not have to find one.
        self.assertIn("pad_incomplete_window", message)
        self.assertIn("distinct", message)

    def test_the_eos_fallback_is_refused_on_the_same_grounds(self) -> None:
        """A tokenizer with no padding token used to get the EOS id here,
        which is the same defect reached by a different route."""

        with self.assertRaises(ValueError) as caught:
            _padding_token_id(
                tokenizer=_Tokenizer(None, EOS),
                eos_token_id=EOS,
                pad_incomplete=True,
            )
        self.assertIn("pad_incomplete_window", str(caught.exception))

    def test_a_tokenizer_with_neither_token_still_says_so(self) -> None:
        with self.assertRaisesRegex(ValueError, "padding or EOS token"):
            _padding_token_id(
                tokenizer=_Tokenizer(None, None),
                eos_token_id=None,
                pad_incomplete=True,
            )


class PaddingStillNeedsAnIdTests(unittest.TestCase):
    """The window builder's own check, so "padding off records nothing" cannot
    become "padding on records nothing"."""

    def test_building_padded_windows_without_an_id_is_refused(self) -> None:
        import torch

        with self.assertRaisesRegex(ValueError, "padding_token_id is required"):
            build_next_token_examples(
                torch.arange(5, dtype=torch.long),
                sequence_length=4,
                stride=4,
                pad_incomplete_window=True,
                padding_token_id=None,
            )


if __name__ == "__main__":
    unittest.main()
