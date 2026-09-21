"""``model.pad_token_id`` refuses a value the generated dataset contradicts.

Three model keys are cross-checked against the manifest the dataset was
generated with. Two of them raise on a mismatch --
``hf_causal_lm._expected_sequence_length`` and ``_expected_vocab_size`` -- and
the third overwrote the configured value in silence.

The overwrite happened in ``factory._add_causal_manifest_metadata``, before
the builder ran, so no later check could catch it: by the time
``_validate_padding_token`` looked, the configured value was gone and it was
range-checking the manifest's own token against the manifest's own vocabulary.
A config asking for a padding token the data does not use was indistinguishable
from one that asked for nothing, and the number reaching the attention mask and
the loss mask was the manifest's either way.

That mattered more than the other two rather than less. A wrong
``sequence_length`` or ``vocab_size`` fails loudly downstream; a wrong padding
token is a mask over the wrong positions, which trains and reports a number.
"""

from __future__ import annotations

import unittest

import pytest

from fedbrew.core.factory import _add_causal_manifest_metadata

pytestmark = pytest.mark.fast

METADATA = {"padding_token_id": 50256, "sequence_length": 128, "vocab_size": 50257}


class ConfiguredAndGeneratedAgreeTest(unittest.TestCase):
    def test_an_unset_key_takes_the_generated_value(self) -> None:
        values: dict[str, object] = {}
        _add_causal_manifest_metadata(values, METADATA)
        self.assertEqual(values["pad_token_id"], 50256)

    def test_the_same_value_configured_is_accepted(self) -> None:
        values: dict[str, object] = {"pad_token_id": 50256}
        _add_causal_manifest_metadata(values, METADATA)
        self.assertEqual(values["pad_token_id"], 50256)

    def test_the_manifest_value_is_still_carried_under_its_own_name(self) -> None:
        """``dataset_padding_token_id`` is what the builder range-checks."""

        values: dict[str, object] = {}
        _add_causal_manifest_metadata(values, METADATA)
        self.assertEqual(values["dataset_padding_token_id"], 50256)


class ConfiguredAndGeneratedDisagreeTest(unittest.TestCase):
    def test_a_different_value_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _add_causal_manifest_metadata({"pad_token_id": 0}, METADATA)
        self.assertIn("pad_token_id", str(caught.exception))

    def test_the_message_names_both_numbers(self) -> None:
        """A mismatch message that does not say which is which sends the
        reader to find out where each came from."""

        with self.assertRaises(ValueError) as caught:
            _add_causal_manifest_metadata({"pad_token_id": 0}, METADATA)
        message = str(caught.exception)
        self.assertIn("0", message)
        self.assertIn("50256", message)

    def test_it_reads_like_its_two_siblings(self) -> None:
        """The wording is the sequence_length message with one word changed;
        three cross-checks that fail three different ways is the defect this
        replaced, in a smaller form."""

        with self.assertRaises(ValueError) as caught:
            _add_causal_manifest_metadata({"pad_token_id": 0}, METADATA)
        self.assertIn("does not match the generated dataset", str(caught.exception))


class NothingToCheckTest(unittest.TestCase):
    def test_a_manifest_without_a_padding_token_leaves_the_key_alone(self) -> None:
        values: dict[str, object] = {"pad_token_id": 7}
        _add_causal_manifest_metadata(values, {"sequence_length": 128})
        self.assertEqual(values["pad_token_id"], 7)

    def test_an_explicit_null_is_not_a_mismatch(self) -> None:
        """``null`` is absence, as it is for sequence_length: both use
        ``is not None`` rather than a membership test."""

        values: dict[str, object] = {"pad_token_id": None}
        _add_causal_manifest_metadata(values, METADATA)
        self.assertEqual(values["pad_token_id"], 50256)


if __name__ == "__main__":
    unittest.main()
