"""`tiny_gpt2` builds against a manifest that declares no padding token.

`model.pad_token_id` is the one key in this builder's table that the factory
writes rather than the author: `_add_causal_manifest_metadata` copies the
manifest's `padding_token_id` over whatever the config said, and **both SFT
generators write that key as `null`** (`generic_sft.py`, `oasst1_sft.py`). So
the value reaching the builder for an SFT dataset is an explicit `None`.

`int(values.get("pad_token_id", 0))` turned that into a bare `TypeError` raised
from inside a model factory, with no message naming the key, the config or the
manifest. The pairing is legal -- `tiny_gpt2` is a `causal_lm` model and the SFT
manifests are `causal_lm` data -- and no shipped config happens to make it, so
nothing failed and nothing said the combination was unsupported either.

`hf_causal_lm` has accepted `None` for the same key all along
(`_validate_padding_token` returns early on it), and `GPT2Config` takes `None`
as "no padding token". This is the two builders agreeing.

An absent key still takes the documented default of 0; only an explicit `None`
passes through. Chapter 06 §2.5 and §3.3.
"""

from __future__ import annotations

import importlib.util
import unittest

import pytest

from fedbrew.core.factory import _add_causal_manifest_metadata

pytestmark = pytest.mark.fast

_HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None

#: What both SFT generators write into the manifest.
SFT_METADATA = {
    "task": "causal_lm_sft",
    "padding_token_id": None,
    "ignore_index": -100,
    "sequence_length": 8,
}


class FactoryHandsTheBuilderAnExplicitNoneTests(unittest.TestCase):
    """No optional dependency needed: this half is the config plumbing."""

    def test_an_sft_manifest_sets_pad_token_id_to_none(self) -> None:
        values: dict[str, object] = {"name": "tiny_gpt2", "vocab_size": 258}

        _add_causal_manifest_metadata(values, SFT_METADATA)

        self.assertIn("pad_token_id", values)
        self.assertIsNone(values["pad_token_id"])


@unittest.skipUnless(_HAS_TRANSFORMERS, "transformers is an optional LLM dependency")
class TinyGPT2NullPaddingTokenTests(unittest.TestCase):
    def test_an_explicit_none_builds_a_model_with_no_padding_token(self) -> None:
        from fedbrew.models.tiny_gpt2 import build_tiny_gpt2

        values: dict[str, object] = {
            "name": "tiny_gpt2",
            "vocab_size": 258,
            "sequence_length": 8,
            "n_embd": 16,
            "n_layer": 1,
            "n_head": 2,
        }
        _add_causal_manifest_metadata(values, SFT_METADATA)

        model = build_tiny_gpt2(values)

        self.assertIsNone(model.config.pad_token_id)

    def test_an_absent_key_still_takes_the_documented_default(self) -> None:
        from fedbrew.models.tiny_gpt2 import build_tiny_gpt2

        model = build_tiny_gpt2(
            {"vocab_size": 258, "sequence_length": 8, "n_embd": 16, "n_layer": 1, "n_head": 2}
        )

        self.assertEqual(model.config.pad_token_id, 0)

    def test_a_configured_id_is_still_carried(self) -> None:
        from fedbrew.models.tiny_gpt2 import build_tiny_gpt2

        model = build_tiny_gpt2(
            {
                "vocab_size": 258,
                "sequence_length": 8,
                "n_embd": 16,
                "n_layer": 1,
                "n_head": 2,
                "pad_token_id": 7,
            }
        )

        self.assertEqual(model.config.pad_token_id, 7)

    def test_a_non_integer_padding_token_is_refused_by_name(self) -> None:
        from fedbrew.models.tiny_gpt2 import build_tiny_gpt2

        for bad in ("0", 1.5, True):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(ValueError, "pad_token_id"):
                    build_tiny_gpt2({"vocab_size": 258, "pad_token_id": bad})


if __name__ == "__main__":
    unittest.main()
