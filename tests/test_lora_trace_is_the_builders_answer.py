"""`run.json`'s LoRA block is what the builder built, not a second derivation.

`build_hf_causal_lm_trace` wrote `lora_config` by re-reading the `model` block
with its own copies of the builder's defaults -- `8`, `16`, `0.05`, `"none"`,
and `"default"` for the adapter name. A default changed in one file and not the
other would have left `run.json` describing an adapter that never ran, and
nothing compared the two. It also recorded `target_modules` verbatim, where the
builder strips and de-duplicates, so a config with a stray space wrote one list
into the record and trained on another. P10-F17.

Both now come from `lora_config_from_model_values`. The tie is checked twice:
here, against the trace and the helper, and -- where PEFT is installed, which
is CI -- against `_fl_lora_config` on a model that was actually built.
"""

from __future__ import annotations

import inspect
import unittest

import pytest

from fedbrew.models.hf_causal_lm_lora import (
    HFCausalLMLoRAConfigError,
    build_hf_causal_lm_lora,
    lora_adapter_name,
    lora_config_from_model_values,
)

pytestmark = pytest.mark.fast

try:  # pragma: no cover - import cost depends on the installed extras.
    import peft  # noqa: F401
    import transformers  # noqa: F401

    _HAS_LLM_DEPS = True
except Exception:  # pragma: no cover - the core-only environment.
    _HAS_LLM_DEPS = False

MINIMAL = {"target_modules": ["c_attn"]}

#: The defaults, written once here so that changing one in the builder fails
#: this test rather than silently changing what run.json claims.
DEFAULTS = {
    "task_type": "CAUSAL_LM",
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "bias": "none",
}


class TheHelperIsTheOneDefinitionTest(unittest.TestCase):
    def test_the_defaults_are_what_they_have_always_been(self) -> None:
        resolved = lora_config_from_model_values(MINIMAL)
        for key, value in DEFAULTS.items():
            with self.subTest(key=key):
                self.assertEqual(resolved[key], value)
        self.assertEqual(resolved["target_modules"], ["c_attn"])
        self.assertEqual(lora_adapter_name({}), "default")

    def test_configured_values_win_over_every_default(self) -> None:
        resolved = lora_config_from_model_values(
            {"target_modules": ["c_attn"], "r": 2, "lora_alpha": 4, "lora_dropout": 0.0}
        )
        self.assertEqual(resolved["r"], 2)
        self.assertEqual(resolved["lora_alpha"], 4)
        self.assertEqual(resolved["lora_dropout"], 0.0)
        self.assertEqual(lora_adapter_name({"adapter_name": "federated"}), "federated")

    def test_target_modules_are_normalised_not_echoed(self) -> None:
        """The old trace echoed the raw list; the builder strips it."""

        resolved = lora_config_from_model_values({"target_modules": [" c_attn ", "c_proj"]})
        self.assertEqual(resolved["target_modules"], ["c_attn", "c_proj"])

    def test_the_result_is_json_writable(self) -> None:
        """It goes into run.json, so task_type is the string, not the enum."""

        import json

        json.dumps(lora_config_from_model_values(MINIMAL))

    def test_it_refuses_what_the_builder_refuses(self) -> None:
        for label, values in {
            "no target modules": {},
            "empty target modules": {"target_modules": []},
            "duplicate target modules": {"target_modules": ["c_attn", "c_attn"]},
            "a trainable bias": {**MINIMAL, "bias": "all"},
            "r zero": {**MINIMAL, "r": 0},
            "dropout above one": {**MINIMAL, "lora_dropout": 1.5},
        }.items():
            with self.subTest(case=label):
                with self.assertRaises(HFCausalLMLoRAConfigError):
                    lora_config_from_model_values(values)


class TheTraceAsksRatherThanDerivesTest(unittest.TestCase):
    def test_the_trace_carries_no_default_of_its_own(self) -> None:
        """The duplication itself: none of these literals may reappear."""

        from fedbrew.core import run_metadata

        source = inspect.getsource(run_metadata.build_hf_causal_lm_trace)
        branch = source[source.index('if config.model.name == "hf_causal_lm_lora"') :]
        code = "\n".join(line for line in branch.splitlines() if not line.strip().startswith("#"))
        for literal in ("8", "16", "0.05", '"none"', '"default"'):
            with self.subTest(literal=literal):
                self.assertNotIn(literal, code)
        self.assertIn("lora_config_from_model_values", code)
        self.assertIn("lora_adapter_name", code)

    def test_the_builder_hands_peft_what_the_helper_returned(self) -> None:
        source = inspect.getsource(build_hf_causal_lm_lora)
        self.assertIn("normalized_config = lora_config_from_model_values(values)", source)
        for derived in ('_positive_int(values.get("r"', '_dropout(values.get("lora_dropout"'):
            with self.subTest(derivation=derived):
                self.assertNotIn(derived, source)


@unittest.skipUnless(_HAS_LLM_DEPS, "PEFT and Transformers are optional dependencies")
class TheBuiltModelCarriesTheSameConfigTest(unittest.TestCase):
    """The definitive tie, where the dependencies exist to build a model."""

    def test_fl_lora_config_equals_what_the_trace_would_record(self) -> None:
        import tempfile
        from pathlib import Path

        from tests.test_lora_adapter_federation import _lora_config, _write_model_fixture

        with tempfile.TemporaryDirectory() as directory:
            config = _lora_config(_write_model_fixture(Path(directory)))
            model = build_hf_causal_lm_lora(config)
            self.assertEqual(model._fl_lora_config, lora_config_from_model_values(config))
            self.assertEqual(model._fl_adapter_name, lora_adapter_name(config))


if __name__ == "__main__":
    unittest.main()
