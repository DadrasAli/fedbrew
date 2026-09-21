"""docs/06-models-and-tasks.md must list each builder's real key set.

The model surface is where an undocumented key does the most damage: every
builder resolved parameters with values.get(name, default), so `lora_alph: 32`
was dropped by the dict and the run trained at 16, under a config that said 32.
reject_unknown_model_keys turned that into a load error. This turns the chapter
into a readable form of the same frozensets.

Bidirectional, per builder. A key the builder accepts and the chapter's table
omits is a key nobody can discover; a key the table lists and the builder
rejects sends a reader to write a config that will not load.

The tables are matched by section, not against the whole file, so a key
documented for the wrong builder fails -- which is the mistake most likely to
be made here, since three of the eight builders share most of their keys.
"""

from __future__ import annotations

import importlib
import re
import unittest
from pathlib import Path

import pytest
from docs_sections import fenced_block_after

from fedbrew.core import registry
from fedbrew.models.config_keys import _INJECTED_KEYS, _INJECTED_PREFIX

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER = REPO_ROOT / "docs" / "06-models-and-tasks.md"

#: Registered model name -> (module, the chapter heading its table sits under).
#: cnn and small_cnn share a module, a key set and one section.
BUILDERS = {
    "mlp": ("fedbrew.models.torch_mlp", "### 2.1 `mlp`"),
    "cnn": ("fedbrew.models.torch_cnn", "### 2.2 `cnn` and `small_cnn`"),
    "small_cnn": ("fedbrew.models.torch_cnn", "### 2.2 `cnn` and `small_cnn`"),
    "femnist_resnet18": (
        "fedbrew.models.femnist_resnet",
        "### 2.3 `femnist_resnet18`",
    ),
    "openimage_shufflenet": (
        "fedbrew.models.openimage_shufflenet",
        "### 2.4 `openimage_shufflenet`",
    ),
    "tiny_gpt2": ("fedbrew.models.tiny_gpt2", "### 2.5 `tiny_gpt2`"),
    "hf_causal_lm": ("fedbrew.models.hf_causal_lm", "### 2.6 `hf_causal_lm`"),
    "hf_causal_lm_lora": (
        "fedbrew.models.hf_causal_lm_lora",
        "### 2.7 `hf_causal_lm_lora`",
    ),
}

#: A builder whose section documents only the keys it adds, because it says in
#: prose that it accepts the base builder's set as well. The inherited half is
#: checked separately, by test_the_lora_builder_accepts_the_base_builder_keys,
#: so nothing goes unchecked -- the split just follows how a reader reads it.
INHERITS = {"hf_causal_lm_lora": "fedbrew.models.hf_causal_lm"}


def _chapter_text() -> str:
    return CHAPTER.read_text(encoding="utf-8")


def _section(heading: str) -> str:
    """The chapter text from `heading` to the next heading of any level."""

    text = _chapter_text()
    start = text.index(heading) + len(heading)
    following = re.search(r"^#{2,3} ", text[start:], flags=re.MULTILINE)
    return text[start : start + following.start()] if following else text[start:]


def _documented_keys(heading: str) -> set[str]:
    """The first column of every table row in a builder's section."""

    return set(re.findall(r"^\| `(\w+)` \|", _section(heading), flags=re.MULTILINE))


def _declared_keys(module_path: str) -> frozenset[str]:
    module = importlib.import_module(module_path)
    keys = getattr(module, "_KNOWN_KEYS", None)
    assert keys is not None, f"{module_path} declares no _KNOWN_KEYS"
    return keys


class ChapterShapeTest(unittest.TestCase):
    def test_present_and_agent_facing(self) -> None:
        self.assertTrue(CHAPTER.is_file())
        self.assertIn("\n## For agents\n", _chapter_text())


class BuilderCoverageTest(unittest.TestCase):
    def test_every_registered_builder_has_a_section(self) -> None:
        registry.register_builtin_components()
        self.assertEqual(
            set(registry.models.builtin()),
            set(BUILDERS),
            "a model was registered or removed; the chapter needs a section "
            "for it and this map needs the entry",
        )
        text = _chapter_text()
        for name, (_, heading) in BUILDERS.items():
            with self.subTest(model=name):
                self.assertIn(heading, text)


class KeyTableTest(unittest.TestCase):
    """Each section's table must equal that builder's _KNOWN_KEYS."""

    def test_tables_match_the_declared_key_sets(self) -> None:
        for name, (module_path, heading) in BUILDERS.items():
            declared = set(_declared_keys(module_path))
            if name in INHERITS:
                declared -= set(_declared_keys(INHERITS[name]))
            documented = _documented_keys(heading)
            with self.subTest(model=name):
                # Injected keys are supplied by the factory and need no row;
                # a builder may still declare one it also reads itself.
                self.assertEqual(
                    documented - _INJECTED_KEYS,
                    declared - _INJECTED_KEYS,
                    f"the {name} table and {module_path}._KNOWN_KEYS disagree",
                )

    def test_the_lora_builder_accepts_the_base_builder_keys(self) -> None:
        """The chapter says LoRA takes all seven base keys plus six."""

        base = _declared_keys("fedbrew.models.hf_causal_lm")
        lora = _declared_keys("fedbrew.models.hf_causal_lm_lora")
        self.assertTrue(
            base <= lora,
            f"LoRA no longer accepts every base key: {sorted(base - lora)}",
        )
        self.assertEqual(len(base), 7)
        self.assertEqual(len(lora - base), 6)
        text = _chapter_text()
        self.assertIn("accepts all seven keys above, plus six", text)

    def test_the_stated_key_counts_match(self) -> None:
        """The overview table prints a count per builder."""

        text = _chapter_text()
        for name, (module_path, _) in BUILDERS.items():
            count = len(_declared_keys(module_path))
            with self.subTest(model=name):
                self.assertIsNotNone(
                    re.search(
                        rf"^\| `{re.escape(name)}` \| \w+ \| `[^`]+` \| {count} \|$",
                        text,
                        flags=re.MULTILINE,
                    ),
                    f"{name} declares {count} keys; the overview table says otherwise",
                )


class InjectedKeyTest(unittest.TestCase):
    def test_the_injected_list_matches_the_code(self) -> None:
        # Scoped to the fenced block by index, not by a character count. The
        # previous form took 400 characters from the anchor, which reached past
        # the block and into the sentence after it, so a key moved out of the
        # list but mentioned nearby still passed.
        block = fenced_block_after(_chapter_text(), "never need declaring")
        self.assertTrue(_INJECTED_KEYS, "no injected keys declared; the import is broken")
        for key in _INJECTED_KEYS:
            with self.subTest(key=key):
                self.assertIn(key, block)
        # And the other direction: the block may not invent one.
        self.assertEqual(
            set(block.split()) - set(_INJECTED_KEYS),
            set(),
            "the block lists keys factory._model_config does not inject",
        )
        self.assertIn(_INJECTED_PREFIX, _chapter_text())


class CitedPathsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _chapter_text()

    def test_cited_modules_exist(self) -> None:
        cited = set(re.findall(r"`(fedbrew/[\w/]+\.py)", self.text))
        self.assertGreaterEqual(len(cited), 10)
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
        self.assertEqual(missing, [])

    def test_cited_tests_exist(self) -> None:
        cited = set(re.findall(r"`(tests/test_\w+\.py)`", self.text))
        self.assertTrue(cited)
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
        self.assertEqual(missing, [])

    def test_cited_configs_exist(self) -> None:
        cited = set(re.findall(r"(configs/[\w/]+\.yaml)", self.text))
        self.assertTrue(cited)
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
