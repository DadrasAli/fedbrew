"""`client_splits` is a declared section, not one every generator is assumed to read.

It was the third member of `_SHARED_SECTIONS`, which exempted it from the rule
the rest of the section allow-list enforces. Both SFT generators took it,
`_parse_client_splits` validated it, and the first statement of each generator
deleted it -- their splits are cut in `tree_splits`, at the conversation tree by
SHA. All three shipped OASST1 configs set `client_splits.eval_ratio: 0.1` under
a comment saying it was kept "for the common generator schema": a ratio that
could be edited to any value with no effect on the data and nothing printed.

The section is declared per generator now. A `tensors` generator gets it
implicitly, because the shared writer in `generate.py` cuts the slices rather
than the generator, and a `shards` generator declares it only if it cuts by it.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import pytest
import yaml

from fedbrew.core.extensions import load_extensions
from fedbrew.core.registry import generators, register_builtin_components
from fedbrew.data.generate import (
    _SHARED_SECTIONS,
    _extensions_of,
    _sections_read_by,
    _validate_generator_keys,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "data" / "configs"

#: The two whose splits come from `tree_splits`, and which therefore must not
#: accept `client_splits`.
TREE_SPLIT_GENERATORS = ("generic_sft", "oasst1_sft")


@pytest.mark.fast
class TheSharedSetTests(unittest.TestCase):
    def test_only_dataset_and_partition_are_shared(self) -> None:
        self.assertEqual(_SHARED_SECTIONS, frozenset({"dataset", "partition"}))


@pytest.mark.fast
class WhichGeneratorsReadItTests(unittest.TestCase):
    def setUp(self) -> None:
        register_builtin_components()

    def test_every_tensors_generator_gets_it_without_declaring_it(self) -> None:
        """Its splits are cut by the shared writer, not by the generator, so
        declaring it on each would be a hand-kept copy of "every tensors
        generator" -- and the first one added without the copy would have a
        working section refused."""

        tensors = [name for name in generators.list() if generators.get(name).kind == "tensors"]
        self.assertTrue(tensors, "no tensors generator found; the registry changed")
        for name in tensors:
            with self.subTest(generator=name):
                spec = generators.get(name)
                self.assertNotIn("client_splits", spec.sections)
                self.assertIn("client_splits", _sections_read_by(spec))

    def test_the_shards_generators_that_cut_by_it_declare_it(self) -> None:
        for name in ("femnist", "tiny_causal_lm", "hf_causal_lm_text"):
            with self.subTest(generator=name):
                self.assertIn("client_splits", generators.get(name).sections)

    def test_the_tree_split_generators_do_not(self) -> None:
        for name in TREE_SPLIT_GENERATORS:
            with self.subTest(generator=name):
                spec = generators.get(name)
                self.assertNotIn("client_splits", _sections_read_by(spec))
                self.assertIn("tree_splits", spec.sections)


@pytest.mark.fast
class ASetSectionIsRefusedTests(unittest.TestCase):
    def setUp(self) -> None:
        register_builtin_components()

    def test_a_config_that_sets_it_for_a_tree_split_generator_is_refused(self) -> None:
        for name in TREE_SPLIT_GENERATORS:
            with self.subTest(generator=name):
                config = {
                    "dataset": {"name": name, "output_dir": "unused"},
                    "partition": {"num_clients": 4},
                    "tree_splits": {"train_ratio": 0.8},
                    "client_splits": {"train_ratio": 0.9, "eval_ratio": 0.1},
                }
                with self.assertRaises(ValueError) as caught:
                    _validate_generator_keys(config, name)
                message = str(caught.exception)
                self.assertIn("client_splits", message)
                # And what is actually in force, which is the whole point of
                # refusing rather than dropping it.
                self.assertIn("tree_splits", message)

    def test_the_same_config_without_the_section_passes_validation(self) -> None:
        """So the guard cannot pass by refusing every config."""

        for name in TREE_SPLIT_GENERATORS:
            with self.subTest(generator=name):
                _validate_generator_keys(
                    {
                        "dataset": {"name": name, "output_dir": "unused"},
                        "partition": {"num_clients": 4},
                        "tree_splits": {"train_ratio": 0.8},
                    },
                    name,
                )

    def test_a_generator_that_cuts_by_it_still_accepts_it(self) -> None:
        _validate_generator_keys(
            {
                "dataset": {"name": "mnist", "output_dir": "unused"},
                "partition": {"strategy": "iid", "num_clients": 4},
                "client_splits": {"train_ratio": 0.8, "eval_ratio": 0.2},
            },
            "mnist",
        )


class NoShippedConfigSetsItWhereItIsInertTests(unittest.TestCase):
    def test_every_shipped_generator_config_validates(self) -> None:
        """The three OASST1 configs carried the inert section; each now says in
        a comment where its splits actually come from.

        Each config is checked the way `generate_from_config` checks it, with
        its own `dataset.extensions` loaded first, and a name no generator
        answers to fails. It used to be skipped: the ten example configs get
        their generators from their extensions, which this test never loaded,
        so alone it checked 14 of the 24 configs and passed, and it checked all
        24 only when an earlier test in the same process had loaded them.
        FINDINGS.csv POST-F17."""

        register_builtin_components()
        configs = [
            path
            for path in sorted(CONFIG_DIR.rglob("*.yaml"))
            if "assets" not in path.relative_to(CONFIG_DIR).parts
        ]
        self.assertTrue(configs, "no generator configs found")
        for path in configs:
            with self.subTest(config=path.relative_to(REPO_ROOT)):
                document = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertIsInstance(document, dict)
                self.assertIn("dataset", document)
                name = str(document["dataset"]["name"])
                load_extensions(_extensions_of(document["dataset"]))
                _validate_generator_keys(document, name)
                if name in TREE_SPLIT_GENERATORS:
                    self.assertNotIn("client_splits", document)


if __name__ == "__main__":
    unittest.main()
