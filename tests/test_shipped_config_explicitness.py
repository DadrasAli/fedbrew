"""Settings that change the numbers must be written down, not defaulted into.

They are the ``numerics`` block, which every shipped config states in full,
resolved. Two of them were once left implicit across the shipped configs, and
both defaults are invisible unless you read source:

``numerics.matmul_precision`` was set to "high" by 20 configs and
omitted by 10. Absent means torch's own default, "highest" -- full fp32
matmuls, where "high" puts them on TensorFloat32. So the two groups were not
running the same arithmetic, and a new arm started by copying a dev or MNIST
config would silently not be comparable with the FEMNIST baselines.

``numerics.deterministic`` was set by 28 and omitted by 2, where absent means
false. A run that is not deterministic should say so.

This walks the shipped tree rather than a hand-kept list, so a new config
cannot quietly reintroduce either gap. It reads each config resolved -- its
``extends`` chain merged (``load_config_mapping``), which is what the run
gets -- so a setting stated once in a family base counts for every arm that
extends it, and a family base is not itself a config it checks. It also pins
the three Delta-SGD
constants that were written into their config in the same pass, against the
drift that omitting them was meant to avoid.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import pytest
import yaml

from fedbrew.clients.torch_delta_sgd_client import (
    DEFAULT_DELTA,
    DEFAULT_GAMMA,
    DEFAULT_THETA_0,
)
from fedbrew.core.config import (
    MATMUL_PRECISIONS,
    NUMERICS_KEYS,
    implied_strategy,
    is_family_base,
    load_config_mapping,
)
from fedbrew.core.inferred import inferred_name, inferred_output_dir

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO_ROOT / "configs"


def _run_configs() -> list[tuple[Path, dict[str, Any]]]:
    """Every shipped run config, resolved: the ones carrying a `runtime` block.

    configs/llm_assets/ holds asset-preparation configs, a different schema
    with no runtime section, and is excluded by that test rather than by name.
    """

    configs = []
    for path in sorted(CONFIG_ROOT.rglob("*.yaml")):
        if is_family_base(path):
            continue
        loaded = load_config_mapping(path)
        if isinstance(loaded, dict) and "runtime" in loaded:
            configs.append((path.relative_to(REPO_ROOT), loaded))
    return configs


class ExperimentHeaderConventionTest(unittest.TestCase):
    """One header shape across every run config.

    Three shapes shipped: nineteen configs set use_run_subdir/tags/notes and
    left the name to the filename, six dev configs set neither, and five set
    the name and none of the rest. The split was by directory rather than by
    anything about the runs, so a reader comparing two arms met two different
    headers and could not tell which difference was meaningful.

    ``name`` is the one field that may be absent OR present. load_config
    infers it from the config's path -- ``<directory>-<file stem>`` under
    ``configs/``, fedbrew/core/inferred.py -- and restating an inferred value
    invites drift: the 69 example arms each wrote exactly that until it was
    inferred. So the rule is that a config sets it only when the recorded name
    must differ from the inferred one, and each such config says why.
    """

    def setUp(self) -> None:
        self.configs = _run_configs()

    def test_every_config_states_the_run_metadata_fields(self) -> None:
        """``output_dir`` may be the one the config's path gives instead.

        79 arms wrote exactly ``outputs/<their path under configs/>``, which
        the loader infers (fedbrew/core/inferred.py); a config stating it
        states a directory of its own.
        """

        for path, config in self.configs:
            for field in ("seed", "output_dir", "use_run_subdir", "tags", "notes"):
                with self.subTest(config=str(path), field=field):
                    stated = field in config["experiment"]
                    if field == "output_dir" and not stated:
                        stated = inferred_output_dir(REPO_ROOT / path) is not None
                    self.assertTrue(
                        stated,
                        "every run config carries the same header; a run "
                        "missing tags or notes is one that cannot be found "
                        "again in runs_index.jsonl",
                    )

    def test_a_config_names_itself_only_to_differ_from_the_inferred_name(self) -> None:
        restating = [
            str(path)
            for path, config in self.configs
            if config["experiment"].get("name") == inferred_name(REPO_ROOT / path)
        ]
        self.assertEqual(
            restating,
            [],
            f"these set experiment.name to the name their path gives: {restating}. "
            "load_config already infers it; writing it again is a second copy "
            "that can drift from the first.",
        )

    def test_a_config_states_its_strategy_only_to_differ_from_the_implied_one(self) -> None:
        """``server.strategy`` is inferred from the rule (fedbrew/core/inferred.py).

        60 configs wrote exactly the strategy their rule implies; the ones that
        state it now name a FedOpt server over a FedAvg-family rule.
        """

        restating = [
            str(path)
            for path, config in self.configs
            if "strategy" in (config.get("server") or {})
            and config["server"]["strategy"] == implied_strategy(config["client"]["update_rule"])
        ]
        self.assertEqual(
            restating,
            [],
            f"these set server.strategy to the one their update_rule implies: {restating}",
        )

    def test_a_config_that_names_itself_says_why(self) -> None:
        for path, config in self.configs:
            if "name" not in config["experiment"]:
                continue
            with self.subTest(config=str(path)):
                lines = (REPO_ROOT / path).read_text(encoding="utf-8").splitlines()
                index = next(
                    number for number, line in enumerate(lines) if line.startswith("  name:")
                )
                # The contiguous comment block immediately above the line, not
                # a `#` anywhere earlier in the file: every config has comments
                # somewhere, so the looser check passed on a config whose
                # explanation had been deleted.
                preceding = []
                cursor = index - 1
                while cursor >= 0 and lines[cursor].strip().startswith("#"):
                    preceding.append(lines[cursor].strip())
                    cursor -= 1
                self.assertNotEqual(
                    preceding,
                    [],
                    "a config whose recorded name differs from the inferred one "
                    "must say why, in the comment directly above the line "
                    "that sets it",
                )


class ExplicitNumericSettingsTest(unittest.TestCase):
    """Every resolved config states the numerics block, and states it in full.

    The block is ``NumericsConfig``: every key that changes the numbers a run
    produces, and nothing else. One assertion covers all of them, and a key
    added to the block is covered by the commit that adds it. The keys used
    to be three in ``runtime.performance`` -- checked from a declared set --
    and two in ``runtime`` checked one at a time, while ``use_amp`` was
    required by the loader instead.
    """

    def setUp(self) -> None:
        self.configs = _run_configs()
        self.assertGreaterEqual(len(self.configs), 30, "expected 30 run configs")

    def test_every_config_states_the_whole_numerics_block(self) -> None:
        self.assertTrue(NUMERICS_KEYS, "the block has no keys")
        for path, config in self.configs:
            with self.subTest(config=str(path)):
                numerics = config.get("numerics")
                self.assertIsInstance(
                    numerics, dict, "a run config states the numerics block, resolved"
                )
                self.assertEqual(
                    sorted(NUMERICS_KEYS - set(numerics)),
                    [],
                    "every numerics key changes the numbers, so a config that "
                    "omits one is not comparable with one that sets it and the "
                    "difference is invisible in the file. See "
                    "docs/10-reproducibility.md",
                )

    def test_no_numerics_key_is_left_in_runtime(self) -> None:
        for path, config in self.configs:
            runtime = config.get("runtime") or {}
            stray = (set(runtime) | set(runtime.get("performance") or {})) & NUMERICS_KEYS
            with self.subTest(config=str(path)):
                self.assertEqual(stray, set())

    def test_the_matmul_precision_written_is_one_torch_accepts(self) -> None:
        """Value validity, which is per-key rather than per-set.

        torch warns and keeps the current setting for an unknown value rather
        than raising, so a typo runs at the previous precision while run.json
        records the typo as though it applied.
        """

        for path, config in self.configs:
            with self.subTest(config=str(path)):
                self.assertIn(config["numerics"]["matmul_precision"], MATMUL_PRECISIONS)

    def test_the_three_switches_are_bools(self) -> None:
        for path, config in self.configs:
            for key in ("deterministic", "deterministic_warn_only", "use_amp"):
                with self.subTest(config=str(path), key=key):
                    self.assertIsInstance(config["numerics"][key], bool)


class DeltaSgdConstantsTest(unittest.TestCase):
    """The config now states the paper's defaults; these must stay the same.

    They were deliberately omitted before, so that the config could not drift
    from clients/torch_delta_sgd_client.py. Writing them made the values
    visible and made drift possible, so this closes it.
    """

    CONFIG = CONFIG_ROOT / "femnist" / "delta_sgd.yaml"

    def test_the_written_values_match_the_client_defaults(self) -> None:
        client = yaml.safe_load(self.CONFIG.read_text(encoding="utf-8"))["client"]

        for key, default in (
            ("theta_0", DEFAULT_THETA_0),
            ("gamma", DEFAULT_GAMMA),
            ("delta", DEFAULT_DELTA),
        ):
            with self.subTest(key=key):
                self.assertEqual(client[key], default)


if __name__ == "__main__":
    unittest.main()
