"""Settings that change the numbers must be written down, not defaulted into.

Two of them were left implicit across the shipped configs, and both defaults
are invisible unless you read source:

``runtime.performance.matmul_precision`` was set to "high" by 20 configs and
omitted by 10. Absent means torch's own default, "highest" -- full fp32
matmuls, where "high" puts them on TensorFloat32. So the two groups were not
running the same arithmetic, and a new arm started by copying a dev or MNIST
config would silently not be comparable with the FEMNIST baselines.

``runtime.deterministic`` was set by 28 and omitted by 2, where absent means
false. A run that is not deterministic should say so.

This walks the shipped tree rather than a hand-kept list, so a new config
cannot quietly reintroduce either gap. It also pins the three Delta-SGD
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
from fedbrew.core.config import MATMUL_PRECISIONS, NUMERICS_PERFORMANCE_KEYS

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO_ROOT / "configs"


def _run_configs() -> list[tuple[Path, dict[str, Any]]]:
    """Every shipped run config: the ones carrying a `runtime` block.

    configs/llm_assets/ holds asset-preparation configs, a different schema
    with no runtime section, and is excluded by that test rather than by name.
    """

    configs = []
    for path in sorted(CONFIG_ROOT.rglob("*.yaml")):
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
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

    ``name`` is the one field that may be absent OR present. load_config fills
    it from the filename stem, and config.py says restating it invites drift --
    three of the five that set it restated the stem exactly. So the rule is
    that a config sets it only when the recorded name must differ from the
    filename, and each such config says why.
    """

    def setUp(self) -> None:
        self.configs = _run_configs()

    def test_every_config_states_the_run_metadata_fields(self) -> None:
        for path, config in self.configs:
            for field in ("seed", "output_dir", "use_run_subdir", "tags", "notes"):
                with self.subTest(config=str(path), field=field):
                    self.assertIn(
                        field,
                        config["experiment"],
                        "every run config carries the same header; a run "
                        "missing tags or notes is one that cannot be found "
                        "again in runs_index.jsonl",
                    )

    def test_a_config_names_itself_only_to_differ_from_its_filename(self) -> None:
        restating = [
            str(path)
            for path, config in self.configs
            if config["experiment"].get("name") == Path(path).stem
        ]
        self.assertEqual(
            restating,
            [],
            f"these set experiment.name to their own filename stem: {restating}. "
            "load_config already does that; writing it again is a second copy "
            "that can drift from the first.",
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
                    "a config whose recorded name differs from its filename "
                    "must say why, in the comment directly above the line "
                    "that sets it",
                )


class ExplicitNumericSettingsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.configs = _run_configs()
        self.assertGreaterEqual(len(self.configs), 30, "expected 30 run configs")

    def test_every_config_states_every_numerics_changing_key(self) -> None:
        """Derived from NUMERICS_PERFORMANCE_KEYS, not from a list here.

        This test was written for ``matmul_precision`` and named it, and
        ``cudnn_benchmark`` sat in the same declared set at the same 20-of-30
        split the test exists to prevent -- one instance of a plural thing,
        fixed one instance at a time. Reading the set means the third key is
        covered by the commit that declares it, not by someone remembering
        this file.
        """

        self.assertTrue(NUMERICS_PERFORMANCE_KEYS, "the declared set is empty")
        for key in sorted(NUMERICS_PERFORMANCE_KEYS):
            for path, config in self.configs:
                with self.subTest(key=key, config=str(path)):
                    performance = config["runtime"].get("performance") or {}
                    self.assertIn(
                        key,
                        performance,
                        f"runtime.performance.{key} changes the numbers, so a "
                        "config that omits it is not comparable with one that "
                        "sets it and the difference is invisible in the file. "
                        "See docs/10-reproducibility.md",
                    )

    def test_the_matmul_precision_written_is_one_torch_accepts(self) -> None:
        """Value validity, which is per-key rather than per-set.

        torch warns and keeps the current setting for an unknown value rather
        than raising, so a typo runs at the previous precision while run.json
        records the typo as though it applied.
        """

        for path, config in self.configs:
            with self.subTest(config=str(path)):
                performance = config["runtime"].get("performance") or {}
                self.assertIn(performance.get("matmul_precision"), MATMUL_PRECISIONS)

    def test_every_config_states_whether_it_is_deterministic(self) -> None:
        for path, config in self.configs:
            with self.subTest(config=str(path)):
                self.assertIn(
                    "deterministic",
                    config["runtime"],
                    "absent means false; say so rather than omitting it",
                )
                self.assertIsInstance(config["runtime"]["deterministic"], bool)

    def test_a_deterministic_config_states_how_strict_it_is(self) -> None:
        """warn_only is only meaningful under deterministic: true, where it is
        the difference between a guarantee and a warning nobody reads."""

        for path, config in self.configs:
            runtime = config["runtime"]
            if not runtime.get("deterministic"):
                continue
            with self.subTest(config=str(path)):
                self.assertIn("deterministic_warn_only", runtime)


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
