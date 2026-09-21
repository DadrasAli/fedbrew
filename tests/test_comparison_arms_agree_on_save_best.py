"""Arms compared against each other select their checkpoint the same way.

An arm with `save_best` on is read at its best validation round; an arm with it
off can only be read at its final one. A table quoting a "best" figure across
the two gives the first arm a maximum over every validation evaluation and the
second a single draw -- one-sided, in the first arm's favour.

The flag writes a file; it does not move a number. Measured by running
`configs/dev/synthetic.yaml` twice under `deterministic: true`, identical but
for `save_best`: all three per-round checkpoints came back byte-identical, and
of `round_metrics.csv`'s 52 columns over 3 rounds the only ones that differed
were the six wall-clock ones. So this is a reporting protocol that has to
match across arms, not a training setting -- which is why it can be fixed by
turning it on rather than by re-running anything.

What is pinned is agreement within a family, not a particular value: three
families ship it off (`medmcqa`, and every `examples/` problem) and three on
(`mnist`, `femnist`, `oasst1`), and each is internally consistent.
"""

from __future__ import annotations

import collections
import unittest
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO_ROOT / "configs"

#: Not a comparison family. Seven unrelated smoke and fixture configs that are
#: never plotted against one another -- two of them exist to exercise the
#: checkpointing block itself, so they must be free to disagree.
NOT_A_COMPARISON_FAMILY = frozenset({"configs/dev"})


def _run_configs() -> list[tuple[Path, dict[str, Any]]]:
    """Every shipped run config: the ones carrying a `runtime` block.

    configs/llm_assets/ holds asset-preparation configs, a different schema
    with no runtime section, and is excluded by that rather than by name --
    the same walker tests/test_shipped_config_explicitness.py uses.
    """

    configs = []
    for path in sorted(CONFIG_ROOT.rglob("*.yaml")):
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and "runtime" in loaded:
            configs.append((path.relative_to(REPO_ROOT), loaded))
    return configs


def _checkpointing(config: dict[str, Any]) -> dict[str, Any]:
    return (config.get("runtime") or {}).get("checkpointing") or {}


def _families() -> dict[str, dict[str, dict[str, Any]]]:
    """Run configs grouped by directory, one group per comparison family."""

    families: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for path, config in _run_configs():
        families[path.parent.as_posix()][path.name] = config
    return families


class EveryFamilySelectsTheSameWayTest(unittest.TestCase):
    def test_the_arms_of_a_family_agree_on_save_best(self) -> None:
        for family, configs in sorted(_families().items()):
            if len(configs) < 2 or family in NOT_A_COMPARISON_FAMILY:
                continue
            with self.subTest(family=family):
                chosen = {
                    name: bool(_checkpointing(config).get("save_best", True))
                    for name, config in sorted(configs.items())
                }
                self.assertEqual(
                    len(set(chosen.values())),
                    1,
                    f"{family} arms disagree on save_best: {chosen}. An arm with it "
                    "off can only be read at its final round while an arm with it on "
                    "is read at its best validation round.",
                )

    def test_the_walk_actually_reaches_the_multi_arm_families(self) -> None:
        """A filter that matched nothing would pass the test above silently."""

        compared = {
            family: len(configs)
            for family, configs in _families().items()
            if len(configs) >= 2 and family not in NOT_A_COMPARISON_FAMILY
        }
        self.assertGreaterEqual(len(compared), 10)
        self.assertEqual(compared.get("configs/mnist"), 3)
        self.assertEqual(compared.get("configs/femnist"), 10)

    def test_selecting_on_a_metric_means_naming_it(self) -> None:
        """`best_metric` defaults in code; an arm that selects must say on what."""

        for path, config in _run_configs():
            checkpointing = _checkpointing(config)
            if not bool(checkpointing.get("save_best", True)):
                continue
            with self.subTest(config=path.as_posix()):
                self.assertIsInstance(checkpointing.get("best_metric"), str)


class TheMnistArmsTest(unittest.TestCase):
    """The MNIST arms, named, so they cannot regress quietly.

    Kept separate from the family sweep because widening
    NOT_A_COMPARISON_FAMILY would silence that sweep for these two.
    """

    def test_both_mnist_arms_keep_a_selected_checkpoint(self) -> None:
        for name in ("fedavg.yaml", "centralized.yaml"):
            with self.subTest(config=name):
                config = yaml.safe_load((CONFIG_ROOT / "mnist" / name).read_text(encoding="utf-8"))
                self.assertIs(_checkpointing(config).get("save_best"), True)

    def test_they_select_on_the_same_metric(self) -> None:
        """A shared metric is what makes the two selections comparable.

        Not asserted family-wide: `configs/femnist/fedavg_ft.yaml` selects on
        `personal_val_...` because a personalized arm has no other choice, so
        agreement on the metric is an MNIST fact, not a general one.
        """

        metrics = {
            name: _checkpointing(
                yaml.safe_load((CONFIG_ROOT / "mnist" / name).read_text(encoding="utf-8"))
            ).get("best_metric")
            for name in ("fedavg.yaml", "centralized.yaml")
        }
        self.assertEqual(set(metrics.values()), {"val_accuracy_sample_weighted_avg"})


if __name__ == "__main__":
    unittest.main()
