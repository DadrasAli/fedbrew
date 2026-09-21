"""Every arm whose paper averages uniformly says so when a run does not.

The notice existed for two arms and was missing from the two that needed it
most. `fedlalr` and `delta_sgd` each reported, at `info`, that the run weighted
clients by example count where the published algorithm weights them equally.
SCAFFOLD and the FedOpt family carry the same deviation and said nothing.

SCAFFOLD is the sharp case, and the reason this is a finding rather than a
tidying. Its server does not apply one weighting: `scaffold.py` folds the model
through `WeightedStateAccumulator` at `_result_weight` -- example counts by
default -- and then scales the summed control delta by `1 / num_clients`
unconditionally. So `x` moves along the example-weighted mean of the local
models while `c` estimates their uniform mean, and `c = (1/N) sum(c_i)` -- the
invariant the `-c_i + c` correction rests on -- is stated in the uniform one.
Under FEMNIST's writer-size spread those are different directions, so the
drift correction is not the one Karimireddy et al. analyse. The behaviour
stays -- this arm is read against fedavg, fedprox and the three fedopt arms,
all example-weighted -- but what was missing was anything that says so.

The FEMNIST sweep is not uniform in either direction, which is worth stating
because the finding assumed otherwise: `fedlalr.yaml` and `delta_sgd.yaml`
ship `aggregation_weighting: uniform`, so those two arms match their papers
and their notices never fire as shipped. The two that deviate are exactly the
two that had no notice.

The guard that would have caught it is not another hand-written case. Two
notices could not notice a third arm lacked one, so the shape here is a table
every registered strategy must appear in: `AGGREGATION_WEIGHTING_NOTICE` maps a
strategy either to the code it emits or to `None` with a reason. A strategy
registered tomorrow and left out of it fails `test_every_registered_strategy_is_classified`,
which is the check the two hand-written notices could not provide.
"""

from __future__ import annotations

import copy
import glob
import unittest
from typing import Any

import pytest

from fedbrew.core.config import load_config, validate_config
from fedbrew.core.registry import register_builtin_components, server_strategies
from fedbrew.core.validation import AGGREGATION_WEIGHTING_NOTICE, validate_full_config

#: The delta_sgd notice, which is keyed by client rule rather than by server
#: strategy and so cannot live in the table above. Checked here so the four
#: arms that carry a notice are checked in one place.
DELTA_SGD_NOTICE = "algorithm.delta_sgd_aggregation_weighting"

#: One shipped config per arm that carries a notice, and whether that config
#: ships `uniform`. Loading the real config rather than building one keeps the
#: test measuring what ships; the flag is why the notice test deletes the key
#: rather than asserting on the file as it stands, since two of the six ship
#: the setting that silences their own notice.
SHIPS_UNIFORM = frozenset({"configs/femnist/fedlalr.yaml", "configs/femnist/delta_sgd.yaml"})

ARM_CONFIGS = {
    "configs/femnist/scaffold.yaml": "algorithm.scaffold_aggregation_weighting",
    "configs/femnist/fedadam.yaml": "algorithm.fedopt_aggregation_weighting",
    "configs/femnist/fedadagrad.yaml": "algorithm.fedopt_aggregation_weighting",
    "configs/femnist/fedyogi.yaml": "algorithm.fedopt_aggregation_weighting",
    "configs/femnist/fedlalr.yaml": "algorithm.fedlalr_aggregation_weighting",
    "configs/femnist/delta_sgd.yaml": DELTA_SGD_NOTICE,
}


def _issues(config: Any) -> dict[str, str]:
    return {issue.code: issue.severity for issue in validate_full_config(config).issues}


@pytest.mark.fast
class ClassificationTest(unittest.TestCase):
    def test_every_registered_strategy_is_classified(self) -> None:
        """The check the two hand-written notices could not provide."""

        register_builtin_components()
        unclassified = sorted(
            name for name in server_strategies.list() if name not in AGGREGATION_WEIGHTING_NOTICE
        )
        self.assertEqual(
            unclassified,
            [],
            "server strategies with no aggregation-weighting classification: "
            f"{unclassified}. Add each to AGGREGATION_WEIGHTING_NOTICE, mapped "
            "either to the code it emits or to None with the reason beside it.",
        )

    def test_the_table_classifies_no_strategy_that_is_not_registered(self) -> None:
        register_builtin_components()
        stale = sorted(
            name for name in AGGREGATION_WEIGHTING_NOTICE if not server_strategies.exists(name)
        )
        self.assertEqual(stale, [], f"classified but not registered: {stale}")


class NoticeTest(unittest.TestCase):
    """Each arm's own shipped config, which is what a reader would run."""

    def test_each_arm_reports_example_weighting_as_a_deviation(self) -> None:
        for path, code in ARM_CONFIGS.items():
            with self.subTest(config=path):
                config = copy.deepcopy(load_config(path))
                config.server.extra.pop("aggregation_weighting", None)
                self.assertEqual(_issues(config).get(code), "info", f"{path} is missing {code}")

    @pytest.mark.fast
    def test_the_two_arms_that_ship_uniform_still_do(self) -> None:
        """`SHIPS_UNIFORM` is a claim about the tree, so it is checked against it.

        If one of these ever drops the setting, the arm starts deviating from
        its own paper and `test_the_deviation_is_never_an_error` below is what
        notices -- but only if this list has not silently gone stale first.
        """

        for path in ARM_CONFIGS:
            with self.subTest(config=path):
                ships_uniform = (
                    copy.deepcopy(load_config(path)).server.extra.get("aggregation_weighting")
                    == "uniform"
                )
                self.assertEqual(ships_uniform, path in SHIPS_UNIFORM)

    def test_a_uniform_run_is_not_told_it_deviates(self) -> None:
        for path, code in ARM_CONFIGS.items():
            with self.subTest(config=path):
                config = copy.deepcopy(load_config(path))
                config.server.extra["aggregation_weighting"] = "uniform"
                validate_config(config)
                self.assertNotIn(code, _issues(config))

    def test_the_deviation_is_never_an_error(self) -> None:
        """Example weighting is a comparability choice, not a misconfiguration.

        Raising it to a warning or an error would refuse the FEMNIST arms as
        they ship, which is the wrong end of the trade the configs make.
        """

        for path, code in ARM_CONFIGS.items():
            if path in SHIPS_UNIFORM:
                continue
            with self.subTest(config=path):
                config = copy.deepcopy(load_config(path))
                validate_config(config)
                self.assertEqual(_issues(config).get(code), "info")


class ShippedConfigTest(unittest.TestCase):
    def test_no_shipped_config_carrying_a_notice_arm_is_silent(self) -> None:
        """The finding's own shape: an arm that deviates and says nothing.

        Scoped to every config in the tree rather than to the six above, so a
        config added for a notice-carrying strategy is covered without an edit
        here.
        """

        for path in sorted(glob.glob("configs/**/*.yaml", recursive=True)):
            try:
                config = copy.deepcopy(load_config(path))
            except Exception:  # noqa: BLE001 - config-loading coverage is elsewhere.
                continue
            code = AGGREGATION_WEIGHTING_NOTICE.get(config.server.strategy)
            if code is None or config.server.extra.get("aggregation_weighting") == "uniform":
                continue
            with self.subTest(config=path, strategy=config.server.strategy):
                self.assertEqual(_issues(config).get(code), "info")


@pytest.mark.fast
class ScaffoldConfigTest(unittest.TestCase):
    def test_the_femnist_arm_records_why_it_stays_example_weighted(self) -> None:
        """The notice tells a reader running it; the config tells a reader reading it.

        Both halves are the fix. Without the config note, the first person to
        see the preflight line has no way to tell a deliberate choice from an
        oversight, and the obvious response -- set uniform -- silently makes
        this arm incomparable with the nine it is tabled against.
        """

        text = open("configs/femnist/scaffold.yaml", encoding="utf-8").read()
        self.assertIn("aggregation_weighting", text)
        self.assertIn("1/num_clients", text)
        self.assertNotIn("\n  aggregation_weighting:", text)


if __name__ == "__main__":
    unittest.main()
