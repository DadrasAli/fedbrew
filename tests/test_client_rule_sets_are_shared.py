"""The client-rule sets have one definition, and it is `config.py`'s.

`UPDATE_MODE_CLIENT_RULES` and `FIXED_LR_SGD_CLIENT_RULES` were defined twice,
in two modules that import from each other, with different members. The
factory's copy included `fedavg_ft`; `config.py`'s did not.

Every consumer therefore answered a different question depending on which
module it imported from. `factory._training_client_kwargs` used its own copy
and built `fedavg_ft` with `momentum`, `weight_decay`, `nesterov`,
`learning_rate_schedule`, `min_learning_rate`, `update_mode` and
`frozen_gradient_weighting`; `config._validate_local_sgd_options` used the
other and returned before checking any of them.

Measured on `configs/femnist/fedavg_ft.yaml`, which sets all seven: deleting
`client.momentum` was **accepted** by `validate_config` and caught only by
`TorchSGDClient.__init__` raising several seconds into the run. Same for
`learning_rate_schedule` and `update_mode`. `TheGapThatWasClosedTest` runs
those three, so the fix is a measurement rather than a claim.

That redundancy was the whole safety net, and it is the shape the audit warned
about: the moment a check exists only in the validator, `fedavg_ft` bypasses it
silently. Two constants with one name and two bodies is also the kind of thing
a reasonable edit resolves in the wrong direction -- which is why
`OneDefinitionTest` scans the tree rather than trusting the import.

Widening `config.py`'s sets turns three validators on for `fedavg_ft` that were
previously skipped. The one shipped config in that shape still validates;
`TheShippedConfigStillValidatesTest` keeps checking, because "no config
breaks" is what made this fix free to make.

The fixed-rate set was also derived from the update-mode set, and the
update-mode check ran from inside the fixed-rate one, so taking an
`update_mode` meant taking `momentum`, `weight_decay`, `nesterov` and a
schedule as well. `TakingAModeIsNotTakingAFixedRateTest` takes a rule out of
the fixed-rate set and requires its mode to be checked and forwarded anyway.
"""

from __future__ import annotations

import ast
import copy
import glob
import unittest
from pathlib import Path

import pytest

from fedbrew.core import config as config_module
from fedbrew.core import factory as factory_module
from fedbrew.core.config import load_config, validate_config
from fedbrew.core.factory import _training_client_kwargs
from fedbrew.core.refusal import RunRefused

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "fedbrew"

#: The names that had two bodies, and the two that feed them.
SHARED_NAMES = (
    "CENTRALIZED_CLIENT_RULES",
    "FEDAVG_FT_CLIENT_RULES",
    "FEDAVG_ENGINE_CLIENT_RULES",
    "UPDATE_MODE_CLIENT_RULES",
    "FIXED_LR_SGD_CLIENT_RULES",
    "FROZEN_WEIGHTING_CLIENT_RULES",
    "UPDATE_MODE_OPTIONAL_CLIENT_RULES",
)

#: Where the one definition lives.
HOME = "fedbrew/core/config.py"


def _assignments(name: str) -> list[str]:
    """Every module under fedbrew/ that assigns `name` at module level."""

    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, ast.AnnAssign)
                else []
            )
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                found.append(str(path.relative_to(REPO_ROOT)))
    return found


@pytest.mark.fast
class OneDefinitionTest(unittest.TestCase):
    def test_each_name_is_assigned_exactly_once_and_in_config(self) -> None:
        for name in SHARED_NAMES:
            with self.subTest(name=name):
                self.assertEqual(_assignments(name), [HOME])

    def test_the_factory_imports_them_rather_than_holding_a_copy(self) -> None:
        for name in SHARED_NAMES:
            with self.subTest(name=name):
                self.assertIs(getattr(factory_module, name), getattr(config_module, name))

    def test_the_members_are_what_the_factory_builds_for(self) -> None:
        """The wider set won, because it is the one the builder acts on."""

        self.assertEqual(
            config_module.UPDATE_MODE_CLIENT_RULES,
            {
                "fedavg",
                "centralized",
                "fedavg_ft",
                "fedprox",
                "scaffold",
                "fedlalr",
                "local_sgd",
                "local_adamw",
            },
        )
        self.assertEqual(
            config_module.FIXED_LR_SGD_CLIENT_RULES,
            {"local_sgd", "fedavg", "centralized", "fedavg_ft"},
        )

    def test_splitting_the_sets_changed_no_member(self) -> None:
        """Every rule receives what it received before the split."""

        fedavg_engine = {"fedavg", "centralized", "fedavg_ft"}
        self.assertEqual(config_module.FEDAVG_ENGINE_CLIENT_RULES, fedavg_engine)
        self.assertEqual(config_module.FROZEN_WEIGHTING_CLIENT_RULES, fedavg_engine)
        own_loop = {"fedprox", "scaffold", "fedlalr", "local_sgd", "local_adamw"}
        self.assertEqual(
            config_module.UPDATE_MODES_BY_CLIENT_RULE,
            {
                **{rule: frozenset(config_module.UPDATE_MODES) for rule in fedavg_engine},
                **dict.fromkeys(own_loop, frozenset({"sequential_epoch", "full_gradient"})),
            },
        )
        self.assertEqual(config_module.UPDATE_MODE_OPTIONAL_CLIENT_RULES, {*own_loop})


@pytest.mark.fast
class TheGapThatWasClosedTest(unittest.TestCase):
    """Each of these was accepted before and is refused now."""

    def setUp(self) -> None:
        self.config = load_config("configs/femnist/fedavg_ft.yaml")

    def test_a_missing_option_the_factory_would_pass_is_refused(self) -> None:
        for key in ("momentum", "weight_decay", "nesterov", "learning_rate_schedule"):
            with self.subTest(key=key):
                broken = copy.deepcopy(self.config)
                self.assertIn(key, broken.client.extra, "the shipped config sets it")
                broken.client.extra.pop(key)
                with self.assertRaises(ValueError) as caught:
                    validate_config(broken)
                self.assertIn(key, str(caught.exception))

    def test_a_missing_update_mode_option_is_refused(self) -> None:
        for key in ("update_mode", "frozen_gradient_weighting"):
            with self.subTest(key=key):
                broken = copy.deepcopy(self.config)
                broken.client.extra.pop(key)
                with self.assertRaises(ValueError) as caught:
                    validate_config(broken)
                self.assertIn(key, str(caught.exception))

    def test_the_narrow_set_would_have_accepted_them(self) -> None:
        """The contrast, so the test above is not passing for another reason."""

        narrow_update_mode = {"fedavg", "centralized"}
        narrow_fixed = {"local_sgd", *narrow_update_mode}
        broken = copy.deepcopy(self.config)
        broken.client.extra.pop("momentum")

        original = (
            config_module.UPDATE_MODE_CLIENT_RULES,
            config_module.FIXED_LR_SGD_CLIENT_RULES,
        )
        config_module.UPDATE_MODE_CLIENT_RULES = narrow_update_mode
        config_module.FIXED_LR_SGD_CLIENT_RULES = narrow_fixed
        try:
            validate_config(broken)
        finally:
            (
                config_module.UPDATE_MODE_CLIENT_RULES,
                config_module.FIXED_LR_SGD_CLIENT_RULES,
            ) = original


@pytest.mark.fast
class TheClientStillRefusesItTooTest(unittest.TestCase):
    """The redundancy the validator gap was hiding behind stays in place."""

    def test_the_client_constructor_is_still_the_second_line(self) -> None:
        from fedbrew.clients.fedavg_ft_client import FedAvgFTClient
        from tests.test_empty_training_batches import _data, _Task

        with self.assertRaises((ValueError, TypeError)):
            FedAvgFTClient(
                client_id="c0",
                task=_Task(),
                model_config={},
                client_data=_data(8),
                local_iterations=1,
                batch_size=4,
                learning_rate=0.1,
                update_mode="not_a_mode",
                frozen_gradient_weighting="examples",
                finetune_epochs=1,
            )


class _StubDataset:
    """Only `get_client_data` is reached; the gates branch on the config."""

    def get_client_data(self, client_id: str) -> dict[str, object]:
        return {}


@pytest.mark.fast
class TakingAModeIsNotTakingAFixedRateTest(unittest.TestCase):
    """A rule outside the fixed-rate set keeps its update_mode, checked and forwarded.

    `fedavg` is taken out of the fixed-rate set here, as a rule whose step size
    comes from somewhere else would be. Before the split its update_mode then
    went unchecked, because the check ran from inside the fixed-rate one.
    """

    def setUp(self) -> None:
        self.config = load_config("configs/mnist/fedavg.yaml")
        self.patched = (config_module, factory_module)
        self.original = [module.FIXED_LR_SGD_CLIENT_RULES for module in self.patched]
        for module in self.patched:
            module.FIXED_LR_SGD_CLIENT_RULES = {"local_sgd"}

    def tearDown(self) -> None:
        for module, original in zip(self.patched, self.original, strict=True):
            module.FIXED_LR_SGD_CLIENT_RULES = original

    def test_the_fixed_rate_settings_are_no_longer_required(self) -> None:
        broken = copy.deepcopy(self.config)
        broken.client.extra.pop("momentum")
        validate_config(broken)

    def test_the_mode_is_still_required_and_checked(self) -> None:
        for change in ("drop", "unknown"):
            with self.subTest(change=change):
                broken = copy.deepcopy(self.config)
                if change == "drop":
                    broken.client.extra.pop("update_mode")
                else:
                    broken.client.extra["update_mode"] = "not_a_mode"
                with self.assertRaises(RunRefused) as caught:
                    validate_config(broken)
                self.assertIn("update_mode", str(caught.exception))

    def test_the_mode_is_still_forwarded_and_the_fixed_rate_settings_are_not(self) -> None:
        forwarded = _training_client_kwargs(self.config, object(), _StubDataset(), "c0", {})
        self.assertEqual(forwarded["update_mode"], self.config.client.extra["update_mode"])
        self.assertIn("frozen_gradient_weighting", forwarded)
        self.assertNotIn("momentum", forwarded)


class TheShippedConfigStillValidatesTest(unittest.TestCase):
    """Why the widening was free: nothing that ships is in the closed gap."""

    def test_every_shipped_fedavg_ft_config_validates(self) -> None:
        checked = 0
        for path in sorted(glob.glob("configs/**/*.yaml", recursive=True)):
            try:
                config = load_config(path)
            except Exception:  # not a run config, or invalid for another reason
                continue
            if config.client.update_rule != "fedavg_ft":
                continue
            checked += 1
            with self.subTest(config=path):
                validate_config(config)
        self.assertGreater(checked, 0, "no shipped fedavg_ft config; the premise is untested")


if __name__ == "__main__":
    unittest.main()
