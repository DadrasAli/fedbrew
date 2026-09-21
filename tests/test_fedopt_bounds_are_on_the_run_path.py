"""The FedOpt bounds are checked where every run reaches them.

`tau: 0` passed every gate and produced a NaN model. `FedOptServer` allowed it
(`tau < 0.0` is the only thing it refused), `validate_config` checked the four
hyperparameters for numeric-ness alone, and the bounds lived in `validation.py`,
which only `--validate-only` reaches. With `tau = 0` the initialisation
`v_{-1} = tau**2` is zero, so a coordinate whose delta is exactly zero computes
`0 / (sqrt(0) + 0)` -- and an exactly-zero delta is what every client returns
for a parameter that receives no gradient. P04-F05.

Two claims are guarded, and the first is the reason for the second: that
`tau = 0` really does produce NaN, measured rather than asserted from the
arithmetic, and that every bound is now refused on the run path with the same
words `--validate-only` reports.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from fedbrew.core.config import (
    FEDOPT_STRATEGIES,
    fedopt_optimizer_name,
    load_config,
    validate_config,
)
from fedbrew.core.validation import validate_full_config
from fedbrew.servers.fedopt import (
    FEDOPT_BOUNDS,
    FEDOPT_HYPERPARAMETERS,
    SUPPORTED_FEDOPT_OPTIMIZERS,
    FedOptServer,
    fedopt_bound_violation,
    unread_fedopt_hyperparameters,
)

#: A value outside each bound, and one inside it.
OUTSIDE: dict[str, tuple[float, ...]] = {
    "server_learning_rate": (0.0, -0.1),
    "beta1": (1.0, -0.1, 2.0),
    "beta2": (1.0, -0.1),
    "tau": (0.0, -1e-8),
}
INSIDE: dict[str, float] = {
    "server_learning_rate": 0.1,
    "beta1": 0.9,
    "beta2": 0.99,
    "tau": 1e-3,
}


def _run_configs() -> list[Path]:
    paths = []
    for path in sorted(Path("configs").rglob("*.yaml")):
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and "runtime" in loaded:
            paths.append(path)
    return paths


def _fedopt_configs() -> list[tuple[Path, Any]]:
    configs = []
    for path in _run_configs():
        config = load_config(str(path))
        if config.server.strategy in FEDOPT_STRATEGIES:
            configs.append((path, config))
    return configs


@pytest.mark.fast
class TauZeroIsTheNaNTest(unittest.TestCase):
    """Why the bound is `> 0` and not `>= 0`, measured on the update itself."""

    def _update(self, optimizer: str, tau: float) -> list[float]:
        _, unread = unread_fedopt_hyperparameters(optimizer)
        server = FedOptServer.__new__(FedOptServer)
        FedOptServer.__init__(
            server,
            server_optimizer=optimizer,
            server_learning_rate=0.1,
            beta1=0.9,
            beta2=None if "beta2" in unread else 0.99,
            tau=None if "tau" in unread else INSIDE["tau"],
            participation_rate=1.0,
            seed=0,
        )
        # Past the constructor's refusal: the claim under test is what the
        # arithmetic does at zero, which is why the refusal exists.
        server.tau = tau
        server._model_state = {"w": torch.zeros(4)}
        return server._apply_fedopt_update({"w": torch.zeros(4)})["w"].tolist()

    def test_a_zero_delta_at_tau_zero_is_nan_for_every_optimizer_with_a_denominator(
        self,
    ) -> None:
        for optimizer in sorted(SUPPORTED_FEDOPT_OPTIMIZERS):
            _, unread = unread_fedopt_hyperparameters(optimizer)
            if "tau" in unread:
                continue
            with self.subTest(optimizer=optimizer):
                self.assertTrue(
                    all(value != value for value in self._update(optimizer, 0.0)),
                    "tau=0 no longer produces NaN, so the bound needs a new reason",
                )
                self.assertEqual(self._update(optimizer, INSIDE["tau"]), [0.0, 0.0, 0.0, 0.0])


@pytest.mark.fast
class TheServerRefusesEveryBoundTest(unittest.TestCase):
    def _build(self, name: str, value: float) -> None:
        values = dict(INSIDE)
        values[name] = value
        FedOptServer(
            server_optimizer="fedadam",
            participation_rate=1.0,
            seed=0,
            **values,
        )

    def test_a_value_outside_its_bound_is_refused(self) -> None:
        for name, values in OUTSIDE.items():
            for value in values:
                with self.subTest(hyperparameter=name, value=value):
                    with self.assertRaises(ValueError) as caught:
                        self._build(name, value)
                    self.assertIn(name, str(caught.exception))

    def test_a_non_finite_value_is_refused(self) -> None:
        for name in FEDOPT_HYPERPARAMETERS:
            for value in (float("nan"), float("inf")):
                with self.subTest(hyperparameter=name, value=value):
                    self.assertIsNotNone(fedopt_bound_violation(name, value))

    def test_every_hyperparameter_has_a_bound(self) -> None:
        self.assertEqual(set(FEDOPT_BOUNDS), set(FEDOPT_HYPERPARAMETERS))

    def test_the_inside_values_really_are_inside(self) -> None:
        """A table of values that all failed would pass a check that refuses everything."""

        for name, value in INSIDE.items():
            with self.subTest(hyperparameter=name):
                self.assertIsNone(fedopt_bound_violation(name, value))


class TheRunPathRefusesWhatPreflightReportsTest(unittest.TestCase):
    BASE = "configs/femnist/fedadam.yaml"

    def _codes(self, config: Any, name: str) -> list[str]:
        return [
            issue.message
            for issue in validate_full_config(config).issues
            if issue.code == f"algorithm.fedopt_{name}_invalid"
        ]

    def test_every_bound_is_refused_by_validate_config_in_preflight_s_words(self) -> None:
        for name, values in OUTSIDE.items():
            for value in values:
                with self.subTest(hyperparameter=name, value=value):
                    config = load_config(self.BASE)
                    config.server.extra[name] = value
                    with self.assertRaises(ValueError) as caught:
                        validate_config(config)
                    self.assertEqual(self._codes(config, name), [str(caught.exception)])

    def test_a_shipped_config_is_accepted_by_both(self) -> None:
        for path, config in _fedopt_configs():
            with self.subTest(config=str(path)):
                validate_config(config)
                self.assertEqual(
                    [
                        issue.code
                        for issue in validate_full_config(config).issues
                        if issue.severity == "error" and issue.code.startswith("algorithm.fedopt")
                    ],
                    [],
                    "preflight objects to a config the run path accepts",
                )

    def test_preflight_still_reports_a_hyperparameter_the_optimizer_needs(self) -> None:
        """Not silence: the skip is for the unread ones, not for a missing one."""

        config = load_config(self.BASE)
        del config.server.extra["tau"]
        codes = [issue.code for issue in validate_full_config(config).issues]
        self.assertIn("algorithm.fedopt_tau_missing", codes)

    def test_an_unread_hyperparameter_is_never_reported_missing(self) -> None:
        for path, config in _fedopt_configs():
            _, unread = unread_fedopt_hyperparameters(fedopt_optimizer_name(config))
            if not unread:
                continue
            with self.subTest(config=str(path)):
                codes = [issue.code for issue in validate_full_config(config).issues]
                for name in unread:
                    self.assertNotIn(f"algorithm.fedopt_{name}_missing", codes)


if __name__ == "__main__":
    unittest.main()
