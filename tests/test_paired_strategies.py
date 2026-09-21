"""Three algorithms need both halves, and both halves are refused at load.

Naming one half of a matched pair used to load clean for SCAFFOLD. A scaffold
client on a FedAvg server ran until the server read a fit payload
with no model_state; a scaffold server on a FedAvg client until it found no
control_delta. Round one -- after the data was staged, the clients were built
and the first local iterations were spent.

validate_full_config reported all of it, and reported it in a module reached
only by --validate-only, so an ordinary run was not stopped by any of it. The
refusal is now in validate_config, which load_config calls.

The table is checked against the pairing constants in fedbrew/core/factory.py,
so it cannot quietly disagree with the code that builds the components, and
every direction is probed against a real shipped config rather than a
hand-built one -- the earlier version of this investigation was misled by a
config whose refusal came from an unrelated check about client.momentum.
"""

from __future__ import annotations

import copy
import glob
import unittest

import pytest

from fedbrew.core.config import PAIRED_STRATEGIES, load_config, validate_config
from fedbrew.core.factory import (
    CENTRALIZED_CLIENT_RULES,
    CENTRALIZED_SERVER_STRATEGIES,
    FEDLALR_CLIENT_RULES,
    FEDLALR_SERVER_STRATEGIES,
    SCAFFOLD_CLIENT_RULES,
    SCAFFOLD_SERVER_STRATEGIES,
)
from fedbrew.core.validation import validate_full_config

#: A shipped config that runs each paired algorithm, so the probe changes one
#: field and nothing else is wrong with the config.
SHIPPED = {
    "centralized": "configs/femnist/centralized.yaml",
    "scaffold": "configs/femnist/scaffold.yaml",
    "fedlalr": "configs/femnist/fedlalr.yaml",
}

#: What the other half is set to when one half is knocked out. fedavg for the
#: server; for the client, a rule that is itself unpaired.
UNPAIRED_SERVER = "fedavg"
UNPAIRED_CLIENT = "fedprox"


def _refusal(config) -> str | None:
    try:
        validate_config(config)
    except ValueError as error:
        return str(error)
    return None


@pytest.mark.fast
class BothHalvesAreRequiredTest(unittest.TestCase):
    def test_the_shipped_config_for_each_is_accepted_as_it_ships(self) -> None:
        """The probes below mean nothing if the baseline is already refused."""

        for name, path in sorted(SHIPPED.items()):
            with self.subTest(algorithm=name):
                config = load_config(path)
                self.assertEqual(config.server.strategy, name)
                self.assertEqual(config.client.update_rule, name)
                self.assertIsNone(_refusal(config))

    def test_the_client_alone_is_refused(self) -> None:
        for name, path in sorted(SHIPPED.items()):
            with self.subTest(algorithm=name):
                config = copy.deepcopy(load_config(path))
                config.server.strategy = UNPAIRED_SERVER
                message = _refusal(config)
                self.assertIsNotNone(message, f"{name} client alone loads clean")
                assert message is not None
                self.assertIn(f"client.update_rule={name}", message)
                self.assertIn(f"requires server.strategy={name}", message)

    def test_the_server_alone_is_refused(self) -> None:
        for name, path in sorted(SHIPPED.items()):
            with self.subTest(algorithm=name):
                config = copy.deepcopy(load_config(path))
                config.client.update_rule = UNPAIRED_CLIENT
                message = _refusal(config)
                self.assertIsNotNone(message, f"{name} server alone loads clean")
                assert message is not None
                self.assertIn(f"server.strategy={name}", message)
                self.assertIn(f"requires client.update_rule={name}", message)

    def test_the_message_says_why_and_what_was_found(self) -> None:
        """A refusal without a reason is one people work around."""

        config = copy.deepcopy(load_config(SHIPPED["scaffold"]))
        config.server.strategy = UNPAIRED_SERVER
        message = _refusal(config)
        assert message is not None
        self.assertIn(f"got {UNPAIRED_SERVER!r}", message)
        self.assertIn("control", message, "the message must carry the reason")


@pytest.mark.fast
class TheTableMatchesTheFactoryTest(unittest.TestCase):
    """One row per paired algorithm, agreeing with what builds the components."""

    def test_every_paired_constant_has_a_row(self) -> None:
        paired = (
            SCAFFOLD_SERVER_STRATEGIES
            | SCAFFOLD_CLIENT_RULES
            | CENTRALIZED_SERVER_STRATEGIES
            | CENTRALIZED_CLIENT_RULES
            | FEDLALR_SERVER_STRATEGIES
            | FEDLALR_CLIENT_RULES
        )
        self.assertEqual(
            set(PAIRED_STRATEGIES),
            paired,
            "the pairing table and factory.py's pairing constants disagree; a "
            "paired algorithm with no row here loads with one half missing",
        )

    def test_every_row_carries_a_reason(self) -> None:
        for name, reason in sorted(PAIRED_STRATEGIES.items()):
            with self.subTest(algorithm=name):
                self.assertGreater(len(reason), 40, "the reason goes in the message")
                self.assertFalse(reason.endswith("."), "it is interpolated mid-sentence")


class PreflightReportsWhatTheRunPathRefusesTest(unittest.TestCase):
    """The two must not disagree, in either direction.

    A preflight missing one half calls a config READY TO RUN that load_config
    refuses, which is worse than either being wrong alone.
    """

    def test_both_directions_of_every_pairing_are_reported(self) -> None:
        for name, path in sorted(SHIPPED.items()):
            base = load_config(path)
            for field, replacement in (
                ("server.strategy", UNPAIRED_SERVER),
                ("client.update_rule", UNPAIRED_CLIENT),
            ):
                with self.subTest(algorithm=name, knocked_out=field):
                    config = copy.deepcopy(base)
                    block, attribute = field.split(".")
                    setattr(getattr(config, block), attribute, replacement)
                    codes = [
                        issue.code
                        for issue in validate_full_config(config).issues
                        if issue.severity == "error" and issue.code.startswith("algorithm.")
                    ]
                    self.assertTrue(
                        any(name in code for code in codes),
                        f"preflight reports no {name} pairing error for a "
                        f"config load_config refuses: {codes}",
                    )


class NoShippedConfigIsHalfPairedTest(unittest.TestCase):
    def test_every_config_still_loads(self) -> None:
        offenders = []
        checked = 0
        for path in sorted(glob.glob("configs/**/*.yaml", recursive=True)):
            try:
                load_config(path)
            except ValueError as error:
                if "requires" in str(error) and "update_rule" in str(error):
                    offenders.append(f"{path}  {error}")
                continue
            except Exception:  # noqa: BLE001 - generator configs are not run configs
                continue
            checked += 1
        self.assertGreater(checked, 20, "the config scan found almost nothing")
        self.assertEqual(offenders, [], f"shipped configs are half-paired: {offenders}")


if __name__ == "__main__":
    unittest.main()
