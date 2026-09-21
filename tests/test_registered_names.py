"""A component name nothing is registered under is refused at load.

A typo in server.strategy used to load clean. The run then died inside
build_components on a registry lookup, which reports the name it could not
find and not the names it has -- and only after the config was accepted, the
output directory prepared and, for a staged dataset, the data copied.

validate_full_config did report it, with the valid names, and lives in the
module only --validate-only reaches, so it stopped nothing. Same finding as
the divergence.metric check and the strategy pairings: a check that does not
run on the run path does not gate.

The five lookups had one shape and five copies, which is how one of them ends
up different -- data.name and model.name returned early after reporting empty,
the other three did not. One table now, checked here against the registries
themselves so a sixth registry cannot be added without a row.
"""

from __future__ import annotations

import copy
import glob
import tempfile
import unittest
from pathlib import Path

import pytest

from fedbrew.core import registry
from fedbrew.core.config import (
    REGISTERED_NAMES,
    REGISTRIES_CHECKED_ELSEWHERE,
    load_config,
    validate_config,
)

BASE = "configs/dev/smoke.yaml"


def _refusal(config) -> str | None:
    try:
        validate_config(config)
    except ValueError as error:
        return str(error)
    return None


def _with(path: str, value: object):
    config = copy.deepcopy(load_config(BASE))
    target = config
    parts = path.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)
    return config


@pytest.mark.fast
class EveryNameIsCheckedTest(unittest.TestCase):
    def test_the_baseline_config_is_accepted(self) -> None:
        """The probes mean nothing if the config is already refused."""

        self.assertIsNone(_refusal(load_config(BASE)))

    def test_a_typo_in_each_is_refused(self) -> None:
        for key, path, _, _ in REGISTERED_NAMES:
            with self.subTest(key=key):
                message = _refusal(_with(path, "definitely_not_registered"))
                self.assertIsNotNone(message, f"a typo in {key} loads clean")
                assert message is not None
                self.assertIn(key, message)
                self.assertIn("definitely_not_registered", message)

    def test_the_message_lists_what_is_registered(self) -> None:
        """The whole reason to refuse here rather than at the registry lookup.

        A KeyError names what was not found. This names what there is, which
        is what someone who mistyped needs.
        """

        for key, path, attribute, _ in REGISTERED_NAMES:
            with self.subTest(key=key):
                message = _refusal(_with(path, "definitely_not_registered"))
                assert message is not None
                registered = sorted(getattr(registry, attribute).list())
                self.assertTrue(registered, f"{attribute} is empty; the probe proves nothing")
                for name in registered:
                    self.assertIn(name, message)

    def test_an_empty_value_is_refused_for_each(self) -> None:
        """Five copies of one check had two of them returning early."""

        for key, path, _, _ in REGISTERED_NAMES:
            with self.subTest(key=key):
                message = _refusal(_with(path, "   "))
                self.assertIsNotNone(message, f"an empty {key} loads clean")
                assert message is not None
                self.assertIn(f"{key} must be set", message)


@pytest.mark.fast
class TheTableCoversEveryRegistryTest(unittest.TestCase):
    def test_one_row_per_registry(self) -> None:
        from fedbrew.core.registry import Registry

        present = {
            name
            for name in dir(registry)
            if not name.startswith("_") and isinstance(getattr(registry, name), Registry)
        }
        self.assertEqual(
            {attribute for _, _, attribute, _ in REGISTERED_NAMES}
            | set(REGISTRIES_CHECKED_ELSEWHERE),
            present,
            "a registry has no row in REGISTERED_NAMES and no entry in "
            "REGISTRIES_CHECKED_ELSEWHERE, so a name it holds is checked nowhere",
        )
        self.assertEqual(
            {attribute for _, _, attribute, _ in REGISTERED_NAMES}
            & set(REGISTRIES_CHECKED_ELSEWHERE),
            set(),
            "a registry is in both tables; it is checked in one place",
        )

    def test_a_registry_checked_elsewhere_really_is(self) -> None:
        """The entry names a function, and the function refuses an unknown name
        with the listing -- the same shape as the load-time check, so the
        exemption from REGISTERED_NAMES is a relocation and not a hole."""

        from importlib import import_module

        for attribute, checker in REGISTRIES_CHECKED_ELSEWHERE.items():
            with self.subTest(registry=attribute):
                module_name, _, function_name = checker.rpartition(".")
                check = getattr(import_module(module_name), function_name)
                with self.assertRaises(ValueError) as caught:
                    check("definitely_not_registered")
                message = str(caught.exception)
                self.assertIn("definitely_not_registered", message)
                for name in getattr(registry, attribute).list():
                    self.assertIn(name, message)

    def test_every_row_points_at_a_real_config_field(self) -> None:
        config = load_config(BASE)
        for key, path, _, _ in REGISTERED_NAMES:
            with self.subTest(key=key):
                self.assertEqual(key, path, "the message key and the attribute path must agree")
                value = config
                for part in path.split("."):
                    value = getattr(value, part)
                self.assertIsInstance(value, str)


@pytest.mark.fast
class RegistrationFailureStandsDownTest(unittest.TestCase):
    """A missing optional extra must not become "unknown model".

    register_builtin_components imports every builder. If one fails on an
    absent extra, refusing here would report the wrong problem in the wrong
    words. The check stands down and the run fails at import with the real
    reason.
    """

    def test_a_raising_registration_does_not_refuse(self) -> None:
        from unittest import mock

        config = _with("model.name", "definitely_not_registered")
        with mock.patch.object(
            registry,
            "register_builtin_components",
            side_effect=ImportError("No module named 'transformers'"),
        ):
            self.assertIsNone(_refusal(config))


@pytest.mark.fast
class ItIsOnTheRunPathTest(unittest.TestCase):
    def test_load_config_itself_refuses(self) -> None:
        import yaml

        raw = yaml.safe_load(Path(BASE).read_text(encoding="utf-8"))
        raw["server"]["strategy"] = "fedavg2"
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            yaml.safe_dump(raw, handle)
            path = handle.name
        try:
            with self.assertRaises(ValueError) as caught:
                load_config(path)
            self.assertIn("server.strategy", str(caught.exception))
        finally:
            Path(path).unlink()


class NoShippedConfigTripsItTest(unittest.TestCase):
    def test_every_config_still_loads(self) -> None:
        offenders = []
        checked = 0
        for path in sorted(glob.glob("configs/**/*.yaml", recursive=True)):
            try:
                load_config(path)
            except ValueError as error:
                if "Registered" in str(error) or "must be set" in str(error):
                    offenders.append(f"{path}  {error}")
                continue
            except Exception:  # noqa: BLE001 - generator configs are not run configs
                continue
            checked += 1
        self.assertGreater(checked, 20, "the config scan found almost nothing")
        self.assertEqual(
            offenders, [], f"shipped configs name unregistered components: {offenders}"
        )


if __name__ == "__main__":
    unittest.main()
