"""``default_override_namespace`` is the supported way to drive ``run`` from Python.

A caller that starts a run in-process rather than through the console script --
an out-of-tree component's entry script, which is the only way to register one,
or a sweep driver -- has to hand ``run`` the namespace ``apply_cli_overrides``
expects. Before this function existed, ``examples/pl-1d/run.py`` did that by
keeping its own dict of all nineteen flag names.

That copy could not fail loudly. ``apply_cli_overrides`` reads every attribute
through ``getattr(args, name, default)``, so a flag missing from the copy is
not an error: the override is simply never applied, and the run proceeds under
the config's value while the caller believes otherwise. The tests here pin the
two properties that make the copy unnecessary -- the namespace is derived from
the parser, and it does not read ``sys.argv`` -- and the one that makes a
misspelled override loud instead of silent.
"""

from __future__ import annotations

import argparse
import re
import unittest
from pathlib import Path
from unittest import mock

import pytest

from fedbrew.core.config import load_config
from fedbrew.core.runner import apply_cli_overrides, default_override_namespace, parse_args

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_SOURCE = REPO_ROOT / "fedbrew" / "core" / "runner.py"

#: Every ``getattr(args, "<name>", ...)`` in fedbrew/core/runner.py. Read from
#: the source rather than listed here, because a list here would be the same
#: hand-kept copy this function exists to delete.
_OVERRIDE_READ = re.compile(r"getattr\(\s*args,\s*\"([a-z_]+)\"")


def _names_apply_cli_overrides_reads() -> set[str]:
    source = RUNNER_SOURCE.read_text(encoding="utf-8")
    body = source.split("def apply_cli_overrides(", 1)[1].split("\ndef ", 1)[0]
    return set(_OVERRIDE_READ.findall(body))


class DerivedFromTheParserTest(unittest.TestCase):
    def test_it_is_exactly_what_parsing_no_arguments_gives(self) -> None:
        self.assertEqual(vars(default_override_namespace()), vars(parse_args([])))

    def test_every_flag_the_override_reader_consumes_is_present(self) -> None:
        """The guard that catches the drift.

        A flag added to ``apply_cli_overrides`` but not to ``parse_args`` would
        be unreachable from the CLI and from here alike, and nothing else in
        the suite notices, because the ``getattr`` default makes it look like
        a flag nobody passed.
        """

        namespace = vars(default_override_namespace())
        missing = sorted(_names_apply_cli_overrides_reads() - set(namespace))
        self.assertEqual(
            missing,
            [],
            f"read by apply_cli_overrides, never defined by parse_args: {missing}",
        )

    def test_the_reader_reads_something_at_all(self) -> None:
        """Guards the guard: a regex that matched nothing would pass silently."""

        self.assertGreater(len(_names_apply_cli_overrides_reads()), 15)


class ItDoesNotReadTheProcessArgumentsTest(unittest.TestCase):
    def test_a_callers_own_flags_do_not_leak_in(self) -> None:
        """``parse_args(None)`` would parse ``sys.argv``; ``[]`` is deliberate.

        The caller is a script with flags of its own, so this is not a corner
        case -- ``examples/pl-1d/run.py`` is invoked with ``--rounds``, which
        ``fedbrew run`` also defines.
        """

        with mock.patch("sys.argv", ["run.py", "--rounds", "7", "--quiet"]):
            namespace = default_override_namespace()
        self.assertIsNone(namespace.rounds)
        self.assertFalse(namespace.quiet)

    def test_a_flag_this_parser_does_not_define_does_not_abort_the_caller(self) -> None:
        """``--algorithm`` is the example script's flag and not ``fedbrew run``'s.

        Parsing ``sys.argv`` would exit the process with argparse's usage
        message, from a call the caller made to ask for defaults.
        """

        with mock.patch("sys.argv", ["run.py", "--algorithm", "fedavg"]):
            namespace = default_override_namespace()
        self.assertFalse(hasattr(namespace, "algorithm"))


class OverridesTest(unittest.TestCase):
    def test_a_named_override_is_set(self) -> None:
        self.assertTrue(default_override_namespace(quiet=True).quiet)
        self.assertEqual(default_override_namespace(rounds=3).rounds, 3)

    def test_the_others_stay_at_their_defaults(self) -> None:
        namespace = default_override_namespace(quiet=True)
        self.assertIsNone(namespace.rounds)
        self.assertEqual(namespace.tag, [])

    def test_an_unknown_override_raises_rather_than_doing_nothing(self) -> None:
        with self.assertRaises(ValueError) as caught:
            default_override_namespace(quite=True)
        self.assertIn("quite", str(caught.exception))

    def test_the_message_names_what_is_accepted(self) -> None:
        with self.assertRaises(ValueError) as caught:
            default_override_namespace(learning_rate=0.1)
        self.assertIn("lr", str(caught.exception))


class AcceptedByTheOverrideReaderTest(unittest.TestCase):
    """The point of the namespace: ``apply_cli_overrides`` takes it, unchanged."""

    def _config(self) -> object:
        return load_config(REPO_ROOT / "configs" / "dev" / "smoke.yaml")

    def test_it_changes_nothing_when_nothing_is_overridden(self) -> None:
        config = self._config()
        self.assertEqual(apply_cli_overrides(config, default_override_namespace()), config)

    def test_an_override_reaches_the_config(self) -> None:
        overridden = apply_cli_overrides(self._config(), default_override_namespace(rounds=4))
        self.assertEqual(overridden.server.global_rounds, 4)

    def test_a_bare_namespace_is_also_accepted(self) -> None:
        """Why the old copy was redundant rather than load-bearing.

        The comment it carried said a missing attribute raised ``AttributeError``.
        It does not, and that is the whole problem: the failure is silent, so
        the copy could rot without any run ever complaining.
        """

        config = self._config()
        self.assertEqual(apply_cli_overrides(config, argparse.Namespace()), config)


if __name__ == "__main__":
    unittest.main()
