"""Tests for the consolidated validation command surface."""

from __future__ import annotations

import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import pytest

from fedbrew.cli.check_hpc_environment import main as check_hpc_main
from fedbrew.cli.cleanup import main as cleanup_main
from fedbrew.cli.inspect_generated_data import main as inspect_data_main
from fedbrew.core.runner import parse_args as parse_runner_args

pytestmark = pytest.mark.fast


class ValidationCommandTests(unittest.TestCase):
    def test_runner_uses_one_validation_option(self) -> None:
        args = parse_runner_args(["--validate-only"])

        self.assertTrue(args.validate_only)

    def test_removed_validation_options_are_rejected(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_runner_args(["--strict"])

    def test_manual_commands_support_help(self) -> None:
        for command in (inspect_data_main, check_hpc_main):
            output = StringIO()
            with redirect_stdout(output):
                command(["--help"])
            self.assertIn("usage:", output.getvalue())

    def test_cleanup_help_names_the_installed_command(self) -> None:
        output = StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit):
            cleanup_main(["--help"])

        self.assertIn("usage: fedbrew cleanup", output.getvalue())


if __name__ == "__main__":
    unittest.main()
