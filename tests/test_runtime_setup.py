"""Tests for deterministic process environment setup."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

import pytest

from fedbrew.core import runner
from fedbrew.core.runtime_setup import configure_deterministic_environment

pytestmark = pytest.mark.fast

_CUBLAS_WORKSPACE_CONFIG = "CUBLAS_WORKSPACE_CONFIG"


class DeterministicEnvironmentTests(unittest.TestCase):
    def test_deterministic_mode_sets_cublas_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_CUBLAS_WORKSPACE_CONFIG, None)

            value = configure_deterministic_environment(True)

            self.assertEqual(value, ":4096:8")
            self.assertEqual(os.environ[_CUBLAS_WORKSPACE_CONFIG], ":4096:8")

    def test_deterministic_mode_preserves_existing_cublas_value(self) -> None:
        with mock.patch.dict(
            os.environ,
            {_CUBLAS_WORKSPACE_CONFIG: ":16:8"},
            clear=False,
        ):
            value = configure_deterministic_environment(True)

            self.assertEqual(value, ":16:8")
            self.assertEqual(os.environ[_CUBLAS_WORKSPACE_CONFIG], ":16:8")

    def test_nondeterministic_mode_does_not_set_cublas_value(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_CUBLAS_WORKSPACE_CONFIG, None)

            value = configure_deterministic_environment(False)

            self.assertIsNone(value)
            self.assertNotIn(_CUBLAS_WORKSPACE_CONFIG, os.environ)

    def test_runner_configures_environment_before_metadata_capture(self) -> None:
        config_path = Path(__file__).parents[1] / "configs" / "dev" / "smoke_eval.yaml"

        class StopAfterAssertion(Exception):
            pass

        def assert_environment_is_ready(config: object) -> None:
            self.assertEqual(os.environ.get(_CUBLAS_WORKSPACE_CONFIG), ":4096:8")
            raise StopAfterAssertion

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_CUBLAS_WORKSPACE_CONFIG, None)
            with mock.patch.object(
                runner,
                "resolve_run_metadata",
                side_effect=assert_environment_is_ready,
            ):
                with self.assertRaises(StopAfterAssertion):
                    runner.run(config_path)


if __name__ == "__main__":
    unittest.main()
