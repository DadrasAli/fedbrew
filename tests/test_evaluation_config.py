"""Tests for the post-aggregation evaluation schedule and client scope."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pytest

from fedbrew.core.config import (
    EvaluationConfig,
    SplitEvaluationConfig,
    load_config,
    validate_config,
)

pytestmark = pytest.mark.fast

_PROJECT_ROOT = Path(__file__).parents[1]
_SMOKE_CONFIG = _PROJECT_ROOT / "configs" / "dev" / "smoke.yaml"


class EvaluationConfigTests(unittest.TestCase):
    def test_missing_evaluation_section_uses_the_built_in_schedule(self) -> None:
        config = load_config(_SMOKE_CONFIG)

        self.assertEqual(config.evaluation.train.clients, "participating")
        self.assertEqual(config.evaluation.val.clients, "all")
        self.assertEqual(config.evaluation.test.clients, "all")
        self.assertEqual(config.evaluation.central_test.every, 10)

    def test_yaml_sets_each_split_independently(self) -> None:
        config_text = _SMOKE_CONFIG.read_text(encoding="utf-8")
        config_text += (
            "\nevaluation:\n"
            "  train:\n    every: 4\n    clients: participating\n"
            "  val:\n    every: 2\n    clients: sample:8\n"
            "  test:\n    every: final\n    clients: all\n"
            "  central_test:\n    every: never\n"
        )

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "per_split.yaml"
            config_path.write_text(config_text, encoding="utf-8")
            config = load_config(config_path)

        self.assertEqual(config.evaluation.train.every, 4)
        self.assertEqual(config.evaluation.val.clients, "sample:8")
        self.assertEqual(config.evaluation.test.every, "final")
        self.assertEqual(config.evaluation.central_test.every, "never")

    def test_removed_single_choice_key_is_rejected(self) -> None:
        config_text = _SMOKE_CONFIG.read_text(encoding="utf-8")
        config_text += "\nevaluation:\n  test_set: server\n"

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "removed_interface.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "has been removed"):
                load_config(config_path)

    def test_an_invalid_schedule_is_rejected(self) -> None:
        config = load_config(_SMOKE_CONFIG)
        invalid = replace(
            config,
            evaluation=EvaluationConfig(
                test=SplitEvaluationConfig(every="sometimes", clients="all")
            ),
        )

        with self.assertRaisesRegex(ValueError, "evaluation.test.every"):
            validate_config(invalid)

    def test_test_may_not_be_evaluated_on_participating_clients_only(self) -> None:
        # The reported test number must not come from the clients the model was
        # just fitted on.
        config = load_config(_SMOKE_CONFIG)
        invalid = replace(
            config,
            evaluation=EvaluationConfig(
                test=SplitEvaluationConfig(every=1, clients="participating")
            ),
        )

        with self.assertRaisesRegex(ValueError, "must not be 'participating'"):
            validate_config(invalid)

    def test_schedule_values_are_resolved_only_from_defaults(self) -> None:
        config = load_config(_SMOKE_CONFIG)

        self.assertEqual(config.server.global_rounds, 1)
        self.assertEqual(config.client.local_iterations, 1)

    def test_component_schedule_fields_are_rejected(self) -> None:
        config_text = _SMOKE_CONFIG.read_text(encoding="utf-8")
        config_text = config_text.replace(
            "  strategy: fedavg\n",
            "  strategy: fedavg\n  global_rounds: 5\n",
        )
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "duplicate_rounds.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "server.global_rounds has been removed",
            ):
                load_config(config_path)

        config_text = _SMOKE_CONFIG.read_text(encoding="utf-8")
        config_text = config_text.replace(
            "  update_rule: local_sgd\n",
            "  update_rule: local_sgd\n  local_iterations: 5\n",
        )
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "duplicate_iterations.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "client.local_iterations has been removed",
            ):
                load_config(config_path)

    def test_the_old_spellings_of_local_iterations_are_refused_by_name(self) -> None:
        """A pre-rename config names the key that replaced its own.

        Refused rather than read under the new name, and refused even beside a
        `defaults.local_iterations`, where it would otherwise be one more
        value nothing reads.
        """

        smoke = _SMOKE_CONFIG.read_text(encoding="utf-8")
        cases = {
            "old defaults key alone": smoke.replace(
                "  local_iterations: 1\n", "  local_epochs: 1\n"
            ),
            "old defaults key beside the new one": smoke.replace(
                "  local_iterations: 1\n", "  local_iterations: 1\n  local_epochs: 5\n"
            ),
            "old client key": smoke.replace(
                "  update_rule: local_sgd\n",
                "  update_rule: local_sgd\n  local_epochs: 5\n",
            ),
        }
        for label, text in cases.items():
            with self.subTest(case=label), tempfile.TemporaryDirectory() as directory:
                config_path = Path(directory) / "pre_rename.yaml"
                config_path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError) as caught:
                    load_config(config_path)
                message = str(caught.exception)
                self.assertIn("local_epochs has been removed", message)
                self.assertIn("defaults.local_iterations", message)
                self.assertIn("predates the rename", message)


if __name__ == "__main__":
    unittest.main()
