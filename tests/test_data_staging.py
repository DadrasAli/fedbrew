"""Staging must fail on a key it cannot honour, not disable itself quietly.

`runtime.data_staging` had no key allow-list. `mode` accepted exactly one
value, copy_tree, and printed "Data staging skipped: only mode=copy_tree is
supported" for anything else -- so the key could only ever turn off the feature
it looked like it configured, and the job still exited 0 having read the
dataset over the network it was trying to avoid. `fallback_local_root` was a
second destination nothing set and nothing documented.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pytest

from fedbrew.core.config import (
    ClientConfig,
    DataConfig,
    ExperimentConfig,
    FullConfig,
    ModelConfig,
    RuntimeConfig,
    ServerConfig,
    TaskConfig,
    validate_config,
)
from fedbrew.core.data_staging import maybe_stage_manifest_dataset, staged_directory_name

pytestmark = pytest.mark.fast


def _config(data_path: str | None, **staging: Any) -> FullConfig:
    return FullConfig(
        experiment=ExperimentConfig(seed=0, output_dir="outputs/test"),
        server=ServerConfig(strategy="fedavg", global_rounds=1, participation_rate=1.0, metrics=[]),
        client=ClientConfig(
            update_rule="fedavg",
            local_iterations=1,
            batch_size=1,
            metrics=[],
            learning_rate=0.1,
            # fedavg's own required extras. validate_config checks unknown
            # keys before these, so the rejection tests would pass without
            # them -- but the "this validates" case has to clear the whole
            # validator, not just the part under test.
            extra={
                "momentum": 0.0,
                "weight_decay": 0.0,
                "nesterov": False,
                "learning_rate_schedule": "constant",
                "min_learning_rate": 0.0,
                "update_mode": "sequential_epoch",
                "frozen_gradient_weighting": "examples",
            },
        ),
        task=TaskConfig(name="classification"),
        data=DataConfig(name="manifest_dataset", path=data_path),
        model=ModelConfig(name="mlp"),
        runtime=RuntimeConfig(device="cpu", use_amp=False, extra={"data_staging": dict(staging)}),
    )


class RemovedStagingKeysTest(unittest.TestCase):
    def test_mode_is_refused_and_says_why(self) -> None:
        config = _config(None, enabled=True, mode="rsync")

        with self.assertRaises(ValueError) as caught:
            validate_config(config)

        message = str(caught.exception)
        self.assertIn("runtime.data_staging.mode has been removed", message)
        self.assertIn("copy_tree was the only supported value", message)

    def test_fallback_local_root_is_refused_and_says_why(self) -> None:
        config = _config(None, enabled=True, fallback_local_root="/scratch")

        with self.assertRaises(ValueError) as caught:
            validate_config(config)

        self.assertIn(
            "runtime.data_staging.fallback_local_root has been removed",
            str(caught.exception),
        )

    def test_an_unrecognised_staging_key_is_refused(self) -> None:
        config = _config(None, enabled=True, local_rooot="/scratch")

        with self.assertRaises(ValueError) as caught:
            validate_config(config)

        message = str(caught.exception)
        self.assertIn("local_rooot", message)
        self.assertIn("enabled, local_root", message)

    def test_the_two_surviving_keys_validate(self) -> None:
        validate_config(_config(None, enabled=True, local_root="/scratch"))
        validate_config(_config(None, enabled=False))


class StagingBehaviourTest(unittest.TestCase):
    """Nothing covered maybe_stage_manifest_dataset before this file."""

    def test_an_unexpanded_local_root_stages_nothing(self) -> None:
        os.environ.pop("FL_NOT_EXPORTED_IN_THIS_TEST", None)
        config = _config(
            "data/generated/x/manifest.json",
            enabled=True,
            local_root="$FL_NOT_EXPORTED_IN_THIS_TEST/staged",
        )

        staged = maybe_stage_manifest_dataset(config)

        self.assertEqual(staged.data.path, config.data.path)

    def test_a_missing_local_root_stages_nothing(self) -> None:
        config = _config("data/generated/x/manifest.json", enabled=True)
        self.assertEqual(maybe_stage_manifest_dataset(config).data.path, config.data.path)

    def test_disabled_staging_leaves_the_path_alone(self) -> None:
        config = _config("data/generated/x/manifest.json", enabled=False)
        self.assertEqual(maybe_stage_manifest_dataset(config).data.path, config.data.path)

    def test_enabled_staging_copies_the_dataset_and_repoints_the_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source" / "dataset"
            source.mkdir(parents=True)
            (source / "manifest.json").write_text("{}", encoding="utf-8")
            (source / "shard_000.pt").write_bytes(b"payload")
            scratch = root / "scratch"

            config = _config(
                str(source / "manifest.json"),
                enabled=True,
                local_root=str(scratch),
            )
            staged = maybe_stage_manifest_dataset(config)

            # The destination carries a digest of the source path, so this
            # asks the same function the code does rather than restating the
            # naming scheme. P10-F25.
            destination = scratch / staged_directory_name(source)
            self.assertEqual(staged.data.path, str(destination / "manifest.json"))
            self.assertEqual((destination / "shard_000.pt").read_bytes(), b"payload")


if __name__ == "__main__":
    unittest.main()
