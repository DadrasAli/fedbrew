"""A missing model snapshot must fail preflight, not the round loop.

The builder has always refused to proceed without one: hf_causal_lm requires
local_files_only, requires asset_manifest, and raises AssetManifestError naming
the prepare-llm command when either asset directory is absent. So a missing
snapshot never caused a silent download or a hang -- that part of the concern
was wrong.

What it did cause is a late failure. The builder runs inside the round loop's
setup, after config load, data staging and the dataset read, so a typo in
asset_manifest surfaced minutes into a job that `--validate-only` had just
called READY TO RUN. Preflight already checks the *data* manifest exactly this
way; this is the model's counterpart.
"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import pytest
import yaml

from fedbrew.core.config import load_config
from fedbrew.core.validation import validate_full_config

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent

#: An LLM config to mutate. Chosen because it names a concrete in-repo asset
#: path rather than an $FL_CACHE_ROOT one, so the resolved-path branch is the
#: one under test.
SOURCE_CONFIG = REPO_ROOT / "configs" / "medmcqa" / "fedavg_lora.yaml"


def _codes(config_values: dict) -> list[tuple[str, str]]:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config_values, handle)
        path = handle.name
    report = validate_full_config(load_config(path))
    return [
        (issue.severity, issue.code)
        for issue in report.issues
        if issue.code.startswith("model.asset_manifest")
    ]


class ModelAssetPreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
        self.assertIn("asset_manifest", self.raw["model"])

    def _with_manifest(self, value: str | None) -> dict:
        edited = copy.deepcopy(self.raw)
        if value is None:
            edited["model"].pop("asset_manifest")
        else:
            edited["model"]["asset_manifest"] = value
        return edited

    def test_a_missing_key_is_an_error(self) -> None:
        self.assertEqual(
            _codes(self._with_manifest(None)),
            [("error", "model.asset_manifest_missing")],
        )

    def test_a_path_that_does_not_exist_is_an_error(self) -> None:
        """The typo case: one character wrong in a directory name."""

        self.assertEqual(
            _codes(self._with_manifest("data/raw/models/qwen2_5_0_5b_bse/asset_manifest.json")),
            [("error", "model.asset_manifest_unusable")],
        )

    def test_an_unexported_variable_is_a_warning_not_an_error(self) -> None:
        """Validating from a login node cannot know what the job will export.

        An $FL_CACHE_ROOT path is the normal shape for an HPC config, so
        refusing it would make --validate-only useless for exactly the configs
        that most need checking before a long job.
        """

        self.assertEqual(
            _codes(self._with_manifest("$FL_CACHE_ROOT/models/x/asset_manifest.json")),
            [("warning", "model.asset_manifest_unresolved_env")],
        )

    def test_a_non_asset_backed_model_is_not_checked(self) -> None:
        """Only the two builders that load a prepared snapshot."""

        from fedbrew.core.validation import _ASSET_BACKED_MODELS

        self.assertEqual(_ASSET_BACKED_MODELS, frozenset({"hf_causal_lm", "hf_causal_lm_lora"}))
        config = load_config(str(REPO_ROOT / "configs" / "dev" / "smoke.yaml"))
        self.assertNotIn(config.model.name, _ASSET_BACKED_MODELS)
        report = validate_full_config(config)
        self.assertEqual(
            [i.code for i in report.issues if i.code.startswith("model.asset")],
            [],
        )


if __name__ == "__main__":
    unittest.main()
