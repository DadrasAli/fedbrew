"""Keys the loader can find out are inferred, marked, and checked when stated.

``fedbrew/core/inferred.py`` fills four keys a config leaves out:
``model.input_dim`` and ``model.num_classes`` from the manifest, for a model
registered as sized by them, and ``experiment.output_dir`` and
``experiment.name`` from where the config sits under ``configs/``. Pinned here:

- each is filled, written into the resolved config, and named with its
  source in ``FullConfig.inferred``, which run.json records and the plan
  header prints;
- a stated value is not marked; a stated dimension that disagrees with the
  manifest is refused at load, not when the model is built;
- a model not sized by a key gets nothing inferred for it;
- outside a ``configs/`` directory the output directory is required and the
  name is the file stem, as every unnamed config was called before;
- ``--output-dir`` is the command line's, not an inference;
- the three client keys required only where they act run when left out.

The manifests here are written by hand, with no shards: the loader reads the
manifest's JSON and nothing else.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import yaml

from fedbrew.core import registry, runner
from fedbrew.core.config import load_config
from fedbrew.core.inferred import FROM_CONFIG_PATH, FROM_MANIFEST
from fedbrew.core.refusal import RunRefused

BASE = textwrap.dedent("""
    experiment:
      seed: 1
    server:
      strategy: fedavg
      participation_rate: 1
    client:
      update_rule: fedavg
      update_mode: sequential_epoch
      batch_size: 4
      learning_rate: 0.05
      learning_rate_schedule: constant
      momentum: 0.0
      weight_decay: 0.0
    data:
      num_clients: 2
      samples_per_client: 8
      input_dim: 4
      num_classes: 2
    model:
      name: mlp
      input_dim: 4
      hidden_dim: 8
      num_classes: 2
    numerics:
      deterministic: true
      use_amp: false
    runtime:
      device: cpu
    evaluation:
      train: {every: never}
      val: {every: never}
      test: {every: never}
      central_test: {every: never}
    defaults:
      global_rounds: 1
      local_iterations: 1
    """)


def _write(path: Path, edit: Any = None) -> Path:
    raw = yaml.safe_load(BASE)
    if edit is not None:
        edit(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


class _Directory(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)


@pytest.mark.fast
class FromTheConfigPathTest(_Directory):
    def test_both_are_inferred_under_a_configs_directory(self) -> None:
        config = load_config(_write(self.root / "configs" / "study" / "arm.yaml"))
        self.assertEqual(config.experiment.output_dir, "outputs/study/arm")
        self.assertEqual(config.experiment.name, "study-arm")
        self.assertEqual(
            config.inferred,
            {"experiment.output_dir": FROM_CONFIG_PATH, "experiment.name": FROM_CONFIG_PATH},
        )

    def test_the_path_below_configs_is_kept_whole(self) -> None:
        config = load_config(_write(self.root / "configs" / "examples" / "family" / "arm.yaml"))
        self.assertEqual(config.experiment.output_dir, "outputs/examples/family/arm")
        self.assertEqual(config.experiment.name, "family-arm")

    def test_a_config_directly_in_configs_is_named_by_its_file(self) -> None:
        config = load_config(_write(self.root / "configs" / "arm.yaml"))
        self.assertEqual(config.experiment.name, "arm")
        self.assertEqual(config.experiment.output_dir, "outputs/arm")

    def test_stated_values_are_kept_and_not_marked(self) -> None:
        def state(raw: dict[str, Any]) -> None:
            raw["experiment"]["output_dir"] = "outputs/elsewhere"
            raw["experiment"]["name"] = "my-run"

        config = load_config(_write(self.root / "configs" / "study" / "arm.yaml", state))
        self.assertEqual(config.experiment.output_dir, "outputs/elsewhere")
        self.assertEqual(config.experiment.name, "my-run")
        self.assertEqual(config.inferred, {})

    def test_outside_a_configs_directory_the_output_dir_is_required(self) -> None:
        with self.assertRaisesRegex(RunRefused, r"experiment\.output_dir is required"):
            load_config(_write(self.root / "elsewhere" / "arm.yaml"))

    def test_outside_a_configs_directory_the_name_is_the_file_stem(self) -> None:
        def state(raw: dict[str, Any]) -> None:
            raw["experiment"]["output_dir"] = str(self.root / "out")

        config = load_config(_write(self.root / "elsewhere" / "arm.yaml", state))
        self.assertEqual(config.experiment.name, "arm")
        self.assertEqual(config.inferred, {"experiment.name": FROM_CONFIG_PATH})

    def test_the_command_line_output_dir_is_not_an_inference(self) -> None:
        config = load_config(_write(self.root / "configs" / "study" / "arm.yaml"))
        effective = runner.apply_cli_overrides(
            config, runner.parse_args(["--output-dir", str(self.root / "cli")])
        )
        self.assertEqual(effective.experiment.output_dir, str(self.root / "cli"))
        self.assertNotIn("experiment.output_dir", effective.inferred)
        self.assertIn("experiment.name", effective.inferred)

    def test_the_plan_header_says_so(self) -> None:
        from fedbrew.core.logging import _identity_rows

        config = load_config(_write(self.root / "configs" / "study" / "arm.yaml"))
        rows = {row.label: row.value for row in _identity_rows(config, None, False, False)}
        self.assertEqual(rows["Experiment"], "study-arm (inferred from the config path)")
        self.assertEqual(rows["Output"], "outputs/study/arm (inferred from the config path)")

    def test_a_stated_value_that_differs_says_what_it_overrides(self) -> None:
        from fedbrew.core.logging import _identity_rows

        def state(raw: dict[str, Any]) -> None:
            raw["experiment"]["output_dir"] = "outputs/elsewhere"
            raw["experiment"]["name"] = "my-run"

        path = _write(self.root / "configs" / "study" / "arm.yaml", state)
        rows = {
            row.label: row.value
            for row in _identity_rows(load_config(path), None, False, False, path)
        }
        self.assertEqual(rows["Experiment"], "my-run (overrides the inferred study-arm)")
        self.assertEqual(
            rows["Output"], "outputs/elsewhere (overrides the inferred outputs/study/arm)"
        )

    def test_a_stated_value_that_agrees_says_nothing(self) -> None:
        from fedbrew.core.logging import _identity_rows

        def state(raw: dict[str, Any]) -> None:
            raw["experiment"]["output_dir"] = "outputs/study/arm"
            raw["experiment"]["name"] = "study-arm"

        path = _write(self.root / "configs" / "study" / "arm.yaml", state)
        rows = {
            row.label: row.value
            for row in _identity_rows(load_config(path), None, False, False, path)
        }
        self.assertEqual(rows["Experiment"], "study-arm")
        self.assertEqual(rows["Output"], "outputs/study/arm")

    def test_the_run_directory_is_not_an_override(self) -> None:
        """use_run_subdir adds the run's own directory below the stated one."""

        from dataclasses import replace

        from fedbrew.core.logging import _identity_rows

        def state(raw: dict[str, Any]) -> None:
            raw["experiment"]["output_dir"] = "outputs/study/arm"
            raw["experiment"]["use_run_subdir"] = True

        path = _write(self.root / "configs" / "study" / "arm.yaml", state)
        config = load_config(path)
        config.experiment = replace(config.experiment, output_dir="outputs/study/arm/run-1")
        rows = {row.label: row.value for row in _identity_rows(config, None, False, False, path)}
        self.assertEqual(rows["Output"], "outputs/study/arm/run-1")

    def test_a_config_may_not_write_the_record(self) -> None:
        def write(raw: dict[str, Any]) -> None:
            raw["inferred"] = {"experiment.name": "by hand"}

        with self.assertRaisesRegex(RunRefused, "unknown top-level config key"):
            load_config(_write(self.root / "configs" / "study" / "arm.yaml", write))


@pytest.mark.fast
class FromTheManifestTest(_Directory):
    def _config(self, model: dict[str, Any], manifest: dict[str, Any] | None = None) -> Path:
        manifest_path = self.root / "data" / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest or {"input_dim": 4, "num_classes": 2}), encoding="utf-8"
        )

        def edit(raw: dict[str, Any]) -> None:
            raw["data"] = {"path": str(manifest_path)}
            raw["model"] = model

        return _write(self.root / "configs" / "study" / "arm.yaml", edit)

    def test_the_dimensions_are_filled_and_marked(self) -> None:
        config = load_config(self._config({"name": "mlp", "hidden_dim": 8}))
        self.assertEqual((config.model.input_dim, config.model.num_classes), (4, 2))
        self.assertEqual(config.inferred["model.input_dim"], FROM_MANIFEST)
        self.assertEqual(config.inferred["model.num_classes"], FROM_MANIFEST)

    def test_run_json_records_the_marks(self) -> None:
        from dataclasses import asdict

        config = load_config(self._config({"name": "mlp", "hidden_dim": 8}))
        recorded = json.loads(json.dumps(asdict(config), default=str))
        self.assertEqual(recorded["model"]["input_dim"], 4)
        self.assertEqual(recorded["inferred"]["model.input_dim"], FROM_MANIFEST)

    def test_the_plan_header_shows_them(self) -> None:
        from fedbrew.core.logging import _algorithm_rows

        config = load_config(self._config({"name": "mlp", "hidden_dim": 8}))
        rows = {row.label: row.value for row in _algorithm_rows(config)}
        self.assertEqual(rows["Model input dimension"], "4 (inferred from the dataset manifest)")
        self.assertEqual(rows["Model classes"], "2 (inferred from the dataset manifest)")

    def test_a_stated_dimension_that_agrees_is_not_marked(self) -> None:
        config = load_config(
            self._config({"name": "mlp", "input_dim": 4, "hidden_dim": 8, "num_classes": 2})
        )
        self.assertEqual(config.inferred.keys() & {"model.input_dim", "model.num_classes"}, set())

    def test_a_stated_dimension_that_disagrees_is_refused_at_load(self) -> None:
        with self.assertRaisesRegex(RunRefused, r"model\.num_classes does not match the dataset"):
            load_config(self._config({"name": "mlp", "hidden_dim": 8, "num_classes": 3}))

    def test_a_model_not_sized_by_a_key_gets_nothing_for_it(self) -> None:
        # Registered first: registering the built-ins inside load_config would
        # write mlp's real shape keys over the patch, so run on its own this
        # test failed while it passed after any test that had registered them.
        registry.register_builtin_components()
        with mock.patch.dict(registry.MODEL_SHAPE_KEYS, {"mlp": ("num_classes",)}):
            config = load_config(self._config({"name": "mlp", "hidden_dim": 8}))
        self.assertIsNone(config.model.input_dim)
        self.assertNotIn("model.input_dim", config.inferred)
        self.assertEqual(config.model.num_classes, 2)

    def test_no_manifest_infers_nothing(self) -> None:
        def edit(raw: dict[str, Any]) -> None:
            raw["data"] = {"path": str(self.root / "absent" / "manifest.json")}
            raw["model"] = {"name": "mlp", "hidden_dim": 8}

        config = load_config(_write(self.root / "configs" / "study" / "arm.yaml", edit))
        self.assertIsNone(config.model.input_dim)
        self.assertNotIn("model.input_dim", config.inferred)

    def test_every_built_in_model_declares_what_it_reads(self) -> None:
        self.assertEqual(registry.model_shape_keys("mlp"), ("input_dim", "num_classes"))
        for name in ("cnn", "small_cnn", "femnist_resnet18", "openimage_shufflenet"):
            with self.subTest(model=name):
                self.assertEqual(registry.model_shape_keys(name), ("num_classes",))
        for name in ("tiny_gpt2", "hf_causal_lm", "hf_causal_lm_lora"):
            with self.subTest(model=name):
                self.assertEqual(registry.model_shape_keys(name), ())

    def test_a_registration_may_declare_only_the_two_keys(self) -> None:
        models = registry.ModelRegistry("models")
        with self.assertRaisesRegex(ValueError, "shape_keys may name only"):
            models.register("m", lambda config: None, task="classification", shape_keys=("dim",))


class TheKeysRequiredOnlyWhereTheyActTest(_Directory):
    """A fedavg run with none of the three builds its client and trains."""

    def test_a_round_runs_without_them(self) -> None:
        def edit(raw: dict[str, Any]) -> None:
            raw["experiment"]["output_dir"] = str(self.root / "run")
            for key in ("nesterov", "min_learning_rate", "frozen_gradient_weighting"):
                raw["client"].pop(key, None)

        path = _write(self.root / "configs" / "study" / "arm.yaml", edit)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            runner.run(path, runner.parse_args(["--quiet"]))
        self.assertTrue((self.root / "run" / "round_metrics.csv").is_file())


if __name__ == "__main__":
    unittest.main()
