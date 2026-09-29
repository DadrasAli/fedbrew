"""An arm file extends its family base, and ``fedbrew config show`` prints it whole.

``extends: <file>`` lays a config over another (``load_config_mapping``,
docs/04 section 2.2): mappings merge key by key, anything else replaces the
base's value whole, a chain may be longer than one, and everything after --
refusals, validation, the path inferences -- sees the merged mapping. A file
named ``_*.yaml`` is a family base, which ``fedbrew run`` refuses and every
listing of shipped run configs skips. Pinned here, with the shipped layout:
every arm in a directory that has a base extends it, and every base is
extended.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
import textwrap
import unittest
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from fedbrew.cli import show_config
from fedbrew.core.config import (
    is_family_base,
    load_config,
    load_config_mapping,
    root_config_keys,
    standalone_config_mapping,
)
from fedbrew.core.refusal import RunRefused

REPO = Path(__file__).resolve().parent.parent

BASE = textwrap.dedent("""
    experiment:
      seed: 1
      tags: [base]
      notes: from the base
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
    defaults:
      global_rounds: 3
      local_iterations: 1
    """)


class _Tree(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.family = Path(directory.name) / "configs" / "study"
        self.family.mkdir(parents=True)
        (self.family / "_base.yaml").write_text(BASE, encoding="utf-8")

    def _arm(self, text: str, name: str = "arm.yaml") -> Path:
        path = self.family / name
        path.write_text(textwrap.dedent(text), encoding="utf-8")
        return path


@pytest.mark.fast
class HowItMergesTest(_Tree):
    def test_a_mapping_merges_key_by_key(self) -> None:
        arm = self._arm("""
            extends: _base.yaml
            client:
              learning_rate: 0.1
            """)
        merged = load_config_mapping(arm)
        self.assertEqual(merged["client"]["learning_rate"], 0.1)
        self.assertEqual(merged["client"]["batch_size"], 4)
        self.assertNotIn("extends", merged)

    def test_a_list_and_a_null_replace_whole(self) -> None:
        arm = self._arm("""
            extends: _base.yaml
            experiment:
              tags: [arm]
              notes: null
            """)
        merged = load_config_mapping(arm)
        self.assertEqual(merged["experiment"]["tags"], ["arm"])
        self.assertIsNone(merged["experiment"]["notes"])

    def test_a_chain_merges_from_the_bottom_up(self) -> None:
        self._arm(
            """
            extends: _base.yaml
            client:
              learning_rate: 0.1
              batch_size: 8
            """,
            "_middle.yaml",
        )
        arm = self._arm("""
            extends: _middle.yaml
            client:
              learning_rate: 0.2
            """)
        client = load_config(arm).client
        self.assertEqual((client.learning_rate, client.batch_size), (0.2, 8))

    def test_the_arm_loads_as_its_merged_mapping_would(self) -> None:
        arm = self._arm("""
            extends: _base.yaml
            client:
              learning_rate: 0.1
            """)
        flat = self.family / "flat.yaml"
        flat.write_text(yaml.safe_dump(load_config_mapping(arm)), encoding="utf-8")
        via_extends, via_flat = asdict(load_config(arm)), asdict(load_config(flat))
        for record in (via_extends, via_flat):
            record["experiment"].pop("name")
            record["experiment"].pop("output_dir")
            record.pop("inferred")
        self.assertEqual(via_extends, via_flat)

    def test_path_inferences_read_the_arm_not_the_base(self) -> None:
        config = load_config(self._arm("extends: _base.yaml\n"))
        self.assertEqual(config.experiment.output_dir, "outputs/study/arm")
        self.assertEqual(config.experiment.name, "study-arm")

    def test_a_standalone_mapping_carries_what_the_path_gave(self) -> None:
        mapping = standalone_config_mapping(self._arm("extends: _base.yaml\n"))
        self.assertEqual(mapping["experiment"]["output_dir"], "outputs/study/arm")
        self.assertEqual(mapping["experiment"]["name"], "study-arm")

    def test_extends_is_a_root_key_the_loader_accepts(self) -> None:
        self.assertIn("extends", root_config_keys())


@pytest.mark.fast
class WhatItRefusesTest(_Tree):
    def test_a_loop(self) -> None:
        self._arm("extends: arm.yaml\n", "_loop.yaml")
        arm = self._arm("extends: _loop.yaml\n")
        with self.assertRaisesRegex(RunRefused, "loops back on itself"):
            load_config(arm)

    def test_a_base_that_is_not_there(self) -> None:
        with self.assertRaisesRegex(RunRefused, "does not exist"):
            load_config(self._arm("extends: _missing.yaml\n"))

    def test_extends_that_is_not_one_file(self) -> None:
        with self.assertRaisesRegex(RunRefused, "must name one base file"):
            load_config(self._arm("extends: [_base.yaml]\n"))

    def test_running_a_family_base(self) -> None:
        with self.assertRaisesRegex(RunRefused, "is a family base"):
            load_config(self.family / "_base.yaml")

    def test_a_removed_key_in_the_base_is_named_as_if_written_in_the_arm(self) -> None:
        base = yaml.safe_load(BASE)
        base["client"]["local_epochs"] = 1
        (self.family / "_base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
        with self.assertRaisesRegex(RunRefused, r"client\.local_epochs has been removed"):
            load_config(self._arm("extends: _base.yaml\n"))

    def test_a_family_base_is_recognised_by_its_name(self) -> None:
        self.assertTrue(is_family_base("configs/femnist/_base.yaml"))
        self.assertFalse(is_family_base("configs/femnist/fedavg.yaml"))


@pytest.mark.fast
class ShowTest(_Tree):
    def _show(self, path: Path) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            show_config.main(["show", str(path)])
        return out.getvalue()

    def test_it_prints_the_merged_config_with_the_inferred_values_marked(self) -> None:
        arm = self._arm("""
            extends: _base.yaml
            client:
              learning_rate: 0.1
            """)
        text = self._show(arm)
        self.assertIn(f"# extends {self.family / '_base.yaml'}", text)
        self.assertIn("  output_dir: outputs/study/arm  # inferred from the config path", text)
        self.assertIn("  name: study-arm  # inferred from the config path", text)
        printed = yaml.safe_load(text)
        self.assertEqual(printed["client"]["learning_rate"], 0.1)
        self.assertEqual(printed["model"]["hidden_dim"], 8)

    def test_what_it_prints_is_a_config_that_loads_to_the_same_run(self) -> None:
        arm = self._arm("""
            extends: _base.yaml
            client:
              learning_rate: 0.1
            """)
        flat = self.family / "printed.yaml"
        flat.write_text(self._show(arm), encoding="utf-8")
        via_arm, via_printed = asdict(load_config(arm)), asdict(load_config(flat))
        # Stated in the printed file, so no longer inferred there.
        self.assertEqual(via_printed.pop("inferred"), {})
        via_arm.pop("inferred")
        self.assertEqual(via_arm, via_printed)

    def test_it_refuses_what_the_run_would(self) -> None:
        with self.assertRaisesRegex(RunRefused, "is a family base"):
            self._show(self.family / "_base.yaml")

    def test_the_dispatcher_knows_it(self) -> None:
        from fedbrew.cli.dispatch import COMMANDS

        self.assertEqual(COMMANDS["config"][0], "fedbrew.cli.show_config")


@pytest.mark.fast
class TheShippedLayoutTest(unittest.TestCase):
    def test_every_arm_beside_a_base_extends_it_and_every_base_is_extended(self) -> None:
        bases = sorted((REPO / "configs").rglob("_base.yaml"))
        self.assertEqual(len(bases), 10)
        extended = set()
        for base in bases:
            for arm in sorted(base.parent.glob("*.yaml")):
                if is_family_base(arm):
                    continue
                with self.subTest(arm=str(arm.relative_to(REPO))):
                    named = yaml.safe_load(arm.read_text(encoding="utf-8")).get("extends")
                    self.assertEqual(named, "_base.yaml")
                    extended.add(base)
        smooth = REPO / "configs/examples/fed-lasso-smooth/fedavg.yaml"
        self.assertEqual(
            yaml.safe_load(smooth.read_text(encoding="utf-8"))["extends"],
            "../fed-lasso/_base.yaml",
        )
        self.assertEqual(extended, set(bases))


class AShippedArmShownTest(unittest.TestCase):
    """Not fast: loading an example arm runs its extension's reference solve."""

    def test_the_drift_quad_fedavg_arm(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            show_config.main(["show", "examples/drift-quad/fedavg"])
        text = out.getvalue()
        printed = yaml.safe_load(text)
        self.assertEqual(printed["client"]["update_rule"], "fedavg")
        self.assertEqual(printed["numerics"]["matmul_precision"], "highest")
        self.assertIn(
            "  output_dir: outputs/examples/drift-quad/fedavg  # inferred from the config path",
            text,
        )


if __name__ == "__main__":
    unittest.main()
