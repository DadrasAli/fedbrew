"""Every setting that changes the numbers is in the numerics block, and only there.

``deterministic``, ``deterministic_warn_only`` and ``use_amp`` sat in
``runtime``, and ``matmul_precision``, ``cudnn_benchmark`` and ``precision``
in ``runtime.performance`` beside keys that change only how long a round
takes. They are ``NumericsConfig`` now (docs/04 section 7.5). Pinned here:

- each old place is refused at load, naming the new one;
- each default is what the key's reader took when a config left it out, so a
  config without the block runs as it did;
- the block is validated like the keys were, and ``precision`` still needs
  the batched executor;
- a run recorded before the block existed is compared on these keys when it
  is resumed, not skipped for lacking them.
"""

from __future__ import annotations

import copy
import tempfile
import textwrap
import unittest
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import yaml

from fedbrew.core.config import NUMERICS_KEYS, load_config, validate_config
from fedbrew.core.refusal import RunRefused
from fedbrew.core.runner import config_differences

pytestmark = pytest.mark.fast

BASE = textwrap.dedent("""
    experiment:
      seed: 1
      output_dir: out
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
      deterministic_warn_only: false
      matmul_precision: highest
      cudnn_benchmark: false
      precision: reference
      use_amp: false
    runtime:
      device: cpu
    schedule:
      rounds: 1
      local_iterations: 1
    """)


class _Directory(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def _load(self, edit: Any = None) -> Any:
        raw = yaml.safe_load(BASE)
        if edit is not None:
            edit(raw)
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return load_config(path)


class EachOldPlaceIsRefusedTest(_Directory):
    OLD = (
        ("runtime", "deterministic", True),
        ("runtime", "deterministic_warn_only", False),
        ("runtime", "use_amp", False),
        ("performance", "matmul_precision", "highest"),
        ("performance", "cudnn_benchmark", False),
        ("performance", "precision", "reference"),
    )

    def test_naming_the_new_place(self) -> None:
        for block, name, value in self.OLD:
            with self.subTest(key=f"{block}.{name}"):

                def edit(
                    raw: dict[str, Any], block: str = block, name: str = name, value: Any = value
                ) -> None:
                    raw["numerics"].pop(name)
                    if block == "runtime":
                        raw["runtime"][name] = value
                    else:
                        raw["runtime"].setdefault("performance", {})[name] = value

                with self.assertRaises(RunRefused) as caught:
                    self._load(edit)
                self.assertIn(f"moved to numerics.{name}", str(caught.exception))

    def test_the_six_are_the_block(self) -> None:
        self.assertEqual(sorted(NUMERICS_KEYS), sorted(name for _, name, _ in self.OLD))


class TheDefaultsAreTheOldReadersTest(_Directory):
    def test_a_config_without_the_block_runs_as_it_did(self) -> None:
        config = self._load(lambda raw: raw.pop("numerics"))
        self.assertEqual(
            asdict(config.numerics),
            {
                # runner: _runtime_extra_bool(config, "deterministic", False) and
                # the same for warn_only; configure_runtime applied matmul and
                # cudnn only when set; the executor read precision with
                # "reference"; use_amp was required, and false is what every
                # shipped config said.
                "deterministic": False,
                "deterministic_warn_only": False,
                "matmul_precision": None,
                "cudnn_benchmark": None,
                "precision": "reference",
                "use_amp": False,
                "extra": {},
            },
        )


class TheBlockIsValidatedTest(_Directory):
    def test_a_bad_value_is_refused(self) -> None:
        for name, value in (
            ("deterministic", "false"),
            ("use_amp", 1),
            ("matmul_precision", "hihg"),
            ("cudnn_benchmark", "false"),
            ("precision", "fp16"),
        ):
            with self.subTest(key=name, value=value):
                config = self._load()
                setattr(config.numerics, name, value)
                with self.assertRaises(RunRefused) as caught:
                    validate_config(config)
                self.assertIn(f"numerics.{name}", str(caught.exception))

    def test_an_unknown_key_in_the_block_is_refused(self) -> None:
        def edit(raw: dict[str, Any]) -> None:
            raw["numerics"]["matmul_precison"] = "high"

        with self.assertRaisesRegex(RunRefused, r"numerics\.matmul_precison"):
            self._load(edit)

    def test_a_step_precision_needs_the_batched_executor(self) -> None:
        def edit(raw: dict[str, Any]) -> None:
            raw["numerics"]["precision"] = "bf16"

        with self.assertRaisesRegex(RunRefused, r"numerics\.precision is a mode of the batched"):
            self._load(edit)

    def test_the_numerics_block_is_what_run_json_records(self) -> None:
        recorded = asdict(self._load())
        self.assertEqual(recorded["numerics"]["matmul_precision"], "highest")
        self.assertNotIn("use_amp", recorded["runtime"])


class ARunRecordedBeforeTheBlockTest(_Directory):
    """Its run.json config holds the six under runtime; a resume still compares them."""

    def _recorded_before(self, config: Any, **old: Any) -> dict[str, Any]:
        recorded = copy.deepcopy(asdict(config))
        numerics = recorded.pop("numerics")
        numerics.pop("extra")
        numerics.update(old)
        runtime = recorded["runtime"]
        runtime["use_amp"] = numerics.pop("use_amp")
        runtime["extra"]["deterministic"] = numerics.pop("deterministic")
        runtime["extra"]["deterministic_warn_only"] = numerics.pop("deterministic_warn_only")
        runtime["extra"]["performance"] = numerics
        return recorded

    def test_the_same_numerics_are_no_difference(self) -> None:
        config = self._load()
        self.assertEqual(config_differences(self._recorded_before(config), config), [])

    def test_a_changed_numerics_key_is_a_difference(self) -> None:
        config = self._load()
        recorded = self._recorded_before(config, matmul_precision="high", use_amp=True)
        self.assertEqual(
            sorted(key for key, _, _ in config_differences(recorded, config)),
            ["numerics.matmul_precision", "numerics.use_amp"],
        )


if __name__ == "__main__":
    unittest.main()
