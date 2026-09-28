"""``runtime.performance.precision`` trains in lower precision, within its measured bound.

Each mode changes only the batched executor's training step; the update
arithmetic and every evaluation stay at the model's precision (chapter 11
§11). What is held, against the batched reference run of the same config:

- ``f32_f64``: a float64 linear example stepped in float32, within ``1e-4``
  per model tensor (of its scale) and per non-timing cell, on the smooth
  examples; measured at most 5.6e-7 and 6.0e-6 over four rounds (pl-1d,
  2026-09-28). fed-lasso's L1 kink amplifies the rounding to 1.1e-3 of the
  model, so it is not held;
- ``bf16``: the MLP's loss under bfloat16 autocast, within ``5e-2`` per model
  tensor and per loss cell; measured 9.5e-3 and 9.1e-3 (a loss spread,
  2026-09-28). Accuracy cells move by whole examples and are not held;
- ``tf32``: the MLP on CUDA, within ``1e-3`` per model tensor and loss cell;
  measured 9.1e-5 and 2.8e-4 (a fit_loss cell) on an A100 (2026-09-28);
- a mode that does not apply -- f32_f64 on a float32 model, bf16 or tf32 on
  a float64 one, tf32 on the CPU -- runs the reference, bit for bit, and
  run.json says why.
"""

from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from fedbrew.core.checkpointing import load_checkpoint
from fedbrew.core.config import load_config
from fedbrew.core.refusal import RunRefused
from fedbrew.core.settings_group import run_group
from tests.test_batched_executor_tolerance import (
    CSVS,
    ExecutorRuns,
    classification_rule_config,
    example_config,
)
from tests.test_fed_lasso_evaluates_on_its_device import _cuda_usable
from tests.test_reproducibility import TIMING

#: f32_f64 on the smooth linear examples.
F32_F64_TOLERANCE = 1e-4
#: bf16 on the MLP, per model tensor and loss cell.
BF16_TOLERANCE = 5e-2
#: tf32 on the MLP on CUDA, per model tensor and loss cell.
TF32_TOLERANCE = 1e-3
SMOOTH = ("fed-lasso-l2", "simplex-lsq", "pl-1d")


def _mlp() -> dict[str, Any]:
    return classification_rule_config({"update_rule": "local_sgd"})


def executor_record(output: Path) -> dict[str, Any]:
    return json.loads((output / "run.json").read_text())["reproducibility"]["executor"]


class ModeRuns(ExecutorRuns):
    def reference_and(self, config: dict[str, Any], **performance: Any) -> tuple[Path, Path]:
        """(the mode's run, the batched reference run) of one config."""

        return (
            self.run_config(config, "batched", **performance),
            self.run_config(config, "batched"),
        )

    def assertTrainedOtherwise(self, mode: Path, reference: Path) -> None:
        """The mode ran: its final model is not the reference's, bit for bit."""

        ours = load_checkpoint(mode / "checkpoints" / "latest.pt")["model_state"]
        theirs = load_checkpoint(reference / "checkpoints" / "latest.pt")["model_state"]
        self.assertFalse(all(torch.equal(ours[key], value) for key, value in theirs.items()))

    def assertLossesAgree(self, mode: Path, reference: Path, tolerance: float) -> None:
        """Every model tensor, and every loss cell, within ``tolerance`` relative."""

        for name in CSVS:
            if not (reference / name).exists():
                continue
            with (reference / name).open() as left, (mode / name).open() as right:
                for row_r, row_m in zip(csv.DictReader(left), csv.DictReader(right), strict=True):
                    for column, value in row_r.items():
                        if "loss" not in column or column in TIMING or value == row_m[column]:
                            continue
                        a, b = float(row_m[column]), float(value)
                        error = abs(a - b) / max(abs(b), 1e-300)
                        self.assertLessEqual(error, tolerance, f"{name} {column}")
        for path in sorted((reference / "checkpoints").glob("round_*.pt")):
            ours = load_checkpoint(mode / "checkpoints" / path.name)["model_state"]
            for key, tensor in load_checkpoint(path)["model_state"].items():
                scale = max(float(tensor.abs().max()), 1e-300)
                error = float((ours[key] - tensor).abs().max()) / scale
                self.assertLessEqual(error, tolerance, f"{path.name} {key}")


class F32F64Test(ModeRuns):
    def test_the_smooth_examples_within_the_bound(self) -> None:
        for name in SMOOTH:
            with self.subTest(example=name):
                mode, reference = self.reference_and(example_config(name), precision="f32_f64")
                self.assertAgree(mode, reference, tolerance=F32_F64_TOLERANCE)
                self.assertTrainedOtherwise(mode, reference)
                self.assertEqual(executor_record(mode)["precision"], {"used": "f32_f64"})

    def test_a_group_steps_each_setting_as_alone(self) -> None:
        base = copy.deepcopy(example_config("fed-lasso-l2"))
        base["runtime"].setdefault("performance", {}).update(
            executor="batched", precision="f32_f64"
        )
        configs = []
        for rate in (0.004, 0.008):
            config = copy.deepcopy(base)
            config["client"]["learning_rate"] = rate
            configs.append(config)
        alone = [self.run_config(config, "batched", precision="f32_f64") for config in configs]
        paths = []
        for number, config in enumerate(configs):
            config["experiment"]["output_dir"] = str(self.root / f"setting{number}")
            path = self.root / f"setting{number}.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            paths.append(path)
        outcomes = run_group(paths, ["client.learning_rate"])
        self.assertEqual([outcome.status for outcome in outcomes], ["completed"] * 2)
        for number, reference in enumerate(alone):
            self.assertAgree(self.root / f"setting{number}", reference, exact=True)


class Bf16Test(ModeRuns):
    def test_the_mlp_within_the_bound(self) -> None:
        mode, reference = self.reference_and(_mlp(), precision="bf16")
        self.assertLossesAgree(mode, reference, BF16_TOLERANCE)
        self.assertTrainedOtherwise(mode, reference)
        self.assertEqual(executor_record(mode)["precision"], {"used": "bf16"})


class Tf32Test(ModeRuns):
    @pytest.mark.cuda
    @unittest.skipUnless(_cuda_usable(), "needs a usable CUDA device")
    def test_the_mlp_on_cuda_within_the_bound(self) -> None:
        config = _mlp()
        config["runtime"]["device"] = "cuda"
        mode, reference = self.reference_and(config, precision="tf32")
        self.assertLossesAgree(mode, reference, TF32_TOLERANCE)
        self.assertTrainedOtherwise(mode, reference)
        self.assertEqual(executor_record(mode)["precision"], {"used": "tf32"})


class AModeThatDoesNotApplyRunsTheReferenceTest(ModeRuns):
    def test_each(self) -> None:
        for label, config, precision, why in (
            ("f32_f64 on float32", _mlp(), "f32_f64", "this model is torch.float32"),
            ("bf16 on float64", example_config("fed-lasso-l2"), "bf16", "torch.float64"),
            ("tf32 on the CPU", _mlp(), "tf32", "this run is on cpu"),
        ):
            with self.subTest(mode=label):
                config = copy.deepcopy(config)
                config["runtime"]["device"] = "cpu"
                mode, reference = self.reference_and(config, precision=precision)
                self.assertAgree(mode, reference, exact=True)
                record = executor_record(mode)["precision"]
                self.assertEqual(record["used"], "reference")
                self.assertIn(why, record["fallback"])


class ConfigTest(unittest.TestCase):
    def _load(self, **performance: Any) -> None:
        config = example_config("fed-lasso")
        config["runtime"].setdefault("performance", {}).update(performance)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            load_config(path)

    def test_a_mode_needs_the_batched_executor(self) -> None:
        with self.assertRaisesRegex(RunRefused, "executor: batched"):
            self._load(precision="bf16")
        self._load(precision="reference")

    def test_an_unknown_precision_is_refused(self) -> None:
        with self.assertRaisesRegex(RunRefused, "precision must be one of"):
            self._load(executor="batched", precision="fp16")


if __name__ == "__main__":
    unittest.main()
