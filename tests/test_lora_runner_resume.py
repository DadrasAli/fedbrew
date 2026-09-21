"""Runner-level regression for offline tiny-LoRA checkpoints and resume."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from fedbrew.core.runner import run
from tests.test_hf_causal_lm_runtime import (
    _write_federated_fixture,
    _write_local_model_fixture,
)

_HAS_LLM_DEPS = (
    importlib.util.find_spec("transformers") is not None
    and importlib.util.find_spec("peft") is not None
)


@unittest.skipUnless(_HAS_LLM_DEPS, "PEFT and Transformers are optional dependencies")
class LoRARunnerResumeTests(unittest.TestCase):
    def test_tiny_lora_round_checkpoint_and_resume_to_round_two(self) -> None:
        project_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Built here rather than read out of the tree. The paths this used
            # to name live under `data/raw/` and `data/generated/`, which
            # .gitignore excludes, so neither has ever existed in a clone and
            # the two assertTrue calls above turned that into a hard failure on
            # every machine nobody had hand-prepared. Preparing them needs a
            # Hugging Face download, which is the wrong dependency for a test
            # whose subject is that loading works under HF_HUB_OFFLINE=1.
            asset_manifest = _write_local_model_fixture(root / "assets")
            data_manifest = _write_federated_fixture(root / "generated", asset_manifest)
            output_dir = root / "output"
            config_path = _write_run_config(
                root,
                project_root=project_root,
                asset_manifest=asset_manifest,
                data_manifest=data_manifest,
                output_dir=output_dir,
                rounds=1,
            )
            with patch.dict(
                os.environ,
                {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            ):
                first = run(config_path)

            self.assertEqual([record.round_id for record in first.metrics_history], [1])
            checkpoint_path = output_dir / "checkpoints/round_001.pt"
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(checkpoint["model_state_scope"], "adapter")
            self.assertTrue(checkpoint["model_state"])
            self.assertTrue(all("lora_" in name for name in checkpoint["model_state"]))

            resumed_config = _write_run_config(
                root,
                project_root=project_root,
                asset_manifest=asset_manifest,
                data_manifest=data_manifest,
                output_dir=output_dir,
                rounds=2,
                resume_from=checkpoint_path,
            )
            with patch.dict(
                os.environ,
                {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            ):
                resumed = run(resumed_config)

            self.assertEqual(
                [record.round_id for record in resumed.metrics_history],
                [1, 2],
            )
            self.assertTrue((output_dir / "checkpoints/round_002.pt").is_file())
            summary = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["final_round"], 2)
            self.assertEqual(summary["num_rounds"], 2)


def _write_run_config(
    root: Path,
    *,
    project_root: Path,
    asset_manifest: Path,
    data_manifest: Path,
    output_dir: Path,
    rounds: int,
    resume_from: Path | None = None,
) -> Path:
    server_path = root / f"server_{rounds}.yaml"
    server_path.write_text(
        "strategy: fedavg\nparticipation_rate: 1\nmetrics:\n  - fit_loss\n  - fit_accuracy\n",
        encoding="utf-8",
    )
    client_path = root / "client.yaml"
    client_path.write_text(
        "update_rule: local_adamw\n"
        "batch_size: 1\n"
        "learning_rate: 0.0001\n"
        "learning_rate_schedule: constant\n"
        "min_learning_rate: 0.0\n"
        "beta1: 0.9\n"
        "beta2: 0.999\n"
        "epsilon: 1.0e-8\n"
        "weight_decay: 0.0\n"
        "max_local_steps: 1\n"
        "metrics:\n"
        "  - fit_loss\n"
        "  - fit_accuracy\n",
        encoding="utf-8",
    )
    resume_line = f"  resume_from: {resume_from}\n" if resume_from else ""
    config_path = root / f"experiment_{rounds}.yaml"
    config_path.write_text(
        f"""experiment:
  name: tiny_lora_runner_regression
  seed: 42
  output_dir: {output_dir}
server_config: {server_path}
client_config: {client_path}
data:
  name: manifest_dataset
  path: {data_manifest}
model:
  name: hf_causal_lm_lora
  asset_manifest: {asset_manifest}
  preparation_config: {project_root / "configs/llm_assets/tiny_gpt2.yaml"}
  sequence_length: 4
  local_files_only: true
  trust_remote_code: false
  adapter_name: regression
  r: 2
  lora_alpha: 4
  lora_dropout: 0.0
  target_modules:
    - c_attn
  bias: none
runtime:
  deterministic: true
  deterministic_warn_only: true
  device: cpu
  use_amp: false
{resume_line}  checkpointing:
    enabled: true
    interval: 1
    save_last: true
    save_best: true
    best_metric: val_loss_sample_weighted_avg
    keep_last: 2
    save_every_round: true
evaluation:
  train:
    every: 1
    clients: all
  val:
    every: 1
    clients: all
  test:
    every: never
  central_test:
    every: 1
defaults:
  global_rounds: {rounds}
  local_iterations: 1
""",
        encoding="utf-8",
    )
    return config_path


if __name__ == "__main__":
    unittest.main()
