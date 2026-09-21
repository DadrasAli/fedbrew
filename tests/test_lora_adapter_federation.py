"""Focused offline tests for LoRA adapter-only federated state and AdamW."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import math
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from cpu_only import no_accelerator
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from fedbrew.clients.torch_adamw_client import TorchAdamWClient
from fedbrew.core.factory import _add_causal_manifest_metadata
from fedbrew.core.protocol import FitRequest, FitResult, RoundInfo
from fedbrew.core.torch_utils import get_model_state
from fedbrew.models.hf_causal_lm_lora import (
    HFCausalLMLoRAConfigError,
    build_hf_causal_lm_lora,
)
from fedbrew.servers.fedavg import FedAvgServer
from fedbrew.tasks.base import TaskAdapter
from fedbrew.tasks.causal_lm import TorchCausalLMTask

_HAS_LLM_DEPS = (
    importlib.util.find_spec("transformers") is not None
    and importlib.util.find_spec("peft") is not None
)


@unittest.skipUnless(_HAS_LLM_DEPS, "PEFT and Transformers are optional dependencies")
class LoRAAdapterFederationTests(unittest.TestCase):
    @pytest.mark.fast
    def test_freezes_base_and_round_trips_only_adapter_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _lora_config(_write_model_fixture(Path(directory)))
            task = TorchCausalLMTask(model_config=config)
            first = build_hf_causal_lm_lora(config)

            trainable_names = [
                name for name, parameter in first.named_parameters() if parameter.requires_grad
            ]
            self.assertTrue(trainable_names)
            self.assertTrue(all("lora_" in name for name in trainable_names))

            original = task.get_federated_model_state(first)
            self.assertTrue(original)
            self.assertTrue(all("lora_" in name for name in original))
            self.assertTrue(all(tensor.device.type == "cpu" for tensor in original.values()))
            self.assertTrue(all(not tensor.requires_grad for tensor in original.values()))

            partial = dict(original)
            partial.pop(next(iter(partial)))
            with self.assertRaisesRegex(ValueError, "missing"):
                task.load_federated_model_state(first, partial)

            changed = {
                name: tensor + float(index + 1)
                for index, (name, tensor) in enumerate(original.items())
            }
            second = build_hf_causal_lm_lora(config)
            task.load_federated_model_state(second, changed)
            restored = task.get_federated_model_state(second)
            for name in changed:
                torch.testing.assert_close(restored[name], changed[name])

            metadata = task.federated_model_state_metadata(second)
            self.assertEqual(metadata["model_state_scope"], "adapter")
            self.assertEqual(metadata["adapter_name"], "federated")
            self.assertLess(
                metadata["communicated_parameters"],
                metadata["total_parameters"],
            )
            self.assertEqual(
                metadata["communicated_parameters"],
                metadata["trainable_parameters"],
            )

    @pytest.mark.fast
    def test_rejects_target_that_matches_no_base_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _lora_config(_write_model_fixture(Path(directory)))
            config["target_modules"] = ["definitely_missing_projection"]

            with self.assertRaisesRegex(
                HFCausalLMLoRAConfigError,
                "do not match base-model modules",
            ):
                build_hf_causal_lm_lora(config)

    @pytest.mark.fast
    def test_rejects_bias_modes_that_unfreeze_base_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _lora_config(_write_model_fixture(Path(directory)))
            config["bias"] = "all"

            with self.assertRaisesRegex(
                HFCausalLMLoRAConfigError,
                "requires model.bias=none",
            ):
                build_hf_causal_lm_lora(config)

    @pytest.mark.fast
    def test_fedavg_averages_adapters_and_rejects_mixed_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _lora_config(_write_model_fixture(Path(directory)))
            task = TorchCausalLMTask(model_config=config)
            server = FedAvgServer(
                participation_rate=1.0,
                seed=0,
                task=task,
                model_config=config,
                metrics=["loss"],
            )
            initial = server.initialize()
            metadata = initial["model_state_metadata"]

            bad_metadata = dict(metadata)
            bad_metadata["model_state_scope"] = "full"
            bad_result = FitResult(
                round_id=1,
                client_id="bad",
                num_examples=1,
                payload={
                    "model_state": initial["model_state"],
                    "model_state_scope": "full",
                    "model_state_metadata": bad_metadata,
                },
            )
            with self.assertRaisesRegex(ValueError, "incompatible model state scopes"):
                server.aggregate(RoundInfo(round_id=1), [bad_result])

            first_state = {
                name: torch.ones_like(tensor) for name, tensor in initial["model_state"].items()
            }
            second_state = {
                name: torch.full_like(tensor, 3.0)
                for name, tensor in initial["model_state"].items()
            }
            results = [
                FitResult(
                    round_id=1,
                    client_id="one",
                    num_examples=1,
                    payload={
                        "model_state": first_state,
                        "model_state_scope": "adapter",
                        "model_state_metadata": metadata,
                    },
                    metrics={"loss": 2.0},
                ),
                FitResult(
                    round_id=1,
                    client_id="two",
                    num_examples=3,
                    payload={
                        "model_state": second_state,
                        "model_state_scope": "adapter",
                        "model_state_metadata": metadata,
                    },
                    metrics={"loss": 1.0},
                ),
            ]
            aggregated = server.aggregate(RoundInfo(round_id=1), results)
            for tensor in aggregated["model_state"].values():
                torch.testing.assert_close(tensor, torch.full_like(tensor, 2.5))
            self.assertEqual(aggregated["model_state_scope"], "adapter")

            checkpoint_state = server.save_state()
            self.assertEqual(checkpoint_state["model_state_scope"], "adapter")
            self.assertTrue(all("lora_" in name for name in checkpoint_state["model_state"]))

            resumed = FedAvgServer(
                participation_rate=1.0,
                seed=0,
                task=task,
                model_config=config,
            )
            resumed.load_state(checkpoint_state)
            resumed_payload = resumed.initialize()
            for name, tensor in aggregated["model_state"].items():
                torch.testing.assert_close(
                    resumed_payload["model_state"][name],
                    tensor,
                )

            incompatible_config = dict(config)
            incompatible_config["adapter_name"] = "different"
            incompatible_task = TorchCausalLMTask(model_config=incompatible_config)
            incompatible = FedAvgServer(
                participation_rate=1.0,
                seed=0,
                task=incompatible_task,
                model_config=incompatible_config,
            )
            incompatible.load_state(checkpoint_state)
            with self.assertRaisesRegex(ValueError, "adapter_name"):
                incompatible.initialize()

    def test_sft_client_weight_uses_processed_active_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _lora_config(_write_model_fixture(Path(directory)))
            config.update(
                dataset_task="causal_lm_sft",
                dataset_ignore_index=-100,
                pad_token_id=None,
            )
            task = TorchCausalLMTask(
                model_config=config,
                batch_size=1,
                device="cpu",
            )
            initial_model = task.build_model(config)
            initial_state = task.get_federated_model_state(initial_model)
            metadata = task.federated_model_state_metadata(initial_model)
            client = TorchAdamWClient(
                client_id="client",
                task=task,
                model_config=config,
                client_data={
                    "train": {
                        "x": torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]], dtype=torch.long),
                        "y": torch.tensor(
                            [[-100, -100, 4, 5], [-100, 7, 8, 9]],
                            dtype=torch.long,
                        ),
                    }
                },
                local_iterations=3,
                batch_size=1,
                learning_rate=1e-3,
                weight_decay=0.0,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
                learning_rate_schedule="constant",
                min_learning_rate=0.0,
                total_rounds=1,
                max_local_steps=1,
                train_shuffle=False,
            )

            result = client.fit(
                FitRequest(
                    round_id=1,
                    client_id="client",
                    payload={
                        "model_state": initial_state,
                        "model_state_scope": "adapter",
                        "model_state_metadata": metadata,
                    },
                )
            )

            self.assertEqual(result.num_examples, 2)
            self.assertEqual(result.metrics["optimizer_steps"], 1.0)
            self.assertEqual(result.metrics["active_target_tokens"], 2.0)
            self.assertEqual(
                result.metrics["communicated_parameters"],
                float(metadata["communicated_parameters"]),
            )
            self.assertTrue(math.isfinite(result.metrics["fit_loss"]))


@pytest.mark.fast
class CausalSFTMetricTests(unittest.TestCase):
    def test_factory_propagates_sft_task_and_ignore_index(self) -> None:
        model_config: dict[str, Any] = {}

        _add_causal_manifest_metadata(
            model_config,
            {"task": "causal_lm_sft", "ignore_index": -100},
        )

        self.assertEqual(model_config["dataset_task"], "causal_lm_sft")
        self.assertEqual(model_config["dataset_ignore_index"], -100)

    def test_minus_100_is_ignored_for_loss_accuracy_and_weight(self) -> None:
        task = TorchCausalLMTask(
            model_config={
                "pad_token_id": None,
                "dataset_task": "causal_lm_sft",
                "dataset_ignore_index": -100,
            }
        )
        targets = torch.tensor([[-100, 2, 3]], dtype=torch.long)
        logits = torch.zeros((1, 3, 5), dtype=torch.float32)
        logits[0, 1, 2] = 5.0
        logits[0, 2, 3] = 5.0

        loss, correct, total = task._loss_and_counts(logits, targets)

        self.assertTrue(math.isfinite(float(loss.item())))
        self.assertEqual(correct, 2)
        self.assertEqual(total, 2)
        self.assertEqual(
            task.federated_aggregation_weight(
                [{"total": 2.0}, {"total": 3.0}],
                evaluated_num_examples=99,
            ),
            5,
        )

        legacy = TorchCausalLMTask(model_config={"pad_token_id": None})
        self.assertEqual(
            legacy.federated_aggregation_weight(
                [{"total": 2.0}],
                evaluated_num_examples=99,
            ),
            99,
        )


class AdamWTrainableParameterTests(unittest.TestCase):
    def setUp(self) -> None:
        # These build CPU tensors and step a CPU optimizer, so nothing here
        # concerns a GPU -- but torch's optimizer probes the current
        # accelerator on every step, which raises on a host whose device is
        # visible and unusable. See tests/cpu_only.py.
        #
        # ExitStack rather than TestCase.enterContext, which is 3.11+ while
        # pyproject declares a 3.10 floor.
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(no_accelerator())

    @pytest.mark.fast
    def test_optimizer_uses_only_trainable_parameters_and_rejects_none(self) -> None:
        task = _CountingTask()
        client = _counting_client(task)
        model = task.build_model({})
        optimizer = client._build_optimizer(model, round_id=1)
        optimized_ids = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        expected_ids = {
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        }
        self.assertEqual(optimized_ids, expected_ids)

        for parameter in model.parameters():
            parameter.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "no trainable parameters"):
            client._build_optimizer(model, round_id=1)

    def test_max_local_steps_and_diagnostics_are_reported(self) -> None:
        task = _CountingTask()
        client = _counting_client(task)
        initial = get_model_state(task.build_model({}))

        result = client.fit(
            FitRequest(
                round_id=1,
                client_id="client",
                payload={"model_state": initial},
            )
        )

        self.assertEqual(result.num_examples, 5)
        self.assertEqual(result.metrics["optimizer_steps"], 2.0)
        self.assertEqual(result.metrics["active_target_tokens"], 2.0)
        self.assertEqual(result.metrics["trainable_parameters"], 1.0)
        torch.testing.assert_close(
            result.payload["model_state"]["weight"],
            initial["weight"],
        )


class _CountingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[0.5]]), requires_grad=False)
        self.bias = nn.Parameter(torch.tensor([0.1]))

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs @ self.weight + self.bias


class _CountingTask(TaskAdapter):
    def build_model(self, config: Mapping[str, Any]) -> nn.Module:
        del config
        return _CountingModel()

    def build_dataloader(
        self,
        data: Any,
        config: Mapping[str, Any],
    ) -> DataLoader[tuple[Tensor, Tensor]]:
        return DataLoader(
            TensorDataset(data["x"], data["y"]),  # type: ignore[arg-type]
            batch_size=int(config["batch_size"]),
            shuffle=bool(config.get("shuffle", False)),
        )

    def train_step(
        self,
        model: nn.Module,
        batch: Any,
        optimizer: Optimizer | None = None,
    ) -> dict[str, float]:
        if optimizer is None:
            raise ValueError("optimizer is required")
        optimizer.zero_grad()
        outputs = model(batch[0])
        loss = (outputs - batch[1]).square().mean()
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.detach().item()), "total": float(batch[1].numel())}

    def eval_step(self, model: nn.Module, batch: Any) -> dict[str, float]:
        with torch.no_grad():
            outputs = model(batch[0])
            loss = (outputs - batch[1]).square().mean()
        return {
            "loss": float(loss.item()),
            "correct": 0.0,
            "total": float(batch[1].numel()),
        }

    def compute_metrics(self, outputs: Sequence[Any]) -> dict[str, float]:
        total = sum(float(output["total"]) for output in outputs)
        loss = sum(float(output["loss"]) * float(output["total"]) for output in outputs)
        return {"loss": loss / total, "accuracy": 0.0}


def _counting_client(task: TaskAdapter) -> TorchAdamWClient:
    return TorchAdamWClient(
        client_id="client",
        task=task,
        model_config={},
        client_data={
            "train": {
                "x": torch.arange(5, dtype=torch.float32).reshape(5, 1),
                "y": torch.ones((5, 1), dtype=torch.float32),
            }
        },
        local_iterations=3,
        batch_size=1,
        learning_rate=1e-2,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        learning_rate_schedule="constant",
        min_learning_rate=0.0,
        total_rounds=1,
        max_local_steps=2,
        train_shuffle=False,
    )


def _lora_config(manifest_path: Path) -> dict[str, Any]:
    return {
        "name": "hf_causal_lm_lora",
        "asset_manifest": str(manifest_path),
        "preparation_config": "configs/llm_assets/tiny_gpt2.yaml",
        "sequence_length": 4,
        "local_files_only": True,
        "trust_remote_code": False,
        "adapter_name": "federated",
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": ["c_attn"],
        "bias": "none",
    }


def _write_model_fixture(root: Path) -> Path:
    from transformers import GPT2Config, GPT2LMHeadModel

    model_path = root / "model"
    tokenizer_path = root / "tokenizer"
    tokenizer_path.mkdir(parents=True)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_positions=8,
            n_ctx=8,
            n_embd=8,
            n_layer=1,
            n_head=2,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=None,
            tie_word_embeddings=True,
        )
    )
    model.save_pretrained(model_path)
    manifest_path = root / "asset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_identifier": "local/tiny-gpt2",
                "tokenizer_identifier": "local/tiny-gpt2",
                "requested_revision": "a" * 40,
                "resolved_revision": "a" * 40,
                "cache_path": str(root),
                "model_path": "model",
                "tokenizer_path": "tokenizer",
                "vocabulary_size": 32,
                "model_type": "gpt2",
                "preparation_timestamp": "2026-01-01T00:00:00+00:00",
                "trust_remote_code": False,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


if __name__ == "__main__":
    unittest.main()
