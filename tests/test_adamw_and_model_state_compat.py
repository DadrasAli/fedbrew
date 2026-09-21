"""Focused tests for local AdamW and model-state aggregation compatibility."""

from __future__ import annotations

import contextlib
import unittest
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
import torch
from cpu_only import no_accelerator
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from fedbrew.clients.torch_adamw_client import TorchAdamWClient
from fedbrew.core.protocol import FitRequest
from fedbrew.core.registry import client_updates, register_builtin_components
from fedbrew.core.torch_utils import WeightedStateAccumulator, get_model_state
from fedbrew.tasks.base import TaskAdapter


class _TokenCountingTask(TaskAdapter):
    """Small task stub whose evaluation count excludes zero-valued targets."""

    def build_model(self, config: Mapping[str, Any]) -> nn.Module:
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.5)
        return model

    def build_dataloader(
        self,
        data: Any,
        config: Mapping[str, Any],
    ) -> DataLoader[tuple[Tensor, Tensor]]:
        if not isinstance(data, Mapping):
            raise TypeError("test data must be a mapping")
        features = data["x"]
        targets = data["y"]
        if not isinstance(features, Tensor) or not isinstance(targets, Tensor):
            raise TypeError("test data values must be tensors")
        return DataLoader(
            TensorDataset(features, targets),
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
        loss = sum(parameter.square().sum() for parameter in model.parameters())
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.detach().item())}

    def eval_step(self, model: nn.Module, batch: Any) -> dict[str, float]:
        targets = batch[1]
        total = int((targets != 0).sum().item())
        loss = float(
            sum(parameter.square().sum() for parameter in model.parameters()).detach().item()
        )
        return {"loss": loss, "correct": float(total), "total": float(total)}

    def compute_metrics(self, outputs: Sequence[Any]) -> dict[str, float]:
        records = [output for output in outputs if isinstance(output, dict)]
        total = sum(float(record["total"]) for record in records)
        if total == 0.0:
            return {"loss": 0.0, "accuracy": 0.0}
        loss = sum(float(record["loss"]) * float(record["total"]) for record in records)
        correct = sum(float(record["correct"]) for record in records)
        return {"loss": loss / total, "accuracy": correct / total}


class LocalAdamWClientTests(unittest.TestCase):
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

    def _build_client(self) -> TorchAdamWClient:
        return TorchAdamWClient(
            client_id="client-0",
            task=_TokenCountingTask(),
            model_config={},
            client_data={
                "train": {
                    "x": torch.ones((2, 3), dtype=torch.long),
                    "y": torch.tensor([[2, 3, 0], [4, 0, 5]], dtype=torch.long),
                }
            },
            local_iterations=1,
            batch_size=2,
            learning_rate=0.002,
            weight_decay=0.03,
            beta1=0.8,
            beta2=0.95,
            epsilon=1e-7,
            learning_rate_schedule="constant",
            min_learning_rate=0.0,
            total_rounds=1,
            metrics=["fit_loss", "fit_accuracy"],
        )

    @pytest.mark.fast
    def test_optimizer_uses_configured_adamw_hyperparameters(self) -> None:
        client = self._build_client()

        optimizer = client._build_optimizer(client.task.build_model({}), round_id=1)

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        group = optimizer.param_groups[0]
        self.assertEqual(group["lr"], 0.002)
        self.assertEqual(group["weight_decay"], 0.03)
        self.assertEqual(group["betas"], (0.8, 0.95))
        self.assertEqual(group["eps"], 1e-7)

    def test_fit_num_examples_counts_evaluated_non_padding_targets(self) -> None:
        client = self._build_client()
        initial_state = get_model_state(client.task.build_model({}))

        result = client.fit(
            FitRequest(
                round_id=1,
                client_id="client-0",
                payload={"model_state": initial_state},
            )
        )

        self.assertEqual(result.num_examples, 4)
        self.assertEqual(result.metrics["fit_accuracy"], 1.0)

    def test_fit_preserves_zero_count_for_all_padding_targets(self) -> None:
        client = self._build_client()
        client.client_data = {
            "train": {
                "x": torch.ones((1, 3), dtype=torch.long),
                "y": torch.zeros((1, 3), dtype=torch.long),
            }
        }
        initial_state = get_model_state(client.task.build_model({}))

        result = client.fit(
            FitRequest(
                round_id=1,
                client_id="client-0",
                payload={"model_state": initial_state},
            )
        )

        self.assertEqual(result.num_examples, 0)

    @pytest.mark.fast
    def test_registry_exposes_local_adamw_lazily(self) -> None:
        register_builtin_components()

        self.assertTrue(client_updates.exists("local_adamw"))


def _accumulate(states, weights=None):
    """Fold states through the accumulator the servers aggregate with."""

    accumulator = WeightedStateAccumulator()
    for index, state in enumerate(states):
        accumulator.add(state, 1.0 if weights is None else weights[index])
    return accumulator.result()


@pytest.mark.fast
class ModelStateAggregationCompatibilityTests(unittest.TestCase):
    def test_averages_float_tensors_and_preserves_non_floating_buffers(self) -> None:
        first = {
            "weight": torch.tensor([1.0, 3.0]),
            "step": torch.tensor(7, dtype=torch.long),
            "mask": torch.tensor([True, False]),
        }
        second = {
            "weight": torch.tensor([3.0, 7.0]),
            "step": torch.tensor(7, dtype=torch.long),
            "mask": torch.tensor([True, False]),
        }

        averaged = _accumulate([first, second], weights=[1.0, 3.0])

        torch.testing.assert_close(averaged["weight"], torch.tensor([2.5, 6.0]))
        self.assertTrue(torch.equal(averaged["step"], first["step"]))
        self.assertTrue(torch.equal(averaged["mask"], first["mask"]))
        self.assertEqual(averaged["step"].dtype, torch.long)
        self.assertNotEqual(averaged["step"].data_ptr(), first["step"].data_ptr())

    def test_rejects_incompatible_floating_tensor_dtype_or_shape(self) -> None:
        incompatible_values = {
            "dtype": torch.tensor([2.0], dtype=torch.float64),
            "shape": torch.tensor([2.0, 3.0], dtype=torch.float32),
        }
        for mismatch, value in incompatible_values.items():
            with self.subTest(mismatch=mismatch):
                states = [
                    {"weight": torch.tensor([1.0], dtype=torch.float32)},
                    {"weight": value},
                ]

                with self.assertRaisesRegex(
                    ValueError,
                    "state tensor 'weight' must have matching dtype and shape",
                ):
                    _accumulate(states)

    def test_rejects_different_non_floating_buffers_with_key_in_error(self) -> None:
        states = [
            {"weight": torch.tensor([1.0]), "step": torch.tensor(1)},
            {"weight": torch.tensor([2.0]), "step": torch.tensor(2)},
        ]

        with self.assertRaisesRegex(
            ValueError,
            "non-floating state tensor 'step' differs between clients",
        ):
            _accumulate(states)


if __name__ == "__main__":
    unittest.main()
