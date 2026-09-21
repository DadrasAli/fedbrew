"""Gradient clipping bounds every client's local update."""

from __future__ import annotations

import math
import unittest
from collections.abc import Mapping
from typing import Any

import pytest
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, TensorDataset

from fedbrew.clients.local_update_modes import run_sgd_update_mode
from fedbrew.tasks.base import TaskAdapter

LEARNING_RATE = 0.5
MAX_GRAD_NORM = 1.0


class _HugeGradientTask(TaskAdapter):
    """Linear task whose targets are large enough to explode the gradient."""

    def build_model(self, config: Mapping[str, Any] | None = None) -> nn.Module:
        torch.manual_seed(0)
        model = nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        return model

    def build_dataloader(self, data: Any, config: Mapping[str, Any]) -> DataLoader[Any]:
        return DataLoader(data, batch_size=int(config.get("batch_size", 2)))

    def train_step(
        self,
        model: nn.Module,
        batch: Any,
        optimizer: optim.Optimizer | None = None,
    ) -> dict[str, float]:
        if optimizer is None:
            raise ValueError("optimizer is required")
        features, targets = batch
        optimizer.zero_grad()
        loss = ((model(features).squeeze(-1) - targets) ** 2).mean()
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.detach()), "correct": 0.0, "total": 1.0}

    def eval_step(self, model: nn.Module, batch: Any) -> dict[str, float]:
        raise NotImplementedError

    def compute_metrics(self, outputs: Any, targets: Tensor | None = None) -> dict[str, float]:
        return {"loss": 0.0, "accuracy": 0.0}


def _update_norm(max_grad_norm: float | None, mode: str) -> float:
    task = _HugeGradientTask()
    model = task.build_model()
    features = torch.full((4, 4), 10.0)
    targets = torch.full((4,), 1000.0)
    loader = task.build_dataloader(TensorDataset(features, targets), {"batch_size": 4})
    run_sgd_update_mode(
        task=task,
        model=model,
        train_loader=loader,
        local_iterations=1,
        learning_rate=LEARNING_RATE,
        update_mode=mode,
        frozen_gradient_weighting="examples",
        client_id="c0",
        max_grad_norm=max_grad_norm,
    )
    return float(torch.linalg.vector_norm(model.weight.detach()))


class GradientClippingTest(unittest.TestCase):
    def test_clipping_bounds_the_update_in_every_mode(self) -> None:
        for mode in ("single_batch", "sequential_epoch", "frozen_batch_gradients"):
            with self.subTest(mode=mode):
                unclipped = _update_norm(None, mode)
                clipped = _update_norm(MAX_GRAD_NORM, mode)
                # One SGD step from zero weights gives ||delta|| = lr * ||grad||,
                # so clipping the norm to 1.0 caps the update at the learning rate.
                self.assertLessEqual(clipped, LEARNING_RATE * MAX_GRAD_NORM + 1e-5)
                self.assertGreater(unclipped, clipped * 10)

    def test_clipping_is_off_by_default(self) -> None:
        self.assertGreater(_update_norm(None, "sequential_epoch"), 1.0)

    @pytest.mark.fast
    def test_a_non_positive_threshold_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_grad_norm must be positive"):
            _update_norm(0.0, "single_batch")

    def test_clipped_update_is_finite(self) -> None:
        self.assertTrue(math.isfinite(_update_norm(MAX_GRAD_NORM, "sequential_epoch")))


if __name__ == "__main__":
    unittest.main()
