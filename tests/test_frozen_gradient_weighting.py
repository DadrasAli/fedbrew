"""frozen_batch_gradients normalises by the examples it actually differentiated.

The "examples" weighting is defined as the example-weighted mean gradient over
the train split: the batch weights have to sum to one, or the mode applies a
scaled-down step and calls it the mean. The denominator used to be the client's
``num_examples``, which is the train *and* eval count, so on FEMNIST the weights
summed to 0.90 and on the synthetic dev dataset to 0.50 -- the configured
learning rate, quietly halved.

That definition is the gradient of the pass only when each batch's loss is a
mean over its examples. The causal-LM loss is a mean over active target
tokens, and there the combined update was 53.5% from the gradient of one batch
holding the whole pass (FINDINGS.csv POST-F19). The mode is refused on such a
task now: at load for the built-in tasks in `NON_EXAMPLE_MEAN_TASKS`, and in
both engines for any task whose class overrides `train_loss_denominator`.
`RefusedWhereTheLossIsNotAnExampleMeanTest` holds the set to the task classes.

The weighting is read by that one mode, and `run_sgd_update_mode` required it
under all four (FINDINGS.csv POST-F21). `RequiredOnlyWhereItIsReadTest` runs
the other three without one and the frozen mode refused without one.
"""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, TensorDataset

from fedbrew.clients.local_update_modes import run_delta_sgd_update_mode, run_sgd_update_mode
from fedbrew.core import registry
from fedbrew.core.config import NON_EXAMPLE_MEAN_TASKS, FullConfig, load_config
from fedbrew.core.refusal import RunRefused
from fedbrew.tasks.base import TaskAdapter, loss_averages_over_examples

REPO_ROOT = Path(__file__).resolve().parent.parent

LEARNING_RATE = 0.25


class _LinearTask(TaskAdapter):
    """Least squares on a linear model: the mean gradient is exactly known."""

    def build_model(self, config: Mapping[str, Any] | None = None) -> nn.Module:
        model = nn.Linear(3, 1, bias=False)
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


def _dataset(rows: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(0)
    features = torch.randn(rows, 3, generator=generator)
    targets = torch.randn(rows, generator=generator)
    return TensorDataset(features, targets)


def _full_batch_mean_gradient(data: TensorDataset) -> Tensor:
    """The gradient the "examples" weighting is supposed to reproduce."""

    task = _LinearTask()
    model = task.build_model()
    features, targets = data.tensors
    loss = ((model(features).squeeze(-1) - targets) ** 2).mean()
    loss.backward()
    assert model.weight.grad is not None
    return model.weight.grad.detach().clone()


def _applied_update(data: TensorDataset, batch_size: int, weighting: str) -> Tensor:
    task = _LinearTask()
    model = task.build_model()
    loader = task.build_dataloader(data, {"batch_size": batch_size})
    run_sgd_update_mode(
        task=task,
        model=model,
        train_loader=loader,
        local_iterations=1,
        learning_rate=LEARNING_RATE,
        update_mode="frozen_batch_gradients",
        frozen_gradient_weighting=weighting,
        client_id="c0",
    )
    # The model started at zero, so -w / lr is the combined gradient applied.
    return -model.weight.detach() / LEARNING_RATE


class FrozenGradientWeightingTest(unittest.TestCase):
    def test_examples_weighting_reproduces_the_full_batch_mean_gradient(self) -> None:
        """Equal-sized batches: the scale factor used to be n_train/n_total."""

        data = _dataset(8)
        torch.testing.assert_close(
            _applied_update(data, batch_size=2, weighting="examples"),
            _full_batch_mean_gradient(data),
        )

    def test_a_ragged_last_batch_is_still_the_mean(self) -> None:
        """7 rows at batch_size 2 is 2+2+2+1, so uniform weighting would differ."""

        data = _dataset(7)
        applied = _applied_update(data, batch_size=2, weighting="examples")
        torch.testing.assert_close(applied, _full_batch_mean_gradient(data))
        self.assertFalse(
            torch.allclose(applied, _applied_update(data, 2, "uniform")),
            "the ragged batch must make examples and uniform weighting differ, "
            "or this test cannot tell them apart",
        )

    def test_the_step_does_not_depend_on_how_the_epoch_is_batched(self) -> None:
        data = _dataset(12)
        one_batch = _applied_update(data, batch_size=12, weighting="examples")
        for batch_size in (1, 3, 5):
            with self.subTest(batch_size=batch_size):
                torch.testing.assert_close(
                    _applied_update(data, batch_size, weighting="examples"),
                    one_batch,
                )

    def test_the_other_weightings_keep_their_own_meanings(self) -> None:
        """Only "examples" is a mean; sum and uniform are defined differently."""

        data = _dataset(8)
        mean_gradient = _full_batch_mean_gradient(data)
        # 4 equal batches, so uniform is the same mean by a different route.
        torch.testing.assert_close(
            _applied_update(data, batch_size=2, weighting="uniform"), mean_gradient
        )
        # "sum" adds the four batch gradients: four times the mean.
        torch.testing.assert_close(
            _applied_update(data, batch_size=2, weighting="sum"), mean_gradient * 4.0
        )


class _TokenMeanTask(_LinearTask):
    """A task that declares its own loss denominator, as the causal-LM one does."""

    def train_loss_denominator(self, batch: Any, output: Mapping[str, float]) -> float:
        return float(output["total"])


def _load_with_mode(source: str, mode: str) -> FullConfig:
    document = yaml.safe_load((REPO_ROOT / source).read_text(encoding="utf-8"))
    document["client"]["update_mode"] = mode
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "config.yaml"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        return load_config(path)


@pytest.mark.fast
class RefusedWhereTheLossIsNotAnExampleMeanTest(unittest.TestCase):
    """FINDINGS.csv POST-F19: refused wherever the example weighting is wrong."""

    def test_the_declared_set_is_the_built_in_tasks_that_override_the_count(self) -> None:
        """The set `validate_config` reads is a list kept by hand; this derives
        the same list from the task classes, so the two cannot drift."""

        registry.register_builtin_components()
        overriding = {
            name
            for name in registry.tasks.builtin()
            if not loss_averages_over_examples(registry.tasks.get(name)(model_config={}))
        }
        self.assertIn("causal_lm", overriding)
        self.assertEqual(overriding, set(NON_EXAMPLE_MEAN_TASKS))

    def test_a_causal_lm_config_is_refused_at_load(self) -> None:
        with self.assertRaises(RunRefused) as caught:
            _load_with_mode("configs/oasst1/fedavg_base.yaml", "frozen_batch_gradients")
        self.assertIn("POST-F19", str(caught.exception))
        self.assertIn("full_gradient", str(caught.exception))
        _load_with_mode("configs/oasst1/fedavg_base.yaml", "full_gradient")

    def test_a_classification_config_still_loads(self) -> None:
        config = _load_with_mode("configs/mnist/fedavg.yaml", "frozen_batch_gradients")
        self.assertEqual(config.client.extra["update_mode"], "frozen_batch_gradients")

    def test_both_engines_refuse_a_task_that_declares_its_own_count(self) -> None:
        """The backstop for a task `validate_config` cannot see: an extension
        task's class is not known until it is built. Refused before the first
        gradient, so the model is untouched."""

        task = _TokenMeanTask()
        loader = task.build_dataloader(_dataset(8), {"batch_size": 2})
        runs = {
            "run_sgd_update_mode": lambda model: run_sgd_update_mode(
                task=task,
                model=model,
                train_loader=loader,
                local_iterations=1,
                learning_rate=LEARNING_RATE,
                update_mode="frozen_batch_gradients",
                frozen_gradient_weighting="examples",
                client_id="client_0",
            ),
            "run_delta_sgd_update_mode": lambda model: run_delta_sgd_update_mode(
                task=task,
                model=model,
                train_loader=loader,
                local_iterations=1,
                update_mode="frozen_batch_gradients",
                frozen_gradient_weighting="examples",
                client_id="client_0",
                eta_0=0.2,
                theta_0=1.0,
                gamma=2.0,
                delta=0.1,
            ),
        }
        for name, run in runs.items():
            with self.subTest(engine=name):
                model = task.build_model()
                with self.assertRaises(ValueError) as caught:
                    run(model)
                self.assertIn("POST-F19", str(caught.exception))
                self.assertEqual(float(model.weight.abs().sum()), 0.0)


def _update_under(mode: str, weighting: str | None) -> Tensor:
    """The weights after two iterations of `mode` from zero, over ragged batches."""

    task = _LinearTask()
    model = task.build_model()
    run_sgd_update_mode(
        task=task,
        model=model,
        train_loader=task.build_dataloader(_dataset(7), {"batch_size": 3}),
        local_iterations=2,
        learning_rate=LEARNING_RATE,
        update_mode=mode,
        frozen_gradient_weighting=weighting,
        client_id="c0",
    )
    return model.weight.detach().clone()


class RequiredOnlyWhereItIsReadTest(unittest.TestCase):
    """FINDINGS.csv POST-F21: the engine asks for the weighting under one mode."""

    def test_the_modes_that_read_no_weighting_run_without_one(self) -> None:
        for mode in ("single_batch", "sequential_epoch", "full_gradient"):
            with self.subTest(mode=mode):
                without = _update_under(mode, None)
                self.assertNotEqual(float(without.abs().sum()), 0.0, "the mode did not step")
                for weighting in ("examples", "uniform", "sum"):
                    # Bit for bit: a weighting these modes ignored before is
                    # still ignored, so leaving it out changes nothing.
                    self.assertTrue(torch.equal(without, _update_under(mode, weighting)))

    def test_the_frozen_mode_refuses_to_run_without_one(self) -> None:
        task = _LinearTask()
        model = task.build_model()
        with self.assertRaises(ValueError) as caught:
            run_sgd_update_mode(
                task=task,
                model=model,
                train_loader=task.build_dataloader(_dataset(7), {"batch_size": 3}),
                local_iterations=1,
                learning_rate=LEARNING_RATE,
                update_mode="frozen_batch_gradients",
                client_id="c0",
            )
        self.assertIn("frozen_gradient_weighting", str(caught.exception))
        self.assertEqual(float(model.weight.abs().sum()), 0.0, "refused after stepping")

    def test_a_weighting_that_is_given_is_still_checked(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _update_under("full_gradient", "median")
        self.assertIn("frozen_gradient_weighting must be one of", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
