"""Delta-SGD: the auto-tuned client step size of arXiv:2306.11201.

Algorithm 1, with the delta the paper's Section 4 adds to the second condition,
for local step k of round t:

    x_k     = x_{k-1} - eta_{k-1} * g(x_{k-1})
    eta_k   = min{ gamma * ||x_k - x_{k-1}|| / (2 ||g(x_k) - g(x_{k-1})||),
                   sqrt(1 + delta * theta_{k-1}) * eta_{k-1} }
    theta_k = eta_k / eta_{k-1}

with eta and theta reset to eta_0 / theta_0 at the start of every round.

The tests drive the step-size rule against arithmetic written out in full, and
then check the rule as wired into the real update-mode engine against an
independent loop that measures ||x_k - x_{k-1}|| from parameter snapshots
rather than deriving it.
"""

from __future__ import annotations

import copy
import math
import unittest
from collections.abc import Mapping
from typing import Any

import pytest
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, TensorDataset

from fedbrew.clients.local_update_modes import (
    _DeltaSGDStepper,
    run_delta_sgd_update_mode,
)
from fedbrew.clients.torch_delta_sgd_client import (
    DEFAULT_DELTA,
    DEFAULT_ETA_0,
    DEFAULT_GAMMA,
    DEFAULT_THETA_0,
)
from fedbrew.core.config import load_config, validate_config
from fedbrew.core.factory import build_components
from fedbrew.core.protocol import FitRequest
from fedbrew.core.validation import validate_full_config
from fedbrew.tasks.base import TaskAdapter

_BASE_CONFIG = load_config("configs/dev/synthetic.yaml")

ETA_0 = 0.2
THETA_0 = 1.0
GAMMA = 2.0
DELTA = 0.1


def _stepper(**overrides: Any) -> _DeltaSGDStepper:
    kwargs: dict[str, Any] = {
        "eta_0": ETA_0,
        "theta_0": THETA_0,
        "gamma": GAMMA,
        "delta": DELTA,
        "eta_max": None,
    }
    kwargs.update(overrides)
    return _DeltaSGDStepper(**kwargs)


def _grad(value: float) -> dict[str, Tensor]:
    return {"w": torch.tensor([value])}


@pytest.mark.fast
class StepSizeRuleTest(unittest.TestCase):
    """The rule itself, against closed-form arithmetic."""

    def test_first_step_uses_eta_0_unchanged(self) -> None:
        """Algorithm 1 line 6: there is no previous iterate to measure."""

        stepper = _stepper()
        self.assertEqual(stepper.step_size_for(_grad(3.0)), ETA_0)
        self.assertEqual(stepper.theta, THETA_0)

    def test_second_step_takes_the_smoothness_term_when_it_binds(self) -> None:
        # ||x_1 - x_0|| = eta_0 * ||g_0|| = 0.2 * 1 = 0.2
        # ||g_1 - g_0|| = |-9 - 1|       = 10
        # smoothness    = 2 * 0.2 / (2 * 10) = 0.02, below the growth term.
        stepper = _stepper()
        stepper.step_size_for(_grad(1.0))
        eta = stepper.step_size_for(_grad(-9.0))

        iterate_distance = ETA_0 * 1.0
        gradient_distance = abs(-9.0 - 1.0)
        smoothness = GAMMA * iterate_distance / (2.0 * gradient_distance)
        growth = math.sqrt(1.0 + DELTA * THETA_0) * ETA_0
        self.assertLess(smoothness, growth)
        self.assertAlmostEqual(eta, smoothness)
        self.assertAlmostEqual(stepper.theta, smoothness / ETA_0)

    def test_growth_term_caps_how_fast_the_step_size_can_rise(self) -> None:
        stepper = _stepper()
        stepper.step_size_for(_grad(10.0))
        eta = stepper.step_size_for(_grad(10.0 - 1e-3))

        growth = math.sqrt(1.0 + DELTA * THETA_0) * ETA_0
        self.assertAlmostEqual(eta, growth)
        self.assertGreater(eta, ETA_0)

    def test_identical_gradients_fall_back_to_the_growth_term(self) -> None:
        """The smoothness estimate is undefined at a zero difference, not
        infinite. Matches Malitsky & Mishchenko's reference implementation."""

        stepper = _stepper()
        stepper.step_size_for(_grad(5.0))
        eta = stepper.step_size_for(_grad(5.0))

        self.assertAlmostEqual(eta, math.sqrt(1.0 + DELTA * THETA_0) * ETA_0)
        self.assertEqual(stepper.undefined_curvature_steps, 1)

    def test_zero_gradient_does_not_divide_by_zero(self) -> None:
        stepper = _stepper()
        stepper.step_size_for(_grad(0.0))
        eta = stepper.step_size_for(_grad(0.0))
        self.assertTrue(math.isfinite(eta))
        self.assertEqual(stepper.undefined_curvature_steps, 1)

    def test_eta_max_clamps_and_is_counted(self) -> None:
        clamp = 0.201
        stepper = _stepper(eta_max=clamp)
        stepper.step_size_for(_grad(10.0))
        eta = stepper.step_size_for(_grad(10.0))

        self.assertAlmostEqual(eta, clamp)
        self.assertEqual(stepper.clamped_steps, 1)
        # theta must describe the step size actually taken, not the one the
        # rule wanted before the clamp.
        self.assertAlmostEqual(stepper.theta, clamp / ETA_0)

    def test_theta_is_the_ratio_to_the_step_it_replaces(self) -> None:
        stepper = _stepper()
        etas = [stepper.step_size_for(_grad(g)) for g in (1.0, -9.0, 4.0, -2.0)]
        # theta after the final step is eta_last / eta_previous.
        self.assertAlmostEqual(stepper.theta, etas[-1] / etas[-2])

    def test_invalid_hyperparameters_are_rejected(self) -> None:
        for field in ("eta_0", "theta_0", "gamma", "delta"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                _stepper(**{field: 0.0})
        with self.assertRaises(ValueError):
            _stepper(eta_max=-1.0)


class _CountingLinearTask(TaskAdapter):
    """Deterministic least-squares task that counts its gradient evaluations."""

    def __init__(self) -> None:
        self.train_step_calls = 0

    def build_model(self, config: Mapping[str, Any] | None = None) -> nn.Module:
        model = nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[0.5, -0.25, 1.0]]))
        return model

    def build_dataloader(self, data: Any, config: Mapping[str, Any]) -> DataLoader[Any]:
        return DataLoader(data, batch_size=int(config.get("batch_size", 2)), shuffle=False)

    def train_step(
        self,
        model: nn.Module,
        batch: Any,
        optimizer: optim.Optimizer | None = None,
    ) -> dict[str, float]:
        if optimizer is None:
            raise ValueError("optimizer is required")
        self.train_step_calls += 1
        features, targets = batch
        optimizer.zero_grad()
        loss = ((model(features).squeeze(-1) - targets) ** 2).mean()
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.detach()), "correct": 0.0, "total": float(targets.numel())}

    def eval_step(self, model: nn.Module, batch: Any) -> dict[str, float]:
        raise NotImplementedError

    def compute_metrics(self, outputs: Any, targets: Tensor | None = None) -> dict[str, float]:
        return {"loss": 0.0, "accuracy": 0.0}


def _dataset() -> TensorDataset:
    torch.manual_seed(0)
    return TensorDataset(torch.randn(8, 3), torch.randn(8))


def _run(mode: str, *, local_iterations: int = 2, **overrides: Any) -> Any:
    task = _CountingLinearTask()
    model = task.build_model()
    loader = task.build_dataloader(_dataset(), {"batch_size": 2})
    kwargs: dict[str, Any] = {
        "task": task,
        "model": model,
        "train_loader": loader,
        "local_iterations": local_iterations,
        "update_mode": mode,
        "frozen_gradient_weighting": "examples",
        "client_id": "c0",
        "eta_0": ETA_0,
        "theta_0": THETA_0,
        "gamma": GAMMA,
        "delta": DELTA,
    }
    kwargs.update(overrides)
    return task, model, run_delta_sgd_update_mode(**kwargs)


class UpdateModeEngineTest(unittest.TestCase):
    """The rule as wired into the shared update-mode engine."""

    def test_every_mode_runs_and_reports_its_step_count(self) -> None:
        expected = {
            "single_batch": 2,  # local_iterations steps
            "sequential_epoch": 8,  # local_iterations * 4 batches
            "frozen_batch_gradients": 2,  # one combined step per epoch
        }
        for mode, steps in expected.items():
            with self.subTest(mode=mode):
                _, _, result = _run(mode)
                self.assertEqual(result.optimizer_steps, steps)
                self.assertEqual(len(result.step_sizes), steps)
                self.assertEqual(result.step_sizes[0], ETA_0)

    def test_one_backward_per_local_step(self) -> None:
        """Delta-SGD reuses the gradient it measured to pick the step size, so
        it costs exactly what plain SGD costs."""

        task, _, result = _run("sequential_epoch")
        self.assertEqual(task.train_step_calls, result.optimizer_steps)

    def test_step_sizes_match_a_loop_that_measures_the_iterate_distance(self) -> None:
        """The engine derives ||x_k - x_{k-1}|| as eta * ||g_{k-1}|| instead of
        cloning the parameters. This pins that shortcut against a reference
        loop that snapshots and subtracts the parameters for real."""

        _, _, result = _run("sequential_epoch")

        task = _CountingLinearTask()
        model = task.build_model()
        loader = task.build_dataloader(_dataset(), {"batch_size": 2})

        eta, theta = ETA_0, THETA_0
        previous_parameters: Tensor | None = None
        previous_gradient: Tensor | None = None
        reference: list[float] = []
        for _ in range(2):
            for batch in loader:
                features, targets = batch
                model.zero_grad()
                loss = ((model(features).squeeze(-1) - targets) ** 2).mean()
                loss.backward()
                gradient = model.weight.grad.detach().clone().flatten()
                parameters = model.weight.detach().clone().flatten()

                if previous_gradient is not None:
                    iterate_distance = float(
                        torch.linalg.vector_norm(parameters - previous_parameters)
                    )
                    gradient_distance = float(
                        torch.linalg.vector_norm(gradient - previous_gradient)
                    )
                    growth = math.sqrt(1.0 + DELTA * theta) * eta
                    eta_next = min(GAMMA * iterate_distance / (2.0 * gradient_distance), growth)
                    theta, eta = eta_next / eta, eta_next

                previous_parameters = parameters
                previous_gradient = gradient
                reference.append(eta)
                with torch.no_grad():
                    model.weight.add_(model.weight.grad, alpha=-eta)

        self.assertEqual(len(reference), len(result.step_sizes))
        for index, (produced, expected) in enumerate(
            zip(result.step_sizes, reference, strict=True)
        ):
            with self.subTest(step=index):
                self.assertAlmostEqual(produced, expected, places=6)

    def test_eta_and_theta_reset_between_rounds(self) -> None:
        """Algorithm 1 lines 6-7. This is what keeps the client stateless."""

        task = _CountingLinearTask()
        model = task.build_model()
        loader = task.build_dataloader(_dataset(), {"batch_size": 2})
        common = {
            "task": task,
            "train_loader": loader,
            "local_iterations": 1,
            "update_mode": "sequential_epoch",
            "frozen_gradient_weighting": "examples",
            "client_id": "c0",
            "eta_0": ETA_0,
            "theta_0": THETA_0,
            "gamma": GAMMA,
            "delta": DELTA,
        }
        first = run_delta_sgd_update_mode(model=model, **common)
        second = run_delta_sgd_update_mode(model=model, **common)

        self.assertEqual(first.step_sizes[0], ETA_0)
        self.assertEqual(second.step_sizes[0], ETA_0)
        self.assertNotEqual(first.step_sizes[1:], second.step_sizes[1:])

    def test_eta_max_bounds_every_applied_step(self) -> None:
        clamp = 0.05
        _, _, result = _run("sequential_epoch", eta_max=clamp, eta_0=clamp)
        self.assertTrue(all(step <= clamp + 1e-12 for step in result.step_sizes))

    @pytest.mark.fast
    def test_amp_is_rejected_rather_than_silently_wrong(self) -> None:
        task = _CountingLinearTask()
        task._scaler = object()
        model = task.build_model()
        loader = task.build_dataloader(_dataset(), {"batch_size": 2})
        with self.assertRaises(ValueError):
            run_delta_sgd_update_mode(
                task=task,
                model=model,
                train_loader=loader,
                local_iterations=1,
                update_mode="sequential_epoch",
                frozen_gradient_weighting="examples",
                client_id="c0",
                eta_0=ETA_0,
                theta_0=THETA_0,
                gamma=GAMMA,
                delta=DELTA,
            )


@pytest.mark.fast
class PaperDefaultsTest(unittest.TestCase):
    def test_defaults_match_the_paper(self) -> None:
        """arXiv:2306.11201 uses these unchanged across every experiment."""

        self.assertEqual(DEFAULT_ETA_0, 0.2)
        self.assertEqual(DEFAULT_THETA_0, 1.0)
        self.assertEqual(DEFAULT_GAMMA, 2.0)
        self.assertEqual(DEFAULT_DELTA, 0.1)


@pytest.mark.fast
class ConfigurationTest(unittest.TestCase):
    """Both validation layers: the hard loader check and the preflight report."""

    def _config(self, **client_extra: Any) -> Any:
        config = copy.deepcopy(_BASE_CONFIG)
        config.client.update_rule = "delta_sgd"
        config.client.learning_rate = None
        config.client.extra = {"eta_0": ETA_0, **client_extra}
        config.server.extra["aggregation_weighting"] = "uniform"
        return config

    def _preflight_errors(self, config: Any) -> list[str]:
        return [
            issue.code for issue in validate_full_config(config).issues if issue.severity == "error"
        ]

    def test_only_eta_0_is_required(self) -> None:
        """theta_0, gamma and delta carry the paper's own defaults."""

        config = self._config()
        validate_config(config)
        self.assertEqual(self._preflight_errors(config), [])

    def test_explicit_paper_hyperparameters_are_accepted(self) -> None:
        config = self._config(
            theta_0=DEFAULT_THETA_0,
            gamma=DEFAULT_GAMMA,
            delta=DEFAULT_DELTA,
            eta_max=1.0,
        )
        validate_config(config)
        self.assertEqual(self._preflight_errors(config), [])

    def test_missing_eta_0_is_rejected_by_both_layers(self) -> None:
        config = self._config()
        del config.client.extra["eta_0"]
        with self.assertRaises(ValueError):
            validate_config(config)
        self.assertIn("algorithm.delta_sgd_eta_0_missing", self._preflight_errors(config))

    def test_non_positive_hyperparameters_are_rejected(self) -> None:
        for field in ("eta_0", "theta_0", "gamma", "delta", "eta_max"):
            with self.subTest(field=field):
                config = self._config(**{field: 0.0})
                with self.assertRaises(ValueError):
                    validate_config(config)
                self.assertIn(
                    f"algorithm.delta_sgd_{field}_invalid",
                    self._preflight_errors(config),
                )

    def test_an_explicit_learning_rate_is_rejected(self) -> None:
        """The step size is measured, not configured; accepting the key would
        make it look tunable."""

        config = self._config()
        config.client.learning_rate = 0.1
        with self.assertRaises(ValueError):
            validate_config(config)
        self.assertIn("client.learning_rate_unexpected", self._preflight_errors(config))

    def test_unknown_update_mode_is_rejected(self) -> None:
        config = self._config(update_mode="not_a_mode")
        with self.assertRaises(ValueError):
            validate_config(config)
        self.assertIn("algorithm.delta_sgd_update_mode_invalid", self._preflight_errors(config))

    def test_amp_is_rejected(self) -> None:
        config = self._config()
        config.runtime.use_amp = True
        with self.assertRaises(ValueError):
            validate_config(config)
        self.assertIn("algorithm.delta_sgd_amp_unsupported", self._preflight_errors(config))

    def test_example_weighting_is_flagged_as_a_deviation_not_an_error(self) -> None:
        """The paper averages clients uniformly. Weighting by examples is a
        legitimate choice for comparability, so it is info, not an error."""

        config = self._config()
        del config.server.extra["aggregation_weighting"]
        validate_config(config)
        report = validate_full_config(config)
        codes = {
            issue.code: issue.severity
            for issue in report.issues
            if issue.code == "algorithm.delta_sgd_aggregation_weighting"
        }
        self.assertEqual(codes, {"algorithm.delta_sgd_aggregation_weighting": "info"})

    def test_the_fedavg_server_accepts_the_delta_sgd_client(self) -> None:
        config = self._config()
        self.assertEqual(config.server.strategy, "fedavg")
        self.assertNotIn("algorithm.fedavg_client_incompatible", self._preflight_errors(config))


class EndToEndTest(unittest.TestCase):
    """A real run through the factory, server and loop."""

    def test_a_short_run_completes_and_reports_the_step_size_trace(self) -> None:
        config = copy.deepcopy(_BASE_CONFIG)
        config.client.update_rule = "delta_sgd"
        config.client.learning_rate = None
        config.client.extra = {"eta_0": ETA_0}
        config.server.extra["aggregation_weighting"] = "uniform"
        config.server.global_rounds = 3
        config.client.local_iterations = 2
        step_size_metrics = [
            "client_step_size_mean",
            "client_step_size_min",
            "client_step_size_max",
            "client_step_size_final",
            "step_size_clamp_fraction",
            "undefined_curvature_fraction",
        ]
        config.client.metrics = ["fit_loss", *step_size_metrics]
        config.server.metrics = ["fit_loss", *step_size_metrics]

        components = build_components(config)
        client = components.clients[next(iter(components.clients))]
        payload = components.server.initialize()
        result = client.fit(FitRequest(round_id=1, client_id=client.client_id, payload=payload))

        for name in step_size_metrics:
            self.assertIn(name, result.metrics)
        # The first local step always uses eta_0, so the round's maximum can
        # never fall below it.
        self.assertGreaterEqual(result.metrics["client_step_size_max"], ETA_0)
        self.assertEqual(result.metrics["step_size_clamp_fraction"], 0.0)

    @pytest.mark.fast
    def test_the_client_carries_no_adaptive_state_across_rounds(self) -> None:
        """eta and theta reset every round, so a checkpoint has nothing
        model-sized to store -- this is what makes delta_sgd cheap at 3597
        writers where SCAFFOLD is not."""

        config = copy.deepcopy(_BASE_CONFIG)
        config.client.update_rule = "delta_sgd"
        config.client.learning_rate = None
        config.client.extra = {"eta_0": ETA_0}
        components = build_components(config)
        state = components.clients[next(iter(components.clients))].get_state()

        self.assertNotIn("eta", state)
        self.assertNotIn("theta", state)
        for value in state.values():
            self.assertNotIsInstance(value, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
