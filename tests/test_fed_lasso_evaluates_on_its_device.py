"""fed-lasso measures F(x) on the device its model is on (POST-F33).

``FedLassoTask.eval_step`` used to compute the optimality gap through
``ProblemSpec.objective_at``, which builds the design and the targets on the
CPU and multiplies them by the model's iterate. On CUDA the iterate is on the
GPU, so every evaluation raised. The task now holds the design and targets on
its own device, built once, and evaluates F(x) from those.

Two checks. On any machine: with the spec's design and targets made to raise,
an evaluation still runs, so it no longer reads them per call, and its gap is
``objective_at(x) - F*`` as the spec computes it. Where CUDA exists: a round of
the shipped arm runs and evaluates on the GPU.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest
import torch
import yaml

from fedbrew.core.config import load_config
from fedbrew.core.factory import build_components
from fedbrew.core.runner import run
from tests.test_batched_executor_tolerance import example_config


def _config_file(directory: Path, device: str) -> Path:
    config = example_config("fed-lasso")
    config["defaults"]["global_rounds"] = 1
    config["experiment"]["output_dir"] = str(directory / "run")
    config["runtime"]["device"] = device
    path = directory / "fed-lasso.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _cuda_usable() -> bool:
    """A CUDA device this process can allocate on; one held exclusively elsewhere is not."""

    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
    except RuntimeError:
        return False
    return True


class FedLassoEvaluationDeviceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def test_an_evaluation_reads_no_host_copy_of_the_problem(self) -> None:
        components = build_components(load_config(_config_file(self.root, "cpu")))
        task = components.task
        client = components.dataset.list_clients()[0]
        model = task.build_model(components.clients[client].model_config)
        with torch.no_grad():
            model.x.copy_(torch.linspace(-1.0, 1.0, model.x.numel(), dtype=torch.float64))
        spec = task.spec
        train = components.dataset.get_client_data(client)["train"]
        batch = next(iter(task.build_dataloader(train)))
        expected = spec.objective_at(model.iterate) - spec.optimal_objective()

        refuse = mock.Mock(side_effect=AssertionError("evaluation read the spec's CPU data"))
        with (
            mock.patch.object(type(spec), "design", refuse),
            mock.patch.object(type(spec), "client_targets", refuse),
        ):
            measured = task.eval_step(model, batch)

        self.assertTrue(math.isclose(measured["optimality_gap"], expected, rel_tol=1e-12))

    @pytest.mark.cuda
    @unittest.skipUnless(_cuda_usable(), "needs a usable CUDA device")
    def test_a_round_evaluates_on_cuda(self) -> None:
        run(_config_file(self.root, "cuda"), args=None)
        self.assertTrue((self.root / "run" / "round_metrics.csv").is_file())


if __name__ == "__main__":
    unittest.main()
