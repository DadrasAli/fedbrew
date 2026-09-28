"""examples/fed-logistic-l1: each problem it poses, on data its generator writes.

What is held, per problem (``problem.PROBLEMS``), on a small synthetic
instance generated here (d = 8, 4 clients of 16 rows):

- the gradient autograd takes of the task's ``functional_loss`` is the
  module's analytic ``gradient``, at random points, to 1e-14 relative;
- the manifest's reference: a convex problem's ``f_star`` is ``F`` at its
  ``x_star``, certified to a KKT residual under ``CERTIFICATE``; a nonconvex
  one records neither;
- one FedAvg round, batched, agrees with the sequential round within the
  batched executor's ``1e-12`` (chapter 11 §9), every cell and the model;
- a model block that states another problem than the data is refused.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch
import yaml

from fedbrew.core import extensions
from fedbrew.core.config import load_config
from fedbrew.core.factory import build_components
from tests.test_batched_executor_tolerance import CSVS, TOLERANCE, ExecutorRuns

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTENSION = REPO_ROOT / "examples" / "fed-logistic-l1" / "problem.py"
ARM = REPO_ROOT / "configs" / "examples" / "fed-logistic-l1-synthetic-lambda0.03" / "fedavg.yaml"

problem = extensions._import_file(EXTENSION)

_root = tempfile.TemporaryDirectory()
_manifests: dict[tuple[str, str], Path] = {}


def small_manifest(loss: str, penalty: str) -> Path:
    """A small instance of the problem, generated once per process."""

    if (loss, penalty) not in _manifests:
        from fedbrew.data.generate import generate_from_config

        directory = Path(_root.name) / f"{loss}-{penalty}"
        generator = {
            "dataset": {
                "name": "fed_logistic_l1",
                "output_dir": str(directory / "data"),
                "seed": 42,
                "extensions": [str(EXTENSION)],
            },
            "partition": {"strategy": "analytic", "num_clients": 4},
            "problem": {
                "dim": 8,
                "rows_per_client": 16,
                "loss": loss,
                "penalty": penalty,
                "penalty_strength": 0.03,
                "partition_block": 4,
            },
        }
        directory.mkdir(parents=True)
        path = directory / "generator.yaml"
        path.write_text(yaml.safe_dump(generator), encoding="utf-8")
        _manifests[(loss, penalty)] = Path(generate_from_config(path))
    return _manifests[(loss, penalty)]


def small_config(loss: str, penalty: str) -> dict[str, Any]:
    """The shipped synthetic FedAvg arm, on the small instance, for one round."""

    config = yaml.safe_load(ARM.read_text(encoding="utf-8"))
    config["experiment"]["extensions"] = [str(EXTENSION)]
    config["data"]["path"] = str(small_manifest(loss, penalty))
    config["model"].update(input_dim=8, loss=loss, penalty=penalty, penalty_strength=0.03)
    config["client"].update(learning_rate=0.5, batch_size=16)
    config["defaults"]["global_rounds"] = 1
    config["evaluation"]["test"] = {"every": 1, "clients": "all"}
    config["runtime"]["quiet"] = True
    config["runtime"]["checkpointing"].update(
        enabled=True, save_last=True, save_every_round=True, keep_last=None
    )
    return config


def _reference(loss: str, penalty: str) -> dict[str, Any]:
    manifest = json.loads(small_manifest(loss, penalty).read_text(encoding="utf-8"))
    return manifest["reference"]


class TheGradientIsAnalyticTest(unittest.TestCase):
    def test_autograd_of_the_task_loss_is_the_gradient(self) -> None:
        spec = problem.ProblemSpec(num_clients=4, dim=8, rows_per_client=16, partition_block=4)
        features, labels = spec.design(), spec.labels()
        generator = torch.Generator().manual_seed(7)
        for loss, penalty in problem.PROBLEMS:
            model = problem.LogisticModel(8, 0.03, loss=loss, penalty=penalty)
            for _ in range(5):
                point = torch.randn(8, generator=generator, dtype=torch.float64)
                with self.subTest(problem=f"{loss}+{penalty}"):
                    params = {"x": point.clone().requires_grad_(True)}
                    value, _ = _task(loss, penalty).functional_loss(
                        model, params, {}, (features, labels)
                    )
                    (measured,) = torch.autograd.grad(value, params["x"])
                    expected = problem.gradient(point, features, labels, 0.03, loss, penalty)
                    scale = float(expected.abs().max())
                    self.assertLessEqual(float((measured - expected).abs().max()), 1e-14 * scale)
                    self.assertEqual(
                        float(value),
                        problem.objective(point, features, labels, 0.03, loss, penalty),
                    )


class TheReferenceTest(unittest.TestCase):
    def test_a_convex_problem_is_certified_and_a_nonconvex_one_has_none(self) -> None:
        for loss, penalty in problem.PROBLEMS:
            with self.subTest(problem=f"{loss}+{penalty}"):
                reference = _reference(loss, penalty)
                self.assertEqual(reference["problem"]["loss"], loss)
                self.assertEqual(reference["problem"]["penalty"], penalty)
                if not problem.convex(loss, penalty):
                    self.assertNotIn("f_star", reference)
                    self.assertNotIn("x_star", reference)
                    continue
                self.assertLessEqual(reference["kkt_residual"], problem.CERTIFICATE)
                task = _task(loss, penalty)
                at = task.pooled_objective(torch.tensor(reference["x_star"], dtype=torch.float64))
                self.assertEqual(reference["f_star"], at)


class OneRoundBatchedAgreesTest(ExecutorRuns):
    def test_each_problem_within_the_executor_tolerance(self) -> None:
        for loss, penalty in problem.PROBLEMS:
            with self.subTest(problem=f"{loss}+{penalty}"):
                batched, sequential = self.both(small_config(loss, penalty))
                record = json.loads((batched / "run.json").read_text())["reproducibility"]
                self.assertEqual(record["executor"]["used"], "batched")
                for name in CSVS:
                    self._compare_csv(batched / name, sequential / name, False, TOLERANCE)
                self._compare_checkpoint(
                    batched / "checkpoints" / "round_001.pt",
                    sequential / "checkpoints" / "round_001.pt",
                    False,
                    TOLERANCE,
                )


class AnotherProblemIsRefusedTest(unittest.TestCase):
    def test_the_model_block_must_state_the_data_problem(self) -> None:
        config = small_config("logistic", "l1")
        with tempfile.TemporaryDirectory() as directory:
            for key, value in (("penalty_strength", 0.05),):
                with self.subTest(key=key):
                    edited = copy.deepcopy(config)
                    edited["model"][key] = value
                    edited["experiment"]["output_dir"] = directory
                    path = Path(directory) / "config.yaml"
                    path.write_text(yaml.safe_dump(edited), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "manifest reference"):
                        build_components(load_config(path))


def _task(loss: str, penalty: str) -> Any:
    manifest = small_manifest(loss, penalty)
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    metadata["manifest_path"] = str(manifest)
    return problem.FedLogisticL1Task(
        model_config={"input_dim": 8, "penalty_strength": 0.03, "loss": loss, "penalty": penalty},
        dataset_metadata=metadata,
    )


if __name__ == "__main__":
    unittest.main()
