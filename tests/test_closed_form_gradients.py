"""Every linear example's closed-form gradient is autograd's, and a run on it is the vmap_grad run.

``closed_form_gradient`` (``fedbrew.tasks.base.BatchableTask``) gives a whole
stack's gradients of ``functional_loss`` by formula, and
``runtime.performance.gradient_form: closed_form`` trains on it. Held here, for
each example and each of its problems:

- at random points, some coordinates exactly 0, over a stack of four clients'
  rows, with and without a mask of padded rows: each client's gradient is
  ``torch.func.grad`` of the task's own ``functional_loss`` within ``1e-12`` of
  its largest entry, and each step output (the loss) within ``1e-12``
  relative -- an l1 term included, whose subgradient at exactly 0 is 0 both
  ways; asked for no outputs, as a training step asks, the gradients are the
  same bits and no loss is computed;
- a run of the shipped arm with ``closed_form`` agrees with the same run on
  ``vmap_grad`` within the batched executor's tolerance, in every CSV cell
  and checkpoint, and run.json records the form.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

import torch
import yaml

from fedbrew.clients.torch_sgd_client import _get_train_data
from fedbrew.core.config import load_config, standalone_config_mapping
from fedbrew.core.factory import build_components
from tests.test_batched_executor_tolerance import (
    ROUNDS,
    TOLERANCE,
    ExecutorRuns,
    example_manifest,
)

REPO = Path(__file__).resolve().parent.parent
HETERO = "configs/examples/heterogeneous-quadratic/fedavg_k10.yaml"
KAPPA10 = "configs/examples/fed-logistic-l1-synthetic-kappa10"

#: label -> (the arm, the generator config whose data it runs on).
CASES: dict[str, tuple[str, str]] = {
    "ridge logistic regression": (
        f"{KAPPA10}/logistic-l2sq-lambda0.01.yaml",
        "fed-logistic-l1-synthetic-kappa10",
    ),
    "l1-regularized logistic regression": (
        f"{KAPPA10}/logistic-l1-lambda0.01.yaml",
        "fed-logistic-l1-synthetic-kappa10",
    ),
    "logistic regression with a nonconvex regularizer": (
        f"{KAPPA10}/logistic-nonconvex-lambda0.01.yaml",
        "fed-logistic-l1-synthetic-kappa10",
    ),
    "the sigmoid loss": (
        f"{KAPPA10}/tanh-l2sq-lambda0.01.yaml",
        "fed-logistic-l1-synthetic-kappa10",
    ),
    "heterogeneous quadratic": (HETERO, "heterogeneous-quadratic"),
    "planted lasso": (HETERO, "heterogeneous-quadratic-lasso"),
    "Geman-McClure double well": (HETERO, "heterogeneous-quadratic-double-well"),
    "fed-lasso": ("configs/examples/fed-lasso/fedavg.yaml", "fed-lasso"),
    "fed-lasso-l2": ("configs/examples/fed-lasso-l2/fedavg.yaml", "fed-lasso-l2"),
    "fed-lasso-smooth": ("configs/examples/fed-lasso-smooth/fedavg.yaml", "fed-lasso-smooth"),
    "drift-quad": ("configs/examples/drift-quad/fedavg.yaml", "drift-quad"),
    "drift-quad-rate": ("configs/examples/drift-quad-rate/fedavg.yaml", "drift-quad-rate"),
    "drift-quad-floor": ("configs/examples/drift-quad-floor/fedavg.yaml", "drift-quad-floor"),
    "simplex-lsq": ("configs/examples/simplex-lsq/fedavg.yaml", "simplex-lsq"),
    "simplex-lsq-feasible": (
        "configs/examples/simplex-lsq-feasible/fedavg.yaml",
        "simplex-lsq-feasible",
    ),
    "pl-1d": ("configs/examples/pl-1d/fedavg.yaml", "pl-1d"),
    "nonconvex-simplex": ("configs/examples/nonconvex-simplex/fedavg.yaml", "nonconvex-simplex"),
}

CLIENTS = 4


def _config(arm: str, setting: str, root: Path) -> dict[str, Any]:
    config = standalone_config_mapping(REPO / arm)
    config["data"]["path"] = str(example_manifest(setting))
    config["experiment"]["output_dir"] = str(root / "unused")
    config["experiment"]["extensions"] = [
        str(REPO / path) for path in config["experiment"].get("extensions") or []
    ]
    return config


def _stack(task: Any, dataset: Any) -> tuple[tuple[torch.Tensor, ...], int]:
    """The first rows every one of the first four clients holds, stacked."""

    splits = [
        task.split_rows(_get_train_data(dataset.get_client_data(client)))
        for client in list(dataset.list_clients())[:CLIENTS]
    ]
    rows = min(len(split[0]) for split in splits)
    return tuple(
        torch.stack([split[k][:rows] for split in splits]) for k in range(len(splits[0]))
    ), rows


class TheClosedFormIsAutogradTest(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def _components(self, arm: str, setting: str) -> Any:
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump(_config(arm, setting, self.root)), encoding="utf-8")
        return build_components(load_config(path))

    def test_every_example_and_problem(self) -> None:
        generator = torch.Generator().manual_seed(2026)
        for label, (arm, setting) in CASES.items():
            with self.subTest(case=label):
                components = self._components(arm, setting)
                task = components.task
                client = components.clients[next(iter(components.clients))]
                model = task.build_model(client.model_config)
                buffers = dict(model.named_buffers())
                batch, rows = _stack(task, components.dataset)
                shape = model.x.shape
                for masked in (False, True):
                    x = 0.5 * torch.randn(CLIENTS, *shape, generator=generator, dtype=torch.float64)
                    x.view(CLIENTS, -1)[:, :2] = 0.0  # exact zeros, where the l1 kink is
                    mask = None
                    if masked:
                        mask = torch.ones(CLIENTS, rows, dtype=torch.float64)
                        # Every client keeps a real row: some examples hold one.
                        mask[1, max(1, rows // 2) :] = 0.0
                        if rows > 1:
                            mask[3, -1:] = 0.0
                    self._check(task, model, buffers, x, batch, mask, f"{label} masked={masked}")

    def _check(self, task, model, buffers, x, batch, mask, where) -> None:  # type: ignore[no-untyped-def]
        with torch.no_grad():
            grads, outputs = task.closed_form_gradient(model, {"x": x}, buffers, batch, mask)
            bare, none = task.closed_form_gradient(
                model, {"x": x}, buffers, batch, mask, outputs=False
            )
        self.assertTrue(torch.equal(bare["x"], grads["x"]), f"{where} without outputs")
        self.assertNotIn("loss", none, where)
        for client in range(CLIENTS):

            def loss(params: Any, client: int = client) -> Any:
                one = tuple(tensor[client] for tensor in batch)
                return task.functional_loss(
                    model, params, buffers, one, None if mask is None else mask[client]
                )

            (expected, aux) = torch.func.grad(loss, has_aux=True)({"x": x[client]})
            scale = max(1.0, float(expected["x"].abs().max()))
            error = float((grads["x"][client] - expected["x"]).abs().max())
            self.assertLessEqual(error, 1e-12 * scale, f"{where} client {client} gradient")
            value = float(aux["loss"])
            self.assertLessEqual(
                abs(float(outputs["loss"][client]) - value),
                1e-12 * max(1.0, abs(value)),
                f"{where} client {client} loss",
            )


class ARunOnTheClosedFormTest(ExecutorRuns):
    def test_every_example_within_the_executor_tolerance(self) -> None:
        seen = set()
        for label, (arm, setting) in CASES.items():
            if (arm, setting) in seen:
                continue
            seen.add((arm, setting))
            config = _config(arm, setting, self.root)
            config["schedule"]["rounds"] = ROUNDS
            config["runtime"]["quiet"] = True
            config["runtime"]["checkpointing"].update(
                enabled=True, save_last=True, save_every_round=True, keep_last=None
            )
            with self.subTest(case=label):
                closed = self.run_config(config, "batched", gradient_form="closed_form")
                autograd = self.run_config(config, "batched", gradient_form="vmap_grad")
                record = json.loads((closed / "run.json").read_text())["reproducibility"]
                self.assertEqual(record["executor"]["gradient_form"], "closed_form")
                self.assertAgree(closed, autograd, tolerance=TOLERANCE)


if __name__ == "__main__":
    unittest.main()
