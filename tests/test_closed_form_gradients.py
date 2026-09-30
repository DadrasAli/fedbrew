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
- a run of the shipped arm on its default form -- the closed form, batched
  and sequential -- agrees with the reference, the sequential executor on
  autograd, within the batched executor's tolerance, in every CSV cell and
  checkpoint, as ``closed_form`` stated does with the batched run on
  ``vmap_grad``; run.json records the form and that it was the default;
- the sequential executor's closed-form step (``closed_form_train_step``),
  which every rule's local loop takes through ``take_train_step``: fed-lasso
  under every shipped arm's rule, FedAvg's four update modes, clipping,
  momentum, AdamW, FedProx, SCAFFOLD and Delta-SGD, each within the tolerance
  of the same run on autograd.
"""

from __future__ import annotations

import copy
import json
import unittest
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import torch
import yaml

from fedbrew.clients.torch_sgd_client import _get_train_data
from fedbrew.core.config import load_config, standalone_config_mapping
from fedbrew.core.factory import build_components
from fedbrew.core.runner import run
from tests.test_batched_executor_tolerance import (
    ROUNDS,
    TOLERANCE,
    ExecutorRuns,
    classification_arms,
    classification_rule_config,
    example_config,
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
                reference = self.run_config(config, "sequential", gradient_form="autograd")
                for executor in ("batched", "sequential"):
                    default = self.run_as_written(config, executor)
                    record = json.loads((default / "run.json").read_text())["reproducibility"]
                    self.assertEqual(
                        record["executor"]["gradient_form"],
                        {"used": "closed_form", "default": True},
                    )
                    self.assertAgree(default, reference, tolerance=TOLERANCE)
                closed = self.run_config(config, "batched", gradient_form="closed_form")
                autograd = self.run_config(config, "batched", gradient_form="vmap_grad")
                record = json.loads((closed / "run.json").read_text())["reproducibility"]
                self.assertEqual(
                    record["executor"]["gradient_form"], {"used": "closed_form", "default": False}
                )
                self.assertAgree(closed, autograd, tolerance=TOLERANCE)

    def run_as_written(self, config: dict[str, Any], executor: str) -> Path:
        """The run with ``executor`` and no gradient form stated: the run's default form."""

        config = copy.deepcopy(config)
        config["runtime"].setdefault("performance", {})["executor"] = executor
        self._count += 1
        output = self.root / f"run{self._count}-{executor}-default"
        config["experiment"]["output_dir"] = str(output)
        path = self.root / f"run{self._count}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        run(path, args=None)
        return output


def _rules() -> Iterator[tuple[str, dict[str, Any]]]:
    """fed-lasso's client block under each rule and mode its sequential loop steps through.

    Each shipped arm as it is, and the executor tolerance test's rules on
    fed-lasso's batch size and step size (``classification_arms``).
    """

    for arm in sorted(path.stem for path in (REPO / "configs/examples/fed-lasso").glob("[!_]*")):
        yield arm, {"arm": arm}
    for label, rule in classification_arms():
        client = classification_rule_config({"learning_rate": 0.004, **rule})["client"]
        yield label, {**client, "batch_size": 4}
    yield (
        "delta_sgd",
        {
            "batch_size": 4,
            "update_rule": "delta_sgd",
            "eta_0": 0.004,
            "theta_0": 1.0,
            "gamma": 2.0,
            "delta": 0.1,
            "eta_max": None,
        },
    )


class TheSequentialStepOnTheClosedFormTest(ExecutorRuns):
    def test_every_rule_within_the_executor_tolerance(self) -> None:
        for label, client in _rules():
            config = example_config("fed-lasso", client.pop("arm", "fedavg"))
            if client:
                config["client"] = client
                if client["update_rule"] == "scaffold":
                    config["server"]["strategy"] = "scaffold"
            with self.subTest(rule=label):
                with _counted() as closed_steps:
                    closed = self.run_config(config, "sequential", gradient_form="closed_form")
                with _counted() as autograd_steps:
                    autograd = self.run_config(config, "sequential", gradient_form="autograd")
                self.assertGreater(closed_steps.call_count, 0)
                self.assertEqual(autograd_steps.call_count, 0)
                self.assertAgree(closed, autograd, tolerance=TOLERANCE)


def _counted() -> Any:
    """``closed_form_train_step``, its calls counted."""

    from fedbrew.tasks import base

    return mock.patch.object(base, "closed_form_train_step", wraps=base.closed_form_train_step)


if __name__ == "__main__":
    unittest.main()
