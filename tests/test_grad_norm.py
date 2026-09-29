"""``evaluation.grad_norm``: ``||grad F||^2`` of the global objective, and nothing else changed.

``grad_norm_sq`` is the squared norm of the gradient of F at the global model,
F the task's training loss over every client's train split, weighted as the
loss averages -- under an l1 term, of F's minimum-norm subgradient
(``fedbrew/core/grad_norm.py``). Pinned here:

- at random points it is autograd's gradient of the pooled objective -- every
  client's train rows as one batch -- on every linear example, in float64 to
  1e-12, with fed-lasso's l1 case checked against the analytic smooth
  gradient soft-thresholded by hand at the zero coordinates; and the
  weighting holds on classification's cross-entropy over clients of unequal
  sizes;
- the batched evaluator's device pass and the sequential reference agree to
  the executor tolerance, and so do the three paths of a run: sequential,
  batched per round and resident;
- on, it changes no other column and no checkpoint; off -- the default -- no
  gradient pass runs in any path and no column is written;
- the plan header's columns are the ones written, for every task;
- a task that declares no gradient refuses the key, and a monitor or a
  selection on the column is refused while it is never measured.
"""

from __future__ import annotations

import copy
import csv
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch
import yaml

from fedbrew.core.batched_evaluator import BatchedEvaluator
from fedbrew.core.config import divergence_direction, load_config
from fedbrew.core.factory import build_components
from fedbrew.core.grad_norm import (
    GRAD_NORM_COLUMN,
    SequentialGradNorm,
    WeightedGradient,
    minimum_norm_gradient,
    trainable,
)
from fedbrew.core.refusal import RunRefused
from tests.test_batched_executor_tolerance import (
    EXAMPLES,
    ROUNDS,
    TOLERANCE,
    ExecutorRuns,
    example_config,
)
from tests.test_resident_round import _clean, _executor, per_round

REPO = Path(__file__).resolve().parent.parent
SMOKE = REPO / "configs" / "dev" / "smoke.yaml"


def _relative(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


@pytest.mark.fast
class TheMinimumNormSubgradientTest(unittest.TestCase):
    def test_zero_coordinates_are_soft_thresholded_and_the_rest_kept(self) -> None:
        x = torch.tensor([0.0, 0.0, 0.0, 2.0, -1.0], dtype=torch.float64)
        g = torch.tensor([0.3, -0.05, -0.8, 0.4, 0.4], dtype=torch.float64)
        minimum = minimum_norm_gradient({"x": g}, {"x": x}, {"x": 0.1})["x"]
        torch.testing.assert_close(
            minimum, torch.tensor([0.2, 0.0, -0.7, 0.4, 0.4], dtype=torch.float64)
        )

    def test_a_smooth_objective_is_left_alone(self) -> None:
        g = {"w": torch.ones(3)}
        self.assertIs(minimum_norm_gradient(g, {"w": torch.zeros(3)}, {})["w"], g["w"])


@pytest.mark.fast
class TheWeightingIsTheTasksTest(unittest.TestCase):
    """Clients of 3, 7 and 12 rows, in batches of 5: the pooled cross-entropy's gradient."""

    def test_classification(self) -> None:
        from fedbrew.core import registry
        from fedbrew.tasks.classification.torch_classification import TorchClassificationTask

        registry.register_builtin_components()
        torch.manual_seed(0)
        task = TorchClassificationTask(
            model_config={"name": "mlp", "input_dim": 4, "hidden_dim": 6, "num_classes": 3},
            batch_size=5,
        )
        model = task.build_model()
        splits = [(torch.randn(n, 4), torch.randint(0, 3, (n,))) for n in (3, 7, 12)]
        params = trainable(model)
        gradient = WeightedGradient(params)
        for features, targets in splits:
            for batch in task.build_dataloader({"x": features, "y": targets}, {"shuffle": False}):
                loss, count = task.objective_loss(model, batch)
                gradient.add(loss, count)
        pooled = torch.nn.functional.cross_entropy(
            model(torch.cat([f for f, _ in splits])), torch.cat([t for _, t in splits])
        )
        expected = torch.autograd.grad(pooled, list(params.values()))
        for (name, mean), wanted in zip(gradient.mean().items(), expected, strict=True):
            with self.subTest(parameter=name):
                torch.testing.assert_close(mean.float(), wanted, rtol=1e-5, atol=1e-7)


class _Examples(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def components(self, name: str) -> Any:
        path = self.root / f"{name}.yaml"
        config = example_config(name)
        config["experiment"]["output_dir"] = str(self.root / name)
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return build_components(load_config(path))


class AtRandomPointsTest(_Examples):
    """Every linear example: the pooled objective's gradient, and batched against sequential."""

    def test_every_example(self) -> None:
        generator = torch.Generator().manual_seed(7)
        for name in EXAMPLES:
            with self.subTest(example=name):
                components = self.components(name)
                server, dataset, task = components.server, components.dataset, components.task
                server.initialize()
                reference, batched = SequentialGradNorm(), BatchedEvaluator()
                for _ in range(3):
                    state = _random_point(server, generator)
                    expected = _pooled(task, server, dataset, state)
                    sequential = reference.measure(server, dataset)[GRAD_NORM_COLUMN]
                    device = batched.evaluate_grad_norm(server, dataset)[GRAD_NORM_COLUMN]
                    self.assertLess(_relative(sequential, expected), 1e-12)
                    self.assertLess(_relative(device, sequential), TOLERANCE)

    def test_fed_lasso_against_its_analytic_subgradient(self) -> None:
        """The smooth part by fed-lasso's own ``gradient``, soft-thresholded here at x_j = 0."""

        components = self.components("fed-lasso")
        server, dataset, task = components.server, components.dataset, components.task
        server.initialize()
        problem = sys.modules[type(task).__module__]
        level = task.spec.penalty_strength
        generator = torch.Generator().manual_seed(11)
        for _ in range(3):
            x = _random_point(server, generator)["x"]
            self.assertTrue(bool((x == 0).any()) and bool((x != 0).any()))
            smooth = torch.stack(
                [
                    problem.gradient(x, task._design.cpu(), row, 0.0, "l1")
                    for row in task._client_targets.cpu()
                ]
            ).mean(dim=0)
            minimum = torch.where(
                x == 0,
                torch.sign(smooth) * torch.clamp(smooth.abs() - level, min=0.0),
                smooth + level * torch.sign(x),
            )
            expected = float(minimum @ minimum)
            measured = SequentialGradNorm().measure(server, dataset)[GRAD_NORM_COLUMN]
            self.assertLess(_relative(measured, expected), 1e-12)


def _random_point(server: Any, generator: torch.Generator) -> dict[str, torch.Tensor]:
    """A random global model, a third of its coordinates exactly zero; set on the server."""

    state = {}
    for key, value in server._model_state.items():
        if torch.is_floating_point(value):
            value = torch.randn(value.shape, generator=generator, dtype=value.dtype)
            value = torch.where(
                torch.rand(value.shape, generator=generator) < 1 / 3, 0.0, value
            ).to(value.dtype)
        state[key] = value
    server._model_state = state
    return state


def _pooled(task: Any, server: Any, dataset: Any, state: dict[str, torch.Tensor]) -> float:
    """``||g||^2`` of autograd's gradient of every train row in one batch, F's l1 made minimal.

    The l1 step is the task's own map (fed-lasso's analytic check is the one
    that does not trust it).
    """

    from fedbrew.core.grad_norm import _train_split

    model = copy.deepcopy(task.build_model(server.model_config))
    task.load_federated_model_state(model, state)
    splits = [
        task.split_rows(_train_split(dataset.get_client_data(client)))
        for client in dataset.list_clients()
    ]
    rows = tuple(torch.cat([split[k] for split in splits]) for k in range(len(splits[0])))
    params = trainable(model)
    loss, _ = task.functional_loss(model, None, None, rows, None)
    grads = dict(zip(params, torch.autograd.grad(loss, list(params.values())), strict=True))
    minimum = minimum_norm_gradient(grads, params, task.objective_l1(model))
    return float(sum(torch.sum(g.double() ** 2) for g in minimum.values()))


class ThePathsOfARunTest(ExecutorRuns):
    """Sequential, batched per round and resident: the same column, and nothing else moved."""

    def test_fed_lasso(self) -> None:
        self._check("fed-lasso")

    def test_drift_quad(self) -> None:
        self._check("drift-quad")

    def _check(self, name: str) -> None:
        off = _clean(example_config(name))
        on = copy.deepcopy(off)
        on.setdefault("evaluation", {})["grad_norm"] = {"every": 1}
        runs = {
            "sequential off": self.run_config(off, "sequential"),
            "sequential on": self.run_config(on, "sequential"),
            "resident off": self.run_config(off, "batched"),
            "resident on": self.run_config(on, "batched"),
        }
        with per_round():
            runs["per round on"] = self.run_config(on, "batched")
        self.assertEqual(_executor(runs["resident on"])["rounds"], {"used": "resident"})
        self.assertEqual(_executor(runs["per round on"])["rounds"]["used"], "per_round")
        rows = {label: _rows(output) for label, output in runs.items()}
        reference = [float(row[GRAD_NORM_COLUMN]) for row in rows["sequential on"]]
        self.assertEqual(len(reference), ROUNDS)
        for label in ("per round on", "resident on"):
            for round_id, (row, value) in enumerate(
                zip(rows[label], reference, strict=True), start=1
            ):
                with self.subTest(path=label, round=round_id):
                    self.assertLess(_relative(float(row[GRAD_NORM_COLUMN]), value), TOLERANCE)
        for path in ("sequential", "resident"):
            with self.subTest(unchanged=path):
                self.assertEqual(
                    _without_timing(rows[f"{path} on"], GRAD_NORM_COLUMN),
                    _without_timing(rows[f"{path} off"]),
                )
                for checkpoint in sorted((runs[f"{path} off"] / "checkpoints").glob("round_*.pt")):
                    self._compare_checkpoint(
                        runs[f"{path} on"] / "checkpoints" / checkpoint.name,
                        checkpoint,
                        True,
                        0.0,
                    )

    def test_only_the_scheduled_rounds_carry_it(self) -> None:
        config = _clean(example_config("fed-lasso"))
        config.setdefault("evaluation", {})["grad_norm"] = {"every": 3}
        for executor in ("sequential", "batched"):
            with self.subTest(executor=executor):
                rows = _rows(self.run_config(config, executor))
                carried = [int(row["round_id"]) for row in rows if row[GRAD_NORM_COLUMN] != ""]
                # Round 1 and the final round are pinned, as for every schedule.
                self.assertEqual(carried, [1, 3, ROUNDS])


class OffCostsNothingTest(ExecutorRuns):
    """Left at never, no gradient pass runs in any path and no column is written."""

    def test_no_path_measures_it(self) -> None:
        from fedbrew.core.resident_evaluation import ResidentEvaluation

        refuse = mock.Mock(side_effect=AssertionError("a gradient pass ran"))
        config = _clean(example_config("fed-lasso"))
        with (
            mock.patch.object(SequentialGradNorm, "measure", refuse),
            mock.patch.object(BatchedEvaluator, "evaluate_grad_norm", refuse),
            mock.patch.object(ResidentEvaluation, "enqueue_grad_norm", refuse),
        ):
            outputs = [self.run_config(config, "sequential"), self.run_config(config, "batched")]
            with per_round():
                outputs.append(self.run_config(config, "batched"))
        refuse.assert_not_called()
        for output in outputs:
            with self.subTest(output=output.name):
                self.assertNotIn(GRAD_NORM_COLUMN, _rows(output)[0])


class ThePlannedColumnsAreWrittenTest(unittest.TestCase):
    """Every task, on, round-trips: the header's column list is the CSV's."""

    def test_every_task(self) -> None:
        from fedbrew.core.logging import _planned_metric_names
        from tests.test_planned_columns_every_task import HAS_LLM, _Data, _runnable, _written

        configs = [REPO / "configs" / "dev" / "synthetic.yaml"]
        configs += [
            REPO / "configs" / "examples" / family / "fedavg.yaml" for family in sorted(EXAMPLES)
        ]
        if HAS_LLM:
            configs.append(REPO / "configs" / "dev" / "tiny_causal_lm.yaml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            data = _Data(root / "data")
            for index, path in enumerate(configs):
                with self.subTest(config=str(path.relative_to(REPO))):
                    runnable = _runnable(path, data, root / f"run{index}")
                    raw = yaml.safe_load(runnable.read_text(encoding="utf-8"))
                    raw.setdefault("evaluation", {})["grad_norm"] = {"every": 1}
                    runnable.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
                    planned = set(_planned_metric_names(load_config(runnable)))
                    self.assertIn(GRAD_NORM_COLUMN, planned)
                    written = _written(runnable)
                    self.assertEqual(planned - written, set(), "planned, not written")
                    self.assertEqual(written - planned, set(), "written, not planned")


@pytest.mark.fast
class TheKeyTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def _load(self, edit: Any) -> Any:
        raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
        raw["experiment"]["output_dir"] = str(self.root / "out")
        edit(raw)
        path = self.root / "smoke.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return load_config(path)

    @staticmethod
    def _measured(raw: dict[str, Any], every: Any = 1) -> None:
        raw.setdefault("evaluation", {})["grad_norm"] = {"every": every}

    def test_it_is_off_by_default(self) -> None:
        self.assertEqual(load_config(SMOKE).evaluation.grad_norm.every, "never")

    def test_a_task_that_declares_no_gradient_refuses_it(self) -> None:
        from fedbrew.core.registry import tasks

        with (
            mock.patch.object(type(tasks), "grad_norm", return_value=None),
            self.assertRaisesRegex(RunRefused, "declares no gradient of its objective"),
        ):
            self._load(self._measured)

    def test_an_unknown_key_is_refused(self) -> None:
        with self.assertRaisesRegex(RunRefused, "evaluation.grad_norm.evry"):
            self._load(lambda raw: raw.setdefault("evaluation", {}).update(grad_norm={"evry": 1}))

    def test_a_monitor_on_it_needs_it_measured_and_watches_it_fall(self) -> None:
        def watch(raw: dict[str, Any]) -> None:
            raw.setdefault("divergence", {})["metric"] = GRAD_NORM_COLUMN

        with self.assertRaisesRegex(RunRefused, "no round carries grad_norm_sq"):
            self._load(watch)
        config = self._load(lambda raw: (watch(raw), self._measured(raw)))
        self.assertEqual(divergence_direction(config), "min")

    def test_every_shipped_task_declares_it(self) -> None:
        from fedbrew.core import registry

        registry.register_builtin_components()
        for task in ("classification", "causal_lm"):
            with self.subTest(task=task):
                self.assertTrue(registry.tasks.grad_norm(task))
        for problem in sorted((REPO / "examples").glob("*/problem.py")):
            with self.subTest(example=problem.parent.name):
                text = problem.read_text(encoding="utf-8")
                self.assertIn("GRAD_NORM_GLOSS = (", text)
                self.assertIn("grad_norm=", text)
                self.assertIn("def objective_loss(", text)


def _rows(output: Path) -> list[dict[str, str]]:
    with (output / "round_metrics.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def _without_timing(rows: list[dict[str, str]], *dropped: str) -> list[dict[str, str]]:
    return [
        {
            key: value
            for key, value in row.items()
            if not key.endswith("_sec") and key not in dropped
        }
        for row in rows
    ]


if __name__ == "__main__":
    unittest.main()
