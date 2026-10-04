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
import dataclasses
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch
import yaml

from fedbrew.core import extensions
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
        # The example's own file, not sys.modules: whether the module the
        # extension loader put there is still there depends on what else the
        # worker ran before this test.
        problem = extensions._import_file(REPO / "examples" / "fed-lasso" / "problem.py")
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


class TheResidentPassGathersItsRowsOnceTest(ExecutorRuns):
    """The resident pass gathers its chunks of rows once and reads them every round after.

    Where a copy of every train row would not fit in a quarter of the free
    memory, it gathers them each round instead, and the run is the same.
    """

    def test_kept_and_gathered_each_round(self) -> None:
        from fedbrew.core.resident_evaluation import ResidentEvaluation

        config = _clean(example_config("fed-lasso"))
        config.setdefault("evaluation", {})["grad_norm"] = {"every": 1}
        seen: list[Any] = []
        real = ResidentEvaluation._grad_chunks

        def recorded(evaluation: Any) -> Any:
            seen.append(real(evaluation))
            return seen[-1]

        with mock.patch.object(ResidentEvaluation, "_grad_chunks", recorded):
            kept = self.run_config(config, "batched")
        self.assertEqual(_executor(kept)["rounds"], {"used": "resident"})
        self.assertEqual(len(seen), ROUNDS)
        self.assertIsInstance(seen[0], list)
        self.assertTrue(all(chunks is seen[0] for chunks in seen))
        seen.clear()
        with (
            mock.patch.object(ResidentEvaluation, "_grad_chunks", recorded),
            mock.patch("fedbrew.core.resident_evaluation.free_memory", return_value=0),
        ):
            gathered = self.run_config(config, "batched")
        self.assertEqual(len(seen), ROUNDS)
        self.assertFalse(any(isinstance(chunks, list) for chunks in seen))
        self.assertEqual(_without_timing(_rows(kept)), _without_timing(_rows(gathered)))


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

    def test_one_pass_is_the_default_and_a_bool(self) -> None:
        self.assertIs(load_config(SMOKE).evaluation.grad_norm.fused, True)

        def stated(value: Any) -> Any:
            return lambda raw: raw.setdefault("evaluation", {}).update(
                grad_norm={"every": 1, "fused": value}
            )

        self.assertIs(self._load(stated(False)).evaluation.grad_norm.fused, False)
        with self.assertRaisesRegex(RunRefused, "evaluation.grad_norm.fused must be true or false"):
            self._load(stated("yes"))

    def test_the_closed_form_is_the_default_form(self) -> None:
        self.assertEqual(load_config(SMOKE).evaluation.grad_norm.gradient_form, "closed_form")

        def stated(value: Any) -> Any:
            return lambda raw: raw.setdefault("evaluation", {}).update(
                grad_norm={"every": 1, "gradient_form": value}
            )

        loaded = self._load(stated("autograd"))
        self.assertEqual(loaded.evaluation.grad_norm.gradient_form, "autograd")
        for value in ("closed", True, ["autograd"]):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(RunRefused, "evaluation.grad_norm.gradient_form must be"),
            ):
                self._load(stated(value))

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


class WhereOnePassCannotBeTest(unittest.TestCase):
    """``fused_pass``: the pass where F is the central pass's, and otherwise why there is none.

    Where the central pass is measured in parts on every client's train rows
    and its loss keeps its graph (fed-logistic-l1), one pass; each condition
    taken away, the reason it names; with no gradient measured, no record.
    """

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def _plan(self, config: dict[str, Any]) -> tuple[Any, dict[str, Any], Any]:
        from fedbrew.core.grad_norm import fused_pass

        path = self.root / f"config{len(list(self.root.iterdir()))}.yaml"
        config["experiment"]["output_dir"] = str(self.root / "out")
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        loaded = load_config(path)
        components = build_components(loaded)
        fused, record = fused_pass(loaded.evaluation, components.server, components.dataset)
        return fused, record, components

    @staticmethod
    def _logistic(**evaluation: Any) -> dict[str, Any]:
        from tests.test_fed_logistic_l1 import small_config

        config = small_config("logistic", "nonconvex")
        config["evaluation"]["central_test"] = {"every": 1}
        config["evaluation"]["grad_norm"] = {"every": 1}
        config["evaluation"].update(evaluation)
        return config

    def test_where_it_is_and_is_not(self) -> None:
        from fedbrew.core.grad_norm import FusedPass

        fused, record, components = self._plan(self._logistic())
        self.assertIsInstance(fused, FusedPass)
        self.assertTrue(fused.closed)
        self.assertEqual(record, {"pass": "fused", "asked": "fused", "gradient": "closed_form"})
        fused, record, _ = self._plan(
            self._logistic(grad_norm={"every": 1, "gradient_form": "autograd"})
        )
        self.assertFalse(fused.closed)
        self.assertEqual(
            record,
            {
                "pass": "fused",
                "asked": "fused",
                "gradient": "autograd",
                "gradient_reason": "asked for",
            },
        )
        self.assertEqual(self._plan(self._logistic(grad_norm={"every": "never"}))[:2], (None, {}))
        for label, config, reason in (
            ("asked for", self._logistic(grad_norm={"every": 1, "fused": False}), "asked for"),
            (
                "no central pass",
                self._logistic(central_test={"every": "never"}),
                "the central pass is not measured",
            ),
            (
                "fed-lasso",
                {**example_config("fed-lasso"), "evaluation": {"grad_norm": {"every": 1}}},
                "not measured in parts",
            ),
        ):
            with self.subTest(case=label):
                if label == "fed-lasso":
                    config["evaluation"]["central_test"] = {"every": 1}
                fused, record, _ = self._plan(config)
                self.assertIsNone(fused)
                self.assertEqual(record["pass"], "separate")
                self.assertIn(reason, record["reason"])
        self._each_condition_taken_away(components)

    def _each_condition_taken_away(self, components: Any) -> None:
        from fedbrew.core.grad_norm import fused_pass

        evaluation = load_config(next(self.root.glob("config0.yaml"))).evaluation
        server, dataset = components.server, components.dataset

        class Own(type(server)):  # type: ignore[misc]
            def evaluate_global(self, global_data: Any, model: Any = None) -> dict[str, float]:
                return {}

        own = copy.copy(server)
        own.__class__ = Own
        fewer = copy.copy(dataset)
        rows = dataset.get_global_data()
        fewer.get_global_data = lambda: {key: value[:-1] for key, value in rows.items()}
        task_class = type(server.task)
        detached = mock.patch.object(
            task_class, "functional_eval", _detached(task_class.functional_eval)
        )
        autograd = dataclasses.replace(
            evaluation,
            grad_norm=dataclasses.replace(evaluation.grad_norm, gradient_form="autograd"),
        )
        for label, arguments, patch, reason in (
            ("a server's own pass", (own, dataset), nullcontext(), "central pass is its own"),
            ("other rows", (server, fewer), nullcontext(), "not every client's train rows"),
        ):
            with self.subTest(case=label), patch:
                fused, record = fused_pass(evaluation, *arguments)
                self.assertIsNone(fused)
                self.assertEqual(record["pass"], "separate")
                self.assertIn(reason, record["reason"])
        with self.subTest(case="a detached loss"), detached:
            # Autograd's pass needs the loss's graph; the closed form reads none.
            fused, record = fused_pass(autograd, server, dataset)
            self.assertIsNone(fused)
            self.assertIn("without its graph", record["reason"])
            fused, record = fused_pass(evaluation, server, dataset)
            self.assertEqual((record["pass"], record["gradient"]), ("fused", "closed_form"))
        with (
            self.subTest(case="no closed form of F"),
            mock.patch.object(task_class, "closed_form_eval", None),
        ):
            fused, record = fused_pass(evaluation, server, dataset)
            self.assertFalse(fused.closed)
            self.assertEqual((record["pass"], record["gradient"]), ("fused", "autograd"))
            self.assertIn("closed_form_eval", record["gradient_reason"])

    def test_the_closed_form_reads_a_stack_holding_its_rows(self) -> None:
        """``share_rows``: the stack seen as one client where it is the batch, else nothing."""

        from fedbrew.core.grad_norm import FusedPass

        _, _, components = self._plan(self._logistic())
        task = components.server.task
        model = task.build_model(components.server.model_config)
        generator = torch.Generator().manual_seed(4)
        features = torch.randn(3, 5, 8, generator=generator, dtype=torch.float64)
        labels = torch.where(torch.rand(3, 5, generator=generator) < 0.5, -1.0, 1.0).double()
        pooled = (features.reshape(15, 8), labels.reshape(15))
        (stack,) = task.closed_form_rows(model, (features, labels))

        def fused(closed: Any = model) -> FusedPass:
            return FusedPass(task, pooled, 15, True, closed=closed)

        shared = fused()
        self.assertTrue(shared.share_rows((stack,)))
        (view,) = shared.batches[0]
        self.assertEqual(view.data_ptr(), stack.data_ptr())
        self.assertTrue(torch.equal(view, fused().batches[0][0]))
        for label, other in (
            ("other order", (stack.flip(0).contiguous(),)),
            ("not contiguous", (stack.transpose(0, 1),)),
            ("another dtype", (stack.float(),)),
            ("two tensors", (stack, stack)),
        ):
            with self.subTest(case=label):
                self.assertFalse(fused().share_rows(other))
        self.assertFalse(fused(closed=None).share_rows((stack,)))
        self.assertFalse(FusedPass(task, pooled, 5, True, closed=model).share_rows((stack,)))

    def test_other_rows_are_not_f(self) -> None:
        from fedbrew.core.grad_norm import _same_rows

        generator = torch.Generator().manual_seed(3)
        parts = [
            (torch.randn(n, 4, generator=generator), torch.randn(n, generator=generator))
            for n in (3, 5, 2)
        ]
        pooled = tuple(torch.cat([part[k] for part in parts]) for k in range(2))
        order = torch.randperm(10, generator=generator)
        self.assertTrue(_same_rows(pooled, parts))
        self.assertTrue(_same_rows(tuple(t[order] for t in pooled), parts))
        self.assertFalse(_same_rows(tuple(t[:-1] for t in pooled), parts))
        changed = (pooled[0].clone(), pooled[1])
        changed[0][4, 2] += 1.0
        self.assertFalse(_same_rows(changed, parts))
        self.assertFalse(_same_rows(pooled[:1], parts))


def _detached(functional_eval: Any) -> Any:
    def detached(self: Any, *args: Any, **kwargs: Any) -> Any:
        measured = functional_eval(self, *args, **kwargs)
        return {key: value.detach() for key, value in measured.items()}

    return detached


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
