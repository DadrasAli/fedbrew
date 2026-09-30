"""examples/heterogeneous-quadratic: its closed forms against numerics, and runs against them.

Pinned here:

- the construction's identities hold for every member and control, and each
  closed form equals an independent numeric route to it: the quadratic's
  FedAvg fixed point against its exact round map iterated to convergence; the
  lasso's and the double well's client minimisers against a dense grid; the
  floors at the design's cells against the values its numpy checks printed;
- the task's loss over a client's rows has the analytic gradient of f_i, and
  ``grad_norm_sq``'s definition gives 0 at x* -- for the lasso only because
  the minimum-norm subgradient soft-thresholds the off-support coordinates;
- FedAvg with exact gradients, run through fedbrew, ends on the manifest's
  closed-form floor;
- the batched executor (the resident round for FedAvg) against the sequential
  one, on iid minibatches: within the executor tolerance.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from fedbrew.core import extensions
from fedbrew.core.config import standalone_config_mapping
from fedbrew.core.grad_norm import minimum_norm_gradient
from fedbrew.core.runner import run
from fedbrew.data.generate import generate_from_config

REPO = Path(__file__).resolve().parent.parent
EXAMPLE = REPO / "examples" / "heterogeneous-quadratic" / "problem.py"
problem = extensions._import_file(EXAMPLE)
Spec = problem.ProblemSpec

#: Every member at the design's centre, and each control.
CONTROLS = {
    "coupled": {},
    "iid": {"zeta_star": 0.0, "epsilon": 0.0},
    "shift-only": {"epsilon": 0.0},
    "decoupled": {"omega": 0.0},
}
MEMBERS = {
    problem.QUADRATIC: {},
    problem.LASSO: {"lam": 1.0},
    problem.DOUBLE_WELL: {"zeta_star": 0.25},
}


def _spec(member: str, **dials: Any) -> Any:
    return Spec(member=member, **{**MEMBERS[member], **dials})


@pytest.mark.fast
class TheIdentitiesTest(unittest.TestCase):
    def test_every_member_and_control(self) -> None:
        for member in MEMBERS:
            for control, dials in CONTROLS.items():
                with self.subTest(member=member, control=control):
                    problem.check_identities(_spec(member, **dials))

    def test_the_heterogeneity_measures(self) -> None:
        spec = _spec(problem.QUADRATIC)
        self.assertAlmostEqual(spec.zeta_star_squared(), 1.0, places=13)
        self.assertAlmostEqual(spec.delta_hbar_delta(), 1.003826, places=6)
        for value in spec.correlation():
            self.assertAlmostEqual(value, 1.0, places=12)
        decoupled = _spec(problem.QUADRATIC, omega=0.0)
        self.assertLess(float(decoupled.delta_star().abs().max()), 1e-15)
        for value in _spec(problem.QUADRATIC, omega=0.25).correlation():
            self.assertAlmostEqual(value, 0.25, places=12)


@pytest.mark.fast
class TheQuadraticFloorTest(unittest.TestCase):
    """The closed-form fixed point against the round map iterated until it stops moving."""

    def test_against_the_iterated_round_map(self) -> None:
        for omega in (1.0, 0.25):
            spec = _spec(problem.QUADRATIC, omega=omega)
            a, linear = spec.curvature(), spec.shifts()
            for alpha, steps in ((0.1, 10), (0.01, 10), (0.01, 100)):
                with self.subTest(omega=omega, alpha=alpha, K=steps):
                    x = spec.centre().clone()
                    for _ in range(5000):
                        local = x.expand_as(a).clone()
                        for _ in range(steps):
                            local = local - alpha * (a * (local - spec.centre()) + linear)
                        moved = local.mean(dim=0)
                        settled = float((moved - x).abs().max()) <= 1e-17
                        x = moved
                        if settled:
                            break
                    iterated = float(spec.objective(x))
                    closed = spec.fedavg_floor(alpha, steps)
                    self.assertLess(abs(iterated - closed) / closed, 1e-9)

    def test_the_design_table(self) -> None:
        spec = _spec(problem.QUADRATIC)
        for (alpha, steps), expected in {
            (0.1, 10): 3.5603e-02,
            (0.1, 100): 8.8982e-02,
            (0.01, 10): 1.0138e-03,
            (0.01, 100): 3.4717e-02,
            (0.001, 10): 1.0195e-05,
            (0.001, 100): 1.1396e-03,
        }.items():
            with self.subTest(alpha=alpha, K=steps):
                self.assertAlmostEqual(spec.fedavg_floor(alpha, steps) / expected, 1.0, places=3)
        for control in ("iid", "shift-only", "decoupled"):
            with self.subTest(control=control):
                self.assertLess(
                    _spec(problem.QUADRATIC, **CONTROLS[control]).fedavg_floor(0.01, 100), 1e-30
                )


class TheNonquadraticMembersTest(unittest.TestCase):
    """Not fast: the lasso's floors iterate its round map for tens of thousands of rounds."""

    def test_client_minimisers_against_a_grid(self) -> None:
        grid = torch.linspace(-3.0, 3.0, 600001, dtype=torch.float64)
        for member, dials in ((problem.LASSO, {}), (problem.LASSO, {"lam": 0.25, "epsilon": 0.0})):
            spec = _spec(member, **dials)
            optima = spec.client_optima()
            a, linear, centre = spec.curvature(), spec.linear() + spec.shifts(), spec.centre()
            for client in (0, 17, 45):
                for j in range(spec.dim):
                    with self.subTest(member=member, dials=dials, client=client, j=j):
                        values = (
                            0.5 * a[client, j] * (grid - centre[j]) ** 2
                            + linear[client, j] * (grid - centre[j])
                            + spec.lam * grid.abs()
                        )
                        best = float(grid[values.argmin()])
                        self.assertLess(abs(best - float(optima[client, j])), 1.1e-5)
        spec = _spec(problem.DOUBLE_WELL)
        optima = spec.client_optima()
        a, linear = spec.curvature(), spec.shifts()
        for client in (0, 17, 45):
            for j in range(spec.dim):
                with self.subTest(member=problem.DOUBLE_WELL, client=client, j=j):
                    values = (
                        a[client, j] * problem.profile(problem.DOUBLE_WELL, grid, spec.theta)
                        + linear[client, j] * grid
                    )
                    best = float(grid[values.argmin()])
                    self.assertLess(abs(best - float(optima[client, j] - spec.centre()[j])), 1.1e-5)

    def test_the_floors_at_the_design_cells(self) -> None:
        cases = [
            (_spec(problem.LASSO), 0.01, (3.883e-02, 3.365e-02, 4.674e-02)),
            (_spec(problem.LASSO, lam=0.25, epsilon=0.0), 0.01, (2.427e-03, 9.906e-03, 3.323e-02)),
            (_spec(problem.DOUBLE_WELL), 0.01, (None, 7.286e-05, 1.995e-03)),
            (_spec(problem.DOUBLE_WELL, epsilon=0.0), 0.01, (None, 2.725e-09, 2.022e-06)),
        ]
        for spec, alpha, expected in cases:
            for steps, value in zip((1, 10, 100), expected, strict=True):
                if value is None:
                    continue
                with self.subTest(member=spec.member, epsilon=spec.epsilon, lam=spec.lam, K=steps):
                    self.assertAlmostEqual(spec.fedavg_floor(alpha, steps) / value, 1.0, places=3)
        self.assertEqual(_spec(problem.LASSO).support_disagreement(), 0)
        self.assertEqual(_spec(problem.LASSO, lam=0.25).support_disagreement(), 320)
        self.assertTrue(_spec(problem.DOUBLE_WELL).both_wells_kept())
        self.assertFalse(_spec(problem.DOUBLE_WELL, zeta_star=1.0).both_wells_kept())


@pytest.mark.fast
class TheTaskLossTest(unittest.TestCase):
    def _task(self, spec: Any) -> Any:
        # Nothing is registered: the loss needs no model built through the
        # registry, and a registration made here would be the package's, which
        # an extension load of the same file in this process then refuses.
        reference = {"problem": problem._problem_of(spec)}
        return problem.HeterogeneousQuadraticTask(dataset_metadata={"reference": reference})

    def test_a_clients_rows_have_the_gradient_of_f_i(self) -> None:
        generator = torch.Generator().manual_seed(3)
        for member in MEMBERS:
            spec = _spec(member, sigma=20.0)
            task = self._task(spec)
            rows = spec.client_rows()
            a, linear = spec.curvature(), spec.linear() + spec.shifts()
            for client in (0, 33):
                with self.subTest(member=member, client=client):
                    x = torch.randn(spec.dim, generator=generator, dtype=torch.float64)
                    x.requires_grad_(True)
                    model = torch.nn.Module()
                    model.x = x
                    loss, _ = task.functional_loss(
                        model, None, None, (rows[client], torch.zeros(len(rows[client])))
                    )
                    (autograd,) = torch.autograd.grad(loss, x)
                    u = x.detach() - spec.centre()
                    expected = a[client] * problem.profile_slope(member, u, spec.theta)
                    expected = expected + linear[client]
                    if member == problem.LASSO:
                        expected = expected + spec.lam * torch.sign(x.detach())
                    torch.testing.assert_close(autograd, expected, rtol=1e-12, atol=1e-12)

    def test_the_minimum_norm_subgradient_vanishes_at_the_optimum(self) -> None:
        for member in MEMBERS:
            spec = _spec(member, sigma=20.0)
            task = self._task(spec)
            pooled = spec.client_rows().reshape(-1, 2 * spec.dim)
            model = torch.nn.Module()
            model.x = torch.nn.Parameter(spec.optimum().clone())
            loss, _ = task.functional_loss(model, None, None, (pooled, torch.zeros(len(pooled))))
            (gradient,) = torch.autograd.grad(loss, model.x)
            minimum = minimum_norm_gradient(
                {"x": gradient}, {"x": model.x}, task.objective_l1(model)
            )["x"]
            with self.subTest(member=member):
                self.assertLess(float(minimum.abs().max()), 1e-11)
                if member == problem.LASSO:
                    # Autograd's own subgradient is not zero off the support:
                    # there it is the smooth gradient, -lam s*_j = -+0.5 lam.
                    self.assertGreater(float(gradient.abs().max()), 0.4)


class _Runs(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def data(self, sigma: float) -> Path:
        config = yaml.safe_load(
            (REPO / "data" / "configs" / "examples" / "heterogeneous-quadratic.yaml").read_text()
        )
        config["dataset"]["output_dir"] = str(self.root / f"data-{sigma}")
        config["dataset"]["extensions"] = [str(EXAMPLE)]
        config["problem"]["sigma"] = sigma
        path = self.root / f"generate-{sigma}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return Path(generate_from_config(path))

    def run_arm(self, arm: str, manifest: Path, executor: str, **edits: Any) -> list[dict]:
        config = standalone_config_mapping(
            REPO / "configs" / "examples" / "heterogeneous-quadratic" / f"{arm}.yaml"
        )
        output = self.root / f"{arm}-{executor}-{len(list(self.root.iterdir()))}"
        config["experiment"]["output_dir"] = str(output)
        config["experiment"]["extensions"] = [str(EXAMPLE)]
        config["data"]["path"] = str(manifest)
        config["runtime"]["quiet"] = True
        # The reference, and the batched path held to it, on autograd.
        config["runtime"]["performance"].update(executor=executor, gradient_form="autograd")
        for key, value in edits.items():
            section, name = key.split("__")
            config[section][name] = value
        path = output.with_suffix(".yaml")
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        run(path, args=None)
        with (output / "round_metrics.csv").open(newline="") as handle:
            return list(csv.DictReader(handle))


class FedAvgEndsOnItsFloorTest(_Runs):
    """Exact gradients, alpha 0.1, K 10: the gap the run ends at is the closed form's."""

    def test_the_coupled_centre(self) -> None:
        manifest = self.data(0.0)
        reference = yaml.safe_load(manifest.read_text())["reference"]
        floor = next(
            cell["gap"]
            for cell in reference["fedavg_exact_floor"]
            if cell["alpha"] == 0.1 and cell["local_steps"] == 10
        )
        rows = self.run_arm(
            "fedavg_k10",
            manifest,
            "batched",
            client__learning_rate=0.1,
            schedule__rounds=150,
        )
        measured = float(rows[-1]["central_test_optimality_gap"])
        self.assertLess(abs(measured - floor) / floor, 1e-9)
        self.assertAlmostEqual(floor / 3.5603e-02, 1.0, places=3)


class BatchedAgainstSequentialTest(_Runs):
    """iid minibatches at sigma 20: every metric within the executor tolerance."""

    def test_fedavg_and_scaffold(self) -> None:
        manifest = self.data(20.0)
        for arm in ("fedavg_k10", "scaffold_k10", "minibatch_sgd_k10"):
            with self.subTest(arm=arm):
                batched = self.run_arm(arm, manifest, "batched", schedule__rounds=12)
                sequential = self.run_arm(arm, manifest, "sequential", schedule__rounds=12)
                self.assertEqual(len(batched), len(sequential))
                for row_b, row_s in zip(batched, sequential, strict=True):
                    for key, value in row_s.items():
                        if key.endswith("_sec") or value == "":
                            continue
                        a, b = float(value), float(row_b[key])
                        scale = max(abs(a), abs(b), 1e-300)
                        self.assertLess(abs(a - b) / scale, 1e-12, f"{arm} {key}")


if __name__ == "__main__":
    unittest.main()
