"""tools/drift_diagnostic.py: rho-hat against its closed form on the heterogeneous quadratic.

At the optimum x-hat, with exact gradients, client i's H local FedAvg steps at
step eta move coordinate j by ``-(1 - (1 - eta a_ij)^H) zeta_ij / a_ij``, so its
pseudo-gradient is ``(1 - (1 - eta a_ij)^H) zeta_ij / (a_ij eta H)`` and rho-hat
is the norm of their mean -- ``(eta (H - 1) / 2) ||delta*||`` to first order,
and zero in the shift-only control, where every client has the same curvature.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch
import yaml

from fedbrew.core.config import standalone_config_mapping
from tests.test_heterogeneous_quadratic import EXAMPLE, REPO, _Runs, problem

_spec = importlib.util.spec_from_file_location(
    "drift_diagnostic", REPO / "tools" / "drift_diagnostic.py"
)
drift = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = drift
_spec.loader.exec_module(drift)


class DriftAtTheOptimumTest(_Runs):
    ETA, H = 1e-4, 10

    def _measure(self, manifest: Path) -> float:
        config = standalone_config_mapping(
            REPO / "configs" / "examples" / "heterogeneous-quadratic" / "fedavg_k10.yaml"
        )
        config["experiment"]["output_dir"] = str(self.root / "unused")
        config["experiment"]["extensions"] = [str(EXAMPLE)]
        config["data"]["path"] = str(manifest)
        config["runtime"]["performance"]["executor"] = "sequential"
        path = self.root / f"drift-{manifest.parent.name}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        reference = yaml.safe_load(manifest.read_text())["reference"]
        centre = torch.tensor(reference["x_hat"], dtype=torch.float64)
        result = drift.measure(
            path, model_state={"x": centre}, local_iterations=self.H, learning_rate=self.ETA
        )
        self.assertEqual(result["local_steps"], [self.H])
        self.assertEqual(result["clients"], 64)
        self.last = result
        return result["rho_hat"]

    def _dataset(self, **dials: float) -> Path:
        config = yaml.safe_load(
            (REPO / "data" / "configs" / "examples" / "heterogeneous-quadratic.yaml").read_text()
        )
        name = "-".join(f"{k}{v}" for k, v in dials.items()) or "centre"
        config["dataset"]["output_dir"] = str(self.root / name)
        config["dataset"]["extensions"] = [str(EXAMPLE)]
        config["problem"].update(sigma=0.0, **dials)
        path = self.root / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        from fedbrew.data.generate import generate_from_config

        return Path(generate_from_config(path))

    def test_the_coupled_centre_against_the_closed_form(self) -> None:
        spec = problem.ProblemSpec()
        a, zeta = spec.curvature(), spec.shifts()
        weight = (1.0 - (1.0 - self.ETA * a) ** self.H) / (a * self.ETA * self.H)
        closed = float(torch.linalg.vector_norm((weight * zeta).mean(dim=0)))
        first_order = (
            self.ETA * (self.H - 1) / 2 * float(torch.linalg.vector_norm(spec.delta_star()))
        )
        measured = self._measure(self._dataset())
        # The mean cancels to about 1e-3 of the clients' own pseudo-gradients,
        # each carrying w - w_c's rounding divided by eta H = 1e-3.
        self.assertLess(abs(measured - closed) / closed, 1e-8)
        # eta kappa (1 + eps)(H - 1) = 0.0135: second order is about 1% here.
        self.assertLess(abs(measured - first_order) / first_order, 0.01)

    def test_the_shift_only_control_has_none(self) -> None:
        # Zero up to the rounding of w - w_c, divided by eta H: against the
        # clients' own pseudo-gradients, which are of order zeta*.
        rho = self._measure(self._dataset(epsilon=0.0))
        self.assertLess(rho / self.last["mean_client_pseudo_gradient_norm"], 1e-10)


if __name__ == "__main__":
    unittest.main()
