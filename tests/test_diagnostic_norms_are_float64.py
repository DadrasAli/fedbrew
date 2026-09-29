"""SCAFFOLD's and FedLALR's diagnostic norms are finite while the state they measure is.

``squared_l2_norm_model_state`` is the square of every one of them:
``control_delta_norm``, ``client_control_norm``, ``server_control_norm``,
``mean_client_control_delta_norm``, ``momentum_norm`` and
``second_moment_norm``. It squared each entry in float32, so one entry above
about 1.8e19 made the norm ``inf`` while a float64 state was finite: on
``configs/examples/nonconvex-simplex/scaffold.yaml`` the SCAFFOLD norms were
``inf`` from round 68 or 69 while the iterate grew finitely (metrics audit
§7.4). It squares in float64 now. Nothing trains on it: over 80 rounds of that
arm every other column was identical before and after (checked 2026-09-29).
"""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

import pytest
import torch
import yaml

from fedbrew.core import runner
from fedbrew.core.config import standalone_config_mapping
from fedbrew.core.torch_utils import squared_l2_norm_model_state
from fedbrew.data.generate import generate_from_config

REPO = Path(__file__).resolve().parent.parent
NORMS = (
    "control_delta_norm",
    "client_control_norm",
    "server_control_norm",
    "mean_client_control_delta_norm",
)


@pytest.mark.fast
class TheSquareIsTakenInFloat64Test(unittest.TestCase):
    def test_an_entry_past_float32s_square_root_of_its_maximum(self) -> None:
        state = {"w": torch.tensor([3.0e19, 4.0e19], dtype=torch.float64)}
        squared = squared_l2_norm_model_state(state)
        self.assertTrue(math.isfinite(squared))
        self.assertTrue(math.isclose(squared, 2.5e39, rel_tol=1e-15))

    def test_a_float32_state_is_squared_in_float64_too(self) -> None:
        # 3e19 is a float32 value; its square is not.
        state = {"w": torch.tensor([3.0e19], dtype=torch.float32)}
        squared = squared_l2_norm_model_state(state)
        self.assertTrue(math.isfinite(squared))
        self.assertAlmostEqual(math.sqrt(squared) / float(state["w"][0]), 1.0, places=12)


class TheScaffoldArmThatOverflowedTest(unittest.TestCase):
    """nonconvex-simplex/scaffold past round 68, where the float32 norms became inf."""

    def test_every_norm_is_finite_through_round_70(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator = yaml.safe_load(
                (REPO / "data/configs/examples/nonconvex-simplex.yaml").read_text()
            )
            generator["dataset"]["output_dir"] = str(root / "data")
            generator["dataset"]["extensions"] = [
                str(REPO / path) for path in generator["dataset"].get("extensions", [])
            ]
            (root / "generator.yaml").write_text(yaml.safe_dump(generator))
            manifest = generate_from_config(root / "generator.yaml")

            config = standalone_config_mapping(
                REPO / "configs/examples/nonconvex-simplex/scaffold.yaml"
            )
            config["data"]["path"] = str(manifest)
            config["experiment"]["output_dir"] = str(root / "run")
            config["experiment"]["extensions"] = [
                str(REPO / path) for path in config["experiment"]["extensions"]
            ]
            config["schedule"]["rounds"] = 70
            (root / "run.yaml").write_text(yaml.safe_dump(config))
            runner.run(root / "run.yaml", runner.parse_args(["--quiet"]))

            with (root / "run" / "round_metrics.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 70)
        # The state really is past float32's range by then: the norms the
        # float32 square overflowed on are above 1.8e19.
        self.assertGreater(float(rows[-1]["server_control_norm"]), 1.8e19)
        for row in rows:
            for name in NORMS:
                with self.subTest(round=row["round_id"], metric=name):
                    self.assertTrue(math.isfinite(float(row[name])))


if __name__ == "__main__":
    unittest.main()
