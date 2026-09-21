"""FedLALR's learning-rate diagnostics: one name per quantity, each checked by value.

The client reports statistics of its per-coordinate rate alpha / sqrt(v_hat)
over its own coordinates, and the server a spread across clients of each
client's coordinate mean. Both used to be called
``client_effective_learning_rate_{mean,min,max}``, and under a client.metrics
that kept the client's ``_min`` but not ``_mean`` the round record carried the
clients' averaged minimum under the name documented as the server's minimum
across clients (FINDINGS.csv POST-F14). They are now
``effective_learning_rate_coordinate_*`` and
``effective_learning_rate_across_clients_*``; the old names are refused in a
config and on a resume.

Two clients with known v_hat and alpha = 1, so every rate is exact in binary:

  client a, 1 example:  v_hat = [1, 4, 16, 64]      rates [1, 1/2, 1/4, 1/8]
  client b, 3 examples: v_hat = [1/4, 1/4, 1, 4]    rates [2, 2, 1, 1/2]

  coordinate mean  a 15/32, b 11/8;  min a 1/8, b 1/2;  max a 1, b 2

Round record:
  effective_learning_rate_coordinate_mean  (1*15/32 + 3*11/8) / 4 = 147/128
  effective_learning_rate_coordinate_min   (1*1/8   + 3*1/2)  / 4 = 13/32
  effective_learning_rate_coordinate_max   (1*1     + 3*2)    / 4 = 7/4
    -- example-weighted across clients, like every client metric (chapter 08 §4.1)
  effective_learning_rate_across_clients_mean  (15/32 + 11/8) / 2 = 59/64
  effective_learning_rate_across_clients_std   |11/8 - 15/32| / 2 = 29/64
  effective_learning_rate_across_clients_min   15/32
  effective_learning_rate_across_clients_max   11/8
    -- unweighted, one value per client, population std
"""

from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pytest
import torch
import yaml

from fedbrew.clients.torch_fedlalr_client import _learning_rate_metrics
from fedbrew.core import runner
from fedbrew.core.config import load_config
from fedbrew.core.metrics import RETIRED_METRIC_NAMES
from fedbrew.core.protocol import FitResult, RoundInfo
from fedbrew.core.refusal import RunRefused
from fedbrew.servers.fedlalr import FedLALRServer

V_HAT = {
    "a": ({"w": torch.tensor([1.0, 4.0]), "b": torch.tensor([16.0, 64.0])}, 1),
    "b": ({"w": torch.tensor([0.25, 0.25]), "b": torch.tensor([1.0, 4.0])}, 3),
}
COORDINATE = ("mean", "min", "max")
COORDINATE_NAMES = [f"effective_learning_rate_coordinate_{s}" for s in COORDINATE]
ACROSS_NAMES = [
    f"effective_learning_rate_across_clients_{s}" for s in ("mean", "std", "min", "max")
]
EXPECTED_ROUND = {
    "effective_learning_rate_coordinate_mean": 147 / 128,
    "effective_learning_rate_coordinate_min": 13 / 32,
    "effective_learning_rate_coordinate_max": 7 / 4,
    "effective_learning_rate_across_clients_mean": 59 / 64,
    "effective_learning_rate_across_clients_std": 29 / 64,
    "effective_learning_rate_across_clients_min": 15 / 32,
    "effective_learning_rate_across_clients_max": 11 / 8,
}


def _server(metrics: list[str]) -> FedLALRServer:
    server = FedLALRServer(epsilon=1e-8, participation_rate=1.0, seed=0, metrics=metrics)
    server._model_state = {"w": torch.zeros(2), "b": torch.zeros(2)}
    server._model_state_scope = "full"
    server._model_state_metadata = {"model_state_scope": "full"}
    server._ensure_optimizer_state()
    return server


def _round(client_metric_names: list[str], server_metrics: list[str]) -> dict[str, float]:
    results = []
    for client_id, (v_hat, examples) in V_HAT.items():
        reported = _learning_rate_metrics(v_hat, 1.0)
        results.append(
            FitResult(
                round_id=1,
                client_id=client_id,
                num_examples=examples,
                payload={
                    "model_state": {"w": torch.zeros(2), "b": torch.zeros(2)},
                    "model_state_scope": "full",
                    "model_state_metadata": {"model_state_scope": "full"},
                    "momentum_state": {"w": torch.zeros(2), "b": torch.zeros(2)},
                    "second_moment_state": v_hat,
                },
                metrics={name: reported[name] for name in client_metric_names},
            )
        )
    round_info = RoundInfo(round_id=1, total_rounds=1)
    _server(server_metrics).aggregate(round_info, results)
    return dict(round_info.metrics)


@pytest.mark.fast
class EachColumnIsItsEstimandTest(unittest.TestCase):
    def test_the_client_reports_its_coordinate_statistics(self) -> None:
        self.assertEqual(
            _learning_rate_metrics(V_HAT["a"][0], 1.0),
            dict(zip(COORDINATE_NAMES, (15 / 32, 1 / 8, 1.0), strict=True)),
        )
        self.assertEqual(
            _learning_rate_metrics(V_HAT["b"][0], 1.0),
            dict(zip(COORDINATE_NAMES, (11 / 8, 1 / 2, 2.0), strict=True)),
        )

    def test_every_round_column_equals_its_hand_computed_value(self) -> None:
        metrics = _round(COORDINATE_NAMES, COORDINATE_NAMES)
        for name, expected in EXPECTED_ROUND.items():
            with self.subTest(column=name):
                self.assertEqual(metrics[name], expected)

    def test_the_audited_shape_no_longer_puts_one_quantity_under_another_s_name(self) -> None:
        """client.metrics keeps the coordinate minimum and not the mean."""

        metrics = _round(
            ["effective_learning_rate_coordinate_min"],
            ["effective_learning_rate_coordinate_min"],
        )
        self.assertEqual(metrics["effective_learning_rate_coordinate_min"], 13 / 32)
        for name in ACROSS_NAMES:
            self.assertNotIn(name, metrics)
        for name in RETIRED_METRIC_NAMES:
            self.assertNotIn(name, metrics)

    def test_no_name_is_both_a_coordinate_and_an_across_clients_statistic(self) -> None:
        self.assertFalse(set(COORDINATE_NAMES) & set(ACROSS_NAMES))
        self.assertEqual(set(RETIRED_METRIC_NAMES) & set(EXPECTED_ROUND), set())


def _smoke_fedlalr(directory: Path, **client_extra: object) -> dict:
    raw = yaml.safe_load(Path("configs/dev/smoke.yaml").read_text(encoding="utf-8"))
    raw["experiment"]["output_dir"] = str(directory / "run")
    raw["server"]["strategy"] = "fedlalr"
    raw["server"]["metrics"] = ["fit_loss", *COORDINATE_NAMES]
    raw["client"] = {
        "update_rule": "fedlalr",
        "batch_size": 4,
        "learning_rate": 0.01,
        "metrics": ["fit_loss", *COORDINATE_NAMES],
        **client_extra,
    }
    raw["runtime"]["checkpointing"] = {
        "enabled": True,
        "save_last": True,
        "save_best": False,
        "save_every_round": False,
        "keep_last": 0,
    }
    raw["client_statistics"] = {"per_client_csv": True}
    raw["defaults"]["global_rounds"] = 2
    return raw


@pytest.mark.fast
class ARetiredNameIsRefusedInAConfigTest(unittest.TestCase):
    def _refused(self, raw: dict) -> str:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaises(RunRefused) as caught:
                load_config(path)
        return str(caught.exception)

    def test_each_retired_name_in_each_place_names_its_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = _smoke_fedlalr(Path(directory))
        for old, guidance in RETIRED_METRIC_NAMES.items():
            for place in ("server.metrics", "client.metrics", "divergence.metric", "best_metric"):
                with self.subTest(name=old, place=place):
                    raw = yaml.safe_load(yaml.safe_dump(base))
                    if place == "server.metrics":
                        raw["server"]["metrics"].append(old)
                    elif place == "client.metrics":
                        raw["client"]["metrics"].append(old)
                    elif place == "divergence.metric":
                        raw["divergence"] = {"metric": old, "non_finite": True}
                    else:
                        raw["runtime"]["checkpointing"].update(save_best=True, best_metric=old)
                    message = self._refused(raw)
                    self.assertIn(f"{old!r} is a retired FedLALR diagnostic name", message)
                    self.assertIn(guidance, message)
                    self.assertIn("effective_learning_rate_", guidance)

    def test_no_shipped_config_names_one(self) -> None:
        for path in sorted(Path("configs").rglob("*.yaml")):
            with self.subTest(path=str(path)):
                text = path.read_text(encoding="utf-8")
                for old in RETIRED_METRIC_NAMES:
                    self.assertNotIn(old, text)


class ARunWritesTheNewNamesAndResumesOnlyOntoThemTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls._directory.name)
        cls.config = cls.root / "fedlalr.yaml"
        cls.config.write_text(yaml.safe_dump(_smoke_fedlalr(cls.root)), encoding="utf-8")
        with redirect_stdout(StringIO()):
            runner.run(cls.config, runner.parse_args(["--quiet"]))
        cls.output = cls.root / "run"

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def _header(self, path: Path) -> list[str]:
        with path.open(encoding="utf-8", newline="") as file:
            return next(csv.reader(file))

    def test_the_csvs_carry_the_new_names_only(self) -> None:
        round_header = self._header(self.output / "round_metrics.csv")
        self.assertTrue(set(ACROSS_NAMES) <= set(round_header))
        self.assertTrue(set(COORDINATE_NAMES) <= set(round_header))
        update_header = self._header(self.output / "client_update_metrics.csv")
        self.assertTrue(set(COORDINATE_NAMES) <= set(update_header))
        for header in (round_header, update_header):
            self.assertFalse(set(RETIRED_METRIC_NAMES) & set(header))

    def test_a_resume_onto_csvs_with_a_retired_name_is_refused_and_changes_nothing(self) -> None:
        for csv_name, new, old in (
            (
                "round_metrics.csv",
                "effective_learning_rate_across_clients_min",
                "client_effective_learning_rate_min",
            ),
            (
                "client_update_metrics.csv",
                "effective_learning_rate_coordinate_mean",
                "client_effective_learning_rate_mean",
            ),
        ):
            with self.subTest(file=csv_name), tempfile.TemporaryDirectory() as directory:
                copy = Path(directory) / "run"
                shutil.copytree(self.output, copy)
                path = copy / csv_name
                text = path.read_text(encoding="utf-8")
                header, rest = text.split("\n", 1)
                path.write_text(header.replace(new, old) + "\n" + rest, encoding="utf-8")
                before = {p: p.read_bytes() for p in sorted(copy.rglob("*")) if p.is_file()}
                args = runner.parse_args(
                    [
                        "--quiet",
                        "--output-dir",
                        str(copy),
                        "--resume-from",
                        str(copy / "checkpoints" / "latest.pt"),
                        "--rounds",
                        "3",
                    ]
                )
                with redirect_stdout(StringIO()), self.assertRaises(RunRefused) as caught:
                    runner.run(self.config, args)
                message = str(caught.exception)
                self.assertIn("retired FedLALR metric names", message)
                self.assertIn(f"{csv_name}: {old}", message)
                self.assertIn(RETIRED_METRIC_NAMES[old], message)
                after = {p: p.read_bytes() for p in sorted(copy.rglob("*")) if p.is_file()}
                self.assertEqual(after, before, "a refused resume changed the run directory")

    def test_the_same_resume_onto_the_new_names_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "run"
            shutil.copytree(self.output, copy)
            args = runner.parse_args(
                [
                    "--quiet",
                    "--output-dir",
                    str(copy),
                    "--resume-from",
                    str(copy / "checkpoints" / "latest.pt"),
                    "--rounds",
                    "3",
                ]
            )
            with redirect_stdout(StringIO()):
                state = runner.run(self.config, args)
            self.assertEqual(state.status, "completed")
            self.assertEqual([r.round_id for r in state.metrics_history], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
