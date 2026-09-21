"""participation_probability: each client independently, for every strategy.

Algorithm 1 samples each of the m clients independently with probability p, so
a round's count is a draw and can be zero. For that to be usable by any
strategy three things must hold, and each has a class here: the draw is what it
says -- independent, at p, reproducible, not moved by the roster's order; a
server block names exactly one of the two schemes; and a round that selects
nobody is not aggregated yet still leaves a complete record, which the
end-to-end runs below read back from disk.
"""

from __future__ import annotations

import csv
import random
import statistics
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import torch
import yaml

from fedbrew.core import runner
from fedbrew.core.config import load_config
from fedbrew.core.protocol import ClientInfo
from fedbrew.servers.fedavg import FedAvgServer
from fedbrew.servers.fedlalr import FedLALRServer
from fedbrew.servers.fedopt import FedOptServer
from fedbrew.servers.scaffold import ScaffoldServer

#: Every tolerance below is at least four standard errors wide at this count.
ROUNDS = 4000


def _roster(count: int) -> list[ClientInfo]:
    return [ClientInfo(client_id=f"client_{index}", num_examples=10) for index in range(count)]


def _bernoulli(probability: object, seed: int = 42) -> FedAvgServer:
    return FedAvgServer(participation_rate=None, seed=seed, participation_probability=probability)


def _ids(server: FedAvgServer, roster: list[ClientInfo], round_id: int) -> list[str]:
    return [client.client_id for client in server.sample_clients(roster, round_id)]


class TheDrawTest(unittest.TestCase):
    def test_each_client_joins_at_the_configured_probability(self) -> None:
        server, roster = _bernoulli(0.3), _roster(10)
        joined = dict.fromkeys((client.client_id for client in roster), 0)
        for round_id in range(1, ROUNDS + 1):
            for client_id in _ids(server, roster, round_id):
                joined[client_id] += 1
        for client_id, count in joined.items():
            with self.subTest(client_id=client_id):
                self.assertAlmostEqual(count / ROUNDS, 0.3, delta=0.03)

    def test_the_count_varies_as_independent_draws_would(self) -> None:
        """Binomial(10, 0.3): mean 3, variance 2.1, and two given clients together
        in 9% of rounds. A fixed-size draw has variance 0; a correlated one misses
        the 9%."""

        server, roster = _bernoulli(0.3), _roster(10)
        counts: list[int] = []
        together = 0
        for round_id in range(1, ROUNDS + 1):
            selected = set(_ids(server, roster, round_id))
            counts.append(len(selected))
            together += {"client_0", "client_1"} <= selected
        self.assertAlmostEqual(statistics.fmean(counts), 3.0, delta=0.15)
        self.assertAlmostEqual(statistics.variance(counts), 2.1, delta=0.3)
        self.assertAlmostEqual(together / ROUNDS, 0.09, delta=0.02)

    def test_a_round_selects_nobody_as_often_as_it_should(self) -> None:
        server, roster = _bernoulli(0.2), _roster(5)
        empty = sum(not _ids(server, roster, round_id) for round_id in range(1, ROUNDS + 1))
        self.assertAlmostEqual(empty / ROUNDS, 0.8**5, delta=0.03)

    def test_probability_one_selects_what_rate_one_does_in_the_same_order(self) -> None:
        roster = _roster(7)
        fixed = FedAvgServer(participation_rate=1.0, seed=42)
        for round_id in range(1, 6):
            with self.subTest(round_id=round_id):
                self.assertEqual(
                    _ids(_bernoulli(1.0), roster, round_id), _ids(fixed, roster, round_id)
                )

    def test_the_schedule_is_a_function_of_the_seed(self) -> None:
        roster = _roster(20)

        def schedule(seed: int) -> list[list[str]]:
            server = _bernoulli(0.25, seed)
            return [_ids(server, roster, round_id) for round_id in range(1, 51)]

        self.assertEqual(schedule(42), schedule(42))
        self.assertNotEqual(schedule(42), schedule(43))

    def test_the_roster_order_moves_the_order_returned_but_not_the_draw(self) -> None:
        roster = _roster(30)
        shuffled = list(roster)
        random.Random(7).shuffle(shuffled)
        position = {client.client_id: index for index, client in enumerate(shuffled)}
        server = _bernoulli(0.4)
        for round_id in range(1, 21):
            with self.subTest(round_id=round_id):
                drawn = _ids(server, shuffled, round_id)
                self.assertEqual(set(drawn), set(_ids(server, roster, round_id)))
                self.assertEqual(drawn, sorted(drawn, key=position.__getitem__))

    def test_a_duplicate_client_id_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "ids must be unique"):
            _bernoulli(0.5).sample_clients(_roster(3) + _roster(1), 1)


class ExactlyOneSchemeTest(unittest.TestCase):
    def test_the_server_refuses_both_neither_and_a_value_outside_the_interval(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one of participation_rate"):
            FedAvgServer(participation_rate=None, seed=0)
        with self.assertRaisesRegex(ValueError, "exactly one of participation_rate"):
            FedAvgServer(participation_rate=0.5, seed=0, participation_probability=0.5)
        for bad in (0, -0.1, 1.5, True, float("nan")):
            with (
                self.subTest(probability=bad),
                self.assertRaisesRegex(
                    ValueError, r"participation_probability must be in \(0, 1\]"
                ),
            ):
                _bernoulli(bad)

    def test_every_built_in_server_forwards_the_probability(self) -> None:
        """The factory hands every strategy the same common kwargs, and each
        subclass spells out FedAvg's; one that dropped this would sample by a
        rule nobody configured."""

        common: dict[str, Any] = {
            "participation_rate": None,
            "seed": 0,
            "participation_probability": 0.5,
        }
        servers = {
            "fedavg": FedAvgServer(**common),
            "scaffold": ScaffoldServer(**common),
            "fedlalr": FedLALRServer(epsilon=1e-8, **common),
            "fedopt": FedOptServer(
                server_optimizer="fedadam",
                server_learning_rate=0.01,
                beta1=0.9,
                beta2=0.99,
                tau=1e-3,
                **common,
            ),
        }
        roster = _roster(20)
        for name, server in servers.items():
            with self.subTest(strategy=name):
                self.assertEqual(server.participation_probability, 0.5)
                self.assertIsNone(server.participation_rate)
                counts = {len(server.sample_clients(roster, round_id)) for round_id in range(1, 21)}
                self.assertGreater(len(counts), 1, "a fixed count is not a Bernoulli draw")


def _write_config(root: Path, *, seed: int = 42, rounds: int = 3, **participation: object) -> Path:
    config = {
        "experiment": {"seed": seed, "output_dir": str(root / "run"), "use_run_subdir": False},
        "server": {"strategy": "fedavg", "metrics": ["fit_loss"], **participation},
        "client": {
            "update_rule": "local_sgd",
            "batch_size": 4,
            "learning_rate": 0.1,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "nesterov": False,
            "learning_rate_schedule": "constant",
            "min_learning_rate": 0.0,
            "metrics": ["fit_loss"],
        },
        "data": {"num_clients": 2, "samples_per_client": 8, "input_dim": 4, "num_classes": 2},
        "model": {"name": "mlp", "input_dim": 4, "hidden_dim": 4, "num_classes": 2},
        "runtime": {
            "deterministic": True,
            "device": "cpu",
            "use_amp": False,
            "checkpointing": {
                "enabled": True,
                "save_last": False,
                "save_best": False,
                "save_every_round": True,
                "keep_last": None,
            },
        },
        "defaults": {"global_rounds": rounds, "local_iterations": 1},
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


class TheConfigNamesExactlyOneSchemeTest(unittest.TestCase):
    def _refusal(self, **participation: object) -> str:
        with tempfile.TemporaryDirectory() as directory:
            path = _write_config(Path(directory), **participation)
            with self.assertRaises(ValueError) as caught:
                load_config(str(path))
        return str(caught.exception)

    def test_both_are_refused(self) -> None:
        message = self._refusal(participation_rate=0.5, participation_probability=0.5)
        self.assertIn("exactly one of participation_rate", message)
        self.assertIn("both are set", message)

    def test_neither_is_refused(self) -> None:
        self.assertIn("neither is set", self._refusal())

    def test_a_probability_outside_the_interval_is_refused(self) -> None:
        for bad in (0, 1.5, "0.5", True):
            with self.subTest(probability=bad):
                self.assertIn(
                    "server.participation_probability must be in (0, 1]",
                    self._refusal(participation_probability=bad),
                )

    def test_a_probability_loads_with_the_rate_unset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write_config(Path(directory), participation_probability=0.25)
            config = load_config(str(path))
        self.assertEqual(config.server.participation_probability, 0.25)
        self.assertIsNone(config.server.participation_rate)


def _seed_for(pattern: tuple[bool, ...], probability: float, clients: int) -> int:
    """The first seed whose rounds select a client exactly where ``pattern`` says."""

    roster = _roster(clients)
    for seed in range(10_000):
        server = _bernoulli(probability, seed)
        drawn = tuple(bool(_ids(server, roster, r)) for r in range(1, len(pattern) + 1))
        if drawn == pattern:
            return seed
    raise AssertionError(f"no seed below 10000 selects clients as {pattern}")


def _run(
    root: Path, **overrides: object
) -> tuple[list[dict[str, str]], dict[int, dict[str, torch.Tensor]], list[tuple[Any, ...]]]:
    """Run a config through runner.run and read back what it wrote."""

    calls: list[tuple[Any, ...]] = []
    path = _write_config(root, **overrides)
    with mock.patch.object(
        runner, "client_progress_reporter", return_value=lambda *call: calls.append(call)
    ):
        runner.run(path, runner.parse_args(["--quiet"]))
    with (root / "run" / "round_metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    checkpoints = {}
    for file in (root / "run" / "checkpoints").glob("*.pt"):
        payload = torch.load(file, map_location="cpu", weights_only=False)
        checkpoints[int(payload["round_id"])] = payload["model_state"]
    return rows, checkpoints, calls


def _same_model(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> bool:
    return first.keys() == second.keys() and all(torch.equal(first[k], second[k]) for k in first)


class ARoundThatSelectsNobodyTest(unittest.TestCase):
    """End to end, on disk: the loop, the checkpoints, the metrics CSV, the footer."""

    def test_it_is_not_aggregated_and_is_still_recorded(self) -> None:
        pattern = (False, True, False)
        seed = _seed_for(pattern, probability=0.3, clients=2)
        with tempfile.TemporaryDirectory() as directory:
            rows, checkpoints, calls = _run(
                Path(directory), seed=seed, rounds=3, participation_probability=0.3
            )

        self.assertEqual([int(row["round_id"]) for row in rows], [1, 2, 3])
        counts = [int(row["num_clients"]) for row in rows]
        self.assertEqual([count > 0 for count in counts], list(pattern))
        # No client trained, so there is no fit loss, and nothing stands in for it.
        self.assertEqual([row["fit_loss"] == "" for row in rows], [True, False, True])

        self.assertEqual(sorted(checkpoints), [1, 2, 3])
        # Round 2 trained, so its model moved; round 3 selected nobody, so its
        # checkpointed model is round 2's, tensor for tensor.
        self.assertFalse(_same_model(checkpoints[1], checkpoints[2]))
        self.assertTrue(_same_model(checkpoints[2], checkpoints[3]))

        announced = [call for call in calls if call[3] == "fit" and call[1] == 0]
        self.assertEqual(announced, [(1, 0, 0, "fit"), (2, 0, counts[1], "fit"), (3, 0, 0, "fit")])


class AtProbabilityOneTest(unittest.TestCase):
    def test_a_run_is_the_run_at_rate_one(self) -> None:
        """Same selection in the same order, so the same arithmetic: every number
        recorded agrees, and so does every checkpointed tensor."""

        runs = []
        for participation in ({"participation_rate": 1.0}, {"participation_probability": 1.0}):
            with tempfile.TemporaryDirectory() as directory:
                rows, checkpoints, _ = _run(Path(directory), rounds=2, **participation)
            untimed = [{k: v for k, v in row.items() if not k.endswith("_sec")} for row in rows]
            runs.append((untimed, checkpoints))

        (rate_rows, rate_models), (probability_rows, probability_models) = runs
        self.assertEqual(rate_rows, probability_rows)
        self.assertEqual(sorted(rate_models), sorted(probability_models))
        for round_id, state in rate_models.items():
            with self.subTest(round_id=round_id):
                self.assertTrue(_same_model(state, probability_models[round_id]))


if __name__ == "__main__":
    unittest.main()
