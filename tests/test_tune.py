"""``fedbrew tune``: the two protocols reproduce picks worked out by hand.

The search (``fedbrew/core/tuning.py``) is first driven by hand-made score
landscapes, which pin each rule: the extension past an edge, the stop on a tie,
the cap on steps, a grid that cannot be extended, the tie broken toward the
centre, two dials extended at once, the median over seeds with a failed run.

Then both protocols run for real, through ``fedbrew sweep``, on drift-quad
(``examples/drift-quad``) at full participation and one local step, where the
FedAvg round is exactly gradient descent on ``F(x) = x^T A x / 2`` with
``A = diag(a_j)``, ``a_j = 100^(j/15)``, from ``x_0,j = 1/sqrt(a_j)``. So the gap
after round ``t`` is, by hand,

    F(x_t) - F* = sum_j (1/2) (1 - eta a_j)^(2t),

and every candidate's score follows: the mean over rounds of its log10
(grid_and_edge) or its mean over rounds (pilot's running mean). Worked out from
that formula over 20 rounds, the mean of log10 of the gap is 0.6732, 0.5619 and
0.4147 at eta = 2^-9, 2^-8 and 2^-7: the best is the grid's top, so it is
extended to 2^-6, 0.2146, better by 0.20 decades, and again to 2^-5, which
diverges (|1 - eta a_15| = 2.1). The pick is 2^-6, interior. For the pilot,
centre 1e-3 over 10 rounds, the running means are 7.801, 6.505 and 3.256 at
1e-4, 1e-3 and 1e-2: the top again, extended to 1e-1, which diverges; the pick
is 1e-2, interior.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import io
import json
import math
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from fedbrew.core.config import TuningConfig
from fedbrew.core.refusal import RunRefused
from fedbrew.core.tuning import (
    METHODS,
    Candidate,
    RunScore,
    Settings,
    Tuner,
    aggregate_scores,
    internal,
    parse_axis,
    pick,
    validate_section,
)

GAP = "central_test_optimality_gap"


class Landscape(Tuner):
    """The search with a score per point in place of runs."""

    def __init__(
        self, method: str, dials: dict[str, Any], score: Callable[..., float | None], **given: Any
    ) -> None:
        chosen = METHODS[method]
        self.axes = [parse_axis(key, spec, chosen) for key, spec in dials.items()]
        self.keys = [axis.key for axis in self.axes]
        self.settings = Settings(
            method=chosen,
            metric="m",
            direction=given.get("direction", "min"),
            seeds=given.get("seeds", [0]),
            rounds=None,
            aggregate=given.get("aggregate", "median"),
            tie=given.get("tie", chosen.tie),
            tie_kind=given.get("tie_kind", chosen.tie_kind),
            max_steps=given.get("max_steps", 6),
            floor=1e-16,
        )
        self.initial = {axis.key: list(axis.values) for axis in self.axes}
        self.candidates: dict[tuple[Any, ...], Candidate] = {}
        self.steps: list[dict[str, Any]] = []
        self.log = lambda line: None
        self.landscape = score
        self.ran: list[list[tuple[Any, ...]]] = []
        self.out = Path(".")

    def run_candidates(self, new: Any) -> None:
        self.ran.append([candidate.point for candidate in new])
        for candidate in new:
            values = []
            for seed in self.settings.seeds:
                value = self.landscape(*candidate.point, seed=seed)
                candidate.runs.append(RunScore(seed, None, "completed", value))
                values.append(internal(value, self.settings.direction))
            candidate.aggregate = aggregate_scores(values, self.settings.aggregate)


def log2_distance(target: float) -> Callable[..., float]:
    return lambda x, seed=0: abs(math.log2(x) - target)


@pytest.mark.fast
class TheSearchTest(unittest.TestCase):
    def test_extended_past_the_edge_until_the_pick_is_interior(self) -> None:
        search = Landscape(
            "grid_and_edge", {"lr": {"base": 2, "exponents": [-8, -6]}}, log2_distance(-2)
        )
        selection = search.tune()
        self.assertEqual(selection.selected.point, (2.0**-2,))
        self.assertTrue(selection.interior)
        self.assertEqual(selection.stop["code"], "tie")
        # Steps: the grid, then -5, -4, -3, -2 each better by a whole decade, then -1 worse.
        self.assertEqual(search.ran[1:], [[(2.0**e,)] for e in (-5, -4, -3, -2, -1)])
        self.assertEqual(selection.final["lr"], [2.0**e for e in range(-8, 0)])

    def test_the_cap_on_extensions(self) -> None:
        search = Landscape(
            "grid_and_edge",
            {"lr": {"base": 2, "exponents": [-8, -6]}},
            log2_distance(-2),
            max_steps=2,
        )
        selection = search.tune()
        self.assertEqual(selection.selected.point, (2.0**-4,))
        self.assertFalse(selection.interior)
        self.assertEqual(selection.stop["code"], "max_steps")

    def test_an_extension_that_gains_no_more_than_a_tie_keeps_the_pick_before_it(self) -> None:
        scores = {2.0**-8: 2.0, 2.0**-7: 1.5, 2.0**-6: 1.0, 2.0**-5: 0.96}
        search = Landscape(
            "grid_and_edge", {"lr": {"base": 2, "exponents": [-8, -6]}}, lambda x, seed=0: scores[x]
        )
        selection = search.tune()
        self.assertEqual(selection.selected.point, (2.0**-6,))
        self.assertTrue(selection.interior)
        self.assertEqual(selection.stop["code"], "tie")
        self.assertEqual([c.point for c in selection.tied], [(2.0**-5,), (2.0**-6,)])

    def test_a_stated_grid_that_is_not_geometric_is_not_extended(self) -> None:
        search = Landscape(
            "grid_and_edge", {"beta": {"values": [0.0, 0.5, 0.9]}}, lambda x, seed=0: -x
        )
        selection = search.tune()
        self.assertEqual(selection.selected.point, (0.9,))
        self.assertEqual(selection.stop["code"], "not_extendable")
        self.assertFalse(selection.interior)

    def test_a_geometric_stated_grid_is_extended_by_its_ratio(self) -> None:
        search = Landscape(
            "grid_and_edge",
            {"gamma": {"values": [0.1, 1, 10]}},
            lambda x, seed=0: abs(math.log10(x) - 2),
        )
        selection = search.tune()
        self.assertEqual(selection.selected.point, (100.0,))
        self.assertTrue(selection.interior)

    def test_pilot_ties_go_toward_the_centre(self) -> None:
        # 1e-1 is best by 5e-4 relative: within the pilot's 1e-3, so tied with the centre.
        scores = {1e-3: 1.0005, 1e-2: 1.0, 1e-1: 0.9995}
        search = Landscape(
            "pilot", {"lr": {"centre": 1e-2, "decades": 1}}, lambda x, seed=0: scores[x]
        )
        selection = search.tune()
        self.assertEqual(selection.selected.point, (1e-2,))
        self.assertEqual(selection.stop["code"], "interior")
        self.assertEqual(sorted(c.point for c in selection.tied), [(1e-2,), (1e-1,)])
        # grid_and_edge on the same scores takes the best and so extends past it.
        best = Landscape(
            "grid_and_edge",
            {"lr": {"values": [1e-3, 1e-2, 1e-1]}},
            lambda x, seed=0: scores.get(x, 2.0),
        )
        self.assertEqual(best.tune().selected.point, (1e-1,))
        self.assertEqual(best.ran[1], [(1.0,)])

    def test_two_dials_at_a_corner_are_extended_together(self) -> None:
        search = Landscape(
            "grid_and_edge",
            {
                "lr": {"base": 2, "exponents": [-3, -1]},
                "gamma": {"values": [1, 10, 100]},
            },
            lambda x, g, seed=0: abs(math.log2(x) - 0) + abs(math.log10(g) - 3),
        )
        selection = search.tune()
        self.assertEqual(selection.selected.point, (1.0, 1000.0))
        # The first extension adds a row and a column: 4 x 4 - 3 x 3 = 7 new candidates.
        self.assertEqual(len(search.ran[1]), 7)
        self.assertTrue(selection.interior)

    def test_a_metric_better_higher(self) -> None:
        search = Landscape(
            "grid_and_edge",
            {"lr": {"base": 2, "exponents": [-3, -1]}},
            lambda x, seed=0: -abs(math.log2(x) + 2),
            direction="max",
        )
        self.assertEqual(search.tune().selected.point, (0.25,))

    def test_a_failed_seed_and_the_median(self) -> None:
        # Seed 2 fails (None) on the top point: its median over three seeds is the worse of two.
        def score(x: float, seed: int = 0) -> float | None:
            if x == 2.0**-1 and seed == 2:
                return None
            return abs(math.log2(x) + 1) + (0.5 if x == 2.0**-1 and seed == 1 else 0.0)

        search = Landscape(
            "grid_and_edge", {"lr": {"base": 2, "exponents": [-3, -1]}}, score, seeds=[0, 1, 2]
        )
        candidate = search.tune().candidates
        top = next(c for c in candidate if c.point == (0.5,))
        self.assertEqual(top.aggregate, 0.5)

    def test_every_candidate_failing_is_an_error(self) -> None:
        from fedbrew.core.tuning import TuneError

        search = Landscape(
            "grid_and_edge", {"lr": {"base": 2, "exponents": [-3, -1]}}, lambda x, seed=0: None
        )
        with self.assertRaisesRegex(TuneError, "every candidate failed"):
            search.tune()


@pytest.mark.fast
class ThePiecesTest(unittest.TestCase):
    def test_grids(self) -> None:
        grid = METHODS["grid_and_edge"]
        self.assertEqual(
            parse_axis("a", {"base": 2, "exponents": [-3, -1]}, grid).values, [0.125, 0.25, 0.5]
        )
        pilot = METHODS["pilot"]
        axis = parse_axis("a", {"centre": 0.003, "decades": 2}, pilot)
        self.assertEqual(axis.values, [3e-05, 0.0003, 0.003, 0.03, 0.3])
        self.assertEqual(axis.ratio, 10.0)
        axis.extend("low")
        self.assertEqual((axis.values[0], axis.first), (3e-06, -1))
        self.assertEqual(axis.index(0.003), 2)

    def test_method_and_dial_refusals(self) -> None:
        grid, pilot = METHODS["grid_and_edge"], METHODS["pilot"]
        with self.assertRaisesRegex(RunRefused, "powers of 10 around a stated centre"):
            parse_axis("a", {"values": [1, 2, 3]}, pilot)
        with self.assertRaisesRegex(RunRefused, "takes a grid"):
            parse_axis("a", {"centre": 1}, grid)
        with self.assertRaisesRegex(RunRefused, "at least 3"):
            parse_axis("a", {"values": [1, 2]}, grid)
        with self.assertRaisesRegex(RunRefused, "exactly one of"):
            parse_axis("a", {"values": [1, 2, 3], "centre": 2}, grid)
        with self.assertRaisesRegex(RunRefused, "unknown keys"):
            parse_axis("a", {"values": [1, 2, 3], "step": 2}, grid)

    def test_the_section(self) -> None:
        validate_section(TuningConfig())
        with self.assertRaisesRegex(RunRefused, "methods are"):
            validate_section(
                TuningConfig(method="random", metric="m", dials={"a": {"values": [1, 2, 3]}})
            )
        with self.assertRaisesRegex(RunRefused, "pilot horizon"):
            validate_section(TuningConfig(method="pilot", metric="m", dials={"a": {"centre": 1}}))
        with self.assertRaisesRegex(RunRefused, "repeats a seed"):
            validate_section(
                TuningConfig(
                    method="grid_and_edge",
                    metric="m",
                    dials={"a": {"values": [1, 2, 3]}},
                    seeds=[1, 1],
                )
            )
        with self.assertRaisesRegex(RunRefused, "no method"):
            validate_section(TuningConfig(metric="m"))

    def test_the_median_with_failed_runs(self) -> None:
        self.assertEqual(aggregate_scores([1.0, math.inf, 3.0], "median"), 3.0)
        self.assertEqual(aggregate_scores([1.0, math.inf], "median"), math.inf)
        self.assertEqual(aggregate_scores([1.0, 2.0], "median"), 1.5)
        self.assertEqual(aggregate_scores([math.inf, math.inf], "median"), math.inf)
        self.assertEqual(aggregate_scores([1.0, math.inf], "mean"), math.inf)

    def test_pick(self) -> None:
        axes = [parse_axis("a", {"values": [1, 2, 4, 8, 16]}, METHODS["grid_and_edge"])]
        scores = {(1,): 1.0, (2,): 0.5, (4,): 0.52, (8,): 0.6, (16,): 0.9}
        chosen, tied = pick(scores, axes, METHODS["grid_and_edge"], 0.05, "absolute")
        self.assertEqual((chosen, sorted(tied)), ((2,), [(2,), (4,)]))
        chosen, _ = pick(scores, axes, METHODS["pilot"], 0.05, "absolute")
        self.assertEqual(chosen, (4,))


def drift_quad(evaluation: dict[str, Any] | None = None) -> dict[str, Any]:
    """drift-quad's FedAvg arm as gradient descent: full participation, one local step."""

    from tests.test_batched_executor_tolerance import example_config

    config = example_config("drift-quad", "fedavg")
    config["server"]["participation_rate"] = 1.0
    config["schedule"]["local_iterations"] = 1
    config["runtime"]["checkpointing"].update(save_every_round=False, enabled=False)
    config["reporting"]["per_client_csv"] = False
    if evaluation:
        config.setdefault("evaluation", {}).update(evaluation)
    return config


def gap(eta: float, t: int) -> float:
    curvature = [100.0 ** (j / 15) for j in range(16)]
    return math.fsum(0.5 * (1.0 - eta * a) ** (2 * t) for a in curvature)


class TuneRuns(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def tune(self, config: dict[str, Any], *flags: str) -> tuple[str, Path]:
        from fedbrew.cli import tune as cli

        config = copy.deepcopy(config)
        config["experiment"]["output_dir"] = str(self.root / "base")
        path = self.root / "base.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        out = self.root / "tune"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli.main(["--config", str(path), "--out", str(out), *flags])
        return stdout.getvalue(), out

    def candidates(self, out: Path) -> dict[float, dict[str, Any]]:
        selection = json.loads((out / "selection.json").read_text())
        return {c["values"]["client.learning_rate"]: c for c in selection["candidates"]}


class GridAndEdgeOnDriftQuadTest(TuneRuns):
    def test_the_pick_worked_out_by_hand(self) -> None:
        config = drift_quad({"central_test": {"every": 1}})
        config["schedule"]["rounds"] = 20
        config["tuning"] = {
            "method": "grid_and_edge",
            "metric": "optimality_gap",
            "dials": {"client.learning_rate": {"base": 2, "exponents": [-9, -7]}},
        }
        printed, out = self.tune(config)
        selection = json.loads((out / "selection.json").read_text())
        self.assertEqual(selection["selected"]["values"], {"client.learning_rate": 2.0**-6})
        self.assertTrue(selection["selected"]["interior"])
        self.assertEqual(selection["stop"]["code"], "tie")
        self.assertEqual(selection["metric"], GAP)
        self.assertEqual(
            selection["dials"]["client.learning_rate"]["final"], [2.0**e for e in range(-9, -4)]
        )
        for eta, candidate in self.candidates(out).items():
            if eta == 2.0**-5:
                # Diverged, or so far above the others that it is worst either way.
                self.assertTrue(candidate["aggregate"] is None or candidate["aggregate"] > 1.0)
                continue
            by_hand = math.fsum(math.log10(gap(eta, t)) for t in range(1, 21)) / 20
            self.assertAlmostEqual(candidate["aggregate"], by_hand, delta=1e-9, msg=eta)
        resolved = yaml.safe_load((out / "selected.yaml").read_text())
        self.assertEqual(resolved["client"]["learning_rate"], 2.0**-6)
        self.assertNotIn("tuning", resolved)
        self.assertEqual(resolved["schedule"]["rounds"], 20)
        evidence = list(csv.DictReader((out / "evidence.csv").open()))
        self.assertEqual(len(evidence), 5)
        self.assertEqual([r["selected"] for r in evidence].count("True"), 1)
        self.assertTrue((out / "analysis" / "groups.csv").is_file())
        self.assertIn("picked client.learning_rate=0.015625", printed)
        # Run again: every run is reused and the pick is the same.
        again, _ = self.tune(config)
        self.assertIn("0 runs to do, 3 finished and reused", again)
        self.assertEqual(
            json.loads((out / "selection.json").read_text())["selected"], selection["selected"]
        )


class PilotOnDriftQuadTest(TuneRuns):
    def test_the_pick_worked_out_by_hand(self) -> None:
        config = drift_quad()
        config["schedule"]["rounds"] = 200
        config["tuning"] = {
            "method": "pilot",
            "metric": "optimality_gap",
            "rounds": 10,
            "dials": {"client.learning_rate": {"centre": 1e-3, "decades": 1}},
        }
        _, out = self.tune(config)
        selection = json.loads((out / "selection.json").read_text())
        self.assertEqual(selection["selected"]["values"], {"client.learning_rate": 1e-2})
        self.assertTrue(selection["selected"]["interior"])
        self.assertEqual(selection["tie"], {"value": 1e-3, "kind": "relative"})
        for eta, candidate in self.candidates(out).items():
            if eta == 1e-1:
                self.assertTrue(candidate["aggregate"] is None or candidate["aggregate"] > 1e6)
                continue
            by_hand = math.fsum(gap(eta, t) for t in range(1, 11)) / 10
            self.assertAlmostEqual(candidate["aggregate"] / by_hand, 1.0, delta=1e-12, msg=eta)
        # The pilot ran 10 rounds; the resolved config keeps the full horizon.
        run = Path(self.candidates(out)[1e-2]["runs"][0]["run"])
        self.assertEqual(json.loads((run / "run.json").read_text())["final_round"], 10)
        resolved = yaml.safe_load((out / "selected.yaml").read_text())
        self.assertEqual(resolved["schedule"]["rounds"], 200)
        self.assertEqual(resolved["client"]["learning_rate"], 1e-2)
        self.assertNotIn("convergence", resolved)

    def test_the_plan_runs_nothing(self) -> None:
        config = drift_quad()
        config["tuning"] = {
            "method": "pilot",
            "metric": "optimality_gap",
            "rounds": 10,
            "seeds": [1, 2],
            "dials": {"client.learning_rate": {"centre": 1e-3}},
        }
        printed, out = self.tune(config, "--plan")
        self.assertIn("3 candidates x 2 seeds", printed)
        self.assertIn("group of 3", printed)
        self.assertEqual(len(list((out / "configs").glob("*.yaml"))), 6)
        self.assertFalse((out / "runs").exists())
        written = yaml.safe_load(next((out / "configs").glob("*seed1.yaml")).read_text())
        self.assertEqual(written["convergence"], {"metrics": [GAP]})
        self.assertEqual(written["schedule"]["rounds"], 10)
        self.assertEqual(written["experiment"]["seed"], 1)
        self.assertNotIn("tuning", written)

    def test_a_metric_the_run_cannot_mean_is_refused(self) -> None:
        config = drift_quad()
        config["tuning"] = {
            "method": "pilot",
            "metric": "accuracy",
            "rounds": 10,
            "dials": {"client.learning_rate": {"centre": 1e-3}},
        }
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            self.tune(config)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("does not evaluate", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
