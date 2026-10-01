"""``fedbrew analyze``: its statistics on hand-made CSVs, worked out by hand.

Five runs in two groups (the config is the learning rate; the seed is the
other thing that differs). Group lr-0.1 holds three runs whose
``grad_norm_sq`` halves every round from 4, 8 and 2, so every number below is a
power of two or a mean of them:

    seed 1:  4    2    1    0.5      log2 mean 0.5    mean 1.875
    seed 2:  8    4    2    1        log2 mean 1.5    mean 3.75
    seed 3:  2    1    0.5  0.25     log2 mean -0.5   mean 0.9375

Last iterate 0.5, 1, 0.25: median 0.5, min 0.25, max 1; the 0.25 quantile of
the three means (0.9375, 1.875, 3.75) is 0.9375 + 0.5 * (1.875 - 0.9375) =
1.40625 (linear interpolation at position 0.5) and the 0.75 quantile 2.8125.
The other group's two runs make an even count, whose median is the mean of
the middle two.
"""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pytest
import yaml

from fedbrew.cli import analyze as cli
from fedbrew.core.analysis import (
    Analysis,
    AnalysisError,
    analyze,
    default_direction,
    median,
    quantile,
    quantile_name,
    spread,
    write_tables,
)

pytestmark = pytest.mark.fast

NAN = math.nan
LOG2 = math.log10(2.0)


def make_run(
    root: Path,
    name: str,
    seed: int,
    lr: float,
    columns: dict[str, list[float | None]],
    *,
    record: bool = True,
) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    rounds = len(next(iter(columns.values())))
    with (directory / "round_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["round_id", *columns])
        for index in range(rounds):
            writer.writerow(
                [
                    index + 1,
                    *("" if v is None else repr(v) for v in (c[index] for c in columns.values())),
                ]
            )
    if record:
        config = {
            "experiment": {"seed": seed, "name": name, "output_dir": str(directory), "tags": []},
            "client": {"learning_rate": lr},
            "runtime": {"extra": {"flush_every": 1}},
        }
        (directory / "run.json").write_text(
            json.dumps({"status": "completed", "config": config}), encoding="utf-8"
        )
    return directory


NORM = "grad_norm_sq"
ACC = "central_test_accuracy"
GAP = "central_test_optimality_gap"


class HandMade(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        runs = self.root / "runs"
        make_run(runs, "a1", 1, 0.1, {NORM: [4, 2, 1, 0.5], ACC: [None, 0.5, 0.4, 0.9]})
        make_run(runs, "a2", 2, 0.1, {NORM: [8, 4, 2, 1], ACC: [None, 0.6, 0.6, 0.7]})
        make_run(runs, "a3", 3, 0.1, {NORM: [2, 1, 0.5, 0.25], ACC: [None, 0.1, 0.2, 0.3]})
        make_run(runs, "b1", 1, 0.2, {NORM: [1, 1, 1, 1], ACC: [None, 0.0, 0.0, 0.0]})
        make_run(runs, "b2", 2, 0.2, {NORM: [3, 3, 3, 3], ACC: [None, 0.0, 0.0, 0.0]})
        self.runs = runs

    def result(self, **options: Any) -> Analysis:
        return analyze([self.runs], metrics=[NORM], **options)

    def group(self, result: Analysis, lr: str) -> str:
        return next(label for label in result.groups if lr in label)

    def row(self, result: Analysis, lr: str, metric: str, statistic: str) -> dict[str, Any]:
        group = self.group(result, lr)
        (row,) = [
            r
            for r in result.group_rows()
            if (r["group"], r["metric"], r["statistic"]) == (group, metric, statistic)
        ]
        return row


class TheStatisticsAreTheHandWorkedOnesTest(HandMade):
    def test_the_groups_are_the_configs_without_the_seed_and_labelled_by_what_differs(self) -> None:
        result = self.result()
        self.assertEqual(
            sorted(result.groups), ["client.learning_rate=0.1", "client.learning_rate=0.2"]
        )
        self.assertEqual(len(result.groups["client.learning_rate=0.1"]), 3)
        self.assertEqual(len(result.groups["client.learning_rate=0.2"]), 2)

    def test_per_run_statistics(self) -> None:
        by_run = {s.run: s for s in self.result().runs}
        a1, a2, a3 = by_run["a1"], by_run["a2"], by_run["a3"]
        self.assertEqual((a1.last, a2.last, a3.last), (0.5, 1.0, 0.25))
        self.assertEqual((a1.last_round, a1.best_round), (4, 4))
        self.assertEqual((a1.best, a2.best, a3.best), (0.5, 1.0, 0.25))
        for run, exponent in ((a1, 0.5), (a2, 1.5), (a3, -0.5)):
            self.assertAlmostEqual(run.mean_log10, exponent * LOG2, places=14)
        self.assertEqual((a1.running_mean, a2.running_mean, a3.running_mean), (1.875, 3.75, 0.9375))
        self.assertEqual({s.running_mean_source for s in (a1, a2, a3)}, {"reconstructed"})
        self.assertEqual((a1.seed, a2.seed, a3.seed), (1, 2, 3))

    def test_across_seeds_at_the_end(self) -> None:
        result = self.result(quantiles=[0.25, 0.75])
        last = self.row(result, "0.1", NORM, "last")
        self.assertEqual(
            (last["n"], last["median"], last["min"], last["max"]), (3.0, 0.5, 0.25, 1.0)
        )
        mean = self.row(result, "0.1", NORM, "running_mean")
        self.assertEqual((mean["q25"], mean["median"], mean["q75"]), (1.40625, 1.875, 2.8125))
        self.assertEqual((mean["min"], mean["max"]), (0.9375, 3.75))
        log = self.row(result, "0.1", NORM, "mean_log10")
        self.assertAlmostEqual(log["median"], 0.5 * LOG2, places=14)

    def test_an_even_count_has_the_mean_of_the_middle_two_as_its_median(self) -> None:
        result = self.result()
        last = self.row(result, "0.2", NORM, "last")
        self.assertEqual(
            (last["n"], last["median"], last["min"], last["max"]), (2.0, 2.0, 1.0, 3.0)
        )

    def test_per_round_across_seeds(self) -> None:
        result = self.result(quantiles=[0.25])
        curve = {
            (r["group"], r["variable"], r["round_id"]): r
            for r in result.curve_rows()
            if r["metric"] == NORM
        }
        group = self.group(result, "0.1")
        first = curve[(group, "value", 1)]
        self.assertEqual(
            (first["n"], first["median"], first["min"], first["max"]), (3.0, 4.0, 2.0, 8.0)
        )
        self.assertEqual(first["q25"], 3.0)  # 2, 4, 8 at position 0.5: 2 + 0.5 * (4 - 2)
        last = curve[(group, "value", 4)]
        self.assertEqual((last["median"], last["min"], last["max"]), (0.5, 0.25, 1.0))
        # The running mean of round 2 is (4+2)/2, (8+4)/2, (2+1)/2: 3, 6, 1.5.
        second = curve[(group, "running_mean", 2)]
        self.assertEqual((second["median"], second["min"], second["max"]), (3.0, 1.5, 6.0))
        self.assertEqual(curve[(group, "best_so_far", 3)]["median"], 1.0)


class EvaluatedRowsOnlyTest(HandMade):
    def test_a_blank_is_not_a_value_and_the_reconstruction_is_warned_about(self) -> None:
        result = analyze([self.runs], metrics=[ACC])
        a1 = next(s for s in result.runs if s.run == "a1")
        self.assertEqual((a1.rounds, a1.evaluated), (4, 3))
        self.assertEqual((a1.last, a1.last_round), (0.9, 4))
        self.assertEqual((a1.best, a1.best_round, a1.direction), (0.9, 4, "max"))
        self.assertEqual(a1.running_mean, math.fsum([0.5, 0.4, 0.9]) / 3)
        self.assertEqual(a1.curves["best_so_far"], {2: 0.5, 3: 0.5, 4: 0.9})
        self.assertEqual(sorted(a1.curves["value"]), [2, 3, 4])
        warned = [w for w in result.warnings if "a1" in w and "reconstructed" in w]
        self.assertEqual(len(warned), 1)
        self.assertIn("3 of 4 rounds evaluated", warned[0])

    def test_the_written_running_mean_column_is_the_one_used_and_is_not_warned_about(self) -> None:
        root = self.root / "convergence"
        values = [4.0, 2.0, 1.0, 0.5]
        means = [math.fsum(values[: t + 1]) / (t + 1) for t in range(4)]
        make_run(root, "c1", 1, 0.1, {NORM: [4, None, None, 0.5], f"{NORM}_running_mean": means})
        result = analyze([root], metrics=[NORM])
        (run,) = result.runs
        self.assertEqual((run.running_mean, run.running_mean_source), (1.875, "column"))
        self.assertEqual(run.curves["running_mean"], dict(enumerate(means, start=1)))
        self.assertEqual([w for w in result.warnings if "reconstructed" in w], [])

    def test_a_value_that_is_not_positive_is_left_out_of_the_log_and_warned_about(self) -> None:
        root = self.root / "zero"
        make_run(root, "z1", 1, 0.1, {GAP: [1.0, 0.1, 0.0, 0.01]})
        result = analyze([root], metrics=["optimality_gap"])
        (run,) = result.runs
        self.assertEqual(result.metrics, [GAP])
        self.assertAlmostEqual(run.mean_log10, (0.0 - 1.0 - 2.0) / 3, places=14)
        self.assertEqual((run.nonpositive, run.best, run.best_round), (1, 0.0, 3))
        self.assertTrue(any("not positive" in w for w in result.warnings))

    def test_a_value_that_is_not_finite_poisons_the_reconstructed_mean_not_the_best(self) -> None:
        root = self.root / "inf"
        make_run(root, "i1", 1, 0.1, {NORM: [1.0, 2.0, math.inf, 3.0]})
        (run,) = analyze([root], metrics=[NORM]).runs
        self.assertTrue(math.isnan(run.running_mean))
        self.assertEqual((run.best, run.last), (1.0, 3.0))


class TheStatisticsFunctionsTest(unittest.TestCase):
    def test_quantiles_are_linear_between_order_statistics(self) -> None:
        values = [4.0, 1.0, 3.0, 2.0]
        self.assertEqual(quantile(values, 0.0), 1.0)
        self.assertEqual(quantile(values, 1.0), 4.0)
        self.assertEqual(quantile(values, 0.5), 2.5)
        self.assertEqual(quantile(values, 0.25), 1.75)  # position 0.75: 1 + 0.75 * (2 - 1)
        self.assertEqual(median([5.0]), 5.0)
        self.assertTrue(math.isnan(quantile([], 0.5)))

    def test_the_spread_ignores_what_is_not_finite(self) -> None:
        out = spread([1.0, math.nan, 3.0, math.inf], [0.5])
        self.assertEqual(
            (out["n"], out["median"], out["min"], out["max"], out["q50"]), (2.0, 2.0, 1.0, 3.0, 2.0)
        )

    def test_names(self) -> None:
        self.assertEqual(
            (quantile_name(0.25), quantile_name(0.025), quantile_name(0.9)), ("q25", "q2.5", "q90")
        )
        self.assertEqual(default_direction("central_test_accuracy"), "max")
        self.assertEqual(default_direction("central_test_support_f1"), "max")
        self.assertEqual(default_direction("grad_norm_sq"), "min")


class TheTablesTest(HandMade):
    def test_four_files_tidy_and_blank_where_nothing_exists(self) -> None:
        result = analyze([self.runs], metrics=[NORM, ACC], quantiles=[0.1, 0.9])
        out = self.root / "out"
        paths = write_tables(result, out)
        self.assertEqual(
            [p.name for p in paths], ["runs.csv", "groups.csv", "curves.csv", "analysis.json"]
        )
        runs = list(csv.DictReader((out / "runs.csv").open()))
        self.assertEqual(len(runs), 10)  # five runs, two metrics
        self.assertEqual(
            list(runs[0]),
            [
                "group", "run", "seed", "status", "metric", "direction", "rounds", "evaluated",
                "last_round", "last", "best_round", "best", "mean_log10", "nonpositive",
                "running_mean", "running_mean_source",
            ],
        )  # fmt: skip
        groups = list(csv.DictReader((out / "groups.csv").open()))
        self.assertEqual(len(groups), 2 * 2 * 4)  # groups x metrics x statistics
        self.assertEqual(list(groups[0])[:3], ["group", "metric", "statistic"])
        self.assertEqual(list(groups[0])[3:], ["n", "median", "min", "max", "q10", "q90"])
        curves = list(csv.DictReader((out / "curves.csv").open()))
        # ACC is evaluated on rounds 2 to 4 only: no value row for round 1.
        self.assertFalse(
            [
                r
                for r in curves
                if r["metric"] == ACC and r["variable"] == "value" and r["round_id"] == "1"
            ]
        )
        document = json.loads((out / "analysis.json").read_text())
        self.assertEqual(document["settings"]["quantiles"], [0.1, 0.9])
        self.assertEqual(document["settings"]["directions"], {NORM: "min", ACC: "max"})
        self.assertEqual(len(document["runs"]), 10)
        self.assertEqual(len(document["groups"]), 2)
        self.assertTrue(document["warnings"])

    def test_a_direction_can_be_overridden(self) -> None:
        result = analyze([self.runs], metrics=[ACC], directions={ACC: "min"})
        a1 = next(s for s in result.runs if s.run == "a1")
        self.assertEqual((a1.best, a1.best_round), (0.4, 3))


class FindingRunsTest(HandMade):
    def test_a_directory_below_which_runs_are_a_sweep(self) -> None:
        self.assertEqual(len(analyze([self.root], metrics=[NORM]).runs), 5)

    def test_one_run_and_a_repeat_is_counted_once(self) -> None:
        result = analyze([self.runs / "a1", self.runs / "a1", self.runs], metrics=[NORM])
        self.assertEqual(len(result.runs), 5)

    def test_a_config_file_is_its_output_directory(self) -> None:
        config = self.root / "a1.yaml"
        config.write_text(
            yaml.safe_dump({"experiment": {"output_dir": str(self.runs / "a1")}}), encoding="utf-8"
        )
        result = analyze([config], metrics=[NORM])
        self.assertEqual([s.run for s in result.runs], ["a1"])

    def test_a_run_that_cannot_be_read_is_noted_and_the_others_are_analyzed(self) -> None:
        broken = self.runs / "broken"
        broken.mkdir()
        (broken / "round_metrics.csv").write_text("loss\n1\n", encoding="utf-8")
        result = analyze([self.runs], metrics=[NORM])
        self.assertEqual(len(result.runs), 5)
        self.assertTrue(any("broken" in note and "round_id" in note for note in result.notes))

    def test_a_run_without_a_record_is_its_directorys_own_group_and_says_so(self) -> None:
        make_run(self.root / "bare", "r1", 1, 0.1, {NORM: [1.0, 2.0]}, record=False)
        result = analyze([self.root / "bare"], metrics=[NORM])
        self.assertEqual(list(result.groups), ["bare"])
        self.assertIsNone(result.runs[0].seed)
        self.assertTrue(any("no run.json" in w for w in result.warnings))

    def test_nothing_to_analyze_and_unknown_metrics_are_errors_not_tracebacks(self) -> None:
        with self.assertRaisesRegex(AnalysisError, "no run to analyze"):
            analyze([self.root / "nowhere"])
        with self.assertRaisesRegex(AnalysisError, "no run has a 'nothing' column"):
            analyze([self.runs], metrics=["nothing"])
        with self.assertRaisesRegex(AnalysisError, "quantile"):
            analyze([self.runs], metrics=[NORM], quantiles=[1.5])
        bare = make_run(self.root / "x", "x1", 1, 0.1, {"other": [1.0]})
        with self.assertRaisesRegex(AnalysisError, "none of the default metrics"):
            analyze([bare])


class TheCommandTest(HandMade):
    def test_it_writes_the_tables_and_prints_a_summary(self) -> None:
        out = self.root / "cli-out"
        import contextlib
        import io

        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            cli.main(
                [str(self.runs), "--metrics", NORM, "--quantiles", "0.1", "0.9", "--out", str(out)]
            )
        text = stdout.getvalue()
        self.assertIn("5 runs in 2 groups", text)
        self.assertIn("client.learning_rate=0.1  grad_norm_sq  n=3", text)
        self.assertTrue((out / "groups.csv").is_file())

    def test_it_exits_2_with_a_message_when_there_is_nothing_to_do(self) -> None:
        import contextlib
        import io

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            cli.main([str(self.root / "nowhere")])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("no run to analyze", stderr.getvalue())

    def test_a_malformed_direction_is_refused(self) -> None:
        import contextlib
        import io

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            cli.main([str(self.runs), "--direction", "grad_norm_sq"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
