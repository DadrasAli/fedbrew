"""Every split aggregate says how many clients it is over.

`evaluation.{split}.clients` takes `all`, `participating`, `sample:N` or
`resample:N`, and every setting produces the same column names. So
`val_accuracy_sample_weighted_avg` from a run over 200 clients and one over
3,597 are one column name over two populations, differing in standard error by
`sqrt(3597/200) = 4.2x`. `configs/femnist/fedavg_ft.yaml` is the one shipped
FEMNIST arm that samples, and it selects `best.pt` off that column, against ten
arms that do not.

`{split}_num_clients` makes the difference legible from `round_metrics.csv`
without opening two configs. P07-F06.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import pytest

from fedbrew.core.checkpointing import validate_selection_metric
from fedbrew.core.config import ClientStatisticsConfig, client_metric_names, load_config
from fedbrew.core.logging import _planned_metric_names, classify_metric
from fedbrew.core.loop import _aggregate_client_split_metrics
from fedbrew.core.metrics import metric_gloss
from fedbrew.core.protocol import EvalResult

pytestmark = pytest.mark.fast

SPLITS = ("train", "val", "test")
FEMNIST = sorted(Path("configs/femnist").glob("*.yaml"))


def _result(client_id: str, *, count: int, split: str = "val") -> EvalResult:
    metrics: dict[str, float] = {}
    if count:
        metrics[f"{split}_loss"] = 0.5
        metrics[f"{split}_accuracy"] = 0.75
    return EvalResult(
        round_id=1,
        client_id=client_id,
        num_examples=count,
        metrics=metrics,
        payload={"num_examples_by_split": {split: count}},
    )


class TheCountIsEmittedTest(unittest.TestCase):
    def test_it_counts_the_clients_the_averages_are_over(self) -> None:
        results = [_result(f"c{i}", count=4) for i in range(7)]
        aggregated = _aggregate_client_split_metrics(results, "val", ClientStatisticsConfig())
        self.assertEqual(aggregated["val_num_clients"], 7.0)

    def test_a_client_with_no_examples_is_not_counted(self) -> None:
        """The averages drop it, so counting it would describe another population."""

        results = [_result(f"c{i}", count=4) for i in range(5)]
        results.append(_result("empty", count=0))
        aggregated = _aggregate_client_split_metrics(results, "val", ClientStatisticsConfig())
        self.assertEqual(aggregated["val_num_clients"], 5.0)

    def test_two_population_sizes_are_distinguishable_by_the_column_alone(self) -> None:
        """The finding: the same column name over two populations."""

        statistics = ClientStatisticsConfig()
        small = _aggregate_client_split_metrics(
            [_result(f"c{i}", count=4) for i in range(200)], "val", statistics
        )
        large = _aggregate_client_split_metrics(
            [_result(f"c{i}", count=4) for i in range(3597)], "val", statistics
        )
        self.assertEqual(
            small["val_accuracy_sample_weighted_avg"],
            large["val_accuracy_sample_weighted_avg"],
            "this test needs the aggregates to agree, so that the count is the tell",
        )
        self.assertNotEqual(small["val_num_clients"], large["val_num_clients"])

    def test_every_split_emits_one(self) -> None:
        for split in SPLITS:
            with self.subTest(split=split):
                aggregated = _aggregate_client_split_metrics(
                    [_result("c0", count=3, split=split)], split, ClientStatisticsConfig()
                )
                self.assertIn(f"{split}_num_clients", aggregated)


class TheColumnIsDeclaredEverywhereItHasToBeTest(unittest.TestCase):
    def test_it_is_in_the_column_vocabulary(self) -> None:
        for split in SPLITS:
            with self.subTest(split=split):
                self.assertIn(
                    f"{split}_num_clients",
                    client_metric_names(split, ClientStatisticsConfig()),
                )

    def test_the_plan_header_lists_it(self) -> None:
        """A column written and not listed makes the header worse than none."""

        # FEMNIST's fedavg arm evaluates train and val, not test -- its test
        # numbers come from central_test -- so those two are what to assert.
        config = load_config("configs/femnist/fedavg.yaml")
        self.assertEqual(config.evaluation.test.every, "never")
        planned = _planned_metric_names(config)
        self.assertIn("val_num_clients", planned)
        self.assertIn("train_num_clients", planned)
        self.assertNotIn("test_num_clients", planned)

    def test_the_personal_pass_gets_its_own(self) -> None:
        config = load_config("configs/femnist/fedavg_ft.yaml")
        self.assertEqual(config.evaluation.model_scope, "both")
        planned = _planned_metric_names(config)
        self.assertIn("val_num_clients", planned)
        self.assertIn("personal_val_num_clients", planned)

    def test_it_has_a_gloss_and_a_group(self) -> None:
        for split in SPLITS:
            for name in (f"{split}_num_clients", f"personal_{split}_num_clients"):
                with self.subTest(column=name):
                    self.assertTrue(metric_gloss(name).endswith("."))
                    group, _, _, _ = classify_metric(name)
                    self.assertEqual(group, "spread")

    def test_selecting_a_checkpoint_on_it_is_refused(self) -> None:
        """A client count has no better or worse direction."""

        with self.assertRaises(ValueError):
            validate_selection_metric("val_num_clients")


class TheFemnistArmSaysWhyItSamplesTest(unittest.TestCase):
    """The other half of the fix: the config states what sampling costs."""

    def _clients(self, path: Path) -> Any:
        return load_config(str(path)).evaluation.val.clients

    def test_fedavg_ft_is_still_the_only_femnist_arm_that_samples(self) -> None:
        sampling = {path.name for path in FEMNIST if self._clients(path) != "all"}
        self.assertEqual(
            sampling,
            {"fedavg_ft.yaml"},
            "the set of FEMNIST arms sampling val has changed; the comment in "
            "fedavg_ft.yaml claims it is the only one",
        )

    def test_its_val_block_says_the_column_is_not_comparable(self) -> None:
        text = (Path("configs/femnist/fedavg_ft.yaml")).read_text(encoding="utf-8")
        for claim in (
            "val_accuracy_sample_weighted_avg",
            "3597/200",
            "val_num_clients",
            "P07-F06",
        ):
            with self.subTest(claim=claim):
                self.assertIn(claim, text)

    def test_the_writer_counts_the_comment_names_are_the_shipped_ones(self) -> None:
        """200 and 3,597 are in the comment; neither may drift silently."""

        text = (Path("configs/femnist/fedavg_ft.yaml")).read_text(encoding="utf-8")
        self.assertEqual(self._clients(Path("configs/femnist/fedavg_ft.yaml")), "sample:200")
        self.assertIn("200", text)


if __name__ == "__main__":
    unittest.main()
