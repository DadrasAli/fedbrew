"""evaluation.model_scope: which model the client passes measure.

Personalized update rules produce a per-client model, so the round's client
evaluation has to be able to measure that instead of -- or alongside -- the
aggregated global model. The personalized pass reports under a "personal_"
SPLIT prefix (personal_val_accuracy, not val_accuracy_personal), which is what
lets it reuse the existing aggregation, dispersion and worst-percent machinery
verbatim: everything downstream keys off f"{split}_{metric}".

"global" is the default, so every config written before this field existed
evaluates exactly what it evaluated before.
"""

from __future__ import annotations

import copy
import re
import unittest
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from fedbrew.clients.base import ClientUpdate
from fedbrew.core.checkpointing import (
    selection_mode_for_metric,
    validate_selection_metric,
)
from fedbrew.core.config import (
    EVALUATION_MODEL_SCOPES,
    PERSONAL_SPLIT_PREFIX,
    CentralTestConfig,
    ClientStatisticsConfig,
    EvaluationConfig,
    SplitEvaluationConfig,
    client_metric_names,
    load_config,
    validate_config,
)
from fedbrew.core.loop import (
    _aggregate_client_split_metrics,
    _scope_split_names,
    _validate_client_evaluation,
    run_fl_loop,
)
from fedbrew.core.protocol import (
    ClientInfo,
    EvalRequest,
    EvalResult,
    FitRequest,
    FitResult,
    RoundInfo,
)
from fedbrew.data.dataset import FederatedDataset
from fedbrew.servers.base import ServerStrategy

pytestmark = pytest.mark.fast

#: Global and personal numbers are deliberately different so a test cannot
#: pass by reading the wrong one. The personal model is the better one, which
#: is what fine-tuning on the client's own data is supposed to buy.
_GLOBAL = {"c0": {"loss": 4.0, "accuracy": 0.25}, "c1": {"loss": 2.0, "accuracy": 0.75}}
_PERSONAL = {"c0": {"loss": 1.0, "accuracy": 0.50}, "c1": {"loss": 1.0, "accuracy": 1.0}}
_COUNTS = {"c0": {"train": 10, "test": 10}, "c1": {"train": 30, "test": 30}}


class ScopeSplitNamesTest(unittest.TestCase):
    def test_each_scope_names_the_passes_it_runs(self) -> None:
        splits = ["train", "val", "test"]
        self.assertEqual(_scope_split_names("global", splits), splits)
        self.assertEqual(
            _scope_split_names("personal", splits),
            ["personal_train", "personal_val", "personal_test"],
        )
        self.assertEqual(_scope_split_names("both", ["val"]), ["val", "personal_val"])

    def test_the_prefix_is_the_documented_one(self) -> None:
        self.assertEqual(PERSONAL_SPLIT_PREFIX, "personal_")


class ValidationTest(unittest.TestCase):
    """A client that skips the personalized pass must fail the round."""

    def _result(self, metrics: Mapping[str, float], counts: Mapping[str, int]) -> EvalResult:
        return EvalResult(
            round_id=1,
            client_id="c0",
            num_examples=sum(counts.values()),
            metrics=dict(metrics),
            payload={"num_examples_by_split": dict(counts)},
        )

    def test_global_scope_is_unchanged(self) -> None:
        result = self._result(
            {"train_loss": 1.0, "train_accuracy": 0.5},
            {"train": 4},
        )
        _validate_client_evaluation(result, ["train"], "global")

    def test_personal_scope_requires_the_prefixed_metrics(self) -> None:
        result = self._result(
            {"train_loss": 1.0, "train_accuracy": 0.5},
            {"train": 4},
        )
        with self.assertRaises(ValueError) as caught:
            _validate_client_evaluation(result, ["train"], "personal")
        self.assertIn("personal_train_loss", str(caught.exception))

    def test_personal_scope_requires_the_prefixed_counts(self) -> None:
        result = self._result(
            {"personal_train_loss": 1.0, "personal_train_accuracy": 0.5},
            {"train": 4},
        )
        with self.assertRaises(ValueError) as caught:
            _validate_client_evaluation(result, ["train"], "personal")
        self.assertIn("personal_train", str(caught.exception))

    def test_both_requires_each_side(self) -> None:
        complete = {
            "train_loss": 1.0,
            "train_accuracy": 0.5,
            "personal_train_loss": 0.5,
            "personal_train_accuracy": 0.9,
        }
        counts = {"train": 4, "personal_train": 4}
        _validate_client_evaluation(self._result(complete, counts), ["train"], "both")

        for dropped in ("train_loss", "personal_train_loss"):
            with self.subTest(dropped=dropped):
                partial = {k: v for k, v in complete.items() if k != dropped}
                with self.assertRaises(ValueError):
                    _validate_client_evaluation(self._result(partial, counts), ["train"], "both")

    def test_val_stays_optional_in_both_scopes(self) -> None:
        """A client too small to hold out a validation split reports zero and is
        dropped from the aggregate rather than failing the round -- that has to
        hold for the personalized pass too."""

        result = self._result(
            {
                "train_loss": 1.0,
                "train_accuracy": 0.5,
                "personal_train_loss": 0.5,
                "personal_train_accuracy": 0.9,
            },
            {"train": 4, "personal_train": 4, "val": 0, "personal_val": 0},
        )
        _validate_client_evaluation(result, ["train", "val"], "both")


class _ScopedClient(ClientUpdate):
    """Reports whichever scope it was asked for, and records the request."""

    def __init__(self, client_id: str) -> None:
        self.client_id = client_id
        self.requested_scopes: list[str] = []

    def setup(self, client_info: ClientInfo) -> None:
        return None

    def fit(self, request: FitRequest) -> FitResult:
        return FitResult(
            round_id=request.round_id,
            client_id=self.client_id,
            num_examples=_COUNTS[self.client_id]["train"],
            payload={"model_state": {"version": request.round_id}},
            metrics={"fit_loss": 1.0},
        )

    def evaluate(self, request: EvalRequest) -> EvalResult:
        scope = str(request.payload.get("model_scope", "global"))
        self.requested_scopes.append(scope)
        splits = list(request.payload.get("splits", []))

        metrics: dict[str, float] = {}
        counts: dict[str, int] = {}
        for split in splits:
            if scope in {"global", "both"}:
                counts[split] = _COUNTS[self.client_id][split]
                for name, value in _GLOBAL[self.client_id].items():
                    metrics[f"{split}_{name}"] = value
            if scope in {"personal", "both"}:
                counts[f"{PERSONAL_SPLIT_PREFIX}{split}"] = _COUNTS[self.client_id][split]
                for name, value in _PERSONAL[self.client_id].items():
                    metrics[f"{PERSONAL_SPLIT_PREFIX}{split}_{name}"] = value

        return EvalResult(
            round_id=request.round_id,
            client_id=self.client_id,
            num_examples=sum(counts.values()),
            metrics=metrics,
            payload={"model_scope": scope, "num_examples_by_split": counts},
        )

    def get_state(self) -> dict[str, Any]:
        return {"client_id": self.client_id}

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.client_id = str(state.get("client_id", self.client_id))


class _ScopedDataset(FederatedDataset):
    def list_clients(self) -> list[str]:
        return list(_COUNTS)

    def get_client_data(self, client_id: str) -> dict[str, Any]:
        raise AssertionError("the fake clients already own their data")

    def get_client_metadata(self, client_id: str) -> dict[str, Any]:
        counts = _COUNTS[client_id]
        return {
            "client_id": client_id,
            "num_examples": counts["train"] + counts["test"],
            "num_train_examples": counts["train"],
            "num_eval_examples": counts["test"],
        }

    def get_global_data(self) -> Any:
        raise AssertionError("central global data must not be used")

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "scope-test"}


class _ScopedServer(ServerStrategy):
    def initialize(self) -> dict[str, Any]:
        return {"model_state": {"version": 0}}

    def configure_round(
        self, round_info: RoundInfo, clients: Sequence[ClientInfo]
    ) -> Sequence[FitRequest]:
        return [
            FitRequest(
                round_id=round_info.round_id,
                client_id=client.client_id,
                payload={"model_state": {"version": round_info.round_id - 1}},
            )
            for client in clients
        ]

    def aggregate(self, round_info: RoundInfo, results: Sequence[FitResult]) -> dict[str, Any]:
        round_info.metrics["fit_loss"] = 1.0
        return {
            "model_state": {"version": round_info.round_id},
            "metrics": dict(round_info.metrics),
        }

    def evaluate(self, round_info: RoundInfo, results: Sequence[EvalResult]) -> dict[str, float]:
        return {}

    def save_state(self) -> dict[str, Any]:
        return {}

    def load_state(self, state: Mapping[str, Any]) -> None:
        return None


def _run(model_scope: str) -> tuple[dict[str, float], dict[str, _ScopedClient]]:
    clients = {client_id: _ScopedClient(client_id) for client_id in _COUNTS}
    state = run_fl_loop(
        _ScopedServer(),
        clients,
        _ScopedDataset(),
        global_rounds=1,
        checkpointing={"enabled": False},
        evaluation=EvaluationConfig(
            train=SplitEvaluationConfig(every="never", clients="all"),
            val=SplitEvaluationConfig(every="never", clients="all"),
            test=SplitEvaluationConfig(every=1, clients="all"),
            central_test=CentralTestConfig(every="never"),
            model_scope=model_scope,
        ),
    )
    return state.metrics_history[0].metrics, clients


class LoopIntegrationTest(unittest.TestCase):
    # sample-weighted: (4*10 + 2*30)/40 = 2.5 global, (1*10 + 1*30)/40 = 1.0 personal
    GLOBAL_TEST_LOSS = 2.5
    PERSONAL_TEST_LOSS = 1.0

    def test_global_is_the_default_and_emits_no_personal_columns(self) -> None:
        metrics, clients = _run("global")
        self.assertEqual(EvaluationConfig().model_scope, "global")
        self.assertAlmostEqual(metrics["test_loss_sample_weighted_avg"], self.GLOBAL_TEST_LOSS)
        self.assertFalse([name for name in metrics if name.startswith("personal_")])
        self.assertEqual(clients["c0"].requested_scopes, ["global"])

    def test_personal_replaces_the_global_columns(self) -> None:
        metrics, clients = _run("personal")
        self.assertAlmostEqual(
            metrics["personal_test_loss_sample_weighted_avg"], self.PERSONAL_TEST_LOSS
        )
        self.assertNotIn("test_loss_sample_weighted_avg", metrics)
        self.assertEqual(clients["c0"].requested_scopes, ["personal"])

    def test_both_emits_the_two_columns_side_by_side(self) -> None:
        metrics, _ = _run("both")
        self.assertAlmostEqual(metrics["test_loss_sample_weighted_avg"], self.GLOBAL_TEST_LOSS)
        self.assertAlmostEqual(
            metrics["personal_test_loss_sample_weighted_avg"], self.PERSONAL_TEST_LOSS
        )

    def test_the_personal_split_gets_the_full_statistics_family(self) -> None:
        """The point of prefixing the split rather than suffixing the metric:
        dispersion and worst-percent come for free, with no changes to
        _aggregate_client_split_metrics or _client_distribution_statistics."""

        metrics, _ = _run("both")
        for suffix in (
            "sample_weighted_avg",
            "avg",
            "std",
            "min",
            "max",
            "worst10",
        ):
            with self.subTest(suffix=suffix):
                self.assertIn(f"personal_test_accuracy_{suffix}", metrics)
                self.assertIn(f"test_accuracy_{suffix}", metrics)


class ConfigurationTest(unittest.TestCase):
    def test_default_is_global(self) -> None:
        config = load_config("configs/dev/synthetic.yaml")
        validate_config(config)
        self.assertEqual(config.evaluation.model_scope, "global")

    def test_every_scope_validates(self) -> None:
        base = load_config("configs/dev/synthetic.yaml")
        for scope in sorted(EVALUATION_MODEL_SCOPES):
            with self.subTest(scope=scope):
                config = copy.deepcopy(base)
                config.evaluation.model_scope = scope
                validate_config(config)

    def test_an_unknown_scope_is_rejected(self) -> None:
        config = load_config("configs/dev/synthetic.yaml")
        config.evaluation.model_scope = "per_client"
        with self.assertRaises(ValueError):
            validate_config(config)


class CheckpointSelectionTest(unittest.TestCase):
    """best.pt has to be selectable on whichever pass the scope actually runs.

    A best_metric the scope never emits does not fail loudly at runtime --
    best.pt simply never gets written -- so it is worth failing at config load,
    before a multi-day run starts.
    """

    def _config(self, scope: str, best_metric: str) -> Any:
        config = load_config("configs/femnist/fedavg.yaml")
        config.evaluation.model_scope = scope
        config.runtime.extra["checkpointing"]["best_metric"] = best_metric
        return config

    def test_each_scope_accepts_the_metric_it_emits(self) -> None:
        for scope, metric in (
            ("global", "val_accuracy_sample_weighted_avg"),
            ("personal", "personal_val_accuracy_sample_weighted_avg"),
            ("both", "val_accuracy_sample_weighted_avg"),
            ("both", "personal_val_accuracy_sample_weighted_avg"),
        ):
            with self.subTest(scope=scope, metric=metric):
                validate_config(self._config(scope, metric))

    def test_a_metric_the_scope_never_emits_is_rejected(self) -> None:
        for scope, metric in (
            ("global", "personal_val_accuracy_sample_weighted_avg"),
            ("personal", "val_accuracy_sample_weighted_avg"),
        ):
            with self.subTest(scope=scope, metric=metric):
                with self.assertRaises(ValueError):
                    validate_config(self._config(scope, metric))

    def test_selection_still_may_not_read_the_test_set(self) -> None:
        """Widening the prefix must not widen the val-vs-test guard: selecting
        best.pt on a test metric biases the reported test score."""

        for metric in (
            "test_accuracy_sample_weighted_avg",
            "personal_test_accuracy_sample_weighted_avg",
        ):
            with self.subTest(metric=metric), self.assertRaises(ValueError):
                validate_config(self._config("both", metric))

    def test_the_personal_metric_direction_is_still_derived(self) -> None:
        self.assertEqual(
            selection_mode_for_metric("personal_val_accuracy_sample_weighted_avg"), "max"
        )
        self.assertEqual(selection_mode_for_metric("personal_val_loss_sample_weighted_avg"), "min")


class EmittedSelectionMetricTest(unittest.TestCase):
    """best.pt has to be selectable on a column the run actually writes.

    Prefix plus direction is not enough: val_accuracy_bottom10 passes both
    checks and names nothing. The loop then reads None out of the metrics dict
    every round and skips the update, so the run ends with best_checkpoint
    null and no other sign that selection never happened.
    """

    def _config(self, best_metric: str) -> Any:
        config = load_config("configs/femnist/fedavg.yaml")
        config.runtime.extra["checkpointing"]["best_metric"] = best_metric
        return config

    def test_a_metric_no_column_will_carry_is_rejected(self) -> None:
        for metric in (
            # The name the validator's own help text used to recommend.
            "val_accuracy_bottom10",
            # A percentage other than the configured worst_percent: 10.
            "val_accuracy_worst5",
            # A statistic whose client_statistics toggle is off.
            "val_accuracy_variance",
            # Abbreviated: "acc" gives a direction, but no column is named it.
            "val_acc_avg",
        ):
            with self.subTest(metric=metric):
                with self.assertRaises(ValueError) as caught:
                    validate_config(self._config(metric))
                self.assertIn("never emitted", str(caught.exception))

    def test_turning_the_statistic_on_makes_its_metric_selectable(self) -> None:
        config = self._config("val_accuracy_variance")
        config.client_statistics.variance = True
        validate_config(config)

        config = self._config("val_accuracy_worst5")
        config.client_statistics.worst_percent = 5
        validate_config(config)

    def test_a_fractional_worst_percent_keeps_the_loop_spelling(self) -> None:
        config = self._config("val_accuracy_worst2p5")
        config.client_statistics.worst_percent = 2.5
        validate_config(config)

    def test_the_personal_pass_is_checked_against_its_own_columns(self) -> None:
        config = self._config("personal_val_accuracy_variance")
        config.evaluation.model_scope = "personal"
        with self.assertRaises(ValueError):
            validate_config(config)
        config.client_statistics.variance = True
        validate_config(config)

    def test_the_predicted_names_are_the_names_the_loop_emits(self) -> None:
        """The check is a mirror of the aggregation, so pin it to the original.

        Everything above only proves the validator is self-consistent. This is
        what proves it agrees with the code that writes the columns.
        """

        statistics = ClientStatisticsConfig(
            std=True, variance=True, min=True, max=True, worst_percent=2.5
        )
        results = [
            EvalResult(
                round_id=1,
                client_id=f"c{index}",
                num_examples=4,
                metrics={"val_loss": 1.0 + index, "val_accuracy": 0.5 + index / 10},
                payload={"num_examples_by_split": {"val": 4}},
            )
            for index in range(4)
        ]
        emitted = _aggregate_client_split_metrics(results, "val", statistics)
        self.assertEqual(set(emitted), client_metric_names("val", statistics))

    def test_every_metric_the_help_text_names_can_exist(self) -> None:
        """The defect this fixes was in the error message, not only the check."""

        try:
            validate_selection_metric("central_test_accuracy")
        except ValueError as exc:
            message = str(exc)
        else:  # pragma: no cover - the guard under test refuses this name.
            self.fail("a test metric must not pass validate_selection_metric")

        every_toggle_on = ClientStatisticsConfig(
            std=True, variance=True, min=True, max=True, worst_percent=10
        )
        producible = client_metric_names("val", every_toggle_on)
        named = set(re.findall(r"\bval_[a-z0-9_]+", message))
        self.assertTrue(named, "the help text should name some metrics")
        self.assertEqual(named - producible, set())


if __name__ == "__main__":
    unittest.main()


class PerClientCsvUnderPersonalScopeTest(unittest.TestCase):
    """client_metrics.csv must carry data under every model_scope.

    Under `personal` it carried none. The personalized pass reports its splits
    and its counts under a `personal_` prefix, the record builder read the
    unprefixed names, found nothing, and wrote a row whose three counts were 0
    and whose six metric columns were empty -- for every client, every round,
    in a file the user had switched on deliberately with per_client_csv.

    Under `both` the global pass is recorded, which is what these columns have
    always held; the personalized numbers reach round_metrics.csv through the
    personal_ aggregates either way.
    """

    def _result(self, prefix: str) -> EvalResult:
        return EvalResult(
            round_id=1,
            client_id="c0",
            num_examples=20,
            metrics={
                f"{prefix}train_loss": 0.3,
                f"{prefix}train_accuracy": 0.9,
                f"{prefix}test_loss": 0.4,
                f"{prefix}test_accuracy": 0.8,
            },
            payload={"num_examples_by_split": {f"{prefix}train": 10, f"{prefix}test": 10}},
        )

    def _record(self, prefix: str, scope: str):
        from fedbrew.core.loop import _build_client_evaluation_record

        return _build_client_evaluation_record(
            self._result(prefix),
            participated=True,
            evaluated_splits=["train", "test"],
            model_scope=scope,
        )

    def test_personal_scope_records_the_personal_numbers(self) -> None:
        record = self._record("personal_", "personal")
        self.assertEqual(record.train_num_examples, 10)
        self.assertEqual(record.test_num_examples, 10)
        self.assertEqual(record.global_model_train_loss, 0.3)
        self.assertEqual(record.global_model_test_accuracy, 0.8)
        self.assertEqual(record.model_scope, "personal")

    def test_global_scope_is_unchanged(self) -> None:
        record = self._record("", "global")
        self.assertEqual(record.train_num_examples, 10)
        self.assertEqual(record.global_model_test_accuracy, 0.8)
        self.assertEqual(record.model_scope, "global")

    def test_both_records_the_global_pass(self) -> None:
        record = self._record("", "both")
        self.assertEqual(record.global_model_train_loss, 0.3)
        self.assertEqual(record.model_scope, "global")

    def test_the_csv_row_says_which_model_it_measured(self) -> None:
        from fedbrew.core.artifacts import (
            _CLIENT_EVALUATION_FIELDS,
            _client_evaluation_payload,
        )

        self.assertEqual(_CLIENT_EVALUATION_FIELDS[-1], "model_scope")
        row = _client_evaluation_payload(self._record("personal_", "personal"))
        self.assertEqual(set(row), set(_CLIENT_EVALUATION_FIELDS))
        self.assertEqual(row["model_scope"], "personal")
        # No column shifted: the scope was appended, not inserted.
        self.assertEqual(_CLIENT_EVALUATION_FIELDS[0], "round_id")
        self.assertEqual(_CLIENT_EVALUATION_FIELDS[2], "participated")
