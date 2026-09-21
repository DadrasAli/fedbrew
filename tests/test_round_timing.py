"""Tests for per-round wall-clock instrumentation and its artifacts."""

from __future__ import annotations

import csv
import json
import tempfile
import time
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from fedbrew.clients.base import ClientUpdate
from fedbrew.core.artifacts import (
    load_round_metrics_csv,
    save_round_metrics_csv,
    save_run_json,
)
from fedbrew.core.config import (
    CentralTestConfig,
    EvaluationConfig,
    SplitEvaluationConfig,
)
from fedbrew.core.logging import _format_duration, _wall_clock_summary
from fedbrew.core.loop import run_fl_loop
from fedbrew.core.protocol import (
    ClientInfo,
    EvalRequest,
    EvalResult,
    FitRequest,
    FitResult,
    RoundInfo,
)
from fedbrew.core.state import MetricRecord, RoundTimings
from fedbrew.data.dataset import FederatedDataset
from fedbrew.servers.base import ServerStrategy

# Long enough that a phase is unambiguously distinguishable from scheduler
# noise, short enough that the suite stays fast.
_FIT_SECONDS = 0.05
_CLIENT_EVAL_SECONDS = 0.02
_GLOBAL_EVAL_SECONDS = 0.03


class _TimingDataset(FederatedDataset):
    def list_clients(self) -> list[str]:
        return ["client_0", "client_1"]

    def get_client_data(self, client_id: str) -> dict[str, Any]:
        raise AssertionError("the fake clients already own their data")

    def get_client_metadata(self, client_id: str) -> dict[str, Any]:
        return {
            "client_id": client_id,
            "num_examples": 4,
            "num_train_examples": 2,
            "num_eval_examples": 2,
        }

    def get_global_data(self) -> dict[str, str]:
        return {"scope": "global-test"}

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "timing-test"}


class _TimingClient(ClientUpdate):
    def __init__(self) -> None:
        self.client_id = ""

    def setup(self, client_info: ClientInfo) -> None:
        self.client_id = client_info.client_id

    def fit(self, request: FitRequest) -> FitResult:
        time.sleep(_FIT_SECONDS)
        return FitResult(
            round_id=request.round_id,
            client_id=self.client_id,
            num_examples=2,
            payload={"model_state": {"version": request.round_id}},
            metrics={"loss": 1.0, "accuracy": 0.5},
        )

    def evaluate(self, request: EvalRequest) -> EvalResult:
        time.sleep(_CLIENT_EVAL_SECONDS)
        splits = list(request.payload.get("splits", ["train"]))
        return EvalResult(
            round_id=request.round_id,
            client_id=self.client_id,
            num_examples=2 * len(splits),
            metrics={f"{split}_loss": 1.0 for split in splits}
            | {f"{split}_accuracy": 0.5 for split in splits},
            payload={
                "model_scope": "global",
                "num_examples_by_split": dict.fromkeys(splits, 2),
            },
        )

    def get_state(self) -> dict[str, Any]:
        return {"client_id": self.client_id}

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.client_id = str(state.get("client_id", self.client_id))


class _TimingServer(ServerStrategy):
    def initialize(self) -> dict[str, Any]:
        return {"model_state": {"version": 0}}

    def configure_round(
        self,
        round_info: RoundInfo,
        clients: Sequence[ClientInfo],
    ) -> Sequence[FitRequest]:
        return [
            FitRequest(
                round_id=round_info.round_id,
                client_id=client.client_id,
                payload={"model_state": {"version": round_info.round_id - 1}},
            )
            for client in clients
        ]

    def aggregate(
        self,
        round_info: RoundInfo,
        results: Sequence[FitResult],
    ) -> dict[str, Any]:
        round_info.metrics.update({"loss": 1.0, "accuracy": 0.5})
        return {"model_state": {"version": round_info.round_id}}

    def evaluate(
        self,
        round_info: RoundInfo,
        results: Sequence[EvalResult],
    ) -> dict[str, float]:
        return {}

    def evaluate_global(self, global_data: Any) -> dict[str, float]:
        time.sleep(_GLOBAL_EVAL_SECONDS)
        return {"central_test_loss": 1.0, "central_test_accuracy": 0.5}

    def save_state(self) -> dict[str, Any]:
        return {}

    def load_state(self, state: Mapping[str, Any]) -> None:
        return None


def _run_two_rounds(output_dir: Path) -> Any:
    return run_fl_loop(
        server=_TimingServer(),
        client={"client_0": _TimingClient(), "client_1": _TimingClient()},
        dataset=_TimingDataset(),
        global_rounds=2,
        output_dir=output_dir,
        # Explicit every-round schedule: this test measures the cost of each
        # phase, so every phase has to run in every round. The defaults skip
        # most rounds, which is the point of them.
        evaluation=EvaluationConfig(
            train=SplitEvaluationConfig(every=1, clients="all"),
            val=SplitEvaluationConfig(every="never", clients="all"),
            test=SplitEvaluationConfig(every=1, clients="all"),
            central_test=CentralTestConfig(every=1),
        ),
    )


class RoundTimingTests(unittest.TestCase):
    def test_loop_attributes_wall_clock_to_the_phase_that_spent_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = _run_two_rounds(Path(directory))

        self.assertEqual(len(state.metrics_history), 2)
        for record in state.metrics_history:
            timings = record.timings
            self.assertIsNotNone(timings)
            assert timings is not None
            # Two clients fit per round.
            self.assertGreaterEqual(timings.fit, 2 * _FIT_SECONDS)
            self.assertGreaterEqual(timings.client_eval, 2 * _CLIENT_EVAL_SECONDS)
            self.assertGreaterEqual(timings.global_eval, _GLOBAL_EVAL_SECONDS)
            # Client fits stream through aggregate_stream, so their cost must be
            # attributed to fit rather than inflating the server's own time.
            self.assertLess(timings.aggregate, _FIT_SECONDS)
            phase_total = (
                timings.fit
                + timings.aggregate
                + timings.client_eval
                + timings.global_eval
                + timings.checkpoint
            )
            self.assertGreaterEqual(timings.total, phase_total)

        self.assertEqual(
            [round_state.timings for round_state in state.rounds],
            [record.timings for record in state.metrics_history],
        )

    @pytest.mark.fast
    def test_timings_survive_the_csv_round_trip(self) -> None:
        record = MetricRecord(
            round_id=1,
            metrics={"fit_loss": 0.5},
            num_clients=1,
            num_examples=2,
            timings=RoundTimings(
                total=1.5,
                fit=1.0,
                aggregate=0.1,
                client_eval=0.2,
                global_eval=0.15,
                checkpoint=0.05,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            save_round_metrics_csv([record], directory)
            restored = load_round_metrics_csv(directory)

        self.assertEqual(restored[0].timings, record.timings)
        self.assertEqual(restored[0].metrics, record.metrics)
        self.assertEqual(restored[0].num_clients, 1)

    @pytest.mark.fast
    def test_a_skipped_split_reloads_as_absent_not_as_zero(self) -> None:
        # Under a per-split schedule most rounds measure only some splits.
        # A blank cell means "not measured", which is not the same fact as 0.0.
        history = [
            MetricRecord(
                round_id=1,
                metrics={"fit_loss": 0.5},
                num_clients=1,
                num_examples=2,
            ),
            MetricRecord(
                round_id=2,
                metrics={"fit_loss": 0.4, "test_accuracy_avg": 0.75},
                num_clients=1,
                num_examples=2,
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            save_round_metrics_csv(history, directory)
            restored = load_round_metrics_csv(directory)

        self.assertNotIn("test_accuracy_avg", restored[0].metrics)
        self.assertAlmostEqual(restored[1].metrics["test_accuracy_avg"], 0.75)

    @pytest.mark.fast
    def test_csv_appends_timing_columns_and_blanks_unmeasured_rounds(self) -> None:
        history = [
            MetricRecord(
                round_id=1,
                metrics={"loss": 0.5},
                num_clients=1,
                num_examples=2,
            ),
            MetricRecord(
                round_id=2,
                metrics={"loss": 0.25},
                num_clients=1,
                num_examples=2,
                timings=RoundTimings(total=2.0, fit=1.5, client_eval=0.25),
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            csv_path = save_round_metrics_csv(history, directory)
            rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
            header = rows[0].keys()

        # Timing columns come last so that adding them never shifts the position
        # of a column an existing plot or script already reads.
        self.assertEqual(
            list(header)[-6:],
            [
                "duration_sec",
                "fit_sec",
                "aggregate_sec",
                "client_eval_sec",
                "global_eval_sec",
                "checkpoint_sec",
            ],
        )
        self.assertEqual(rows[0]["duration_sec"], "")
        self.assertEqual(rows[0]["fit_sec"], "")
        self.assertEqual(rows[1]["duration_sec"], "2.0")
        self.assertEqual(rows[1]["fit_sec"], "1.5")
        self.assertNotIn("duration_sec", history[1].metrics)

    @pytest.mark.fast
    def test_summary_reports_round_time_over_measured_rounds_only(self) -> None:
        history = [
            MetricRecord(
                round_id=1,
                metrics={"loss": 0.5},
                num_clients=1,
                num_examples=2,
            ),
            MetricRecord(
                round_id=2,
                metrics={"loss": 0.4},
                num_clients=1,
                num_examples=2,
                timings=RoundTimings(total=10.0, fit=8.0, client_eval=1.0),
            ),
            MetricRecord(
                round_id=3,
                metrics={"loss": 0.3},
                num_clients=1,
                num_examples=2,
                timings=RoundTimings(total=2.0, fit=1.0, client_eval=0.5),
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            summary_path = save_run_json(
                history,
                directory,
                _minimal_config(),
                run_metadata={
                    "started_at": "2026-01-01T00:00:00+00:00",
                    "finished_at": "2026-01-01T00:00:12+00:00",
                    "duration_sec": 12.5,
                },
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        timing = summary["timing"]
        self.assertEqual(timing["timed_rounds"], 2)
        self.assertEqual(timing["total_round_sec"], 12.0)
        self.assertEqual(timing["mean_sec_per_round"], 6.0)
        self.assertEqual(timing["median_sec_per_round"], 6.0)
        self.assertEqual(timing["min_sec_per_round"], 2.0)
        self.assertEqual(timing["max_sec_per_round"], 10.0)
        self.assertEqual(timing["phase_sec"]["fit"], 9.0)
        self.assertEqual(timing["run_duration_sec"], 12.5)
        # Bookends are top-level now; "timing" holds per-round statistics only.
        self.assertEqual(summary["started_at"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(summary["finished_at"], "2026-01-01T00:00:12+00:00")
        self.assertNotIn("started_at", timing)

    @pytest.mark.fast
    def test_summary_omits_round_statistics_when_nothing_was_timed(self) -> None:
        history = [
            MetricRecord(
                round_id=1,
                metrics={"loss": 0.5},
                num_clients=1,
                num_examples=2,
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            summary_path = save_run_json(history, directory, _minimal_config())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["timing"]["timed_rounds"], 0)
        self.assertNotIn("mean_sec_per_round", summary["timing"])


def _minimal_config():
    """Smallest FullConfig save_run_json will accept, for artifact-shape tests."""

    from fedbrew.core.config import (
        ClientConfig,
        DataConfig,
        ExperimentConfig,
        FullConfig,
        ModelConfig,
        RuntimeConfig,
        ServerConfig,
        TaskConfig,
    )

    return FullConfig(
        experiment=ExperimentConfig(seed=0, output_dir="outputs/test"),
        server=ServerConfig(strategy="fedavg", global_rounds=1, participation_rate=1.0, metrics=[]),
        client=ClientConfig(update_rule="fedavg", local_iterations=1, batch_size=1, metrics=[]),
        task=TaskConfig(name="classification"),
        data=DataConfig(),
        model=ModelConfig(name="mlp"),
        runtime=RuntimeConfig(device="cpu", use_amp=False),
    )


@pytest.mark.fast
class DurationFormattingTests(unittest.TestCase):
    def test_durations_render_at_the_precision_a_time_budget_needs(self) -> None:
        self.assertEqual(_format_duration(0.042), "0.04s")
        self.assertEqual(_format_duration(24.0), "24.0s")
        self.assertEqual(_format_duration(95.0), "1m 35s")
        self.assertEqual(_format_duration(6720.0), "1h 52m")

    def test_wall_clock_summary_covers_only_timed_rounds(self) -> None:
        history = [
            MetricRecord(
                round_id=1,
                metrics={},
                num_clients=1,
                num_examples=0,
            ),
            MetricRecord(
                round_id=2,
                metrics={},
                num_clients=1,
                num_examples=0,
                timings=RoundTimings(total=30.0),
            ),
            MetricRecord(
                round_id=3,
                metrics={},
                num_clients=1,
                num_examples=0,
                timings=RoundTimings(total=10.0),
            ),
        ]
        self.assertEqual(
            _wall_clock_summary(history),
            "40.0s over 2 timed rounds (20.0s/round)",
        )
        self.assertIsNone(_wall_clock_summary(history[:1]))


if __name__ == "__main__":
    unittest.main()
