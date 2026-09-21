"""Tests for the divergence/stall termination notice.

Two things: the on_termination hook in run_fl_loop (fired the moment the
divergence monitor decides to stop a run via a plain NaN metric, as opposed
to test_non_finite_aggregation.py's separate NonFiniteStateError break
point), and the console output it drives through
fedbrew/core/logging.print_termination_notice. Before this existed, a run
stopped by the monitor gave no live signal at all -- the reason string
reached the terminal only in the final print_experiment_end footer, after the
process had already finished.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest

from fedbrew.clients.base import ClientUpdate
from fedbrew.core.config import (
    CentralTestConfig,
    ClientConfig,
    ClientStatisticsConfig,
    DataConfig,
    DivergenceConfig,
    EvaluationConfig,
    ExperimentConfig,
    FullConfig,
    ModelConfig,
    RuntimeConfig,
    ServerConfig,
    SplitEvaluationConfig,
    TaskConfig,
)
from fedbrew.core.divergence import DivergenceVerdict
from fedbrew.core.logging import print_termination_notice
from fedbrew.core.loop import run_fl_loop
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


def _config(**runtime_extra: Any) -> FullConfig:
    return FullConfig(
        experiment=ExperimentConfig(seed=0, output_dir="outputs/test"),
        server=ServerConfig(strategy="fedavg", global_rounds=1, participation_rate=1.0, metrics=[]),
        client=ClientConfig(update_rule="fedavg", local_iterations=1, batch_size=1, metrics=[]),
        task=TaskConfig(name="classification"),
        data=DataConfig(),
        model=ModelConfig(name="mlp"),
        runtime=RuntimeConfig(device="cpu", use_amp=False, extra=runtime_extra),
    )


def _verdict(**overrides: Any) -> DivergenceVerdict:
    fields: dict[str, Any] = {
        "status": "diverged",
        "detector": "blowup",
        "round_id": 7,
        "metric": "fit_loss",
        "value": 41.2,
        "threshold": 10.0,
        "reason": "round 7: fit_loss=41.2 exceeds 10.0x its first observed value",
    }
    fields.update(overrides)
    return DivergenceVerdict(**fields)


@pytest.mark.fast
class PrintTerminationNoticeTests(unittest.TestCase):
    """Forces the plain-text fallback throughout: rich's own Console output
    capture is not exercised anywhere else in this suite, and the content
    asserted here is identical either way -- only the styling differs."""

    def test_quiet_suppresses_it_entirely(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_termination_notice(_verdict(), _config(quiet=True, no_rich=True))
        self.assertEqual(buffer.getvalue(), "")

    def test_plain_mode_names_the_detector_metric_round_and_reason(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_termination_notice(_verdict(), _config(no_rich=True))
        output = buffer.getvalue()
        self.assertIn("STOPPING EARLY: DIVERGED", output)
        self.assertIn("round 7: fit_loss=41.2 exceeds 10.0x its first observed value", output)
        self.assertIn("Detector: blowup on fit_loss (round 7)", output)

    def test_a_stalled_verdict_is_labelled_stalled_not_diverged(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_termination_notice(
                _verdict(status="stalled", detector="patience", reason="plateaued"),
                _config(no_rich=True),
            )
        self.assertIn("STOPPING EARLY: STALLED", buffer.getvalue())


class _Dataset(FederatedDataset):
    def list_clients(self) -> list[str]:
        return ["client_0"]

    def get_client_data(self, client_id: str) -> dict[str, Any]:
        raise AssertionError("the fake client owns its data")

    def get_client_metadata(self, client_id: str) -> dict[str, Any]:
        return {
            "client_id": client_id,
            "num_examples": 2,
            "num_train_examples": 2,
            "num_eval_examples": 0,
        }

    def get_global_data(self) -> None:
        return None

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "termination-test"}


class _Client(ClientUpdate):
    def __init__(self) -> None:
        self.client_id = ""

    def setup(self, client_info: ClientInfo) -> None:
        self.client_id = client_info.client_id

    def fit(self, request: FitRequest) -> FitResult:
        return FitResult(
            round_id=request.round_id,
            client_id=self.client_id,
            num_examples=2,
            metrics={"fit_loss": 1.0},
            payload={"model_state": {}},
        )

    def evaluate(self, request: EvalRequest) -> EvalResult:
        raise AssertionError("this test schedules no evaluation")

    def get_state(self) -> dict[str, Any]:
        return {"client_id": self.client_id}

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.client_id = str(state.get("client_id", self.client_id))


class _GoesToNanOnRound(ServerStrategy):
    """Reports a NaN fit_loss on one chosen round via ordinary aggregation --
    caught by DivergenceMonitor.update, not by the separate
    NonFiniteStateError/WeightedStateAccumulator path
    test_non_finite_aggregation.py covers."""

    def __init__(self, bad_round: int) -> None:
        self.bad_round = bad_round

    def initialize(self) -> dict[str, Any]:
        return {"model_state": {}}

    def configure_round(
        self, round_info: RoundInfo, clients: Sequence[ClientInfo]
    ) -> Sequence[FitRequest]:
        return [
            FitRequest(round_id=round_info.round_id, client_id=c.client_id, payload={})
            for c in clients
        ]

    def aggregate(self, round_info: RoundInfo, results: Sequence[FitResult]) -> dict[str, Any]:
        round_info.metrics["fit_loss"] = (
            float("nan") if round_info.round_id == self.bad_round else 1.0
        )
        return {"model_state": {}}

    def evaluate(self, round_info: RoundInfo, results: Sequence[EvalResult]) -> dict[str, float]:
        return {}

    def save_state(self) -> dict[str, Any]:
        return {}

    def load_state(self, state: Mapping[str, Any]) -> None:
        return None


_NO_EVAL = EvaluationConfig(
    train=SplitEvaluationConfig(every="never", clients="all"),
    val=SplitEvaluationConfig(every="never", clients="all"),
    test=SplitEvaluationConfig(every="never", clients="all"),
    central_test=CentralTestConfig(every="never"),
)


def _run(bad_round: int, rounds: int, on_termination: Any = None) -> Any:
    with tempfile.TemporaryDirectory() as directory:
        return run_fl_loop(
            server=_GoesToNanOnRound(bad_round),
            client={"client_0": _Client()},
            dataset=_Dataset(),
            global_rounds=rounds,
            output_dir=Path(directory),
            evaluation=_NO_EVAL,
            client_statistics=ClientStatisticsConfig(per_client_csv=False),
            # metric="fit_loss", non_finite=True by default -- exactly what
            # _GoesToNanOnRound reports.
            divergence=DivergenceConfig(),
            on_termination=on_termination,
        )


class OnTerminationHookTests(unittest.TestCase):
    def test_default_is_a_no_op(self) -> None:
        state = _run(bad_round=2, rounds=4)
        self.assertEqual(state.status, "diverged")

    def test_fires_once_with_the_verdict_the_run_stopped_on(self) -> None:
        calls: list[DivergenceVerdict] = []
        state = _run(bad_round=2, rounds=4, on_termination=calls.append)

        self.assertEqual(len(calls), 1)
        verdict = calls[0]
        self.assertEqual(verdict.status, "diverged")
        self.assertEqual(verdict.detector, "non_finite")
        self.assertEqual(verdict.round_id, 2)
        self.assertEqual(verdict.as_dict(), state.termination)

    def test_never_fires_on_a_healthy_run(self) -> None:
        calls: list[DivergenceVerdict] = []
        state = _run(bad_round=0, rounds=3, on_termination=calls.append)

        self.assertEqual(calls, [])
        self.assertEqual(state.status, "completed")


if __name__ == "__main__":
    unittest.main()
