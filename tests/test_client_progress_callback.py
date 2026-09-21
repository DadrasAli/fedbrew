"""Tests for the round loop's per-client progress callback.

fit and client evaluation are the only stages in a round that visit a known,
bounded sequence of clients one at a time -- but nothing surfaced that before
this hook existed; print_round_metrics only reports once the whole round is
done. These tests exercise the hook's contract directly, not any renderer:
this repository intentionally has none for it yet.
"""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from fedbrew.clients.base import ClientUpdate
from fedbrew.core.config import CentralTestConfig, EvaluationConfig, SplitEvaluationConfig
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

_CLIENT_IDS = ["client_0", "client_1", "client_2"]


class _ProgressDataset(FederatedDataset):
    def list_clients(self) -> list[str]:
        return list(_CLIENT_IDS)

    def get_client_data(self, client_id: str) -> dict[str, Any]:
        raise AssertionError("the fake clients already own their data")

    def get_client_metadata(self, client_id: str) -> dict[str, Any]:
        return {
            "client_id": client_id,
            "num_examples": 4,
            "num_train_examples": 2,
            "num_eval_examples": 2,
        }

    def get_global_data(self) -> None:
        return None

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "progress-test"}


class _ProgressClient(ClientUpdate):
    def __init__(self) -> None:
        self.client_id = ""

    def setup(self, client_info: ClientInfo) -> None:
        self.client_id = client_info.client_id

    def fit(self, request: FitRequest) -> FitResult:
        return FitResult(
            round_id=request.round_id,
            client_id=self.client_id,
            num_examples=2,
            payload={"model_state": {"version": request.round_id}},
            metrics={"loss": 1.0, "accuracy": 0.5},
        )

    def evaluate(self, request: EvalRequest) -> EvalResult:
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


class _ProgressServer(ServerStrategy):
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

    def save_state(self) -> dict[str, Any]:
        return {}

    def load_state(self, state: Mapping[str, Any]) -> None:
        return None


def _run(output_dir: Path, on_client_progress: Any) -> Any:
    return run_fl_loop(
        server=_ProgressServer(),
        client={client_id: _ProgressClient() for client_id in _CLIENT_IDS},
        dataset=_ProgressDataset(),
        global_rounds=2,
        output_dir=output_dir,
        on_client_progress=on_client_progress,
        # Only test is scheduled: this test is about the fit/client_eval
        # callback, not about which splits a round evaluates.
        evaluation=EvaluationConfig(
            train=SplitEvaluationConfig(every="never", clients="all"),
            val=SplitEvaluationConfig(every="never", clients="all"),
            test=SplitEvaluationConfig(every=1, clients="all"),
            central_test=CentralTestConfig(every="never"),
        ),
    )


class ClientProgressCallbackTests(unittest.TestCase):
    def test_default_is_a_no_op(self) -> None:
        """Omitting the callback must not change behaviour at all."""

        with tempfile.TemporaryDirectory() as directory:
            state = _run(Path(directory), None)
        self.assertEqual(len(state.metrics_history), 2)

    def test_reports_the_selection_then_once_per_client_per_round(self) -> None:
        """The fit phase opens with done=0: its total is the round's selection,
        which under participation_probability varies and can be zero, and a
        consumer that waited for a finished client would show the last round's."""

        calls: list[tuple[int, int, int, str]] = []

        with tempfile.TemporaryDirectory() as directory:
            _run(
                Path(directory),
                lambda round_id, done, total, phase: calls.append((round_id, done, total, phase)),
            )

        fit_calls = [call for call in calls if call[3] == "fit"]
        eval_calls = [call for call in calls if call[3] == "client_eval"]

        # 3 clients, 2 rounds: fit announces then reports each client, evaluation
        # reports each client, and `done`/`total` reset every round rather than
        # accumulating across the whole run -- total is always this round's
        # client count, not a running count across the run.
        def one_round(round_id: int, phase: str, first: int) -> list[tuple[int, int, int, str]]:
            return [(round_id, done, 3, phase) for done in range(first, 4)]

        self.assertEqual(fit_calls, one_round(1, "fit", 0) + one_round(2, "fit", 0))
        self.assertEqual(
            eval_calls, one_round(1, "client_eval", 1) + one_round(2, "client_eval", 1)
        )

    def test_the_round_comes_from_the_loop_not_from_counting_callbacks(self) -> None:
        """The caller cannot derive it. The only round boundary a caller sees
        is on_round_end, which fires *after* a round, so a resumed run -- whose
        first round is the checkpoint's, not round 1 -- would have that round
        reported under the wrong number by anything counting for itself."""

        calls: list[int] = []

        with tempfile.TemporaryDirectory() as directory:
            _run(
                Path(directory),
                lambda round_id, done, total, phase: calls.append(round_id),
            )

        self.assertEqual(sorted(set(calls)), [1, 2])
        # Every callback within a round carries that round, including the very
        # first one, which fires before any round has ended.
        self.assertEqual(calls[0], 1)

    def test_fit_calls_complete_before_evaluation_starts_within_a_round(self) -> None:
        calls: list[tuple[int, int, int, str]] = []

        with tempfile.TemporaryDirectory() as directory:
            _run(
                Path(directory),
                lambda round_id, done, total, phase: calls.append((round_id, done, total, phase)),
            )

        phases = [phase for *_, phase in calls]
        # Round 1: fit's announcement and its three clients, then three
        # client_eval, then round 2 repeats -- never interleaved.
        self.assertEqual(
            phases,
            ["fit"] * 4 + ["client_eval"] * 3 + ["fit"] * 4 + ["client_eval"] * 3,
        )

    @pytest.mark.fast
    def test_a_raising_callback_propagates_and_stops_the_run(self) -> None:
        """No swallowed exceptions: a broken hook must fail loudly.

        This is deliberately not "the hook must not affect training" --
        run_fl_loop calls on_round_end/on_round_flush the same unguarded way,
        and a callback that misbehaves should surface immediately rather than
        be caught and hidden inside loop internals.
        """

        def _explode(round_id: int, done: int, total: int, phase: str) -> None:
            raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "boom"):
                _run(Path(directory), _explode)


if __name__ == "__main__":
    unittest.main()
