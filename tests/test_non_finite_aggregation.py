"""What happens when a client's model state is not finite.

Every finiteness check in the repository used to be math.isfinite on a Python
scalar -- a hyperparameter, a norm, a metric. torch.isfinite/isnan/isinf
appeared nowhere in fedbrew/ or tools/, so a NaN tensor was treated as data:
it took the round's weighted mean with it, and for the three strategies
carrying second-moment or control state it stayed forever (FedOpt's
v <- b2*v + (1-b2)*d^2 was still NaN after 20 clean rounds) and was written
into latest.pt for --resume-latest to pick up.
"""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch

from fedbrew.clients.base import ClientUpdate
from fedbrew.core.config import (
    CentralTestConfig,
    ClientStatisticsConfig,
    EvaluationConfig,
    SplitEvaluationConfig,
)
from fedbrew.core.loop import run_fl_loop
from fedbrew.core.protocol import (
    ClientInfo,
    EvalRequest,
    EvalResult,
    FitRequest,
    FitResult,
    RoundInfo,
)
from fedbrew.core.torch_utils import (
    NonFiniteStateError,
    WeightedStateAccumulator,
)
from fedbrew.data.dataset import FederatedDataset
from fedbrew.servers.base import ServerStrategy


@pytest.mark.fast
class AccumulatorRefusesNonFiniteTests(unittest.TestCase):
    """One check in the class every strategy aggregates through."""

    def test_a_nan_tensor_is_refused_and_names_the_key(self) -> None:
        accumulator = WeightedStateAccumulator()
        with self.assertRaises(NonFiniteStateError) as caught:
            accumulator.add({"w": torch.tensor([float("nan"), 1.0])}, 100.0)
        self.assertIn("'w'", str(caught.exception))

    def test_an_inf_tensor_is_refused(self) -> None:
        accumulator = WeightedStateAccumulator()
        with self.assertRaises(NonFiniteStateError):
            accumulator.add({"w": torch.tensor([1.0, float("inf")])}, 1.0)

    def test_a_non_finite_weight_is_refused(self) -> None:
        accumulator = WeightedStateAccumulator()
        for weight in (float("nan"), float("inf")):
            with self.subTest(weight=weight):
                with self.assertRaises(NonFiniteStateError):
                    accumulator.add({"w": torch.tensor([1.0])}, weight)

    def test_one_bad_client_cannot_poison_the_round(self) -> None:
        """add(nan, 100) + add(1, 900) used to average to nan, silently."""

        accumulator = WeightedStateAccumulator()
        with self.assertRaises(NonFiniteStateError):
            accumulator.add({"w": torch.tensor([float("nan")])}, 100.0)
        accumulator = WeightedStateAccumulator()
        accumulator.add({"w": torch.tensor([1.0, 3.0])}, 1.0)
        accumulator.add({"w": torch.tensor([3.0, 1.0])}, 3.0)
        self.assertEqual(accumulator.result()["w"].tolist(), [2.5, 1.5])

    def test_integer_buffers_still_pass_through(self) -> None:
        """The check is floating-point only; int buffers are compared, not averaged."""

        accumulator = WeightedStateAccumulator()
        buffer = torch.tensor([7], dtype=torch.long)
        accumulator.add({"n": buffer}, 1.0)
        accumulator.add({"n": buffer}, 2.0)
        self.assertEqual(accumulator.result()["n"].tolist(), [7])


class _Dataset(FederatedDataset):
    def list_clients(self) -> list[str]:
        return ["client_0"]

    def get_client_data(self, client_id: str) -> dict[str, Any]:
        raise AssertionError("the fake client owns its data")

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
        return {"name": "non-finite-test"}


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


class _BlowsUpOnRound(ServerStrategy):
    """Aggregation refuses a non-finite client state on one chosen round."""

    def __init__(self, bad_round: int) -> None:
        self.bad_round = bad_round
        self.version = 0

    def initialize(self) -> dict[str, Any]:
        return {"model_state": {"version": self.version}}

    def configure_round(self, round_info, clients):
        return [
            FitRequest(round_id=round_info.round_id, client_id=c.client_id, payload={})
            for c in clients
        ]

    def aggregate(self, round_info: RoundInfo, results: Sequence[FitResult]):
        if round_info.round_id == self.bad_round:
            raise NonFiniteStateError(
                "client state tensor 'net.0.weight' contains non-finite values"
            )
        round_info.metrics.update({"fit_loss": 1.0})
        self.version = round_info.round_id
        return {"model_state": {"version": self.version}}

    def evaluate(self, round_info, results) -> dict[str, float]:
        return {}

    def evaluate_global(self, global_data: Any) -> dict[str, float]:
        return {}

    def save_state(self) -> dict[str, Any]:
        return {"model_state": {"version": self.version}}

    def load_state(self, state: Mapping[str, Any]) -> None:
        return None


_EVALUATION = EvaluationConfig(
    train=SplitEvaluationConfig(every=1, clients="all"),
    val=SplitEvaluationConfig(every="never", clients="all"),
    test=SplitEvaluationConfig(every="never", clients="all"),
    central_test=CentralTestConfig(every="never"),
)


class TheLoopRecordsItRatherThanCrashingTests(unittest.TestCase):
    """The contract: a dead run exits zero, recorded, never as a crash."""

    def _run(self, bad_round: int, rounds: int = 5, on_termination=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        output_dir = Path(directory.name)
        state = run_fl_loop(
            server=_BlowsUpOnRound(bad_round),
            client={"client_0": _Client()},
            dataset=_Dataset(),
            global_rounds=rounds,
            output_dir=output_dir,
            checkpointing={"enabled": True, "save_every_round": True, "keep_last": None},
            evaluation=_EVALUATION,
            client_statistics=ClientStatisticsConfig(per_client_csv=False),
            on_termination=on_termination,
        )
        return state, output_dir

    def test_the_run_is_marked_diverged_not_raised(self) -> None:
        state, _ = self._run(bad_round=3)

        self.assertEqual(state.status, "diverged")
        self.assertEqual(state.termination["detector"], "non_finite_client_state")
        self.assertEqual(state.termination["round_id"], 3)
        self.assertIn("net.0.weight", state.termination["reason"])

    def test_the_healthy_rounds_before_it_are_kept(self) -> None:
        state, _ = self._run(bad_round=3)

        self.assertEqual([r.round_id for r in state.metrics_history], [1, 2])

    def test_no_checkpoint_is_written_for_the_bad_round(self) -> None:
        """_update_checkpoints runs before the monitor, so the round has to be
        abandoned at aggregation or the poisoned state reaches disk."""

        _, output_dir = self._run(bad_round=3)
        written = sorted(p.name for p in (output_dir / "checkpoints").glob("round_*.pt"))

        self.assertNotIn("round_003.pt", written)

    def test_a_healthy_run_is_unaffected(self) -> None:
        state, _ = self._run(bad_round=0, rounds=4)

        self.assertEqual(state.status, "completed")

    def test_on_termination_fires_with_the_same_verdict_written_to_state(self) -> None:
        """Aggregation's own graceful stop (this class' whole subject) has to
        reach on_termination exactly like the metric-based divergence monitor
        does -- both break points construct a DivergenceVerdict and both must
        call the hook before breaking, or a run stopped by a non-finite client
        state would stay silent at the moment it happens while a run stopped
        by the monitor would not."""

        calls = []
        state, _ = self._run(bad_round=3, on_termination=calls.append)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].as_dict(), state.termination)
        self.assertEqual(calls[0].detector, "non_finite_client_state")
        self.assertEqual(calls[0].round_id, 3)

    def test_on_termination_is_never_called_on_a_healthy_run(self) -> None:
        calls = []
        self._run(bad_round=0, rounds=4, on_termination=calls.append)

        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
