"""A resumed run has to continue the RNG stream, not restart it.

nn.Dropout draws from the process-wide default generator, not from a per-client
stream, so a run's position in that stream is state exactly as much as its
weights are. Weights, server state and client state were all restored on
resume; the RNG position was not. Both runs were internally deterministic and
they disagreed -- on real FEMNIST shards fit_accuracy moved on CPU and on GPU,
and stayed bit-identical with dropout: 0.0, which isolates the cause completely.

This matters because the packed SLURM script runs #SBATCH --requeue with a
per-run --resume-latest guard, so on a long job a resume is the normal path.
"""

from __future__ import annotations

import random
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
from fedbrew.core.runtime_setup import capture_rng_state, restore_rng_state
from fedbrew.data.dataset import FederatedDataset
from fedbrew.servers.base import ServerStrategy


class _Dataset(FederatedDataset):
    def list_clients(self) -> list[str]:
        return ["client_0", "client_1"]

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
        return {"name": "rng-resume-test"}


class _DropoutLikeClient(ClientUpdate):
    """Consumes the global torch RNG on every fit, the way nn.Dropout does."""

    def __init__(self) -> None:
        self.client_id = ""

    def setup(self, client_info: ClientInfo) -> None:
        self.client_id = client_info.client_id

    def fit(self, request: FitRequest) -> FitResult:
        draw = float(torch.rand(1).item())
        return FitResult(
            round_id=request.round_id,
            client_id=self.client_id,
            num_examples=2,
            payload={"model_state": {"version": request.round_id}},
            metrics={"fit_loss": draw, "fit_accuracy": draw / 2.0},
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


class _Server(ServerStrategy):
    def __init__(self) -> None:
        self.version = 0

    def initialize(self) -> dict[str, Any]:
        return {"model_state": {"version": self.version}}

    def configure_round(self, round_info: RoundInfo, clients: Sequence[ClientInfo]):
        return [
            FitRequest(round_id=round_info.round_id, client_id=c.client_id, payload={})
            for c in clients
        ]

    def aggregate(self, round_info: RoundInfo, results: Sequence[FitResult]):
        round_info.metrics["fit_loss"] = sum(r.metrics["fit_loss"] for r in results)
        self.version = round_info.round_id
        return {"model_state": {"version": self.version}}

    def evaluate(self, round_info, results) -> dict[str, float]:
        return {}

    def evaluate_global(self, global_data: Any) -> dict[str, float]:
        return {}

    def save_state(self) -> dict[str, Any]:
        return {"model_state": {"version": self.version}}

    def load_state(self, state: Mapping[str, Any]) -> None:
        model_state = state.get("model_state")
        if isinstance(model_state, Mapping):
            self.version = int(model_state.get("version", self.version))


_EVALUATION = EvaluationConfig(
    train=SplitEvaluationConfig(every="never", clients="all"),
    val=SplitEvaluationConfig(every="never", clients="all"),
    test=SplitEvaluationConfig(every="never", clients="all"),
    central_test=CentralTestConfig(every="never"),
)


def _run(output_dir: Path, rounds: int, resume_from: Path | None = None):
    return run_fl_loop(
        server=_Server(),
        client={"client_0": _DropoutLikeClient(), "client_1": _DropoutLikeClient()},
        dataset=_Dataset(),
        global_rounds=rounds,
        output_dir=output_dir,
        resume_from=resume_from,
        checkpointing={"enabled": True, "save_every_round": True, "keep_last": None},
        evaluation=_EVALUATION,
        client_statistics=ClientStatisticsConfig(per_client_csv=False),
    )


def _losses(state) -> dict[int, float]:
    return {r.round_id: r.metrics["fit_loss"] for r in state.metrics_history}


class ResumeReproducesTheUninterruptedRunTests(unittest.TestCase):
    """A resumed run must continue the RNG stream its checkpoint was written in.

    The state was absent from the checkpoint, so a requeued run silently
    carried on down a different dropout trajectory than the one it claimed to
    resume.
    """

    def _uninterrupted(self, rounds: int) -> dict[int, float]:
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(1234)
            random.seed(1234)
            return _losses(_run(Path(directory), rounds))

    def test_a_resumed_run_matches_the_run_it_continues(self) -> None:
        expected = self._uninterrupted(6)

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            torch.manual_seed(1234)
            random.seed(1234)
            _run(output_dir, rounds=3)

            # A fresh process would reseed here; that is exactly the defect.
            torch.manual_seed(1234)
            random.seed(1234)
            resumed = _losses(
                _run(
                    output_dir,
                    rounds=6,
                    resume_from=output_dir / "checkpoints" / "round_003.pt",
                )
            )

        for round_id in (4, 5, 6):
            with self.subTest(round=round_id):
                self.assertAlmostEqual(resumed[round_id], expected[round_id], places=12)

    def test_the_checkpoint_actually_carries_the_stream(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            _run(output_dir, rounds=2)
            checkpoint = torch.load(
                output_dir / "checkpoints" / "round_002.pt",
                map_location="cpu",
                weights_only=False,
            )

        self.assertIn("rng_state", checkpoint)
        self.assertIn("python", checkpoint["rng_state"])
        self.assertIn("torch", checkpoint["rng_state"])

    def test_an_old_checkpoint_without_rng_state_warns(self) -> None:
        """Silently continuing on a fresh stream is the defect; say so instead."""

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            _run(output_dir, rounds=3)
            path = output_dir / "checkpoints" / "round_003.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            del checkpoint["rng_state"]
            torch.save(checkpoint, path)

            with _capture_stdout() as printed:
                _run(output_dir, rounds=4, resume_from=path)

        self.assertIn("no rng_state", printed.text)


@pytest.mark.fast
class RngStateRoundTripTests(unittest.TestCase):
    def test_capture_then_restore_replays_the_same_draws(self) -> None:
        state = capture_rng_state()
        first = [torch.rand(1).item() for _ in range(5)] + [random.random()]

        restore_rng_state(state)
        second = [torch.rand(1).item() for _ in range(5)] + [random.random()]

        self.assertEqual(first, second)

    def test_restoring_an_empty_state_is_a_no_op(self) -> None:
        self.assertEqual(restore_rng_state({}), [])

    def test_the_restored_streams_are_reported(self) -> None:
        restored = restore_rng_state(capture_rng_state())

        self.assertIn("python", restored)
        self.assertIn("torch", restored)


class _capture_stdout:
    def __init__(self) -> None:
        self.text = ""

    def __enter__(self):
        import contextlib
        import io

        self._buffer = io.StringIO()
        self._redirect = contextlib.redirect_stdout(self._buffer)
        self._redirect.__enter__()
        return self

    def __exit__(self, *exc):
        self._redirect.__exit__(*exc)
        self.text = self._buffer.getvalue()
        return False


if __name__ == "__main__":
    unittest.main()
