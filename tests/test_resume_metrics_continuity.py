"""Tests that round metrics survive an interrupted run and resume cleanly.

Checkpoints have always been written inside the round loop; the metric CSVs
were written once, after the last round returned. A cancelled or preempted job
therefore left a checkpoint at round N next to no round_metrics.csv, which is
precisely the history --resume-latest replays. These tests pin the fix: the CSV
on disk always covers every round that has finished.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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
from fedbrew.core.refusal import RunRefused
from fedbrew.core.state import MetricRecord
from fedbrew.data.dataset import FederatedDataset
from fedbrew.servers.base import ServerStrategy


class _Dataset(FederatedDataset):
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
        return {"name": "resume-continuity-test"}


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


class _Server(ServerStrategy):
    def __init__(self) -> None:
        self.version = 0

    def initialize(self) -> dict[str, Any]:
        return {"model_state": {"version": self.version}}

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
        self.version = round_info.round_id
        return {"model_state": {"version": self.version}}

    def evaluate(
        self,
        round_info: RoundInfo,
        results: Sequence[EvalResult],
    ) -> dict[str, float]:
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
    train=SplitEvaluationConfig(every=1, clients="all"),
    val=SplitEvaluationConfig(every="never", clients="all"),
    test=SplitEvaluationConfig(every=1, clients="all"),
    central_test=CentralTestConfig(every="never"),
)


def _run(
    output_dir: Path,
    global_rounds: int,
    resume_from: Path | None = None,
    on_round_end: Any = None,
    on_round_flush: Any = None,
    checkpointing: dict[str, Any] | None = None,
    per_client_csv: bool = False,
) -> Any:
    return run_fl_loop(
        server=_Server(),
        client={"client_0": _Client(), "client_1": _Client()},
        dataset=_Dataset(),
        global_rounds=global_rounds,
        output_dir=output_dir,
        resume_from=resume_from,
        checkpointing=checkpointing
        or {"enabled": True, "save_every_round": True, "keep_last": None},
        evaluation=_EVALUATION,
        client_statistics=ClientStatisticsConfig(per_client_csv=per_client_csv),
        on_round_end=on_round_end,
        on_round_flush=on_round_flush,
    )


def _drop_rounds(output_dir: Path, drop: set[int]) -> None:
    """Punch a hole in round_metrics.csv, as a partly-lost history would."""

    path = output_dir / "round_metrics.csv"
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row for row in rows if int(row["round_id"]) not in drop)


def _round_ids(output_dir: Path) -> list[int]:
    path = output_dir / "round_metrics.csv"
    with path.open("r", encoding="utf-8", newline="") as file:
        return [int(row["round_id"]) for row in csv.DictReader(file)]


class RoundMetricsFlushTests(unittest.TestCase):
    def test_csv_on_disk_covers_every_round_that_has_finished(self) -> None:
        """The file is current mid-run, not only after the last round."""

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            seen: list[list[int]] = []

            def observe(record: MetricRecord) -> None:
                # on_round_end fires after the flush, so the CSV must already
                # name this round -- that is the whole point of flushing.
                seen.append(_round_ids(output_dir))

            _run(output_dir, global_rounds=4, on_round_end=observe)

        self.assertEqual(seen, [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]])

    def test_flush_leaves_no_temp_file_behind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            _run(output_dir, global_rounds=2)
            self.assertEqual(list(output_dir.glob("*.tmp")), [])

    def test_resume_completes_the_existing_csv_instead_of_restarting_it(self) -> None:
        """An interrupted run's rounds are kept, and the rest are appended."""

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            # Stands in for a job cancelled after round 3: the CSV is on disk
            # because of the per-round flush, and so is the checkpoint.
            _run(output_dir, global_rounds=3)
            self.assertEqual(_round_ids(output_dir), [1, 2, 3])

            state = _run(
                output_dir,
                global_rounds=6,
                resume_from=output_dir / "checkpoints" / "round_003.pt",
            )

            self.assertEqual(_round_ids(output_dir), [1, 2, 3, 4, 5, 6])
            self.assertEqual(
                [record.round_id for record in state.metrics_history],
                [1, 2, 3, 4, 5, 6],
            )

    def test_resume_drops_rounds_at_or_after_the_checkpoint(self) -> None:
        """Rewinding to an earlier checkpoint truncates, it does not duplicate."""

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            _run(output_dir, global_rounds=5)
            self.assertEqual(_round_ids(output_dir), [1, 2, 3, 4, 5])

            _run(
                output_dir,
                global_rounds=4,
                resume_from=output_dir / "checkpoints" / "round_002.pt",
            )

            self.assertEqual(_round_ids(output_dir), [1, 2, 3, 4])


class ResumeGateTests(unittest.TestCase):
    """A resume is only taken when rounds 1..checkpoint are all on record."""

    def test_a_hole_in_the_history_is_refused(self) -> None:
        """Not restarted from round 1, as it was before POST-F25."""

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            _run(output_dir, global_rounds=6)
            _drop_rounds(output_dir, {3, 4})

            with self.assertRaises(RunRefused) as caught:
                _run(
                    output_dir,
                    global_rounds=8,
                    resume_from=output_dir / "checkpoints" / "round_006.pt",
                )

            self.assertIn("missing 2 of rounds 1-6", str(caught.exception))
            self.assertEqual(_round_ids(output_dir), [1, 2, 5, 6])

    def test_a_missing_history_is_refused_not_restarted(self) -> None:
        """It used to delete the attempt and start again from round 1.

        tests/test_a_refused_resume_changes_nothing.py pins the rest of
        POST-F25: nothing on disk changes, and the refusal names the way out.
        """

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            _run(output_dir, global_rounds=4)
            checkpoint = output_dir / "checkpoints" / "round_004.pt"
            (output_dir / "round_metrics.csv").unlink()

            with self.assertRaises(RunRefused):
                _run(output_dir, global_rounds=2, resume_from=checkpoint)

            self.assertFalse((output_dir / "round_metrics.csv").exists())
            self.assertTrue(checkpoint.is_file())


class StaleTempFileTests(unittest.TestCase):
    def test_a_killed_writes_temp_file_is_swept_at_the_next_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            output_dir.mkdir(exist_ok=True)
            orphan = output_dir / "client_metrics.csv.tmp"
            orphan.write_text("half a row", encoding="utf-8")

            _run(output_dir, global_rounds=1)

            self.assertFalse(orphan.exists())
            self.assertEqual(list(output_dir.glob("*.tmp")), [])

    def test_an_intact_history_is_still_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            _run(output_dir, global_rounds=4)

            state = _run(
                output_dir,
                global_rounds=6,
                resume_from=output_dir / "checkpoints" / "round_004.pt",
            )

            self.assertEqual(list(output_dir.glob("discarded_*")), [])
            self.assertEqual(
                [record.round_id for record in state.metrics_history],
                [1, 2, 3, 4, 5, 6],
            )


def _corrupt_first_row(path: Path, damage: str) -> None:
    """Damage the first data row of a per-client CSV, as a bad copy or edit would."""

    lines = path.read_text(encoding="utf-8").splitlines()
    if damage == "short row":
        lines[1] = lines[1].split(",")[0]
    else:
        lines[1] = "not-a-round" + lines[1][lines[1].index(",") :]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class ACorruptClientHistoryRefusesTheResumeTests(unittest.TestCase):
    """POST-F10: a resume deleted every earlier round's per-client rows.

    Measured before the fix through `fedbrew run`: one corrupt row in
    client_metrics.csv, then a resume, exited 0 with a one-line warning and
    rewrote client_metrics.csv and client_update_metrics.csv with the resumed
    rounds only, the second of them intact until then. The resume is refused
    now, naming the file and the line, and both files are left as they were.
    """

    def test_a_corrupt_row_refuses_the_resume_and_keeps_both_files(self) -> None:
        for name, damage in (
            ("client_metrics.csv", "bad field"),
            ("client_metrics.csv", "short row"),
            ("client_update_metrics.csv", "bad field"),
        ):
            with self.subTest(file=name, damage=damage), tempfile.TemporaryDirectory() as directory:
                output_dir = Path(directory)
                _run(output_dir, global_rounds=3, per_client_csv=True)
                _corrupt_first_row(output_dir / name, damage)
                before = {
                    csv_name: (output_dir / csv_name).read_bytes()
                    for csv_name in ("client_metrics.csv", "client_update_metrics.csv")
                }

                with self.assertRaises(RunRefused) as caught:
                    _run(
                        output_dir,
                        global_rounds=4,
                        resume_from=output_dir / "checkpoints" / "round_003.pt",
                        per_client_csv=True,
                    )

                self.assertIn(f"{name}:2", str(caught.exception))
                for csv_name, content in before.items():
                    self.assertEqual((output_dir / csv_name).read_bytes(), content, csv_name)

    def test_a_torn_final_row_is_still_resumed(self) -> None:
        """What an interrupted append leaves: that one row is dropped, the rest replayed."""

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            _run(output_dir, global_rounds=3, per_client_csv=True)
            path = output_dir / "client_metrics.csv"
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join([*lines[:-1], lines[-1].split(",")[0]]), encoding="utf-8")

            state = _run(
                output_dir,
                global_rounds=4,
                resume_from=output_dir / "checkpoints" / "round_003.pt",
                per_client_csv=True,
            )

            rounds = [record.round_id for record in state.client_metrics_history]
            self.assertEqual((rounds.count(1), rounds.count(2), rounds.count(4)), (2, 2, 2))


class FlushCadenceTests(unittest.TestCase):
    def test_per_client_csvs_are_current_every_round(self) -> None:
        """A resume rewinds to latest.pt, and save_last writes that every round.

        These files used to be rewritten only on rounds that wrote a
        checkpoint, on the premise that a resume could not rewind further back
        than one. save_last makes every round a rewind point, so any round the
        CSV lagged behind would lose its per-client rows for good: the resumed
        run rebuilds this history from the file and keeps only rows before the
        checkpoint's round. They are appended to each round instead, which is
        what made writing them every round affordable.
        """

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            seen: list[list[int]] = []
            path = output_dir / "client_metrics.csv"

            def observe(record: MetricRecord) -> None:
                if not path.is_file():
                    seen.append([])
                    return
                with path.open("r", encoding="utf-8", newline="") as file:
                    rows = csv.DictReader(file)
                    seen.append(sorted({int(row["round_id"]) for row in rows}))

            _run(
                output_dir,
                global_rounds=5,
                per_client_csv=True,
                checkpointing={
                    "enabled": True,
                    "interval": 2,
                    "save_last": False,
                    "save_best": False,
                    "save_every_round": False,
                    "keep_last": None,
                },
                on_round_end=observe,
            )

        # Checkpoints land on rounds 2 and 4 only, and save_last is off, yet
        # the file covers every finished round after each one. It used to read
        # [[], [1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4]].
        self.assertEqual(
            seen,
            [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
        )

    def test_round_metrics_flush_every_round_regardless_of_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            seen: list[list[int]] = []

            _run(
                output_dir,
                global_rounds=4,
                checkpointing={
                    "enabled": True,
                    "interval": 3,
                    "save_last": False,
                    "save_best": False,
                    "save_every_round": False,
                    "keep_last": None,
                },
                on_round_end=lambda record: seen.append(_round_ids(output_dir)),
            )

        self.assertEqual(seen, [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]])

    def test_the_run_json_hook_fires_once_per_round(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            rounds: list[int] = []

            _run(
                output_dir,
                global_rounds=3,
                on_round_flush=lambda state: rounds.append(state.metrics_history[-1].round_id),
            )

        self.assertEqual(rounds, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
