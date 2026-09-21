"""When `run.json` exists: after the first completed round, and not before.

The runner hands the loop a writer (`runner._run_json_writer`) that the loop
calls once per completed round, after that round's CSV rows and before its
checkpoints commit; `runner.run` writes the final record when the loop
returns. There is no initial `run.json`: nothing is written before round 1,
so a run that fails while building its components or inside its first round
leaves none. Mid-run writes say `status: "running"` and the round they reach;
the final write replaces them. Chapter 09 §2 says so; the audit (03-final-
release-gate, B03) found the paper promising a `run.json` before round 1.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from unittest import mock

import yaml

from fedbrew.clients.torch_sgd_client import TorchSGDClient
from fedbrew.core import runner


def _config(directory: Path, rounds: int) -> Path:
    raw = yaml.safe_load(Path("configs/dev/smoke.yaml").read_text(encoding="utf-8"))
    raw["experiment"]["output_dir"] = str(directory / "run")
    raw["defaults"]["global_rounds"] = rounds
    raw["runtime"]["checkpointing"] = {"enabled": False}
    path = directory / "run.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


class RunJsonIsWrittenAfterEachCompletedRoundTest(unittest.TestCase):
    def test_absent_before_round_one_then_rewritten_each_round_then_final(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory), rounds=3)
            run_json = Path(directory) / "run" / "run.json"
            at_first_fit: list[bool] = []
            after_round: list[dict[str, Any] | None] = []
            fit = TorchSGDClient.fit
            reporter = runner._round_progress_reporter

            def recording_fit(self: TorchSGDClient, request: Any) -> Any:
                if not at_first_fit:
                    at_first_fit.append(run_json.exists())
                return fit(self, request)

            def recording_reporter(*args: Any, **kwargs: Any) -> Any:
                report = reporter(*args, **kwargs)

                def after(record: Any) -> None:
                    after_round.append(_read(run_json))
                    report(record)

                return after

            with (
                mock.patch.object(TorchSGDClient, "fit", recording_fit),
                mock.patch.object(runner, "_round_progress_reporter", recording_reporter),
                redirect_stdout(StringIO()),
            ):
                runner.run(config, runner.parse_args(["--quiet"]))
            final = _read(run_json)

        self.assertEqual(at_first_fit, [False])
        self.assertEqual(len(after_round), 3)
        for round_id, record in enumerate(after_round, start=1):
            assert record is not None
            self.assertEqual(record["status"], "running")
            self.assertEqual(record["final_round"], round_id)
            self.assertEqual(record["num_rounds"], round_id)
        assert final is not None
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["final_round"], 3)

    def test_a_run_that_fails_in_round_one_leaves_no_run_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory), rounds=2)

            def failing_fit(self: TorchSGDClient, request: Any) -> Any:
                raise RuntimeError("fails inside round 1")

            with (
                mock.patch.object(TorchSGDClient, "fit", failing_fit),
                redirect_stdout(StringIO()),
                redirect_stderr(StringIO()),
                self.assertRaisesRegex(RuntimeError, "fails inside round 1"),
            ):
                runner.run(config, runner.parse_args(["--quiet"]))
            self.assertFalse((Path(directory) / "run" / "run.json").exists())


if __name__ == "__main__":
    unittest.main()
