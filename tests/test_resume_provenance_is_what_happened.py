"""`run.json` must say whether the run continued, not whether it was asked to.

`resumed` was `bool(resume_from)`, and `resume_from` is a config value. So a
run that asked to resume and *could not* recorded `resumed: true` -- and it had
restarted from round 1, having deleted the rejected attempt's artifacts.

Nothing on disk disagreed. Measured on the pre-fix tree, a resume that was
taken and a resume that was refused produced identical provenance:

    case                          resumed  first_round  resume_from
    fresh run                       false            1  null
    resume taken                     true            1  set
    resume refused -> restarted      true            1  set        <- same row

`first_round` is not the tell it looks like: a resume that *is* taken replays
`round_metrics.csv` from round 1, so `first_round` is 1 for both. It only
differs when a resume is refused and there was no history to replay -- which is
the case nobody is confused by.

The fix splits the question in two. `resumed` is what happened; `resume_from`
still records which checkpoint was asked for either way; and `resume_restart`
carries why it was not taken -- shaped like `termination`, null for the
ordinary case.

A restarted run is a *correct* run: every number in it is right. What was wrong
was the record's claim about which run those numbers came from, which is the
one thing a provenance file exists to get right. See FINDINGS.csv POST-F05.

Since POST-F25 the third row cannot occur: a resume that cannot be taken is
refused before anything is written, so it leaves no run.json at all, and
`resume_restart` went with the restart it described. What remains pinned is
the half of POST-F05 that still applies -- a fresh run claims nothing, a taken
resume says so -- and that the refused case writes no record claiming anything.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any

import pytest

from fedbrew.core import runner
from fedbrew.core.artifacts import save_run_json
from fedbrew.core.config import load_config
from fedbrew.core.loop import _initialize_or_resume
from fedbrew.core.refusal import RunRefused
from fedbrew.core.state import ExperimentState

CONFIG = """
experiment:
  seed: 42
  output_dir: {out}
server:
  strategy: fedavg
  participation_rate: 1
  metrics: [fit_loss]
client:
  update_rule: local_sgd
  batch_size: 4
  learning_rate: 0.05
  learning_rate_schedule: constant
  min_learning_rate: 0.0
  momentum: 0.0
  weight_decay: 0.0
  nesterov: false
  metrics: [fit_loss]
data:
  num_clients: 4
  samples_per_client: 16
  input_dim: 8
  num_classes: 4
model:
  name: mlp
  input_dim: 8
  hidden_dim: 16
  num_classes: 4
runtime:
  deterministic: true
  device: cpu
  use_amp: false
  checkpointing: {{enabled: true, interval: 1, save_last: true, save_best: false}}
evaluation:
  train: {{every: 1, clients: all}}
defaults:
  global_rounds: {rounds}
  local_iterations: 1
"""


def _write(directory: Path, name: str, out: Path, rounds: int) -> Path:
    path = directory / f"{name}.yaml"
    path.write_text(textwrap.dedent(CONFIG.format(out=out, rounds=rounds)).lstrip(), "utf-8")
    return path


class ThreeCasesOnDiskTest(unittest.TestCase):
    """One real run each, because the claim is about what reaches disk."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        root = Path(cls._directory.name)

        fresh = root / "fresh"
        runner.run(_write(root, "a", fresh, 2), runner.parse_args(["--quiet"]))

        taken = root / "taken"
        shutil.copytree(fresh, taken)
        runner.run(_write(root, "b", taken, 4), runner.parse_args(["--quiet", "--resume-latest"]))

        # A checkpoint with no metric history beside it: `round_metrics_gap`
        # rejects it, and the run is refused before anything is written.
        cls.refused = root / "refused"
        cls.refused.mkdir()
        shutil.copytree(fresh / "checkpoints", cls.refused / "checkpoints")
        cls.refusal = None
        try:
            runner.run(
                _write(root, "c", cls.refused, 4),
                runner.parse_args(
                    ["--quiet", "--resume-from", str(cls.refused / "checkpoints" / "latest.pt")]
                ),
            )
        except RunRefused as refusal:
            cls.refusal = refusal
        cls.runs = {
            name: json.loads((root / name / "run.json").read_text(encoding="utf-8"))
            for name in ("fresh", "taken")
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def test_a_fresh_run_claims_nothing(self) -> None:
        run = self.runs["fresh"]
        self.assertFalse(run["resumed"])
        self.assertIsNone(run["resume_from"])

    def test_a_resume_that_was_taken_says_so(self) -> None:
        run = self.runs["taken"]
        self.assertTrue(run["resumed"])
        self.assertIsNotNone(run["resume_from"])
        self.assertEqual(run["num_rounds"], 4)

    def test_a_resume_that_cannot_be_taken_is_refused(self) -> None:
        self.assertIsInstance(self.refusal, RunRefused)
        self.assertIn("round_metrics.csv", str(self.refusal))

    def test_the_refused_case_writes_no_record_claiming_anything(self) -> None:
        """Before POST-F25 it wrote a fresh run's run.json; now it writes nothing."""

        self.assertFalse((self.refused / "run.json").exists())
        self.assertEqual([p.name for p in self.refused.iterdir()], ["checkpoints"])

    def test_the_restart_key_is_gone(self) -> None:
        """It described a restart that no longer happens, so it could only be null."""

        for name in ("fresh", "taken"):
            with self.subTest(case=name):
                self.assertNotIn("resume_restart", self.runs[name])

    def test_first_round_is_not_the_tell_it_looks_like(self) -> None:
        """It is 1 for a taken resume too, because a taken resume replays from round 1."""

        self.assertEqual(self.runs["taken"]["first_round"], 1)


@pytest.mark.fast
class TheLoopReportsItTest(unittest.TestCase):
    """`_initialize_or_resume` is the only place that knows."""

    def test_no_resume_requested_reports_no_restart(self) -> None:
        from fedbrew.servers.fedavg import FedAvgServer
        from tests.test_empty_training_batches import _Task

        server = FedAvgServer(participation_rate=1.0, seed=0, task=_Task())
        payload, start_round, checkpoint = _initialize_or_resume(server, None)
        self.assertEqual(start_round, 1)
        self.assertIsNone(checkpoint)
        self.assertIn("model_state", payload)


@pytest.mark.fast
class TheMidRunWriteCarriesItTest(unittest.TestCase):
    """A job killed mid-run must leave run.json saying what the loop decided.

    run.json is rewritten every round with `status: running`, and that write
    takes its own metadata dict. Threading `resumed` only into the final write
    would leave every preempted run carrying the config's answer instead.
    """

    def _written(self, state: ExperimentState) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "run"
            config = load_config(_write(root, "d", out, 1))
            save_run_json(
                state.metrics_history,
                out,
                config,
                run_metadata={"status": "running", "resumed": state.resumed},
            )
            return json.loads((out / "run.json").read_text(encoding="utf-8"))

    def test_a_fresh_attempt_mid_run(self) -> None:
        self.assertFalse(self._written(ExperimentState(resumed=False))["resumed"])

    def test_a_continued_attempt_mid_run(self) -> None:
        self.assertTrue(self._written(ExperimentState(resumed=True))["resumed"])

    def test_the_writer_the_runner_installs_passes_them_through(self) -> None:
        """Read off the source, because the closure is built per run."""

        source = Path(runner.__file__).read_text(encoding="utf-8").split("def _run_json_writer")[1]
        self.assertIn('"resumed": state.resumed', source)


@pytest.mark.fast
class TheDefaultIsNotResumedTest(unittest.TestCase):
    def test_a_bare_state_claims_nothing(self) -> None:
        self.assertFalse(ExperimentState().resumed)


if __name__ == "__main__":
    unittest.main()
