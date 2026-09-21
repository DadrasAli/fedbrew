"""What run.json says about a run that was requeued.

A resume starts a fresh process, so anything measured with perf_counter covers
the last attempt and nothing before it. Reported without that context,
duration_sec is the tail of a run presented as the whole of it, and nothing
else on disk said the run had been resumed at all -- resume_from was generated
on both resume paths and then dropped.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pytest

from fedbrew.cli.make_report import _round_span
from fedbrew.core.artifacts import save_run_json
from fedbrew.core.config import load_config
from fedbrew.core.refusal import RunRefused
from fedbrew.core.runner import (
    _carry_previous_attempt,
    _record_durations,
    _refuse_a_foreign_seed,
    _refuse_to_replace_a_finished_run,
    config_differences,
)
from fedbrew.core.state import MetricRecord

pytestmark = pytest.mark.fast


def _history(first: int, last: int) -> list[MetricRecord]:
    return [
        MetricRecord(round_id=r, metrics={"fit_loss": 1.0}, num_clients=2, num_examples=8)
        for r in range(first, last + 1)
    ]


class RunJsonResumeFieldsTests(unittest.TestCase):
    """run.json must account for every attempt, not just the last one.

    It reported num_rounds, duration_sec, timing.* and scale.* from the final
    attempt's history alone, so a resumed 1000-round run read as a shorter one
    and nothing on disk said it had been resumed.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.config = load_config("configs/dev/smoke.yaml")

    def _write(self, history, **metadata) -> dict:
        save_run_json(
            history,
            self.tmpdir,
            self.config,
            run_metadata={"run_id": "r", "status": "completed", **metadata},
        )
        with (self.tmpdir / "run.json").open(encoding="utf-8") as handle:
            return json.load(handle)

    def test_a_fresh_run_says_it_was_not_resumed(self) -> None:
        run = self._write(_history(1, 5))

        self.assertFalse(run["resumed"])
        self.assertIsNone(run["resume_from"])
        self.assertEqual(run["attempts"], 1)
        self.assertEqual(run["first_round"], 1)
        self.assertEqual(run["num_rounds"], 5)
        self.assertEqual(run["final_round"], 5)

    def test_a_resumed_run_records_the_checkpoint_it_continued_from(self) -> None:
        run = self._write(
            _history(1, 5),
            resumed=True,
            resume_from="/runs/x/checkpoints/round_003.pt",
            attempts=2,
        )

        self.assertTrue(run["resumed"])
        self.assertEqual(run["resume_from"], "/runs/x/checkpoints/round_003.pt")
        self.assertEqual(run["attempts"], 2)

    def test_a_history_that_does_not_start_at_round_one_is_visible(self) -> None:
        """num_rounds equals the rounds run only when first_round is 1."""

        run = self._write(_history(120, 1000))

        self.assertEqual(run["num_rounds"], 881)
        self.assertEqual(run["first_round"], 120)
        self.assertEqual(run["final_round"], 1000)

    def test_both_durations_are_reported(self) -> None:
        run = self._write(_history(1, 5), duration_sec=900.0, attempt_duration_sec=300.0)

        self.assertEqual(run["duration_sec"], 900.0)
        self.assertEqual(run["attempt_duration_sec"], 300.0)
        self.assertEqual(run["timing"]["run_duration_sec"], 900.0)


class AttemptDurationTests(unittest.TestCase):
    """duration_sec covers the run; attempt_duration_sec covers this process."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.config = load_config("configs/dev/smoke.yaml")

    def _config_resuming_from(self, checkpoint: str | None):
        from dataclasses import replace

        extra = dict(self.config.runtime.extra)
        if checkpoint is not None:
            extra["resume_from"] = checkpoint
        return replace(self.config, runtime=replace(self.config.runtime, extra=extra))

    def test_a_fresh_run_carries_nothing(self) -> None:
        metadata: dict = {}
        _carry_previous_attempt(self._config_resuming_from(None), metadata, self.tmpdir)

        self.assertEqual(metadata["previous_duration_sec"], 0.0)
        self.assertEqual(metadata["attempts"], 1)
        self.assertFalse(metadata["resumed"])

    def test_a_resume_adds_the_previous_attempts_time(self) -> None:
        (self.tmpdir / "run.json").write_text(
            json.dumps({"duration_sec": 600.0, "attempts": 2}), encoding="utf-8"
        )
        metadata: dict = {}
        _carry_previous_attempt(self._config_resuming_from("ckpt.pt"), metadata, self.tmpdir)

        self.assertEqual(metadata["previous_duration_sec"], 600.0)
        self.assertEqual(metadata["attempts"], 3)
        self.assertTrue(metadata["resumed"])
        self.assertEqual(metadata["resume_from"], "ckpt.pt")

    def test_a_resume_onto_an_older_run_json_does_not_crash(self) -> None:
        """run.json from before these fields existed, or an unreadable one."""

        (self.tmpdir / "run.json").write_text("{not json", encoding="utf-8")
        metadata: dict = {}
        _carry_previous_attempt(self._config_resuming_from("ckpt.pt"), metadata, self.tmpdir)

        self.assertEqual(metadata["previous_duration_sec"], 0.0)
        self.assertTrue(metadata["resumed"])

    def test_the_totals_add_up(self) -> None:
        import time

        metadata = {"previous_duration_sec": 600.0}
        _record_durations(metadata, time.perf_counter())

        self.assertLess(metadata["attempt_duration_sec"], 1.0)
        self.assertAlmostEqual(
            metadata["duration_sec"],
            600.0 + metadata["attempt_duration_sec"],
            places=2,
        )


class SeedCollisionTests(unittest.TestCase):
    """Regression tests for single-seed comparisons (harness half).

    A comparison built from one run per arm reports a difference with no
    dispersion behind it. Replicates are the fix, and nothing in the output
    path distinguishes them unless the caller puts the seed there -- so two
    seeds of one arm silently resolved to one directory.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.config = load_config("configs/dev/smoke.yaml")

    def _seeded(self, seed: int):
        from dataclasses import replace

        return replace(
            self.config,
            experiment=replace(self.config.experiment, seed=seed),
        )

    def _write_run_json(self, seed: int) -> None:
        (self.tmpdir / "run.json").write_text(
            json.dumps({"config": {"experiment": {"seed": seed}}}), encoding="utf-8"
        )

    def test_an_empty_directory_is_fine(self) -> None:
        _refuse_a_foreign_seed(self._seeded(42), self.tmpdir)

    def test_the_same_seed_may_rerun_in_place(self) -> None:
        self._write_run_json(42)
        _refuse_a_foreign_seed(self._seeded(42), self.tmpdir)

    def test_a_different_seed_is_refused(self) -> None:
        self._write_run_json(42)
        with self.assertRaises(RunRefused) as caught:
            _refuse_a_foreign_seed(self._seeded(43), self.tmpdir)

        message = str(caught.exception)
        self.assertIn("seed 43", message)
        self.assertIn("seed 42", message)
        self.assertIn("seed_43", message)

    def test_an_unreadable_run_json_does_not_block_the_run(self) -> None:
        (self.tmpdir / "run.json").write_text("{not json", encoding="utf-8")
        _refuse_a_foreign_seed(self._seeded(43), self.tmpdir)

    def test_a_run_json_without_a_seed_does_not_block_the_run(self) -> None:
        (self.tmpdir / "run.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
        _refuse_a_foreign_seed(self._seeded(43), self.tmpdir)


class FinishedRunReplacementTests(unittest.TestCase):
    """A fresh start must not replace a finished run of another config. POST-F22.

    The seed check above guards the seed alone. At the same seed a fresh start
    replaced whatever finished run the directory held with a run of different
    settings, and nothing on disk said so. The run.json here is the one
    `save_run_json` writes, so the comparison is made against the real format.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.config = load_config("configs/dev/smoke.yaml")

    def _finished(self, config, status: str = "completed") -> None:
        save_run_json(
            _history(1, 2),
            self.tmpdir,
            config,
            run_metadata={"run_id": "earlier", "status": status},
        )

    def _with(self, config, **sections):
        from dataclasses import replace

        return replace(
            config,
            **{name: replace(getattr(config, name), **values) for name, values in sections.items()},
        )

    def test_an_empty_directory_is_fine(self) -> None:
        _refuse_to_replace_a_finished_run(self.config, self.tmpdir)

    def test_the_same_config_may_rerun_in_place(self) -> None:
        self._finished(self.config)
        _refuse_to_replace_a_finished_run(self.config, self.tmpdir)

    def test_a_different_config_is_refused_whatever_the_run_ended_as(self) -> None:
        changed = self._with(self.config, client={"learning_rate": 0.5})
        for status in ("completed", "diverged", "stalled"):
            with self.subTest(status=status):
                self._finished(self.config, status)
                with self.assertRaises(RunRefused) as caught:
                    _refuse_to_replace_a_finished_run(changed, self.tmpdir)
                message = str(caught.exception)
                self.assertIn("client.learning_rate", message)
                self.assertIn(f"that run {self.config.client.learning_rate!r}", message)
                self.assertIn("this run 0.5", message)
                self.assertIn(status, message)

    def test_labels_and_launch_flags_are_not_configuration(self) -> None:
        self._finished(self.config)
        relabelled = self._with(
            self.config,
            experiment={"name": "renamed", "tags": ["other"], "notes": "n", "output_dir": "x"},
            runtime={
                "extra": {
                    **self.config.runtime.extra,
                    "quiet": True,
                    "verbose": True,
                    "no_rich": True,
                    "print_every": 5,
                }
            },
        )
        self.assertEqual(config_differences(_recorded(self.tmpdir), relabelled), [])
        _refuse_to_replace_a_finished_run(relabelled, self.tmpdir)

    def test_a_key_the_recorded_run_lacks_is_not_compared(self) -> None:
        """As a checkpoint written before a setting existed still resumes."""

        self._finished(self.config)
        recorded = _recorded(self.tmpdir)
        del recorded["client"]["learning_rate"]
        changed = self._with(self.config, client={"learning_rate": 0.5})
        self.assertEqual(config_differences(recorded, changed), [])

    def test_a_run_still_marked_running_may_restart_in_place(self) -> None:
        """A crash leaves `running`; restarting it is ordinary."""

        self._finished(self.config, "running")
        _refuse_to_replace_a_finished_run(
            self._with(self.config, client={"learning_rate": 0.5}), self.tmpdir
        )

    def test_a_resume_is_not_a_fresh_start(self) -> None:
        self._finished(self.config)
        resuming = self._with(
            self.config,
            client={"learning_rate": 0.5},
            runtime={"extra": {**self.config.runtime.extra, "resume_from": "latest.pt"}},
        )
        _refuse_to_replace_a_finished_run(resuming, self.tmpdir)

    def test_an_unreadable_run_json_does_not_block_the_run(self) -> None:
        (self.tmpdir / "run.json").write_text("{not json", encoding="utf-8")
        _refuse_to_replace_a_finished_run(self.config, self.tmpdir)

    def test_a_staged_data_path_is_the_same_data(self) -> None:
        """Under staging, data.path names a per-job copy of the same manifest."""

        staged = self._with(
            self.config,
            runtime={"extra": {**self.config.runtime.extra, "data_staging": {"enabled": True}}},
        )
        self._finished(staged)
        moved = self._with(staged, data={"path": "/scratch/job-2/manifest.json"})
        self.assertEqual(config_differences(_recorded(self.tmpdir), moved), [])


def _recorded(directory: Path) -> dict:
    return json.loads((directory / "run.json").read_text(encoding="utf-8"))["config"]


class ReportRoundSpanTests(unittest.TestCase):
    """fedbrew report printed a bare count with nothing beside it to contradict it."""

    def test_a_complete_history_prints_the_count(self) -> None:
        self.assertEqual(
            _round_span(
                {"num_rounds": 1000, "first_round": 1, "final_round": 1000},
                {"global_rounds": 1000},
            ),
            "1000",
        )

    def test_a_partial_history_prints_the_span(self) -> None:
        self.assertEqual(
            _round_span(
                {"num_rounds": 881, "first_round": 120, "final_round": 1000},
                {"global_rounds": 1000},
            ),
            "881 recorded, rounds 120-1000 of 1000 configured",
        )

    def test_a_missing_count_prints_nothing(self) -> None:
        self.assertEqual(_round_span({}, {}), "")

    def test_a_short_run_is_measured_against_what_was_configured(self) -> None:
        """ "1" alone reads as a one-round run rather than one that died."""

        self.assertEqual(
            _round_span(
                {"num_rounds": 1, "first_round": 1, "final_round": 1},
                {"global_rounds": 500},
            ),
            "1 of 500 configured",
        )


if __name__ == "__main__":
    unittest.main()
