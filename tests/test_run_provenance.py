"""A number with no record of the code that produced it cannot be traced back.

runs_index.jsonl has carried git_commit and git_dirty per row since it was
written, and nothing ever produced code_state: every run recorded null for
both, inside a git checkout. save_run_json read the same key into a
local and then never used it. The same held for the runtime record --
seed_everything and configure_runtime each return exactly what they did, and
both returns were discarded apart from resolved_device, so a swallowed
set_float32_matmul_precision, the effective cudnn_benchmark, and the torch
version reached no file at all.

None of this is a metric error. It is the audit trail that would let one be
found, so these tests pin that the trail exists and says what it means.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from fedbrew.core import run_metadata, runner
from fedbrew.core.artifacts import append_run_index, save_run_json
from fedbrew.core.config import load_config
from fedbrew.core.run_metadata import capture_code_state

REPO_ROOT = Path(runner.__file__).resolve().parents[2]
IN_CHECKOUT = (REPO_ROOT / ".git").exists()

_FULL_COMMIT = re.compile(r"\A[0-9a-f]{40}\Z")

#: A commit that is not this repository's, so a test asserting on it cannot
#: pass by accidentally reading the live checkout.
STAMPED = "0" * 39 + "1"


def assert_code_state_is_wellformed(case: unittest.TestCase, state: dict) -> None:
    """One of exactly three shapes, and the record says which one it is.

    These call sites used to assert `isinstance(git_dirty, bool)`, which is
    true only of the checkout shape. That is what made five tests here fail
    from a release archive -- not a defect in the archive, a test that assumed
    the environment it was written in. Asserting the whole shape instead is
    strictly stronger: it also rejects a commit reported with no source, and a
    source reported with no commit.
    """

    source = state.get("commit_source")
    if source == "git":
        case.assertRegex(state["git_commit"], _FULL_COMMIT)
        case.assertIsInstance(state["git_dirty"], bool)
        case.assertNotIn("git_error", state)
    elif source == "archive":
        case.assertRegex(state["git_commit"], _FULL_COMMIT)
        case.assertIsNone(state["git_dirty"], "an extracted tree has nothing to diff against")
        case.assertNotIn("git_error", state)
    else:
        case.assertIsNone(state["git_commit"])
        case.assertIsNone(state["git_dirty"])
        case.assertIn("git_error", state)


def _init_repository(root: Path) -> None:
    """Create a repository whose own commands write nothing in the background.

    `git commit` asks git to run auto maintenance, and recent git runs it
    detached: on 2.55, the version the public runners ship, the commit returns
    while a `git maintenance run --auto --quiet --detach` child is still
    starting. That child takes .git/objects/maintenance.lock and drops it again
    in under a millisecond on an idle machine, which is short enough to be
    invisible here and long enough, on a loaded runner, to land inside the
    window `test_the_lookup_writes_nothing_to_the_repository` compares -- the
    public core-only job failed once on exactly
    `['.git/objects/maintenance.lock'] != []`, which is that file, written by
    the test's own `git commit` in its own temporary repository rather than by
    the lookup. `capture_code_state` runs `git rev-parse` and `git status`, and
    neither runs maintenance at all, so turning it off here removes the race
    without weakening anything the tests assert.
    """

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for name, value in (
        ("user.email", "t@t"),
        ("user.name", "t"),
        # Either setting alone stops the spawn on 2.55. Both are set so a git
        # that stops consulting one of them cannot bring the race back
        # silently, and they are set before the first commit, which is the one
        # command here that would ask for maintenance.
        ("maintenance.auto", "false"),
        ("gc.auto", "0"),
    ):
        subprocess.run(["git", "config", name, value], cwd=root, check=True)


def _write_config(directory: Path) -> Path:
    config_path = directory / "provenance.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            experiment:
              seed: 42
              output_dir: {directory / "run"}

            server:
              strategy: fedavg
              participation_rate: 1
              metrics:
                - fit_loss

            client:
              update_rule: local_sgd
              batch_size: 4
              learning_rate: 0.05
              learning_rate_schedule: constant
              min_learning_rate: 0.0
              momentum: 0.0
              weight_decay: 0.0
              nesterov: false
              metrics:
                - fit_loss

            data:
              num_clients: 2
              samples_per_client: 8
              input_dim: 4
              num_classes: 2

            model:
              name: mlp
              input_dim: 4
              hidden_dim: 4
              num_classes: 2

            runtime:
              deterministic: true
              device: cpu
              use_amp: false
              performance:
                matmul_precision: high

            evaluation:
              train:
                every: 1
                clients: all

            defaults:
              global_rounds: 1
              local_iterations: 1
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return config_path


class CaptureCodeStateTest(unittest.TestCase):
    """The lookup answers, and says why when it cannot.

    Every test here runs in both environments the package ships into except
    the one marked otherwise, which compares against a value only a checkout
    can produce. The shape each environment must produce is pinned
    unconditionally by `assert_code_state_is_wellformed`.
    """

    def test_the_shape_is_one_of_the_three_wherever_this_runs(self) -> None:
        assert_code_state_is_wellformed(self, capture_code_state())

    @unittest.skipUnless(IN_CHECKOUT, "compares against `git rev-parse`, which needs .git")
    def test_it_reports_the_commit_this_checkout_is_on(self) -> None:
        state = capture_code_state()
        self.assertNotIn("git_error", state)
        expected = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(runner.__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.assertEqual(state["git_commit"], expected)
        self.assertIsInstance(state["git_dirty"], bool)
        self.assertEqual(state["commit_source"], "git")

    def test_a_directory_that_is_not_a_repository_reports_why(self) -> None:
        """Absence must be distinguishable from a lookup that never ran.

        The stamp is forced absent rather than left to the environment: run
        from an exported archive this path answers from the stamp instead, and
        a test that passes for a different reason in each environment is not a
        test of either.
        """

        with mock.patch.object(run_metadata, "source_commit", return_value=None):
            with tempfile.TemporaryDirectory() as directory:
                state = capture_code_state(Path(directory))
        self.assertIsNone(state["git_commit"])
        self.assertIsNone(state["git_dirty"])
        self.assertNotIn("commit_source", state)
        self.assertIn("git_error", state)
        assert_code_state_is_wellformed(self, state)

    def test_an_archive_answers_from_the_stamp_when_there_is_no_git(self) -> None:
        """The release case: no .git, and the commit still recorded."""

        with mock.patch.object(run_metadata, "source_commit", return_value=STAMPED):
            with tempfile.TemporaryDirectory() as directory:
                state = capture_code_state(Path(directory))
        self.assertEqual(state["git_commit"], STAMPED)
        self.assertEqual(state["commit_source"], "archive")
        self.assertIsNone(state["git_dirty"])
        self.assertNotIn("git_error", state)
        assert_code_state_is_wellformed(self, state)

    def test_a_checkout_prefers_the_live_commit_over_the_stamp(self) -> None:
        """The fallback is a fallback. A checkout must not report a stale stamp."""

        if not IN_CHECKOUT:  # pragma: no cover - the archive has no live commit
            self.skipTest("needs a checkout to have a live commit to prefer")
        with mock.patch.object(run_metadata, "source_commit", return_value=STAMPED):
            state = capture_code_state()
        self.assertNotEqual(state["git_commit"], STAMPED)
        self.assertEqual(state["commit_source"], "git")

    def test_it_reads_the_running_code_not_the_working_directory(self) -> None:
        """A sweep launches every arm from the submit directory."""

        with tempfile.TemporaryDirectory() as directory:
            _init_repository(Path(directory))
            here = Path.cwd()
            try:
                import os

                os.chdir(directory)
                state = capture_code_state()
            finally:
                os.chdir(here)
        # The empty repo just created has no HEAD; the real checkout does.
        self.assertIsNotNone(state["git_commit"])

    def test_untracked_files_do_not_mark_the_checkout_dirty(self) -> None:
        """outputs/ and scratch are untracked by design in every run."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _init_repository(root)
            (root / "tracked.py").write_text("x = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "first"], cwd=root, check=True)

            (root / "untracked.json").write_text("{}", encoding="utf-8")
            self.assertFalse(capture_code_state(root)["git_dirty"])

            (root / "tracked.py").write_text("x = 2\n", encoding="utf-8")
            self.assertTrue(capture_code_state(root)["git_dirty"])

    def test_the_lookup_writes_nothing_to_the_repository(self) -> None:
        """A run reads the checkout it records; it does not write to it.

        `git status` refreshes the stat cache in .git/index whenever a tracked
        file's stat data has moved since the index last saw it -- an editor
        save, a checkout, an rsync -- and it takes index.lock to do so. A run
        holding that lock makes the user's own `git add` or `git commit` in the
        same checkout fail. The stale stat is forced with an old mtime: a file
        just committed gives status nothing to refresh, and the write this test
        exists to catch would not happen.

        The repository watched is a fresh one this test creates, so the only
        writer that can reach it is the lookup. `_init_repository` says which
        background writer git itself would otherwise add, and why it is off.
        """

        import os

        def files_under(root: Path) -> dict[str, bytes]:
            return {
                path.relative_to(root).as_posix(): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _init_repository(root)
            tracked = root / "tracked.py"
            tracked.write_text("x = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "first"], cwd=root, check=True)
            os.utime(tracked, (0, 0))

            before = files_under(root)
            state = capture_code_state(root)
            after = files_under(root)

        self.assertEqual(state["commit_source"], "git")
        self.assertFalse(state["git_dirty"], "a moved mtime over unchanged content is not a change")
        changed = sorted(
            key for key in before.keys() | after.keys() if before.get(key) != after.get(key)
        )
        self.assertEqual(changed, [], "the lookup wrote to the repository it was reading")


@unittest.skipUnless(IN_CHECKOUT, "exports an archive, which needs the repository to export")
class TheArchiveIsActuallyStampedTest(unittest.TestCase):
    """The stamp is only worth having if the export writes it.

    `capture_code_state`'s archive branch is exercised above with a patched
    stamp, which proves the branch and proves nothing about `.gitattributes`.
    Dropping the `export-subst` line there would leave every release carrying
    the unexpanded placeholder and every test above still green. So this one
    runs the real export and reads the real file back out of it.
    """

    def _export(self, directory: Path) -> Path:
        archive = directory / "export.tar"
        with archive.open("wb") as file:
            subprocess.run(
                ["git", "archive", "--format=tar", "HEAD"],
                cwd=REPO_ROOT,
                stdout=file,
                check=True,
            )
        tree = directory / "tree"
        tree.mkdir()
        subprocess.run(["tar", "-xf", str(archive), "-C", str(tree)], check=True)
        return tree

    def test_an_exported_archive_carries_the_commit_it_was_exported_from(self) -> None:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        with tempfile.TemporaryDirectory() as directory:
            tree = self._export(Path(directory))
            self.assertFalse((tree / ".git").exists(), "an archive carries no repository")
            stamped = (tree / "fedbrew" / "_build_info.py").read_text(encoding="utf-8")
        self.assertIn(head, stamped, "git archive did not substitute the placeholder")
        self.assertNotIn("Format:", stamped, "the placeholder survived the export unexpanded")


class RunsIndexTest(unittest.TestCase):
    """The two fields the index has always promised now carry a value."""

    def test_a_real_run_writes_a_commit_into_the_index(self) -> None:
        """The finding's literal claim: every row was null in a git checkout."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner.run(_write_config(root), runner.parse_args(["--quiet"]))
            rows = [
                json.loads(line)
                for line in (root / "runs_index.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        self.assertEqual(len(rows), 1)
        self.assertIsNotNone(rows[0]["git_commit"])
        assert_code_state_is_wellformed(self, rows[0])

    def test_the_index_row_carries_the_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            append_run_index(
                {
                    "run_id": "r1",
                    "code_state": capture_code_state(),
                    "final_metrics": {},
                },
                root,
            )
            row = json.loads((root / "runs_index.jsonl").read_text(encoding="utf-8").strip())
        self.assertIsNotNone(row["git_commit"])
        assert_code_state_is_wellformed(self, row)


class RunJsonProvenanceTest(unittest.TestCase):
    """An end-to-end run must leave the record on disk, not just compute it."""

    def _run(self, directory: Path) -> dict:
        args = runner.parse_args(["--quiet"])
        runner.run(_write_config(directory), args)
        written = sorted(directory.rglob("run.json"))
        self.assertEqual(len(written), 1)
        return json.loads(written[0].read_text(encoding="utf-8"))

    def test_run_json_records_the_code_that_produced_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._run(Path(directory))
        code_state = run["reproducibility"]["code_state"]
        self.assertIsNotNone(code_state["git_commit"])
        assert_code_state_is_wellformed(self, code_state)

    def test_run_json_records_what_was_seeded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._run(Path(directory))
        seeding = run["reproducibility"]["seeding"]
        self.assertEqual(seeding["seed"], 42)
        self.assertTrue(seeding["deterministic"])
        # Strict by default, and the file has to say which it was.
        self.assertFalse(seeding["deterministic_warn_only"])
        self.assertIsNotNone(seeding["torch_version"])

    def test_run_json_records_the_flags_torch_actually_held(self) -> None:
        """The request is in config; this is the effective value."""

        with tempfile.TemporaryDirectory() as directory:
            run = self._run(Path(directory))
        runtime = run["reproducibility"]["runtime"]
        self.assertEqual(runtime["matmul_precision"], "high")
        self.assertEqual(runtime["resolved_device"], "cpu")
        self.assertIn("cudnn_benchmark", runtime)
        self.assertIn("torch_deterministic_algorithms", runtime)
        self.assertIsInstance(runtime["torch_available"], bool)

    def test_no_flag_is_recorded_twice_with_two_values(self) -> None:
        """seed_everything reads torch before configure_runtime writes to it.

        Its snapshot says matmul_precision "highest" on a run that trains at
        "high", so recording both records would put the stale value beside the
        effective one for the single key that changes every fp32 matmul.
        """

        with tempfile.TemporaryDirectory() as directory:
            run = self._run(Path(directory))
        record = run["reproducibility"]
        seeding = record["seeding"]
        runtime = record["runtime"]
        self.assertEqual(set(seeding) & set(runtime), set())
        self.assertEqual(runtime["matmul_precision"], "high")
        # What seeding alone decided, and the versions, are still here.
        self.assertEqual(seeding["seed"], 42)
        self.assertIsNotNone(seeding["torch_version"])
        # And nothing states the model scope twice.
        self.assertIn("model_state_scope", record["federated_model_state"])
        self.assertNotIn("model_state_scope", record)

    @pytest.mark.fast
    def test_a_reduced_runtime_record_does_not_erase_the_flags(self) -> None:
        """configure_runtime's exception path returns none of the flags.

        Dropping them from seeding unconditionally would then record them
        nowhere, which is the defect this whole finding is about.
        """

        with tempfile.TemporaryDirectory() as directory:
            save_run_json(
                [],
                Path(directory),
                load_config("configs/dev/smoke.yaml"),
                run_metadata={
                    "run_id": "r1",
                    "seeding": {
                        "seed": 42,
                        "matmul_precision": "highest",
                        "torch_version": "2.5.1",
                    },
                    "runtime": {
                        "resolved_device": "cpu",
                        "runtime_setup_error": "boom",
                    },
                },
            )
            run = json.loads((Path(directory) / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(run["reproducibility"]["seeding"]["matmul_precision"], "highest")

    @pytest.mark.fast
    def test_a_swallowed_setup_error_reaches_disk(self) -> None:
        """The one setup error left, and it still reaches disk.

        This used to drive the case through `torch_num_threads:
        "not-an-int"`, because `configure_runtime` caught every exception
        from every performance setting and returned the text. It no longer
        does: validate_config refuses that value, and a setting that cannot
        be applied stops the run rather than being recorded and skipped
        (P04-F07, tests/test_runtime_setup_applies_or_refuses.py). What is
        still caught and recorded is the torch import, which is optional.
        """

        import builtins

        config = load_config("configs/dev/smoke.yaml")
        from fedbrew.core.runtime_setup import configure_runtime

        real_import = builtins.__import__

        def refuse_torch(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", refuse_torch):
            info = configure_runtime(config, deterministic=False)
        self.assertIn("runtime_setup_error", info)

        with tempfile.TemporaryDirectory() as directory:
            save_run_json(
                [],
                Path(directory),
                config,
                run_metadata={"run_id": "r1", "runtime": dict(info)},
            )
            run = json.loads((Path(directory) / "run.json").read_text(encoding="utf-8"))
        self.assertIn("runtime_setup_error", run["reproducibility"]["runtime"])

    @pytest.mark.fast
    def test_an_absent_record_is_omitted_rather_than_written_as_null(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            save_run_json(
                [],
                Path(directory),
                load_config("configs/dev/smoke.yaml"),
                run_metadata={"run_id": "r1"},
            )
            run = json.loads((Path(directory) / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(run["reproducibility"], {})

    def test_the_config_section_still_comes_last(self) -> None:
        """Section order is the organisation; provenance sits before config."""

        with tempfile.TemporaryDirectory() as directory:
            run = self._run(Path(directory))
        keys = list(run)
        self.assertEqual(keys[-1], "config")
        self.assertEqual(keys[-2], "reproducibility")


if __name__ == "__main__":
    unittest.main()
