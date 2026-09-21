"""A refusal reaches the reader as a reason and an exit status, not a traceback.

`fedbrew run` refuses plenty -- a config value out of range, a checkpoint it
will not resume from, a second seed in one directory -- and nothing between the
console script and `run()` caught anything. So a refusal arrived as a raw Python
traceback whose last line happened to be the reason, except for the one refusal
that raised `SystemExit` itself and so left by a different mechanism from all
the others. Measured through the console script before the fix: of thirteen
refusals on the resume path, twelve printed a traceback of 16 to 30 lines, and a
config refused by validation did too.

`RunRefused` is the type a refusal raises, and `runner.main` is the one place
that catches it, prints it and exits `EXIT_REFUSED`. It is a `ValueError`, so
Python callers of `run()` and every test asserting `ValueError` are unaffected,
and it is caught by exact type, so a genuine bug still arrives with its
traceback: `ItIsNarrowTest`.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from collections.abc import Callable, Mapping
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch
import yaml

from fedbrew.clients.lazy_pool import LazyClientPool
from fedbrew.core import runner
from fedbrew.core.checkpointing import refuse_a_reconfigured_resume
from fedbrew.core.config import ClientStatisticsConfig
from fedbrew.core.loop import (
    _aggregate_client_split_metrics,
    _central_test_metrics,
    _evaluate_central_test_set,
    _require_positive_global_rounds,
)
from fedbrew.core.protocol import EvalResult
from fedbrew.core.refusal import RunRefused
from fedbrew.data.centralized_dataset import CentralizedFederatedDataset
from fedbrew.data.llm_assets.manifest import AssetManifestError, load_asset_manifest
from fedbrew.data.manifest_dataset import ManifestFederatedDataset
from fedbrew.data.writers.manifest import save_clients_jsonl, save_manifest
from fedbrew.data.writers.torch_shards import load_client_shard, save_split_client_shard
from fedbrew.servers.fedavg import FedAvgServer
from tests.test_resume_is_all_or_nothing import CONFIG

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Modules whose every raise refuses what the run was given: config loading and
#: validation, the factory, the runner's flags and resume preconditions,
#: checkpoint policy, the extension loader, the model registry, the loop, the LLM
#: trace, and the dataset and prepared-asset files a run reads. A `ValueError`
#: raised here, under any class name, is a refusal that would print a traceback,
#: unless `NOT_REFUSALS` names it. `runner.main` may raise `SystemExit`, which is
#: how it exits; nothing else in these modules may.
#:
#: The list covers only the families converted so far, not the whole tree. The
#: client, server, model and task code raise from inside a running experiment
#: and keep their tracebacks by decision, so nothing here scans them; FINDINGS.md,
#: under *What is left, and why*, records what remains and why.
REFUSING_MODULES = (
    "fedbrew/core/checkpointing.py",
    "fedbrew/core/config.py",
    "fedbrew/core/extensions.py",
    "fedbrew/core/factory.py",
    "fedbrew/core/loop.py",
    "fedbrew/core/registry.py",
    "fedbrew/core/run_metadata.py",
    "fedbrew/core/runner.py",
    "fedbrew/data/centralized_dataset.py",
    "fedbrew/data/llm_assets/manifest.py",
    "fedbrew/data/manifest_dataset.py",
    "fedbrew/data/writers/torch_shards.py",
)

#: Functions in `REFUSING_MODULES` whose raises are not refusals, as (module,
#: function). Each flags a bug in the Python code calling it rather than input a
#: run was given, which is what a traceback is for, so it keeps `ValueError`.
NOT_REFUSALS = frozenset(
    {
        ("fedbrew/core/runner.py", "default_override_namespace"),
        # Python code calling run_fl_loop: argparse and config refuse a round
        # count below one before a run gets this far.
        ("fedbrew/core/loop.py", "_require_positive_global_rounds"),
        # What a strategy's evaluate_global and a client's evaluation report is
        # a contract with the code that wrote them, not input a run was given.
        ("fedbrew/core/loop.py", "_central_test_metrics"),
        ("fedbrew/core/loop.py", "_validate_client_evaluation"),
        # Registering a component is Python code calling the registry, so a
        # malformed name, key set or generator spec is a bug in that code.
        ("fedbrew/core/registry.py", "__post_init__"),
        ("fedbrew/core/registry.py", "_config_key_set"),
        ("fedbrew/core/registry.py", "_frozen_key_set"),
        ("fedbrew/core/registry.py", "registering_from"),
        # Also the duplicate-name refusal, which is both: two configured
        # extensions claiming one name is input, while a built-in or one extension
        # registering a name twice is a bug. The message names both origins, so
        # either reader can act on it. Deferred, not done: RunRefused when both
        # origins are extensions, ValueError otherwise.
        ("fedbrew/core/registry.py", "register"),
        # Raised into its own `except`, which records it as manifest_error; it
        # never leaves the function.
        ("fedbrew/core/run_metadata.py", "build_dataset_provenance"),
        # The loop, its one caller on the run path, asks for the default split.
        ("fedbrew/data/manifest_dataset.py", "get_global_data"),
        # A generator handing over half a test split; `fedbrew run` writes no shard.
        ("fedbrew/data/writers/torch_shards.py", "save_split_client_shard"),
    }
)


def _write(out: Path, rounds: int, *swaps: tuple[str, str]) -> Path:
    text = textwrap.dedent(CONFIG.format(out=out, rounds=rounds)).lstrip()
    for old, new in swaps:
        assert old in text, old
        text = text.replace(old, new)
    path = out.parent / f"{out.name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _resolve(target: ast.expr, namespace: Mapping[str, Any]) -> Any:
    """The object a raise names, looked up the way its module would look it up."""

    if isinstance(target, ast.Name):
        return (
            namespace[target.id] if target.id in namespace else getattr(builtins, target.id, None)
        )
    if isinstance(target, ast.Attribute):
        owner = _resolve(target.value, namespace)
        return None if owner is None else getattr(owner, target.attr, None)
    return None


def _bare_errors(name: str, tree: ast.Module, namespace: Mapping[str, Any]) -> list[str]:
    """Every raise in one module that would leave `fedbrew run` as a traceback.

    Each raised name is resolved to its class rather than matched as a word:
    `AssetManifestError` subclassed `ValueError` under a name no list of words
    held, so a scan of names could not have seen it. A raise the scan cannot
    resolve is reported too, because passing it would be passing unread code.
    A bare `raise error` of a caught instance names no class and is skipped.
    """

    offenders = []
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if (name, function.name) in NOT_REFUSALS:
            continue
        for node in ast.walk(function):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            call = isinstance(node.exc, ast.Call)
            raised = _resolve(node.exc.func if call else node.exc, namespace)
            offence = _offence(raised, call, function.name)
            if offence is not None:
                offenders.append(f"{name}:{node.lineno} {function.name} {offence}")
    return offenders


def _offence(raised: Any, call: bool, function_name: str) -> str | None:
    """What is wrong with one raise, if anything: the rule `_bare_errors` applies."""

    if not isinstance(raised, type):
        return "raises a name the scan cannot resolve" if call else None
    if issubclass(raised, SystemExit):
        return None if function_name == "main" else f"raises {raised.__name__}"
    if issubclass(raised, FileNotFoundError) or (
        issubclass(raised, ValueError) and not issubclass(raised, RunRefused)
    ):
        return f"raises {raised.__name__}"
    return None


def _manifest_dataset(directory: Path) -> Path:
    """The smallest dataset `ManifestFederatedDataset` opens: a manifest, a roster, two shards."""

    rows = []
    for client_id in ("client_0", "client_1"):
        save_split_client_shard(
            directory / "shards" / f"{client_id}.pt",
            torch.zeros(4, 8),
            torch.zeros(4, dtype=torch.long),
            torch.zeros(2, 8),
            torch.zeros(2, dtype=torch.long),
        )
        rows.append({"client_id": client_id, "shard": f"shards/{client_id}.pt"})
    save_clients_jsonl(directory, rows)
    return save_manifest(directory, {"clients_file": "clients.jsonl"})


class _SourceClients:
    """A source dataset for the centralized view, holding whatever payloads a test hands it."""

    def __init__(self, payloads: dict[str, Any]) -> None:
        self.payloads = payloads

    def list_clients(self) -> list[str]:
        return list(self.payloads)

    def get_client_data(self, client_id: str) -> Any:
        return self.payloads[client_id]


def _refused_through_main(test: unittest.TestCase, *argv: str) -> str:
    """Run `runner.main` quietly, require the refusal exit and banner, return stderr.

    Joined on whitespace, because the banner wraps a long reason across lines.
    """

    stderr = StringIO()
    with (
        redirect_stderr(stderr),
        redirect_stdout(StringIO()),
        test.assertRaises(SystemExit) as caught,
    ):
        runner.main(["--quiet", *argv])
    test.assertEqual(caught.exception.code, runner.EXIT_REFUSED)
    test.assertIn("RUN REFUSED", stderr.getvalue())
    return " ".join(stderr.getvalue().split())


class TheResumePathRefusesInWordsTest(unittest.TestCase):
    """Every refusal the investigation triggered, through `runner.main`.

    Each passes `--quiet`, because a refusal is the one thing `--quiet` must not
    hide: the reader who silenced the round output still has to learn why
    nothing ran.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls._directory.name)
        cls.base = cls.root / "base"
        runner.run(_write(cls.base, 3), runner.parse_args(["--quiet"]))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def _copy(self, name: str) -> Path:
        directory = self.root / name
        shutil.copytree(self.base, directory)
        return directory

    def _doctored(self, name: str, edit: Callable[[dict[str, Any]], Any]) -> tuple[Path, Path]:
        directory = self._copy(name)
        checkpoint = torch.load(directory / "checkpoints" / "latest.pt", weights_only=False)
        target = directory / "checkpoints" / "doctored.pt"
        torch.save(edit(checkpoint), target)
        return directory, target

    def _refused(self, *argv: str) -> str:
        stderr = StringIO()
        with (
            redirect_stderr(stderr),
            redirect_stdout(StringIO()),
            self.assertRaises(SystemExit) as caught,
        ):
            runner.main(["--quiet", *argv])
        self.assertEqual(caught.exception.code, runner.EXIT_REFUSED)
        text = stderr.getvalue()
        self.assertIn("RUN REFUSED", text)
        return text

    def _resume(self, directory: Path, checkpoint: Path, *swaps: tuple[str, str]) -> str:
        return self._refused(
            "--config", str(_write(directory, 6, *swaps)), "--resume-from", str(checkpoint)
        )

    def test_scaffold_from_best_pt(self) -> None:
        directory = self._copy("best")
        text = self._resume(directory, directory / "checkpoints" / "best.pt")
        self.assertIn("no client_states", text)

    def test_resume_latest_with_no_checkpoint(self) -> None:
        text = self._refused("--config", str(_write(self.root / "empty", 3)), "--resume-latest")
        self.assertIn("No checkpoint found", text)

    def test_both_resume_flags(self) -> None:
        directory = self._copy("both_flags")
        text = self._refused(
            "--config",
            str(_write(directory, 6)),
            "--resume-from",
            str(directory / "checkpoints" / "latest.pt"),
            "--resume-latest",
        )
        self.assertIn("cannot be used together", text)

    def test_a_checkpoint_path_that_does_not_exist(self) -> None:
        directory = self._copy("absent")
        text = self._resume(directory, directory / "checkpoints" / "absent.pt")
        self.assertIn("absent.pt", text)

    def test_a_checkpoint_that_is_not_a_dict(self) -> None:
        directory = self._copy("not_a_dict")
        torch.save([1, 2], directory / "checkpoints" / "list.pt")
        text = self._resume(directory, directory / "checkpoints" / "list.pt")
        self.assertIn("must contain a dictionary", text)

    def test_a_round_id_that_is_not_positive(self) -> None:
        directory, checkpoint = self._doctored("round_id", lambda c: {**c, "round_id": 0})
        self.assertIn("round_id must be a positive int", self._resume(directory, checkpoint))

    def test_a_checkpoint_without_a_model(self) -> None:
        def without_model(checkpoint: dict[str, Any]) -> dict[str, Any]:
            checkpoint.pop("model_state", None)
            checkpoint["server_state"].pop("model_state", None)
            return checkpoint

        directory, checkpoint = self._doctored("no_model", without_model)
        self.assertIn("must contain a model_state", self._resume(directory, checkpoint))

    def test_a_server_state_that_is_not_a_mapping(self) -> None:
        directory, checkpoint = self._doctored("server_state", lambda c: {**c, "server_state": [1]})
        self.assertIn("server_state must be a mapping", self._resume(directory, checkpoint))

    def test_client_states_that_are_not_a_mapping(self) -> None:
        directory, checkpoint = self._doctored(
            "client_states", lambda c: {**c, "client_states": [1]}
        )
        self.assertIn("client_states must be a mapping", self._resume(directory, checkpoint))

    def test_one_client_state_that_is_not_a_mapping(self) -> None:
        def one_bad_state(checkpoint: dict[str, Any]) -> dict[str, Any]:
            return {
                **checkpoint,
                "client_states": {cid: [1] for cid in checkpoint["client_states"]},
            }

        directory, checkpoint = self._doctored("client_state", one_bad_state)
        self.assertIn("client state must be a mapping", self._resume(directory, checkpoint))

    def test_a_changed_client_setting(self) -> None:
        directory = self._copy("client_setting")
        swap = ("learning_rate: 0.05", "learning_rate: 0.1")
        text = self._resume(directory, directory / "checkpoints" / "latest.pt", swap)
        self.assertIn("learning_rate", text)

    def test_a_changed_server_setting(self) -> None:
        directory = self._copy("server_setting")
        swap = ("participation_rate: 1", "participation_rate: 1\n  aggregation_weighting: uniform")
        text = self._resume(directory, directory / "checkpoints" / "latest.pt", swap)
        self.assertIn("aggregation_weighting", text)

    def test_a_foreign_seed(self) -> None:
        directory = self._copy("seed")
        text = self._resume(
            directory, directory / "checkpoints" / "latest.pt", ("seed: 42", "seed: 7")
        )
        self.assertIn("refusing to run seed 7", text)

    def test_a_config_refused_by_validation(self) -> None:
        """No resume at all: the surface a reader meets first."""

        swap = ("participation_rate: 1", "participation_rate: 2")
        text = self._refused("--config", str(_write(self.root / "invalid", 3, swap)))
        self.assertIn("participation_rate must be in (0, 1]", text)


class TheConsoleScriptTest(unittest.TestCase):
    def test_a_refusal_through_the_console_script_prints_no_traceback(self) -> None:
        """The real entry point in a real process: what a SLURM log receives."""

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            runner.run(_write(out, 3), runner.parse_args(["--quiet"]))
            environment = dict(os.environ)
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(None, [str(REPO_ROOT), environment.get("PYTHONPATH")])
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "fedbrew.cli.dispatch",
                    "run",
                    "--quiet",
                    "--config",
                    str(_write(out, 6)),
                    "--resume-from",
                    str(out / "checkpoints" / "best.pt"),
                ],
                cwd=directory,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, runner.EXIT_REFUSED, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertIn("RUN REFUSED", result.stderr)
        self.assertIn("no client_states", result.stderr)


class ItIsNarrowTest(unittest.TestCase):
    """Catching more than `RunRefused` would hide the defects a traceback is for."""

    @pytest.mark.fast
    def test_a_defect_keeps_its_traceback(self) -> None:
        for error in (ValueError("a defect, not a refusal"), RuntimeError("a defect")):
            with self.subTest(error=type(error).__name__):
                with (
                    mock.patch.object(runner, "run", side_effect=error),
                    self.assertRaises(type(error)),
                ):
                    runner.main(["--quiet", "--config", "configs/dev/smoke.yaml"])

    @pytest.mark.fast
    def test_run_raises_a_refusal_rather_than_exiting(self) -> None:
        """Only the console entry point turns a refusal into an exit status."""

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            path = _write(out, 3, ("participation_rate: 1", "participation_rate: 2"))
            with self.assertRaises(RunRefused):
                runner.run(path, runner.parse_args(["--quiet"]))

    @pytest.mark.fast
    def test_a_refusal_is_still_a_value_error(self) -> None:
        self.assertTrue(issubclass(RunRefused, ValueError))

    @pytest.mark.fast
    def test_a_callers_bug_is_not_a_refusal(self) -> None:
        """A misnamed override is a bug in the script calling `run`, not input it refuses."""

        with self.assertRaises(ValueError) as caught:
            runner.default_override_namespace(no_such_flag=1)
        self.assertNotIsInstance(caught.exception, RunRefused)

    def test_a_callers_bug_in_the_dataset_code_is_not_a_refusal(self) -> None:
        """The two dataset exemptions: arguments Python code passed, not files a run read."""

        with tempfile.TemporaryDirectory() as directory:
            manifest = _manifest_dataset(Path(directory))
            cases: tuple[tuple[str, Callable[[], Any]], ...] = (
                (
                    "get_global_data",
                    lambda: ManifestFederatedDataset(manifest).get_global_data("eval"),
                ),
                (
                    "save_split_client_shard",
                    lambda: save_split_client_shard(
                        Path(directory) / "half.pt", 1, 1, 1, 1, test_x=1
                    ),
                ),
            )
            for name, case in cases:
                with self.subTest(exemption=name):
                    with self.assertRaises(ValueError) as caught:
                        case()
                    self.assertNotIsInstance(caught.exception, RunRefused)

    @pytest.mark.fast
    def test_a_callers_bug_in_the_loop_is_not_a_refusal(self) -> None:
        """The loop's exemptions: a round count from Python code, a strategy off its contract."""

        cases: tuple[tuple[str, Callable[[], Any]], ...] = (
            ("rounds below one", lambda: _require_positive_global_rounds(0)),
            ("a result that is not a mapping", lambda: _central_test_metrics([1.0])),
            ("a result with no numeric loss", lambda: _central_test_metrics({"accuracy": 0.5})),
        )
        for name, case in cases:
            with self.subTest(case=name):
                with self.assertRaises(ValueError) as caught:
                    case()
                self.assertNotIsInstance(caught.exception, RunRefused)


@pytest.mark.fast
class EveryRefusingModuleRaisesARefusalTest(unittest.TestCase):
    """Derived from the modules rather than listed per site.

    A validator added to `config.py` tomorrow that raises `ValueError` would
    print a traceback again, and a case list would not notice. This reads every
    raise in `REFUSING_MODULES` instead.
    """

    def test_no_refusing_module_raises_a_bare_error(self) -> None:
        offenders = []
        for name in REFUSING_MODULES:
            module = importlib.import_module(name.removesuffix(".py").replace("/", "."))
            tree = ast.parse((REPO_ROOT / name).read_text(encoding="utf-8"))
            offenders += _bare_errors(name, tree, vars(module))
        self.assertEqual(offenders, [])

    def test_the_scan_reads_a_type_not_a_name(self) -> None:
        """A `ValueError` under another name is still one, and a refusal under any name is not."""

        tree = ast.parse("def check():\n    raise Renamed('a refusal under another name')\n")
        as_value_error = {"Renamed": type("Renamed", (ValueError,), {})}
        as_refusal = {"Renamed": type("Renamed", (RunRefused,), {})}
        self.assertEqual(len(_bare_errors("module.py", tree, as_value_error)), 1)
        self.assertEqual(_bare_errors("module.py", tree, as_refusal), [])

        unresolved = ast.parse("def check():\n    raise make_an_error()\n")
        self.assertEqual(len(_bare_errors("module.py", unresolved, {})), 1)

        # Neither rule below fires on the tree today, so nothing else would
        # notice either one switched off.
        missing = ast.parse("def check():\n    raise FileNotFoundError('a file')\n")
        self.assertEqual(len(_bare_errors("module.py", missing, {})), 1)
        exits = ast.parse(
            "def main():\n    raise SystemExit(2)\n\ndef check():\n    raise SystemExit(1)\n"
        )
        self.assertEqual(len(_bare_errors("module.py", exits, {})), 1)

    def test_the_scan_sees_the_refusals_it_guards(self) -> None:
        """A scan that found no raises at all would pass while checking nothing."""

        source = (REPO_ROOT / "fedbrew/core/config.py").read_text(encoding="utf-8")
        self.assertGreater(source.count("raise RunRefused("), 50)

    def test_every_exemption_names_a_function_that_still_raises(self) -> None:
        """An exemption left behind by a rename or a deleted raise would exempt nothing
        visible today and whatever lands under that name tomorrow."""

        for name, function_name in sorted(NOT_REFUSALS):
            with self.subTest(module=name, function=function_name):
                tree = ast.parse((REPO_ROOT / name).read_text(encoding="utf-8"))
                functions = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                    and node.name == function_name
                ]
                # One or more: registry.py defines `register` on three classes.
                self.assertNotEqual(functions, [])
                self.assertTrue(
                    any(
                        isinstance(node, ast.Raise)
                        for function in functions
                        for node in ast.walk(function)
                    )
                )


@pytest.mark.fast
class RefusalsAwayFromTheConsoleTest(unittest.TestCase):
    """The refusals `TheResumePathRefusesInWordsTest` cannot reach on its fixture."""

    def test_a_lazily_restored_client_state_is_a_refusal(self) -> None:
        """A manifest dataset's pool defers a client's state until it is built."""

        pool = LazyClientPool(["a"], lambda client_id: None)
        with self.assertRaises(RunRefused):
            pool.load_state_snapshot({"a": [1]})

    def test_a_changed_setting_on_a_lazily_built_client_is_a_refusal(self) -> None:
        """What that deferred client raises when it is built mid-round."""

        with self.assertRaises(RunRefused):
            refuse_a_reconfigured_resume("x client", {"a": 1}, {"a": 2})

    def test_an_adapter_checkpoint_without_metadata_is_a_refusal(self) -> None:
        server = FedAvgServer(participation_rate=1.0, seed=0)
        with self.assertRaises(RunRefused):
            server.load_state(
                {"model_state": {"w": torch.zeros(2)}, "model_state_scope": "adapter"}
            )


class ADatasetARunCannotReadIsARefusalTest(unittest.TestCase):
    """The dataset and prepared-asset files a run reads, refused in words.

    Measured through `fedbrew run` before this: a truncated manifest, a missing
    `data.path` and a deleted shard each printed 31 to 43 lines of traceback, and
    none of the three came from a raise in fedbrew -- they came from `json.loads`,
    `read_text` and `torch.load`. So they are refused at the reader, where a file
    cannot change between a check and the read.
    """

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.manifest = _manifest_dataset(self.root / "data")
        self.shard = self.manifest.parent / "shards" / "client_0.pt"

    def _refused_by_main(self, manifest: Path) -> str:
        synthetic = "  num_clients: 4\n  samples_per_client: 16\n  input_dim: 8\n  num_classes: 4"
        config = _write(self.root / "run", 3, (synthetic, f"  path: {manifest}"))
        stderr = StringIO()
        with (
            redirect_stderr(stderr),
            redirect_stdout(StringIO()),
            self.assertRaises(SystemExit) as caught,
        ):
            runner.main(["--quiet", "--config", str(config)])
        self.assertEqual(caught.exception.code, runner.EXIT_REFUSED)
        self.assertIn("RUN REFUSED", stderr.getvalue())
        # Joined, because the banner wraps a long reason across lines.
        return " ".join(stderr.getvalue().split())

    def test_a_data_path_that_does_not_exist(self) -> None:
        text = self._refused_by_main(self.root / "absent" / "manifest.json")
        self.assertIn("cannot read the dataset manifest", text)

    def test_a_truncated_manifest(self) -> None:
        text = self.manifest.read_text(encoding="utf-8")
        self.manifest.write_text(text[: len(text) // 2], encoding="utf-8")
        self.assertIn("is not valid JSON", self._refused_by_main(self.manifest))

    def test_a_manifest_that_is_not_an_object(self) -> None:
        self.manifest.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(RunRefused, "Expected JSON object"):
            ManifestFederatedDataset(self.manifest)

    def test_a_roster_that_repeats_a_client(self) -> None:
        roster = self.manifest.parent / "clients.jsonl"
        first = roster.read_text(encoding="utf-8").splitlines()[0]
        with roster.open("a", encoding="utf-8") as handle:
            handle.write(first + "\n")
        with self.assertRaisesRegex(RunRefused, "repeats 1 client_id"):
            ManifestFederatedDataset(self.manifest)

    def test_a_manifest_that_names_no_roster(self) -> None:
        """Through `runner.main`: the key the roster reader looks up first."""

        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        del manifest["clients_file"]
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertIn("names no clients_file", self._refused_by_main(self.manifest))

    def test_a_missing_roster(self) -> None:
        roster = self.manifest.parent / "clients.jsonl"
        roster.unlink()
        with self.assertRaises(RunRefused) as caught:
            ManifestFederatedDataset(self.manifest)
        self.assertIn(f"cannot read the client roster {roster}", str(caught.exception))

    def test_a_truncated_roster(self) -> None:
        roster = self.manifest.parent / "clients.jsonl"
        roster.write_text(roster.read_text(encoding="utf-8")[:-10], encoding="utf-8")
        with self.assertRaisesRegex(RunRefused, "line 2 is not valid JSON"):
            ManifestFederatedDataset(self.manifest)

    def test_a_roster_row_the_dataset_cannot_use(self) -> None:
        roster = self.manifest.parent / "clients.jsonl"
        rows = [json.loads(line) for line in roster.read_text(encoding="utf-8").splitlines()]
        for damage, first, expected in (
            ("not an object", [1, 2], "line 1 is not a JSON object"),
            ("no client_id", {"shard": rows[0]["shard"]}, "line 1 has no client_id"),
            ("no shard", {"client_id": rows[0]["client_id"]}, "line 1 has no shard"),
        ):
            with self.subTest(damage=damage):
                lines = [json.dumps(first), *(json.dumps(row) for row in rows[1:])]
                roster.write_text("\n".join(lines) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(RunRefused, expected):
                    ManifestFederatedDataset(self.manifest)

    def test_a_deleted_shard(self) -> None:
        dataset = ManifestFederatedDataset(self.manifest, shard_cache_bytes=0)
        self.shard.unlink()
        with self.assertRaises(RunRefused) as caught:
            dataset.get_client_data("client_0")
        self.assertIn(str(self.shard), str(caught.exception))

    def test_a_shard_that_is_not_a_torch_file(self) -> None:
        whole = self.shard.read_bytes()
        for damage, contents in (
            ("truncated", whole[: len(whole) // 2]),
            ("empty", b""),
            ("not torch", b"not a torch file"),
        ):
            with self.subTest(damage=damage):
                self.shard.write_bytes(contents)
                with self.assertRaises(RunRefused) as caught:
                    load_client_shard(self.shard)
                self.assertIn(str(self.shard), str(caught.exception))

    def test_a_shard_that_is_not_a_dict(self) -> None:
        torch.save([1], self.shard)
        with self.assertRaisesRegex(RunRefused, "must contain a dictionary"):
            load_client_shard(self.shard)

    def test_centralized_training_over_no_clients(self) -> None:
        with self.assertRaisesRegex(RunRefused, "at least one source client"):
            CentralizedFederatedDataset(_SourceClients({}))

    def test_centralized_training_over_no_train_examples(self) -> None:
        empty = {"train": {"x": torch.zeros(0, 2), "y": torch.zeros(0)}}
        dataset = CentralizedFederatedDataset(_SourceClients({"a": empty}))
        with self.assertRaisesRegex(RunRefused, "no pooled train examples"):
            dataset.get_client_data(dataset.list_clients()[0])

    def test_centralized_training_over_a_split_without_features(self) -> None:
        dataset = CentralizedFederatedDataset(
            _SourceClients({"a": {"train": {"y": torch.zeros(3)}}})
        )
        with self.assertRaisesRegex(RunRefused, "missing x/y tensors"):
            dataset.get_client_data(dataset.list_clients()[0])

    def test_a_missing_prepared_asset_manifest(self) -> None:
        self.assertTrue(issubclass(AssetManifestError, RunRefused))
        with self.assertRaises(RunRefused):
            load_asset_manifest(self.root / "absent" / "asset_manifest.json")


@pytest.mark.fast
class TheRestOfCoreRefusesInWordsTest(unittest.TestCase):
    """What fedbrew/core refuses beyond config, the factory, checkpoints and the runner.

    Each was measured through `fedbrew run` before it became a refusal, and each
    printed a traceback and exited 1: an unknown model.name, four extensions
    entries that cannot be loaded, central and split evaluation that a strategy
    or a dataset cannot serve, and the eight checks the LLM trace makes of its
    model config and its data manifest.
    """

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def _refused(self, *swaps: tuple[str, str]) -> str:
        config = _write(self.root / "run", 3, *swaps)
        return _refused_through_main(self, "--config", str(config))

    def test_an_unknown_model(self) -> None:
        text = self._refused(("  name: mlp", "  name: no_such_model"))
        self.assertIn("Unknown model: no_such_model", text)

    def test_an_extensions_entry_that_cannot_be_loaded(self) -> None:
        no_register = self.root / "no_register.py"
        no_register.write_text("X = 1\n", encoding="utf-8")
        for entry, expected in (
            (str(self.root / "absent.py"), "does not exist"),
            ("not a module!", "is neither a path ending in .py nor a dotted module name"),
            (str(no_register), "defines no register() function"),
            ("no_such_module_anywhere", "names a module that cannot be imported"),
        ):
            with self.subTest(entry=entry):
                swap = ("  seed: 42", f"  seed: 42\n  extensions: [{json.dumps(entry)}]")
                self.assertIn(expected, self._refused(swap))

    def test_central_evaluation_a_strategy_cannot_run(self) -> None:
        with self.assertRaisesRegex(RunRefused, "has no evaluate_global"):
            _evaluate_central_test_set(object(), _SourceClients({}))

    def test_central_evaluation_without_global_test_data(self) -> None:
        class Strategy:
            def evaluate_global(self, global_data: Any) -> dict[str, float]:
                return {"loss": 1.0}

        class NoGlobalTest:
            def get_global_data(self) -> None:
                return None

        class GlobalTestLookupFails:
            def get_global_data(self) -> Any:
                raise KeyError("global_test")

        class WithGlobalTest:
            def get_global_data(self) -> Any:
                return {"x": torch.zeros(1)}

        for dataset in (NoGlobalTest(), GlobalTestLookupFails()):
            with self.subTest(dataset=type(dataset).__name__):
                with self.assertRaisesRegex(RunRefused, "does not provide global_test data"):
                    _evaluate_central_test_set(Strategy(), dataset)
        self.assertEqual(
            _evaluate_central_test_set(Strategy(), WithGlobalTest()), {"central_test_loss": 1.0}
        )

    def test_an_evaluation_split_no_client_has(self) -> None:
        results = [
            EvalResult(
                round_id=1,
                client_id="client_0",
                num_examples=0,
                metrics={"val_loss": 1.0},
                payload={"num_examples_by_split": {"val": 0}},
            )
        ]
        with self.assertRaisesRegex(RunRefused, "no client reported a non-empty val split"):
            _aggregate_client_split_metrics(results, "val", ClientStatisticsConfig())

    def test_the_llm_trace_refuses_its_model_config_and_data_manifest(self) -> None:
        """Through `runner.main`, on a prepared-asset manifest with empty asset directories.

        The trace makes these checks before it loads a model or a dataset, so no
        model files are needed to reach them.
        """

        assets = self.root / "assets"
        (assets / "model").mkdir(parents=True)
        (assets / "tokenizer").mkdir()
        asset_manifest = assets / "asset_manifest.json"
        asset_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "model_identifier": "placeholder/model",
                    "tokenizer_identifier": "placeholder/tokenizer",
                    "requested_revision": "main",
                    "resolved_revision": None,
                    "cache_path": str(assets),
                    "model_path": "model",
                    "tokenizer_path": "tokenizer",
                    "vocabulary_size": 10,
                    "model_type": "gpt2",
                    "preparation_timestamp": "2026-09-15T00:00:00Z",
                    "trust_remote_code": False,
                }
            ),
            encoding="utf-8",
        )
        data_manifests = {}
        for name, payload in (("no_corpus_hash", {}), ("not_an_object", [])):
            data_manifests[name] = self.root / name / "manifest.json"
            data_manifests[name].parent.mkdir()
            data_manifests[name].write_text(json.dumps(payload), encoding="utf-8")

        base = yaml.safe_load((REPO_ROOT / "configs/oasst1/fedavg_base.yaml").read_text("utf-8"))
        base["experiment"]["output_dir"] = str(self.root / "out")
        base["model"]["asset_manifest"] = str(asset_manifest)
        base["data"] = {"path": str(data_manifests["no_corpus_hash"])}
        absent = str(self.root / "absent" / "manifest.json")
        cases: tuple[tuple[str, Callable[[dict[str, Any]], Any], str], ...] = (
            (
                "no asset_manifest",
                lambda c: c["model"].pop("asset_manifest"),
                "requires model.asset_manifest",
            ),
            (
                "preparation_config",
                lambda c: c["model"].update(preparation_config=5),
                "must be a path string",
            ),
            (
                "local_files_only",
                lambda c: c["model"].update(local_files_only=False),
                "requires model.local_files_only=true",
            ),
            (
                "trust_remote_code",
                lambda c: c["model"].update(trust_remote_code=True),
                "requires model.trust_remote_code=false",
            ),
            ("no corpus_hash", lambda c: None, "must contain corpus_hash"),
            (
                "not a manifest dataset",
                lambda c: c.update(data={"name": "synthetic_classification"}),
                "requires a generated manifest_dataset",
            ),
            (
                "data manifest missing",
                lambda c: c["data"].update(path=absent),
                "could not read generated LLM data manifest",
            ),
            (
                "data manifest not an object",
                lambda c: c["data"].update(path=str(data_manifests["not_an_object"])),
                "must be a JSON object",
            ),
        )
        for index, (label, edit, expected) in enumerate(cases):
            with self.subTest(case=label):
                config = json.loads(json.dumps(base))
                edit(config)
                path = self.root / f"llm_{index}.yaml"
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                self.assertIn(expected, _refused_through_main(self, "--config", str(path)))


if __name__ == "__main__":
    unittest.main()
