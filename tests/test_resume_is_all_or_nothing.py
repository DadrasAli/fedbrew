"""A resume restores the server's coupled state and the clients' or neither.

SCAFFOLD defines its server control variate as ``c = (1/N) sum_i c_i`` and
corrects every local step by ``c - c_i``. `_restore_server_state` puts ``c``
back from the checkpoint; `_restore_client_states` returns early when the
checkpoint has no ``client_states``, so every ``c_i`` stays at the zeros it is
allocated with. The correction becomes ``+c`` for every client, and the server
goes on updating from the restored ``c`` -- both sides then move by the same
per-round increments, so the gap is preserved exactly rather than decaying.
The run completes and reports success.

`best.pt` is such a checkpoint on purpose: `_without_client_states` strips
per-client state from it, which is worth 40 GB a file on FEMNIST's 3597
writers, and `--resume-from .../best.pt` is one thing to type. The audit filed
this fragile because nothing shipped does it; the edit that reaches it is a
user typing a path.

Measured before the fix, on the four-client SCAFFOLD run this module builds --
three rounds, checkpoint, three more:

    resumed from   ||c||     ||c - mean(c_i)||   final model
    latest.pt      0.363665            0.000000  --
    best.pt        0.675888            0.466774  7.4% of ||w|| away

0.466774 is exactly ``||c||`` at the checkpoint, which is the shape of the
defect: the clients account for none of it.

The fix is a declaration and a refusal. `FedAvgServer.coupled_client_state`
names the save_state keys whose value only means anything alongside per-client
state from the same checkpoint; it is empty everywhere but SCAFFOLD, and
`ClassificationTest` sweeps the registry so a new strategy has to decide rather
than inherit silence. `_refuse_a_half_restored_resume` raises before any state
is touched.

The refusal is narrow on purpose, and `ItIsNarrowTest` pins each edge: a
strategy with no coupling, a checkpoint that has its client states, and a
checkpoint whose ``c`` is still zeros. Refusing a FedAvg resume from `best.pt`
would be the same defect pointing the other way.
"""

from __future__ import annotations

import importlib
import inspect
import re
import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any

import pytest
import torch

from fedbrew.core import runner
from fedbrew.core.checkpointing import load_checkpoint
from fedbrew.core.loop import (
    _initialize_or_resume,
    _refuse_a_half_restored_resume,
    _restore_client_states,
    _restore_server_state,
    _without_client_states,
)
from fedbrew.core.registry import register_builtin_components, server_strategies
from fedbrew.servers.base import ServerStrategy
from fedbrew.servers.fedavg import FedAvgServer
from fedbrew.servers.scaffold import ScaffoldServer

#: The one strategy class whose server state is defined in terms of its
#: clients'. Every other shipped class declares nothing, which is the answer
#: this test wants recorded rather than assumed.
COUPLED_STRATEGIES = {"ScaffoldServer": {"server_control": "client_control"}}

#: Written out so that a strategy added, renamed or removed moves this list
#: rather than slipping through a sweep that silently found one class fewer.
SHIPPED_STRATEGY_CLASSES = [
    "FedAvgServer",
    "FedLALRServer",
    "FedOptServer",
    "ScaffoldServer",
]

REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIG = """
experiment:
  seed: 42
  output_dir: {out}
server:
  strategy: scaffold
  participation_rate: 1
  metrics: [fit_loss]
client:
  update_rule: scaffold
  batch_size: 4
  learning_rate: 0.05
  metrics: [fit_loss, val_loss]
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
  checkpointing:
    enabled: true
    interval: 1
    save_last: true
    save_best: true
    best_metric: val_loss_avg
    keep_last: 0
evaluation:
  train:
    every: 1
    clients: all
  val:
    every: 1
    clients: all
defaults:
  global_rounds: {rounds}
  local_iterations: 1
"""


def _write_config(directory: Path, out: Path, rounds: int) -> Path:
    path = directory / "scaffold.yaml"
    path.write_text(textwrap.dedent(CONFIG.format(out=out, rounds=rounds)).lstrip())
    return path


def _norm(state: dict[str, Any]) -> float:
    return sum(float((value.double() ** 2).sum()) for value in state.values()) ** 0.5


def _residual(server_control: dict[str, Any], client_states: dict[str, Any]) -> float:
    """``||c - (1/N) sum_i c_i||``, the quantity the algorithm defines as zero."""

    mean = {
        key: torch.stack(
            [client_states[cid]["client_control"][key].double() for cid in client_states]
        ).mean(0)
        for key in server_control
    }
    return _norm({key: server_control[key].double() - mean[key] for key in server_control})


def _checkpoint(server_control: Any, client_states: Any = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "round_id": 3,
        "model_state": {"w": torch.zeros(4)},
        "server_state": {"server_control": server_control},
    }
    if client_states is not None:
        payload["client_states"] = client_states
    return payload


def _scaffold_server() -> ScaffoldServer:
    server = ScaffoldServer(participation_rate=1.0, seed=0)
    server._model_state = {"w": torch.zeros(4)}
    return server


def _strategy_classes() -> dict[str, type]:
    """Every ServerStrategy subclass defined in `fedbrew/servers/`, by name.

    The registry holds builder functions rather than classes, so it cannot
    answer a question about a class attribute; walking the subclass tree can,
    and it covers a class registered under several names -- `centralized` and
    `fedavg` share one builder -- exactly once.

    Filtered to the package, because `__subclasses__` is global: run under the
    whole suite it also returns the eight test doubles other modules subclass
    `FedAvgServer` with, which are not shipped strategies and have no coupling
    to declare. Filtering on `__module__` rather than on a name prefix keeps
    that a fact about where a class lives.
    """

    for module in sorted(path.stem for path in (REPO_ROOT / "fedbrew" / "servers").glob("*.py")):
        importlib.import_module(f"fedbrew.servers.{module}")

    found: dict[str, type] = {}

    def walk(cls: type) -> None:
        for subclass in cls.__subclasses__():
            if subclass.__module__.startswith("fedbrew.servers."):
                found[subclass.__name__] = subclass
            walk(subclass)

    walk(ServerStrategy)
    return found


def _classes_a_builder_reaches(builder: Any, known: set[str], depth: int = 3) -> set[str]:
    """Strategy class names a builder names, following builder-to-builder calls."""

    source = inspect.getsource(builder)
    reached = {name for name in known if f"{name}(" in source}
    if reached or depth <= 0:
        return reached

    module = importlib.import_module(builder.__module__)
    for called in re.findall(r"\b(_build_\w+)\(", source):
        delegate = getattr(module, called, None)
        if delegate is not None and delegate is not builder:
            reached |= _classes_a_builder_reaches(delegate, known, depth - 1)
    return reached


@pytest.mark.fast
class ClassificationTest(unittest.TestCase):
    def test_every_shipped_strategy_declares_its_coupling(self) -> None:
        """So a new strategy decides rather than inheriting an empty default."""

        classes = _strategy_classes()
        self.assertEqual(sorted(classes), SHIPPED_STRATEGY_CLASSES)
        for name, cls in sorted(classes.items()):
            with self.subTest(strategy=name):
                self.assertEqual(dict(cls.coupled_client_state), COUPLED_STRATEGIES.get(name, {}))

    def test_every_registered_strategy_builds_one_of_them(self) -> None:
        """Otherwise the sweep above could miss a strategy a run can select.

        Read off the builder's source rather than by constructing it: every
        builder takes a full config and a task, and what is being asked here
        is which class it names. Delegation is followed, because the four
        FedOpt aliases each set one default and hand off to
        `_build_fedopt_server`.
        """

        register_builtin_components()
        known = set(_strategy_classes())
        for name in server_strategies.builtin():
            with self.subTest(strategy=name):
                self.assertTrue(
                    _classes_a_builder_reaches(server_strategies.get(name), known),
                    f"{name}'s builder constructs no class the sweep above covers",
                )

    def test_the_base_default_is_empty(self) -> None:
        self.assertEqual(dict(FedAvgServer.coupled_client_state), {})


@pytest.mark.fast
class TheRefusalTest(unittest.TestCase):
    def test_a_stranded_server_control_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _refuse_a_half_restored_resume(_scaffold_server(), _checkpoint({"w": torch.ones(4)}))
        message = str(caught.exception)
        self.assertIn("server_control", message)
        self.assertIn("client_control", message)

    def test_it_names_the_checkpoint_to_use_instead(self) -> None:
        """A refusal a reader cannot act on is half a fix."""

        with self.assertRaises(ValueError) as caught:
            _refuse_a_half_restored_resume(_scaffold_server(), _checkpoint({"w": torch.ones(4)}))
        message = str(caught.exception)
        self.assertIn("latest.pt", message)
        self.assertIn("--resume-latest", message)

    def test_it_fires_before_any_state_is_restored(self) -> None:
        """The rejected path has to leave the server untouched."""

        server = _scaffold_server()
        server._model_state = {"w": torch.full((4,), 7.0)}
        with self.assertRaises(ValueError):
            _refuse_a_half_restored_resume(server, _checkpoint({"w": torch.ones(4)}))
        self.assertTrue(torch.equal(server._model_state["w"], torch.full((4,), 7.0)))
        self.assertIsNone(server._server_control)


@pytest.mark.fast
class ItIsNarrowTest(unittest.TestCase):
    """Overstating the refusal would be the same defect in the other direction."""

    def test_a_strategy_with_no_coupling_is_untouched(self) -> None:
        server = FedAvgServer(participation_rate=1.0, seed=0)
        _refuse_a_half_restored_resume(server, _checkpoint({"w": torch.ones(4)}))

    def test_a_checkpoint_carrying_its_client_states_is_fine(self) -> None:
        _refuse_a_half_restored_resume(
            _scaffold_server(),
            _checkpoint(
                {"w": torch.ones(4)},
                client_states={"0": {"client_control": {"w": torch.ones(4)}}},
            ),
        )

    def test_an_all_zero_control_variate_is_fine(self) -> None:
        """Round 1's ``c``. Zero on both sides satisfies the invariant."""

        _refuse_a_half_restored_resume(_scaffold_server(), _checkpoint({"w": torch.zeros(4)}))

    def test_an_absent_or_empty_control_variate_is_fine(self) -> None:
        for server_control in ({}, None):
            with self.subTest(server_control=server_control):
                _refuse_a_half_restored_resume(_scaffold_server(), _checkpoint(server_control))

    def test_a_checkpoint_with_no_server_state_is_left_to_the_next_check(self) -> None:
        """`_restore_server_state` is what refuses a checkpoint with no model."""

        _refuse_a_half_restored_resume(_scaffold_server(), {"round_id": 3})


@pytest.mark.fast
class BestCheckpointIsSuchACheckpointTest(unittest.TestCase):
    """The premise: without this, the refusal guards nothing reachable."""

    def test_best_pt_is_written_without_client_states(self) -> None:
        payload = _checkpoint(
            {"w": torch.ones(4)}, client_states={"0": {"client_control": {"w": torch.ones(4)}}}
        )
        self.assertNotIn("client_states", _without_client_states(payload))
        self.assertIn("server_control", _without_client_states(payload)["server_state"])


class EndToEndTest(unittest.TestCase):
    """A real run, its real checkpoints, and the number the defect produces."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        root = Path(cls._directory.name)
        cls.out = root / "run"
        runner.run(_write_config(root, cls.out, 3), runner.parse_args(["--quiet"]))
        cls.latest = load_checkpoint(cls.out / "checkpoints" / "latest.pt")
        cls.best = load_checkpoint(cls.out / "checkpoints" / "best.pt")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def test_the_run_that_wrote_them_satisfies_the_invariant(self) -> None:
        residual = _residual(
            self.latest["server_state"]["server_control"], self.latest["client_states"]
        )
        control_norm = _norm(self.latest["server_state"]["server_control"])
        self.assertGreater(control_norm, 0.1, "the control variate never moved; nothing is at risk")
        self.assertLess(residual, 1e-6 * max(control_norm, 1.0))

    def test_the_real_best_checkpoint_has_a_stranded_control_variate(self) -> None:
        self.assertNotIn("client_states", self.best)
        self.assertGreater(_norm(self.best["server_state"]["server_control"]), 0.1)

    def test_resuming_from_it_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _initialize_or_resume(
                _scaffold_server(), self.out / "checkpoints" / "best.pt", self.out
            )
        self.assertIn("no client_states", str(caught.exception))

    def test_resuming_from_latest_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            resumed = Path(directory) / "run"
            shutil.copytree(self.out, resumed)
            runner.run(
                _write_config(Path(directory), resumed, 6),
                runner.parse_args(
                    ["--quiet", "--resume-from", str(resumed / "checkpoints" / "latest.pt")]
                ),
            )
            final = load_checkpoint(resumed / "checkpoints" / "latest.pt")
        self.assertEqual(final["round_id"], 6)
        self.assertLess(
            _residual(final["server_state"]["server_control"], final["client_states"]), 1e-6
        )

    def test_what_the_refusal_prevents(self) -> None:
        """Bypass it and the residual is the whole of ``||c||``, for good.

        The restore path itself, not the CLI: `_restore_server_state` puts the
        control variate back and `_restore_client_states` returns early, which
        is the two halves of the defect in the two lines that produce it.
        """

        server = _scaffold_server()
        # Built as the run that wrote the checkpoint was: a server at another
        # seed is refused before the restore reaches the control variate.
        server.seed = self.best["server_state"]["seed"]
        _restore_server_state(server, self.best)
        _restore_client_states({}, self.best)

        restored = server._server_control
        assert restored is not None
        zeroed = {key: torch.zeros_like(value) for key, value in restored.items()}
        stranded = _norm({key: restored[key].double() - zeroed[key].double() for key in restored})
        self.assertAlmostEqual(
            stranded, _norm(self.best["server_state"]["server_control"]), places=6
        )


if __name__ == "__main__":
    unittest.main()
