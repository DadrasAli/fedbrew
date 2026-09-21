"""A resume that changes a hyperparameter is refused, not silently taken.

Every `load_state` on a client or a server wrote the checkpoint's value over
the one the config had just built -- `self.beta1 = float(state.get("beta1",
self.beta1))` and a dozen more like it. That is right for state a run *learns*
and wrong for a value the config *sets*: the run continued at the old
hyperparameter while `run.json` recorded the new one, so the run's own record
of itself was false.

The packed SLURM scripts pass `--resume-latest` whenever a `latest.pt` exists
under the arm's output dir, so editing a config and resubmitting into the same
directory is one command and nothing in the result says which value was used.

Refused rather than resolved in either direction, because neither direction is
supported: chapter 09 §5 and chapter 10 both say not to change config between a
run and its resume. It is the call `runner._refuse_a_foreign_seed` already
makes for `experiment.seed`.

Two findings, one mechanism, facing opposite ways.

P10-F14 is the keys a `load_state` reads *back*: the checkpoint silently
outranking the config. POST-F04 is the rest of what `get_state` saves --
`learning_rate`, `local_iterations`, `momentum` and eight more -- which nothing
restores, so the *config* silently outranks the checkpoint and the change takes
effect from the resumed round on. Chapter 10 states one contract covering both,
so a refusal covering half the keys under a chapter a reader takes as covering
all of them is worse than no refusal: it reads as coverage.

`CHECKED_NOT_RESTORED` is the second set. `test_a_checked_but_unrestored_key_is_
refused` is POST-F04's own case, and `test_the_classification_covers_every_saved_
key` fails if a rule checkpoints a key no table classifies.

POST-F07 is a setting no checkpoint carried, which no comparison can see:
`participation_rate` and `seed` on every server. The classification test derives
its universe from `get_state()`, the side that left them out, so
`EveryServerSettingIsCheckpointedTest` derives what a server must save from what
it is built with instead, less a named `SERVER_WIRING` list.

POST-F08 is the same defect at the client: `max_grad_norm` on `FedAvgClient` and
`TorchDeltaSGDClient`. `EveryClientSettingIsCheckpointedTest` applies the same
derivation to every shipped client class, less `CLIENT_WIRING`, and
`AnEditedClippingThresholdIsRefusedTest` resumes a FedAvg run across a changed
threshold.

`total_rounds` is the one conditional. It is an input to `_round_learning_rate`
and to nothing else, and that method returns before reading it under
`constant` -- so extending a constant-schedule run changes no number a client
computes, while extending a cosine run re-anneals every remaining round.
`TheRunLengthTest` pins both halves.

A checkpoint written before `local_epochs` became `local_iterations` carries
the old key and lacks the new one, so the comparison would skip the setting.
`APreRenameCheckpointIsRefusedTest` pins the refusal at every client and, end
to end, before anything is restored.

The refusal must not catch a resume nobody changed: a run's own checkpoint has
to agree with the config its run.json records. `AWrittenCheckpointIsNotRefusedTest`
writes a run of its own and checks exactly that.

POST-F22 is the same edit without the resume: a fresh start into a directory
holding a finished run. Nothing was compared at all, so the new settings
silently replaced the old run. `AFreshStartOverAFinishedRunTest` runs both.
"""

from __future__ import annotations

import importlib
import inspect
import json
import shutil
import tempfile
import textwrap
import unittest
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch
import yaml

from fedbrew.clients.base import ClientUpdate
from fedbrew.clients.fedavg_ft_client import FedAvgFTClient
from fedbrew.core import runner
from fedbrew.core.checkpointing import (
    load_checkpoint,
    refuse_a_pre_rename_client_state,
    refuse_a_reconfigured_resume,
)
from fedbrew.core.refusal import RunRefused
from fedbrew.servers.fedavg import FedAvgServer
from fedbrew.servers.fedlalr import FedLALRServer
from fedbrew.servers.fedopt import (
    SUPPORTED_FEDOPT_OPTIMIZERS,
    FedOptServer,
    unread_fedopt_hyperparameters,
)
from fedbrew.servers.scaffold import ScaffoldServer
from tests.test_client_communication_cost import (
    BUILDERS,
    SGD_SETTINGS,
    UPDATE_MODE_SETTINGS,
    _kwargs,
)
from tests.test_resume_is_all_or_nothing import _strategy_classes

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Per update rule, the keys `load_state` restores that the *config* sets. A
#: resume that changes one of these is the defect this module guards.
CONFIGURED = {
    # Restored by `load_state`, so a mismatch used to win silently: P10-F14.
    "base_seed": 99,
    "train_shuffle": True,
    "eval_shuffle": True,
    "drop_last": True,
    "max_local_steps": 3,
    # Checkpointed and restored by nothing, so a mismatch used to lose
    # silently: POST-F04. Same contract, opposite direction.
    "client_id": "somebody-else",
    "local_iterations": 99,
    "batch_size": 999,
    "eval_batch_size": 999,
    "learning_rate": 0.999,
    "momentum": 0.5,
    "weight_decay": 0.5,
    "nesterov": True,
    "learning_rate_schedule": "cosine",
    "min_learning_rate": 0.001,
}
CLIENT_HYPERPARAMETERS: dict[str, dict[str, Any]] = {
    "local_sgd": {**CONFIGURED, "update_mode": "full_gradient"},
    "fedavg": {**CONFIGURED, "update_mode": "frozen_batch_gradients"},
    "centralized": {**CONFIGURED, "update_mode": "frozen_batch_gradients"},
    "fedavg_ft": {**CONFIGURED, "update_mode": "frozen_batch_gradients"},
    "local_adamw": {
        **CONFIGURED,
        "beta1": 0.5,
        "beta2": 0.5,
        "epsilon": 1e-3,
        "update_mode": "full_gradient",
    },
    "fedprox": {**CONFIGURED, "proximal_mu": 0.5, "update_mode": "full_gradient"},
    "scaffold": {**CONFIGURED, "update_mode": "full_gradient"},
    "delta_sgd": {
        **CONFIGURED,
        "update_mode": "frozen_batch_gradients",
        "eta_0": 0.5,
        "theta_0": 0.5,
        "gamma": 0.5,
        "delta": 0.5,
        # Default None, so the changed value has to be a number.
        "eta_max": 5.0,
    },
    "fedlalr": {
        **CONFIGURED,
        "beta1": 0.5,
        "beta2": 0.5,
        "epsilon": 1e-3,
        "update_mode": "full_gradient",
    },
}

#: Per update rule, the keys `load_state` restores that a run *learns*. These
#: must keep restoring from the checkpoint: that is what a resume is for.
CLIENT_LEARNED: dict[str, dict[str, Any]] = {
    "local_sgd": {"num_examples": 999},
    "fedavg": {"num_examples": 999},
    "centralized": {"num_examples": 999},
    "fedavg_ft": {"num_examples": 999},
    "local_adamw": {"num_examples": 999},
    "fedprox": {"num_examples": 999},
    "scaffold": {"num_examples": 999, "client_control": {"w": torch.ones(2)}},
    "delta_sgd": {"num_examples": 999},
    "fedlalr": {"num_examples": 999},
}

#: Checkpointed, compared, and restored by nothing. POST-F04's set: before it
#: these were the keys a resume could change without a word, because the config
#: simply won. They are still not restored -- a checkpoint cannot outrank a
#: config here -- but a disagreement is now refused rather than resolved.
CHECKED_NOT_RESTORED = {
    "batch_size",
    "client_id",
    "eval_batch_size",
    "learning_rate",
    "learning_rate_schedule",
    "local_iterations",
    "min_learning_rate",
    "momentum",
    "nesterov",
    "weight_decay",
    # fedavg_ft's two are restored, but only by FedAvgFTClient, which the
    # registry builds for the `fedavg_ft` rule; BUILDERS names FedAvgClient
    # there because that is the class the payload tests need.
    "frozen_gradient_weighting",
    # Compared only under a schedule that reads it; see TheRunLengthTest.
    "total_rounds",
    # Never checkpointed until POST-F08, so no resume could compare it.
    "max_grad_norm",
}

#: The two genuine exemptions, and both are about what the value *is*.
#: `num_examples` is measured from the client's shard rather than configured;
#: `metrics` is which columns the run writes, which the per-client CSV cursor
#: already handles by invalidating on a changed header.
NOT_CONFIGURED = {"num_examples", "metrics"}


def _fedopt(**overrides: Any) -> FedOptServer:
    settings: dict[str, Any] = {
        "server_optimizer": "fedadam",
        "server_learning_rate": 0.01,
        "beta1": 0.9,
        "beta2": 0.99,
        "tau": 1e-3,
        "participation_rate": 1.0,
        "seed": 0,
    }
    settings.update(overrides)
    server = FedOptServer(**settings)
    server._model_state = {"w": torch.zeros(4)}
    server._model_state_scope = "full"
    server._model_state_metadata = {"model_state_scope": "full"}
    return server


@pytest.mark.fast
class TheHelperTest(unittest.TestCase):
    def test_agreement_passes(self) -> None:
        refuse_a_reconfigured_resume("x", {"a": 1, "b": 2}, {"a": 1, "b": 2})

    def test_disagreement_raises_and_names_both_values(self) -> None:
        with self.assertRaises(ValueError) as caught:
            refuse_a_reconfigured_resume("fedopt server", {"beta1": 0.9}, {"beta1": 0.5})
        message = str(caught.exception)
        self.assertIn("fedopt server", message)
        self.assertIn("beta1", message)
        self.assertIn("0.9", message)
        self.assertIn("0.5", message)

    def test_a_key_the_checkpoint_lacks_is_not_compared(self) -> None:
        """So a checkpoint written before a setting existed still loads."""

        refuse_a_reconfigured_resume("x", {}, {"beta1": 0.5})

    def test_a_key_the_config_does_not_declare_is_not_compared(self) -> None:
        """Learned state travels in the same mapping and is not a mismatch."""

        refuse_a_reconfigured_resume("x", {"m": {"w": 1.0}, "beta1": 0.9}, {"beta1": 0.9})

    def test_every_disagreement_is_listed_not_just_the_first(self) -> None:
        with self.assertRaises(ValueError) as caught:
            refuse_a_reconfigured_resume("x", {"a": 1, "b": 2, "c": 3}, {"a": 9, "b": 2, "c": 8})
        message = str(caught.exception)
        self.assertIn("2 hyperparameters", message)
        for key in ("a", "c"):
            self.assertIn(f"  {key}:", message)
        self.assertNotIn("  b:", message)

    def test_one_disagreement_reads_as_one(self) -> None:
        with self.assertRaises(ValueError) as caught:
            refuse_a_reconfigured_resume("x", {"a": 1}, {"a": 9})
        self.assertIn("1 hyperparameter.", str(caught.exception))


@pytest.mark.fast
class EveryClientRefusesTest(unittest.TestCase):
    """One case per registered update rule, not one per hand-picked example."""

    def test_a_changed_hyperparameter_is_refused(self) -> None:
        for rule, changes in sorted(CLIENT_HYPERPARAMETERS.items()):
            for key, value in sorted(changes.items()):
                with self.subTest(rule=rule, key=key):
                    client = BUILDERS[rule]()
                    state = dict(client.get_state())
                    if key not in state:
                        continue
                    self.assertNotEqual(state[key], value, f"{key} was already {value!r}")
                    state[key] = value
                    with self.assertRaises(ValueError) as caught:
                        client.load_state(state)
                    self.assertIn(key, str(caught.exception))

    def test_an_unchanged_state_round_trips(self) -> None:
        for rule in sorted(CLIENT_HYPERPARAMETERS):
            with self.subTest(rule=rule):
                client = BUILDERS[rule]()
                client.load_state(dict(client.get_state()))

    def test_learned_state_still_restores(self) -> None:
        """The half a resume exists for; refusing it would be the same defect."""

        for rule, changes in sorted(CLIENT_LEARNED.items()):
            for key, value in sorted(changes.items(), key=lambda item: item[0]):
                with self.subTest(rule=rule, key=key):
                    client = BUILDERS[rule]()
                    state = dict(client.get_state())
                    if key not in state:
                        continue
                    state[key] = value
                    client.load_state(state)


@pytest.mark.fast
class EveryServerRefusesTest(unittest.TestCase):
    def test_fedopt_refuses_each_of_its_five(self) -> None:
        changes = {
            "server_optimizer": "fedyogi",
            "server_learning_rate": 0.5,
            "beta1": 0.5,
            "beta2": 0.5,
            "tau": 0.5,
        }
        for key, value in sorted(changes.items()):
            with self.subTest(key=key):
                state = dict(_fedopt().save_state())
                state[key] = value
                with self.assertRaises(ValueError) as caught:
                    _fedopt().load_state(state)
                self.assertIn(key, str(caught.exception))

    def test_fedopt_learned_state_still_restores(self) -> None:
        state = dict(_fedopt().save_state())
        state["m"] = {"w": torch.ones(4)}
        state["v"] = {"w": torch.full((4,), 2.0)}
        state["update_step"] = 17
        server = _fedopt()
        server.load_state(state)
        self.assertEqual(server._update_step, 17)
        self.assertTrue(torch.equal(server._m["w"], torch.ones(4)))

    def test_the_base_server_refuses_a_changed_aggregation_weighting(self) -> None:
        """P01-F04's setting: it decides what the average means."""

        for cls in (FedAvgServer, ScaffoldServer):
            with self.subTest(server=cls.__name__):
                server = cls(participation_rate=1.0, seed=0, aggregation_weighting="examples")
                server._model_state = {"w": torch.zeros(4)}
                server._model_state_scope = "full"
                server._model_state_metadata = {"model_state_scope": "full"}
                state = dict(server.save_state())
                state["aggregation_weighting"] = "uniform"
                with self.assertRaises(ValueError) as caught:
                    server.load_state(state)
                self.assertIn("aggregation_weighting", str(caught.exception))

    def test_fedlalr_refuses_a_changed_epsilon(self) -> None:
        server = FedLALRServer(participation_rate=1.0, seed=0, epsilon=1e-8)
        server._model_state = {"w": torch.zeros(4)}
        server._model_state_scope = "full"
        server._model_state_metadata = {"model_state_scope": "full"}
        state = dict(server.save_state())
        state["epsilon"] = 1e-3
        with self.assertRaises(ValueError) as caught:
            server.load_state(state)
        self.assertIn("epsilon", str(caught.exception))

    def test_a_changed_metrics_list_is_not_refused(self) -> None:
        """Which columns a run writes is bookkeeping, not the experiment."""

        server = FedAvgServer(participation_rate=1.0, seed=0, metrics=["fit_loss"])
        server._model_state = {"w": torch.zeros(4)}
        server._model_state_scope = "full"
        server._model_state_metadata = {"model_state_scope": "full"}
        state = dict(server.save_state())
        state["metrics"] = ["fit_loss", "fit_accuracy"]
        server.load_state(state)
        self.assertEqual(server.metrics, ["fit_loss", "fit_accuracy"])


#: Constructor parameters a server strategy needs in order to run and that no
#: checkpoint should carry. Every other parameter is a configured setting, so
#: `save_state` has to write it and `load_state` has to refuse a change to it.
#: Hand-kept, because each entry is an exemption, and each says why.
SERVER_WIRING = {
    "task": "the task adapter object that builds and reads the model",
    "model_config": "the model block, which chapter 10 scopes out of the resume check",
}


def _prepared(server: Any) -> Any:
    server._model_state = {"w": torch.zeros(4)}
    server._model_state_scope = "full"
    server._model_state_metadata = {"model_state_scope": "full"}
    return server


def _server_builds() -> Iterator[tuple[str, Callable[[], Any]]]:
    """One way to build each shipped strategy, and FedOpt once per optimizer.

    Per optimizer, because an optimizer that never reads a hyperparameter does
    not checkpoint it, so what each must save differs.
    """

    common: dict[str, Any] = {"participation_rate": 0.5, "seed": 7}
    yield "FedAvgServer", lambda: _prepared(FedAvgServer(**common))
    yield "ScaffoldServer", lambda: _prepared(ScaffoldServer(**common))
    yield "FedLALRServer", lambda: _prepared(FedLALRServer(epsilon=1e-8, **common))
    for optimizer in sorted(SUPPORTED_FEDOPT_OPTIMIZERS):
        _, unread = unread_fedopt_hyperparameters(optimizer)
        settings: dict[str, Any] = {
            "server_learning_rate": 0.01,
            "beta1": 0.9,
            "beta2": 0.99,
            "tau": 1e-3,
            **dict.fromkeys(unread),
        }
        yield (
            f"FedOptServer[{optimizer}]",
            lambda optimizer=optimizer, settings=settings: _prepared(
                FedOptServer(server_optimizer=optimizer, **settings, **common)
            ),
        )


def _constructor_parameters(cls: type) -> set[str]:
    """Every named parameter of every `__init__` on the class's MRO."""

    names: set[str] = set()
    for klass in cls.__mro__:
        init = vars(klass).get("__init__")
        if init is None or klass is object:
            continue
        for parameter in inspect.signature(init).parameters.values():
            if parameter.name != "self" and parameter.kind not in (
                parameter.VAR_POSITIONAL,
                parameter.VAR_KEYWORD,
            ):
                names.add(parameter.name)
    return names


def _settings(server: Any) -> set[str]:
    """What a server must checkpoint: its constructor, less wiring and what it never reads."""

    unread: tuple[str, ...] = ()
    if isinstance(server, FedOptServer):
        _, unread = unread_fedopt_hyperparameters(server.server_optimizer)
    return _constructor_parameters(type(server)) - set(SERVER_WIRING) - set(unread)


def _changed(value: Any) -> Any:
    # An unset optional setting; any number disagrees with it.
    if value is None:
        return 0.5
    if isinstance(value, bool):
        return not value
    if isinstance(value, int | float):
        return value + 1
    if isinstance(value, str):
        return f"{value}-changed"
    raise TypeError(f"no changed value for {value!r}")


@pytest.mark.fast
class EveryServerSettingIsCheckpointedTest(unittest.TestCase):
    """What a server must save is derived from what it is built with.

    `refuse_a_reconfigured_resume` can only compare a key the checkpoint
    carries, and `test_the_classification_covers_every_saved_key` derives its
    universe from `get_state()` -- the side that decides what is carried. A
    setting no state method wrote was invisible to both, and
    `participation_rate` and `seed` were exactly that on every server: POST-F07.
    A resume at a changed rate exited 0, sampled a different number of clients
    from the resumed round on, and `run.json` recorded the new rate for the
    whole run.

    So this derives from the other side. Every constructor parameter is a
    configured setting unless `SERVER_WIRING` names it, and a new parameter has
    to be saved and compared, or exempted there with its reason.
    """

    def test_every_shipped_strategy_is_built(self) -> None:
        """A strategy nobody builds here is a strategy nobody checks."""

        built = {name.split("[")[0] for name, _ in _server_builds()}
        self.assertEqual(built, set(_strategy_classes()))

    def test_every_setting_is_saved(self) -> None:
        for name, build in _server_builds():
            with self.subTest(server=name):
                server = build()
                missing = _settings(server) - set(server.save_state())
                self.assertEqual(missing, set(), f"{name} is built with these and never saves them")

    def test_a_change_to_any_saved_setting_is_refused(self) -> None:
        for name, build in _server_builds():
            for key in sorted(_settings(build()) - NOT_CONFIGURED):
                with self.subTest(server=name, key=key):
                    state = dict(build().save_state())
                    # An unsaved setting is test_every_setting_is_saved's
                    # failure; reporting it twice would not add a cause.
                    if key not in state:
                        continue
                    state[key] = _changed(state[key])
                    with self.assertRaises(ValueError) as caught:
                        build().load_state(state)
                    self.assertIn(key, str(caught.exception))

    def test_every_wiring_exemption_names_a_real_parameter(self) -> None:
        """An exemption for a parameter nothing takes exempts nothing and hides a rename."""

        taken = set().union(
            *(_constructor_parameters(type(build())) for _, build in _server_builds())
        )
        self.assertEqual(set(SERVER_WIRING) - taken, set())


#: Constructor parameters a client needs in order to run and that no checkpoint
#: should carry, as `SERVER_WIRING` is for servers.
CLIENT_WIRING = {
    "task": "the task adapter object that builds the model and the dataloaders",
    "client_data": "the client's data shard, which the dataset supplies again on every run",
    "model_config": "the model block, which chapter 10 scopes out of the resume check",
    "device": (
        "where the update runs rather than what it computes; a requeued job can land on "
        "other hardware, and restoring the RNG state already warns when device streams differ"
    ),
}

#: Saved settings whose comparison depends on another setting, as
#: key -> (whether this client compares it, why). `TheRunLengthTest` pins it.
CONDITIONALLY_COMPARED: dict[str, tuple[Callable[[Any], bool], str]] = {
    "total_rounds": (
        lambda client: client.learning_rate_schedule != "constant",
        "an input to the learning-rate schedule and to nothing else",
    ),
}


def _client_classes() -> dict[str, type]:
    """Every `ClientUpdate` subclass defined in `fedbrew/clients/`, by name."""

    for module in sorted(path.stem for path in (REPO_ROOT / "fedbrew" / "clients").glob("*.py")):
        importlib.import_module(f"fedbrew.clients.{module}")

    found: dict[str, type] = {}

    def walk(cls: type) -> None:
        for subclass in cls.__subclasses__():
            if subclass.__module__.startswith("fedbrew.clients."):
                found[subclass.__name__] = subclass
            walk(subclass)

    walk(ClientUpdate)
    return found


def _client_builds() -> list[tuple[str, Callable[[], Any]]]:
    """One way to build each shipped client class, keyed by class.

    `BUILDERS` is keyed by update rule and builds `FedAvgClient` for `fedavg_ft`,
    so `FedAvgFTClient` is added here. Keyed by class, because the constructor
    is what this derives from.
    """

    builds: dict[str, Callable[[], Any]] = {}
    for build in BUILDERS.values():
        builds.setdefault(type(build()).__name__, build)
    builds["FedAvgFTClient"] = lambda: FedAvgFTClient(
        **_kwargs(**SGD_SETTINGS), **UPDATE_MODE_SETTINGS, finetune_epochs=1
    )
    return sorted(builds.items())


def _client_settings(client: Any) -> set[str]:
    """What a client must checkpoint: its constructor, less wiring."""

    return _constructor_parameters(type(client)) - set(CLIENT_WIRING)


@pytest.mark.fast
class EveryClientSettingIsCheckpointedTest(unittest.TestCase):
    """The same derivation for clients.

    `FedAvgClient` and `TorchDeltaSGDClient` are built with `max_grad_norm`, the
    bound on the gradient each applied update uses, and neither wrote it:
    POST-F08, POST-F07's defect at the client. Measured before the fix, a FedAvg
    run resumed at 0.05 after two rounds at 1.0 exited 0, recorded 0.05 in
    `run.json`, and ended 4.5% of the model's norm away from the same resume at
    1.0. `test_the_classification_covers_every_saved_key` could not see it for
    POST-F07's reason; this derives from the constructor, less `CLIENT_WIRING`.
    """

    def test_every_shipped_client_class_is_built(self) -> None:
        """A class nobody builds here is a class nobody checks."""

        self.assertEqual({name for name, _ in _client_builds()}, set(_client_classes()))

    def test_every_setting_is_saved(self) -> None:
        for name, build in _client_builds():
            with self.subTest(client=name):
                client = build()
                missing = _client_settings(client) - set(client.get_state())
                self.assertEqual(missing, set(), f"{name} is built with these and never saves them")

    def test_a_change_to_any_saved_setting_is_refused(self) -> None:
        for name, build in _client_builds():
            client = build()
            for key in sorted(_client_settings(client) - NOT_CONFIGURED):
                if key in CONDITIONALLY_COMPARED and not CONDITIONALLY_COMPARED[key][0](client):
                    continue
                with self.subTest(client=name, key=key):
                    state = dict(client.get_state())
                    # As for servers: an unsaved setting is the test above's failure.
                    if key not in state:
                        continue
                    state[key] = _changed(state[key])
                    with self.assertRaises(ValueError) as caught:
                        build().load_state(state)
                    self.assertIn(key, str(caught.exception))

    def test_every_wiring_exemption_names_a_real_parameter(self) -> None:
        """An exemption for a parameter nothing takes exempts nothing and hides a rename."""

        taken = set().union(
            *(_constructor_parameters(type(build())) for _, build in _client_builds())
        )
        self.assertEqual(set(CLIENT_WIRING) - taken, set())


CLIPPING_CONFIG = """
experiment:
  seed: 42
  output_dir: {out}
server:
  strategy: fedavg
  participation_rate: 1
  metrics: [fit_loss]
client:
  update_rule: fedavg
  batch_size: 4
  learning_rate: 0.05
  learning_rate_schedule: constant
  min_learning_rate: 0.0
  update_mode: sequential_epoch
  frozen_gradient_weighting: examples
  momentum: 0.0
  weight_decay: 0.0
  nesterov: false
  max_grad_norm: {max_grad_norm}
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
  checkpointing:
    enabled: true
    interval: 1
    save_last: true
    save_best: false
evaluation:
  train:
    every: 1
    clients: all
defaults:
  global_rounds: {rounds}
  local_iterations: 1
"""


class AnEditedClippingThresholdIsRefusedTest(unittest.TestCase):
    """POST-F08 end to end: a FedAvg run resumed at a different `max_grad_norm`."""

    def _write(self, directory: Path, out: Path, rounds: int, max_grad_norm: float) -> Path:
        path = directory / f"{out.name}.yaml"
        text = CLIPPING_CONFIG.format(out=out, rounds=rounds, max_grad_norm=max_grad_norm)
        path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
        return path

    def _run_then_resume(self, edited: float) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "run"
            runner.run(self._write(root, out, 2, 1.0), runner.parse_args(["--quiet"]))
            resumed = root / "resumed"
            shutil.copytree(out, resumed)
            runner.run(
                self._write(root, resumed, 4, edited),
                runner.parse_args(["--quiet", "--resume-latest"]),
            )

    def test_an_edited_threshold_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._run_then_resume(0.05)
        message = str(caught.exception)
        self.assertIn("max_grad_norm", message)
        self.assertIn("0.05", message)

    def test_an_unedited_threshold_resumes(self) -> None:
        self._run_then_resume(1.0)


@pytest.mark.fast
class TheMessageNamesAPathThatWorksTest(unittest.TestCase):
    """A refusal that names a way out has to name one that exists.

    The first draft told the reader to add `--output-dir <new>` to
    `--resume-from`, which does not continue the run: `round_metrics_gap`
    finds no `round_metrics.csv` in the empty directory. Then the run started
    over, with `run.json` still saying `resumed: true`; since POST-F25 it is
    refused. Either way it is not a continuation, which is why the message
    says to copy the run directory.
    """

    def _message(self) -> str:
        with self.assertRaises(ValueError) as caught:
            refuse_a_reconfigured_resume("x", {"learning_rate": 0.05}, {"learning_rate": 0.01})
        return str(caught.exception)

    def test_it_offers_both_directions(self) -> None:
        message = self._message()
        self.assertIn("Continue THIS experiment", message)
        self.assertIn("Run the NEW settings", message)

    def test_it_does_not_promise_a_warm_start(self) -> None:
        """There is no flag that loads a checkpoint's weights under a new config."""

        self.assertIn("no warm start", self._message())
        flags = (REPO_ROOT / "fedbrew" / "core" / "runner.py").read_text(encoding="utf-8")
        for absent in ('"--init-from"', '"--warm-start"', '"--load-model"'):
            with self.subTest(flag=absent):
                self.assertNotIn(absent, flags)

    def test_the_copy_advice_is_the_advice_that_works(self) -> None:
        """`--resume-from` beside an empty output_dir restarts, it does not continue."""

        from fedbrew.core.artifacts import round_metrics_gap

        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNotNone(round_metrics_gap(Path(directory), 2))
        self.assertIn("copy the whole run directory", self._message())


@pytest.mark.fast
class TheRunLengthTest(unittest.TestCase):
    """`total_rounds` is compared only under a schedule that reads it.

    It is an input to `_round_learning_rate` and to nothing else, and that
    method returns `self.learning_rate` before touching it when the schedule is
    `constant`. So the conditional is the value's actual scope rather than a
    concession: extending a constant-schedule run changes no number a client
    computes, and extending a cosine run moves every remaining learning rate.

    Refusing in both cases would refuse the one resume that is unambiguously
    fine -- and three existing tests simulate an interruption by lowering
    `global_rounds` for the first leg, which is only honest because they use
    `constant`. Refusing in neither would let `global_rounds: 500` -> `1000`
    re-anneal a cosine run from its midpoint without a word.
    """

    def _client(self, schedule: str, total_rounds: int) -> Any:
        from fedbrew.clients.torch_sgd_client import TorchSGDClient
        from tests.test_client_communication_cost import SGD_SETTINGS, _kwargs

        settings = dict(SGD_SETTINGS)
        settings["learning_rate_schedule"] = schedule
        if schedule != "constant":
            settings["min_learning_rate"] = 0.001
        return TorchSGDClient(**_kwargs(**settings, total_rounds=total_rounds))

    def test_extending_a_constant_schedule_run_is_allowed(self) -> None:
        state = dict(self._client("constant", 2).get_state())
        self._client("constant", 4).load_state(state)

    def test_extending_a_cosine_schedule_run_is_refused(self) -> None:
        state = dict(self._client("cosine", 2).get_state())
        with self.assertRaises(ValueError) as caught:
            self._client("cosine", 4).load_state(state)
        self.assertIn("total_rounds", str(caught.exception))

    def test_the_premise_that_constant_ignores_it(self) -> None:
        """If this stops holding, the exemption above stops being true."""

        for total_rounds in (2, 4, 400):
            with self.subTest(total_rounds=total_rounds):
                client = self._client("constant", total_rounds)
                self.assertEqual(client._round_learning_rate(1), client.learning_rate)
                self.assertEqual(client._round_learning_rate(2), client.learning_rate)

    def test_the_premise_that_cosine_reads_it(self) -> None:
        short = self._client("cosine", 2)
        long = self._client("cosine", 4)
        self.assertNotEqual(short._round_learning_rate(2), long._round_learning_rate(2))

    def test_a_changed_schedule_is_refused_on_its_own(self) -> None:
        """So the conditional cannot be reached by flipping the schedule too."""

        state = dict(self._client("cosine", 4).get_state())
        with self.assertRaises(ValueError) as caught:
            self._client("constant", 4).load_state(state)
        self.assertIn("learning_rate_schedule", str(caught.exception))


@pytest.mark.fast
class TheUnreadKeysTest(unittest.TestCase):
    """What `get_state` saves and no `load_state` reads back.

    These override nothing today, which is why they are outside this fix. The
    set is written down so that a future editor who starts restoring one has
    to move it here first, and to classify it while doing so.
    """

    def test_the_classification_covers_every_saved_key(self) -> None:
        for rule in sorted(CLIENT_HYPERPARAMETERS):
            with self.subTest(rule=rule):
                saved = set(BUILDERS[rule]().get_state())
                classified = (
                    set(CLIENT_HYPERPARAMETERS[rule])
                    | set(CLIENT_LEARNED.get(rule, {}))
                    | CHECKED_NOT_RESTORED
                    | NOT_CONFIGURED
                )
                self.assertEqual(
                    saved - classified,
                    set(),
                    f"{rule} checkpoints keys nothing has classified",
                )

    def test_a_checked_but_unrestored_key_is_refused(self) -> None:
        """POST-F04. These used to pass silently, the config winning."""

        for key, value in (("learning_rate", 999.0), ("local_iterations", 99)):
            with self.subTest(key=key):
                client = BUILDERS["local_sgd"]()
                state = dict(client.get_state())
                state[key] = value
                with self.assertRaises(ValueError) as caught:
                    client.load_state(state)
                self.assertIn(key, str(caught.exception))

    def test_the_refusal_did_not_turn_into_a_restore(self) -> None:
        """An agreeing checkpoint must leave the configured value in place."""

        client = BUILDERS["local_sgd"]()
        configured = client.learning_rate
        client.load_state(dict(client.get_state()))
        self.assertEqual(client.learning_rate, configured)

    def test_the_two_exemptions_really_are_unchecked(self) -> None:
        client = BUILDERS["local_sgd"]()
        state = dict(client.get_state())
        state["num_examples"] = 999
        state["metrics"] = ["something_else"]
        client.load_state(state)
        self.assertEqual(client._num_examples, 999)


CONFIG = """
experiment:
  seed: 42
  output_dir: {out}
server:
  strategy: fedadam
  participation_rate: 1
  server_learning_rate: {slr}
  beta1: 0.9
  beta2: 0.99
  tau: 0.001
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
  checkpointing:
    enabled: true
    interval: 1
    save_last: true
    save_best: false
evaluation:
  train:
    every: 1
    clients: all
defaults:
  global_rounds: {rounds}
  local_iterations: 1
"""


class EndToEndTest(unittest.TestCase):
    """The scenario the audit describes: edit a config, resubmit, resume."""

    def _write(self, directory: Path, out: Path, rounds: int, slr: float, rate: str = "1") -> Path:
        path = directory / "fedadam.yaml"
        text = textwrap.dedent(CONFIG.format(out=out, rounds=rounds, slr=slr)).lstrip()
        path.write_text(
            text.replace("participation_rate: 1\n", f"participation_rate: {rate}\n"),
            encoding="utf-8",
        )
        return path

    def _run_then_resume(self, edited: float, rate: str = "1") -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "run"
            runner.run(self._write(root, out, 2, 0.01), runner.parse_args(["--quiet"]))
            resumed = root / "resumed"
            shutil.copytree(out, resumed)
            runner.run(
                self._write(root, resumed, 4, edited, rate),
                runner.parse_args(["--quiet", "--resume-latest"]),
            )

    def test_an_edited_participation_rate_is_refused(self) -> None:
        """POST-F07. The rate was never checkpointed, so this resumed and exited 0."""

        with self.assertRaises(ValueError) as caught:
            self._run_then_resume(0.01, rate="0.5")
        message = str(caught.exception)
        self.assertIn("participation_rate", message)
        self.assertIn("0.5", message)

    def test_an_edited_server_learning_rate_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._run_then_resume(0.05)
        message = str(caught.exception)
        self.assertIn("server_learning_rate", message)
        self.assertIn("0.01", message)
        self.assertIn("0.05", message)
        self.assertIn("output_dir", message)

    def test_an_unedited_config_resumes(self) -> None:
        self._run_then_resume(0.01)


class AFreshStartOverAFinishedRunTest(unittest.TestCase):
    """POST-F22: edit a config, rerun it into the same directory without resuming."""

    def _run_twice(self, edited: float) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "run"
            runner.run(self._write(root, out, 0.01), runner.parse_args(["--quiet"]))
            runner.run(self._write(root, out, edited), runner.parse_args(["--quiet"]))
            return json.loads((out / "run.json").read_text(encoding="utf-8"))

    def _write(self, directory: Path, out: Path, slr: float) -> Path:
        path = directory / "fedadam.yaml"
        path.write_text(
            textwrap.dedent(CONFIG.format(out=out, rounds=1, slr=slr)).lstrip(), encoding="utf-8"
        )
        return path

    def test_an_edited_config_is_refused_before_it_replaces_the_run(self) -> None:
        with self.assertRaises(RunRefused) as caught:
            self._run_twice(0.05)
        message = str(caught.exception)
        self.assertIn("server_learning_rate", message)
        self.assertIn("that run 0.01", message)
        self.assertIn("this run 0.05", message)

    def test_the_same_config_reruns_in_place(self) -> None:
        recorded = self._run_twice(0.01)
        self.assertEqual(recorded["status"], "completed")


def _rename_back(checkpoint_path: Path) -> None:
    """Rewrite a checkpoint's client states as a pre-rename tree wrote them."""

    checkpoint = load_checkpoint(checkpoint_path)
    for state in checkpoint["client_states"].values():
        state["local_epochs"] = state.pop("local_iterations")
    torch.save(checkpoint, checkpoint_path)


class APreRenameCheckpointIsRefusedTest(unittest.TestCase):
    """A checkpoint carrying `local_epochs` is refused, not loaded.

    `refuse_a_reconfigured_resume` compares only the keys a checkpoint has. One
    written before `local_epochs` became `local_iterations` lacks the new name,
    so without this refusal its value would be compared against nothing and a
    resume across an edited value taken silently -- POST-F04's defect, arriving
    through a rename instead of an omission.
    """

    @pytest.mark.fast
    def test_every_client_refuses_the_old_key_even_at_the_same_value(self) -> None:
        for rule in sorted(CLIENT_HYPERPARAMETERS):
            with self.subTest(rule=rule):
                client = BUILDERS[rule]()
                state = dict(client.get_state())
                state["local_epochs"] = state.pop("local_iterations")
                with self.assertRaises(RunRefused) as caught:
                    client.load_state(state)
                message = str(caught.exception)
                self.assertIn("local_epochs -> local_iterations", message)
                self.assertIn("predates the rename", message)

    @pytest.mark.fast
    def test_a_current_state_passes_the_check(self) -> None:
        refuse_a_pre_rename_client_state("local_sgd client", {"local_iterations": 1})

    def test_a_resume_is_refused_before_anything_is_restored(self) -> None:
        """Up front, for every client: the lazy pool builds clients late."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "run"
            config = root / "fedadam.yaml"
            config.write_text(
                textwrap.dedent(CONFIG.format(out=out, rounds=2, slr=0.01)).lstrip(),
                encoding="utf-8",
            )
            runner.run(config, runner.parse_args(["--quiet"]))
            latest = out / "checkpoints" / "latest.pt"
            _rename_back(latest)
            round_metrics = (out / "round_metrics.csv").read_bytes()
            checkpoint = latest.read_bytes()

            with (
                mock.patch(
                    "fedbrew.core.loop._restore_server_state",
                    side_effect=AssertionError("server state was restored first"),
                ),
                self.assertRaises(RunRefused) as caught,
            ):
                runner.run(config, runner.parse_args(["--quiet", "--resume-latest"]))

            self.assertIn("local_iterations", str(caught.exception))
            self.assertIn("predates the rename", str(caught.exception))
            self.assertEqual((out / "round_metrics.csv").read_bytes(), round_metrics)
            self.assertEqual(latest.read_bytes(), checkpoint)


class AWrittenCheckpointIsNotRefusedTest(unittest.TestCase):
    """A run's checkpoint agrees with the config its run.json records.

    This walked `outputs/` instead, and could not fail for two reasons. The
    directory is gitignored, so the test was red in every clone and green in a
    checkout only while some run happened to be on disk. And it read
    `config["server"]["aggregation_weighting"]`, where run.json does not record
    it -- a strategy's own settings are under `server.extra` -- so against a
    run.json this tree writes, the configured value was None, the pair was
    skipped, and the test passed having compared nothing, including against a
    checkpoint whose value had been changed.

    Both records are indexed directly, so a key that moves raises instead of
    skipping. `uniform` rather than the default, so a checkpoint that recorded
    the default whatever the config said disagrees here rather than agreeing by
    coincidence.
    """

    def test_the_checkpoint_matches_the_recorded_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "run"
            config = yaml.safe_load(textwrap.dedent(CONFIG.format(out=out, rounds=1, slr=0.01)))
            config["server"]["aggregation_weighting"] = "uniform"
            path = root / "fedadam.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            runner.run(path, runner.parse_args(["--quiet"]))
            recorded = json.loads((out / "run.json").read_text(encoding="utf-8"))["config"]
            saved = load_checkpoint(out / "checkpoints" / "latest.pt")["server_state"]

        configured = recorded["server"]["extra"]["aggregation_weighting"]
        self.assertEqual(configured, "uniform")
        self.assertEqual(
            saved["aggregation_weighting"],
            configured,
            "a run's own checkpoint would be refused when that run resumes",
        )


if __name__ == "__main__":
    unittest.main()
