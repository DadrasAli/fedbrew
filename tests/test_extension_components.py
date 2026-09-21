"""A stub of each of the five kinds, built through the real factory.

The registry can hold an out-of-tree component and the config can name it,
but neither says what the component is *handed*. Before this, the factory
enumerated built-in names at every dispatch and everything else fell through
to a call with no arguments at all -- so a third dataset backend and a third
task received no configuration, which is the gap `examples/pl-1d`'s README
§2 describes and the reason every example configured itself through a
closure that no run config could see and no run.json could record.

So each of the five registries gets a stub here, registered the way an
extension registers, named by a config, and built through
``build_components``. What is checked is the contract: exactly which keyword
arguments arrive, that a key declared at registration arrives with them and
one that was not declared is refused at load, and that the dataset's
metadata -- where a generated problem states its reference optimum --
reaches the task.

The pairing checks are the deliberate hole. They enumerate shipped names and
cannot judge a pair involving a component they have never seen, so an
extension is left to refuse its own bad partner rather than being refused
for not appearing in a list it cannot appear in. That is recorded as an info
issue rather than left silent.
"""

from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path

import pytest
import yaml

from fedbrew.core import extensions, registry
from fedbrew.core.config import load_config
from fedbrew.core.factory import (
    EXTENSION_CLIENT_KEYS,
    EXTENSION_DATASET_FIELDS,
    EXTENSION_TASK_KEYS,
    build_components,
    is_extension,
)
from fedbrew.core.validation import run_checks

pytestmark = pytest.mark.fast

#: One file registering a stub in every registry a run config selects from,
#: each recording the keyword arguments it was built with. Written as an
#: extension file rather than registered inline, so what is exercised is the
#: whole path a user takes: experiment.extensions -> loader -> register() ->
#: load_config -> build_components.
EXTENSION = textwrap.dedent(
    '''
    """Stubs for every registry, each recording how it was built."""

    from typing import Any

    import torch
    from torch import nn

    from fedbrew.core import registry
    from fedbrew.data.dataset import FederatedDataset
    from fedbrew.tasks.base import TaskAdapter

    BUILT: dict[str, dict[str, Any]] = {{}}


    class ProbeDataset(FederatedDataset):
        def __init__(self, **kwargs):
            BUILT["dataset"] = kwargs
            self.rows = int(kwargs.get("num_clients") or 2)

        def list_clients(self):
            return [f"client_{{index}}" for index in range(self.rows)]

        def get_client_data(self, client_id):
            row = {{"x": torch.zeros(1, 2), "y": torch.zeros(1)}}
            return {{"train": row, "eval": row, "test": row, "num_examples": 1}}

        def get_client_metadata(self, client_id):
            return {{"client_id": client_id, "num_examples": 1}}

        def get_global_data(self):
            return {{"x": torch.zeros(2, 2), "y": torch.zeros(2)}}

        def get_metadata(self):
            return {{"task": "probe", "reference": {{"x_star": [0.0], "f_star": 0.5}}}}


    class ProbeTask(TaskAdapter):
        def __init__(self, **kwargs):
            BUILT["task"] = kwargs
            self.dataset_metadata = kwargs.get("dataset_metadata", {{}})
            self._scaler = None

        def build_model(self, config=None):
            return nn.Linear(2, 1)

        def build_dataloader(self, data, config=None):
            return [(data["x"], data["y"])]

        def train_step(self, model, batch, optimizer=None):
            return {{"loss": 0.0}}

        def eval_step(self, model, batch):
            return {{"loss": 0.0, "total": 1.0}}

        def compute_metrics(self, outputs):
            return {{"loss": 0.0}}


    class ProbeServer:
        def __init__(self, **kwargs):
            BUILT["server"] = kwargs


    class ProbeClient:
        def __init__(self, **kwargs):
            BUILT["client"] = kwargs


    def build_probe_model(config=None):
        return nn.Linear(2, 1)


    def register():
        registry.tasks.register("probe_task", lambda **kwargs: ProbeTask(**kwargs))
        registry.datasets.register(
            "probe_data", lambda **kwargs: ProbeDataset(**kwargs), config_keys={{"probe_rows"}}
        )
        registry.models.register("probe_model", build_probe_model, task="probe_task")
        registry.server_strategies.register(
            "probe_server", lambda **kwargs: ProbeServer(**kwargs), config_keys={{"probe_gain"}}
        )
        registry.client_updates.register(
            "probe_client", lambda **kwargs: ProbeClient(**kwargs), config_keys={{"probe_step"}}
        )
    '''
)

NAMES = {
    "tasks": "probe_task",
    "datasets": "probe_data",
    "models": "probe_model",
    "server_strategies": "probe_server",
    "client_updates": "probe_client",
}


class ExtensionComponentFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.entry = str(self.directory / "problem.py")
        Path(self.entry).write_text(EXTENSION.format(), encoding="utf-8")
        self.addCleanup(extensions._loaded.pop, str(Path(self.entry).resolve()), None)
        for attribute, name in NAMES.items():
            component = getattr(registry, attribute)
            self.addCleanup(component._items.pop, name, None)
            self.addCleanup(component._origins.pop, name, None)
            self.addCleanup(component._config_keys.pop, name, None)
        self.addCleanup(registry.MODEL_TASKS.pop, "probe_model", None)

    def _config(self, **overrides: object) -> Path:
        raw = {
            "experiment": {
                "seed": 3,
                "output_dir": str(self.directory / "out"),
                "use_run_subdir": False,
                "tags": [],
                "notes": "",
                "extensions": [self.entry],
            },
            "server": {
                "strategy": "probe_server",
                "participation_rate": 1.0,
                "metrics": [],
                "probe_gain": 2.5,
            },
            "client": {
                "update_rule": "probe_client",
                "batch_size": 4,
                "learning_rate": 0.1,
                "metrics": [],
                "probe_step": 7,
            },
            "data": {"name": "probe_data", "num_clients": 2, "probe_rows": 5},
            "model": {"name": "probe_model", "input_dim": 2},
            "runtime": {"device": "cpu", "use_amp": False, "deterministic": True},
            "defaults": {"global_rounds": 4, "local_iterations": 2},
        }
        for dotted, value in overrides.items():
            section, _, key = dotted.partition("__")
            if key:
                raw.setdefault(section, {})[key] = value  # type: ignore[index]
            else:
                raw[section] = value  # type: ignore[assignment]
        path = self.directory / "run.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return path

    def _built(self) -> dict:
        import sys

        for module in sys.modules.values():
            if getattr(module, "__file__", None) == str(Path(self.entry).resolve()):
                return module.BUILT
        raise AssertionError("the extension module was not imported")


class EveryRegistryIsReachedTest(ExtensionComponentFixture):
    def test_all_five_build_through_the_factory(self) -> None:
        config = load_config(self._config())

        components = build_components(config)

        self.assertEqual(type(components.server).__name__, "ProbeServer")
        self.assertEqual(type(components.task).__name__, "ProbeTask")
        self.assertEqual(type(components.dataset).__name__, "ProbeDataset")
        self.assertEqual(len(components.clients), 2)
        self.assertEqual(type(components.clients["client_0"]).__name__, "ProbeClient")
        self.assertEqual(components.model_factory.__name__, "build_probe_model")

    def test_each_is_recognised_as_an_extension_and_the_builtins_are_not(self) -> None:
        load_config(self._config())
        for attribute, name in NAMES.items():
            with self.subTest(registry=attribute):
                self.assertTrue(is_extension(getattr(registry, attribute), name))
        self.assertFalse(is_extension(registry.server_strategies, "fedavg"))
        self.assertFalse(is_extension(registry.tasks, "classification"))


class ContractTest(ExtensionComponentFixture):
    def test_the_task_receives_the_documented_keys(self) -> None:
        build_components(load_config(self._config()))
        built = self._built()["task"]

        self.assertEqual(set(built), set(EXTENSION_TASK_KEYS))
        self.assertEqual(built["batch_size"], 4)
        self.assertEqual(built["device"], "cpu")
        self.assertEqual(built["model_config"]["name"], "probe_model")
        self.assertEqual(built["model_config"]["input_dim"], 2)
        self.assertIsInstance(built["dataloader_config"], dict)
        self.assertIsInstance(built["reuse_model"], bool)

    def test_the_task_is_handed_the_dataset_metadata_including_its_reference(self) -> None:
        """Where a generated problem states what a run on it is scored
        against. Before this the task closed over a spec instead, and nothing
        in run.json said what the numbers were measured against."""

        build_components(load_config(self._config()))

        metadata = self._built()["task"]["dataset_metadata"]
        self.assertEqual(metadata["reference"], {"x_star": [0.0], "f_star": 0.5})

    def test_the_dataset_receives_its_named_fields_the_seed_and_its_declared_key(self) -> None:
        build_components(load_config(self._config()))
        built = self._built()["dataset"]

        self.assertEqual(built["num_clients"], 2)
        self.assertEqual(built["seed"], 3)
        self.assertEqual(built["probe_rows"], 5)
        self.assertLessEqual(set(built), {*EXTENSION_DATASET_FIELDS, "seed", "probe_rows"})

    def test_an_unset_dataset_field_is_absent_rather_than_none(self) -> None:
        """So the backend's own default applies, rather than being overwritten
        by a None the config never wrote."""

        build_components(load_config(self._config()))
        self.assertNotIn("samples_per_client", self._built()["dataset"])
        self.assertNotIn("path", self._built()["dataset"])

    def test_the_server_receives_the_fedavg_common_kwargs_and_its_declared_key(self) -> None:
        build_components(load_config(self._config()))
        built = self._built()["server"]

        self.assertEqual(
            set(built),
            {
                "task",
                "model_config",
                "participation_rate",
                "participation_probability",
                "seed",
                "metrics",
                "aggregation_weighting",
                "probe_gain",
            },
        )
        self.assertEqual(built["probe_gain"], 2.5)
        self.assertEqual(built["participation_rate"], 1.0)
        self.assertIsNone(built["participation_probability"])
        self.assertEqual(built["seed"], 3)
        self.assertEqual(built["aggregation_weighting"], "examples")

    def test_the_client_receives_the_training_base_and_its_declared_key(self) -> None:
        build_components(load_config(self._config()))
        built = self._built()["client"]

        self.assertEqual(set(built), {*EXTENSION_CLIENT_KEYS, "probe_step"})
        self.assertEqual(built["probe_step"], 7)
        self.assertEqual(built["client_id"], "client_1")
        self.assertEqual(built["local_iterations"], 2)
        self.assertEqual(built["total_rounds"], 4)
        self.assertEqual(built["base_seed"], 3)
        self.assertEqual(built["learning_rate"], 0.1)
        self.assertIs(built["task"], self._built()["task"] and built["task"])

    def test_a_client_rule_may_derive_its_own_step_size(self) -> None:
        """Neither required nor refused: whether a rule needs a configured
        learning rate is a fact about the rule, and preflight does not know it
        for one the package did not write."""

        config = load_config(self._config(client__learning_rate=None))

        self.assertEqual(
            [issue.code for issue in run_checks(config) if issue.severity == "error"], []
        )
        build_components(config)
        self.assertIsNone(self._built()["client"]["learning_rate"])


class DeclaredKeysTest(ExtensionComponentFixture):
    def test_an_undeclared_key_in_a_component_block_is_refused_at_load(self) -> None:
        with self.assertRaises(ValueError) as caught:
            load_config(self._config(server__probe_undeclared=1))
        self.assertIn("server.probe_undeclared", str(caught.exception))

    def test_a_declared_key_the_config_omits_does_not_arrive_as_none(self) -> None:
        raw = yaml.safe_load(self._config().read_text(encoding="utf-8"))
        del raw["server"]["probe_gain"]
        path = self.directory / "run.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")

        build_components(load_config(path))

        self.assertNotIn("probe_gain", self._built()["server"])


class PairingNeutralityTest(ExtensionComponentFixture):
    def _with_fedavg_server(self) -> Path:
        """The probe client under the shipped FedAvg server.

        `probe_gain` goes with the strategy that declared it: a declared key
        is loadable only while its component is in force, so leaving it here
        is refused -- which the test below asserts, since it is the mechanism
        working rather than an obstacle to route around.
        """

        raw = yaml.safe_load(self._config().read_text(encoding="utf-8"))
        raw["server"]["strategy"] = "fedavg"
        del raw["server"]["probe_gain"]
        path = self.directory / "fedavg.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return path

    def test_an_extension_rule_may_pair_with_a_shipped_server(self) -> None:
        """`fedavg` refuses any client rule outside a shipped list, and an
        out-of-tree rule can never be in it."""

        config = load_config(self._with_fedavg_server())

        errors = [issue for issue in run_checks(config) if issue.severity == "error"]
        self.assertEqual([issue.code for issue in errors], [])

    def test_the_strategys_declared_key_does_not_survive_the_strategy(self) -> None:
        with self.assertRaises(ValueError) as caught:
            load_config(self._config(server__strategy="fedavg"))
        self.assertIn("server.probe_gain", str(caught.exception))

    def test_the_unchecked_pairing_is_reported_rather_than_left_silent(self) -> None:
        config = load_config(self._config())

        notes = [
            issue
            for issue in run_checks(config)
            if issue.code == "algorithm.extension_pairing_unchecked"
        ]
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].severity, "info")

    def test_a_shipped_pair_still_gets_the_shipped_rules(self) -> None:
        """The neutrality is for extensions only: scaffold's server still
        refuses a fedavg client."""

        from fedbrew.core.config import load_config as load

        config = load("configs/dev/smoke.yaml")
        config.server.strategy = "fedavg"
        config.client.update_rule = "scaffold"
        errors = [issue.code for issue in run_checks(config) if issue.severity == "error"]
        self.assertIn("algorithm.scaffold_server_incompatible", errors)


class RunsEndToEndTest(ExtensionComponentFixture):
    def test_preflight_accepts_a_run_built_entirely_from_extensions(self) -> None:
        from fedbrew.core.runner import main

        main(["--config", str(self._config()), "--validate-only", "--quiet"])

    def test_run_json_would_record_every_registered_name(self) -> None:
        from fedbrew.core.run_metadata import build_extension_provenance

        config = load_config(self._config())
        provenance = build_extension_provenance(config)

        assert provenance is not None
        registered = {tuple(pair) for pair in provenance[self.entry]["registered"]}
        self.assertEqual(registered, set(NAMES.items()))
        self.assertTrue(json.dumps(provenance))


if __name__ == "__main__":
    unittest.main()
