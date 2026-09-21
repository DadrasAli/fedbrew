"""A run config names its extensions, and everything downstream sees them.

``experiment.extensions`` is where an out-of-tree component gets its name:
the entries are loaded by ``load_config`` before any name in the config is
checked, so a task, model, strategy, rule or backend an extension registers
passes the same load-time guard the built-ins pass, reaches ``--validate-only``
and the plan header, and is recorded in run.json with the SHA-256 of the file
it came from. A flag or an environment variable would have reached none of
those, which is why it is a config key.

Two consequences are checked beside the loading itself. A model registered
with ``task=`` needs no ``experiment.task``, and that key is now refused by
name with the redirect, because a value nothing reads is the failure this
package's config layer exists to catch. And a component may declare the
config keys it reads at registration; they are accepted only while that
component is the one in force, so a key declared for one strategy does not
load clean under another.

Registrations here go into the process-wide registries, as they would in a
run, and are popped on cleanup so the documentation guards see nothing.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import textwrap
import unittest
from pathlib import Path

import pytest
import yaml

from fedbrew.core import extensions, registry
from fedbrew.core.config import _validate_registered_names, load_config
from fedbrew.core.runner import default_override_namespace, main, run

SMOKE = Path("configs/dev/smoke.yaml")

MODEL_EXTENSION = textwrap.dedent(
    '''
    """Registers one model under a new name, with the task it needs."""

    from fedbrew.core import registry
    from fedbrew.models.torch_mlp import build_torch_mlp


    def register():
        registry.models.register("{model}", build_torch_mlp, task="classification")
    '''
)

STRATEGY_EXTENSION = textwrap.dedent(
    '''
    """Registers one strategy that declares a config key of its own."""

    from fedbrew.core import registry


    def register():
        registry.server_strategies.register(
            "{strategy}", lambda **kwargs: None, config_keys={{"probe_knob"}}
        )
    '''
)

NOOP_EXTENSION = textwrap.dedent(
    '''
    """Registers nothing; exists to be recorded."""


    def register():
        pass
    '''
)


class ExtensionConfigFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.smoke = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))

    def _extension(self, name: str, body: str, **names: str) -> str:
        path = self.directory / name
        path.write_text(body.format(**names), encoding="utf-8")
        self.addCleanup(extensions._loaded.pop, str(path.resolve()), None)
        return str(path)

    def _forget(self, component: registry.Registry, *names: str) -> None:
        for name in names:
            self.addCleanup(component._items.pop, name, None)
            self.addCleanup(component._origins.pop, name, None)
            self.addCleanup(component._config_keys.pop, name, None)
            self.addCleanup(registry.MODEL_TASKS.pop, name, None)

    def _config(self, **overrides: object) -> Path:
        raw = json.loads(json.dumps(self.smoke))
        for dotted, value in overrides.items():
            section, key = dotted.split("__")
            raw.setdefault(section, {})[key] = value
        path = self.directory / "run.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return path


@pytest.mark.fast
class LoadingTest(ExtensionConfigFixture):
    def test_a_model_from_an_extension_loads_and_needs_no_experiment_task(self) -> None:
        entry = self._extension("problem.py", MODEL_EXTENSION, model="ext_probe_model")
        self._forget(registry.models, "ext_probe_model")

        config = load_config(
            self._config(experiment__extensions=[entry], model__name="ext_probe_model")
        )

        self.assertEqual(config.model.name, "ext_probe_model")
        self.assertEqual(config.task.name, "classification", "read from the registration")
        self.assertEqual(config.experiment.extensions, [entry])
        self.assertEqual(registry.models.origin("ext_probe_model"), entry)

    def test_the_same_config_without_the_extension_is_refused_and_says_where_to_look(
        self,
    ) -> None:
        with self.assertRaises(ValueError) as caught:
            load_config(self._config(model__name="ext_probe_model_never_loaded"))
        message = str(caught.exception)
        self.assertIn("ext_probe_model_never_loaded", message)

    def test_an_unknown_name_names_the_key_that_loads_extensions(self) -> None:
        config = load_config(SMOKE)
        config.task.name = "definitely_not_registered"
        with self.assertRaises(ValueError) as caught:
            _validate_registered_names(config)
        self.assertIn("experiment.extensions", str(caught.exception))

    def test_preflight_sees_the_extension(self) -> None:
        """--validate-only loads the config through the same path, so the
        component the extension registered passes preflight too."""

        entry = self._extension("problem.py", MODEL_EXTENSION, model="ext_probe_preflight")
        self._forget(registry.models, "ext_probe_preflight")
        path = self._config(experiment__extensions=[entry], model__name="ext_probe_preflight")

        main(["--config", str(path), "--validate-only", "--quiet"])

    def test_a_missing_extension_file_fails_the_load_by_name(self) -> None:
        with self.assertRaises(ValueError) as caught:
            load_config(self._config(experiment__extensions=[str(self.directory / "absent.py")]))
        self.assertIn("absent.py", str(caught.exception))


@pytest.mark.fast
class RemovedTaskKeyTest(ExtensionConfigFixture):
    def test_experiment_task_is_refused_with_the_redirect(self) -> None:
        with self.assertRaises(ValueError) as caught:
            load_config(self._config(experiment__task="classification"))
        message = str(caught.exception)
        self.assertIn("experiment.task has been removed", message)
        self.assertIn("models.register", message)

    def test_the_field_is_gone_from_the_dataclass(self) -> None:
        from dataclasses import fields

        from fedbrew.core.config import ExperimentConfig

        self.assertNotIn("task", {item.name for item in fields(ExperimentConfig)})


@pytest.mark.fast
class ExtensionsShapeTest(ExtensionConfigFixture):
    def test_a_bare_string_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            load_config(self._config(experiment__extensions="problem.py"))
        self.assertIn("experiment.extensions must be a list", str(caught.exception))

    def test_a_non_string_entry_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            load_config(self._config(experiment__extensions=[3]))
        self.assertIn("non-empty strings", str(caught.exception))

    def test_a_repeated_entry_is_refused(self) -> None:
        entry = self._extension("problem.py", NOOP_EXTENSION)
        with self.assertRaises(ValueError) as caught:
            load_config(self._config(experiment__extensions=[entry, entry]))
        self.assertIn("twice", str(caught.exception))

    def test_an_empty_list_is_the_default_and_loads(self) -> None:
        config = load_config(self._config(experiment__extensions=[]))
        self.assertEqual(config.experiment.extensions, [])


@pytest.mark.fast
class DeclaredConfigKeysTest(ExtensionConfigFixture):
    def test_a_declared_key_is_accepted_while_its_component_is_in_force(self) -> None:
        entry = self._extension("strategy.py", STRATEGY_EXTENSION, strategy="ext_probe_strategy")
        self._forget(registry.server_strategies, "ext_probe_strategy")

        config = load_config(
            self._config(
                experiment__extensions=[entry],
                server__strategy="ext_probe_strategy",
                server__probe_knob=3,
            )
        )

        self.assertEqual(config.server.extra["probe_knob"], 3)
        self.assertEqual(
            registry.server_strategies.config_keys("ext_probe_strategy"), frozenset({"probe_knob"})
        )

    def test_the_same_key_is_refused_under_a_strategy_that_did_not_declare_it(self) -> None:
        entry = self._extension("strategy.py", STRATEGY_EXTENSION, strategy="ext_probe_strategy_b")
        self._forget(registry.server_strategies, "ext_probe_strategy_b")

        with self.assertRaises(ValueError) as caught:
            load_config(
                self._config(
                    experiment__extensions=[entry],
                    server__strategy="fedavg",
                    server__probe_knob=3,
                )
            )
        self.assertIn("server.probe_knob", str(caught.exception))

    def test_a_builtin_declares_nothing(self) -> None:
        registry.register_builtin_components()
        for name in registry.server_strategies.builtin():
            self.assertEqual(registry.server_strategies.config_keys(name), frozenset())


class RunRecordTest(ExtensionConfigFixture):
    def test_run_json_records_each_extension_with_its_hash(self) -> None:
        entry = self._extension("noop.py", NOOP_EXTENSION)
        output = self.directory / "out"
        path = self._config(experiment__extensions=[entry])

        run(path, default_override_namespace(rounds=1, quiet=True, output_dir=str(output)))

        record = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["config"]["experiment"]["extensions"], [entry])
        recorded = record["reproducibility"]["extensions"][entry]
        self.assertEqual(recorded["resolved"], str(Path(entry).resolve()))
        self.assertEqual(recorded["sha256"], hashlib.sha256(Path(entry).read_bytes()).hexdigest())
        self.assertEqual(recorded["registered"], [])

    def test_a_run_without_extensions_records_no_block(self) -> None:
        output = self.directory / "out"
        run(
            self._config(), default_override_namespace(rounds=1, quiet=True, output_dir=str(output))
        )
        record = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertNotIn("extensions", record["reproducibility"])
        self.assertEqual(record["config"]["experiment"]["extensions"], [])


if __name__ == "__main__":
    unittest.main()
