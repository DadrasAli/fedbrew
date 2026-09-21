"""Every example runs through the shipped commands, like anything else.

`examples/drift-quad` was the reason the extension hook was built and is the
test of whether it worked; the other four followed it. Before the migration
each example shipped a `run.py` that imported its own `problem.py`, composed a
run config in Python and called `fedbrew.core.runner.run` -- so their arms
never met `--validate-only`, the plan header, or a single load-time guard, and
no `run.json` recorded what a run was scored against.

What this checks is that the migrated form needs nothing the mechanism does
not give a stranger, and that it is the *same* form in all five. Every
generator config and every arm config loads, names its example's extension,
and resolves; every manifest carries a reference optimum and declares that
nothing is held out; every `problem.py` registers a generator, a task and a
model and registers them only through `register()`, so importing one -- as the
sweep scripts and this file do -- has no effect on the registries. Where an
example cross-checks its model block against its data, that refusal is checked
too.

Deliberately not here: the numbers. Those are the READMEs', they take about
ten minutes across the five, and they are re-measured when an example changes
rather than on every suite run.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

import pytest
import yaml

from fedbrew.core import extensions, registry
from fedbrew.core.config import load_config
from fedbrew.core.factory import build_components
from fedbrew.core.validation import run_checks

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTENSION = "examples/drift-quad/problem.py"
SETTINGS = ("drift-quad", "drift-quad-rate", "drift-quad-floor")
ARMS = (
    "fedavg",
    "fedprox",
    "fedavgm",
    "fedadam",
    "fedyogi",
    "fedadagrad",
    "scaffold",
    "fedlalr",
)

#: example directory -> (its extension file, the dataset/task name it
#: registers, its model name, the generator-config settings it ships).
#: Every entry is a claim that the example is on the hook; the test below
#: reads it in both directions, so an example added to `examples/` without a
#: row here fails rather than going unchecked.
EXAMPLES = {
    "pl-1d": ("pl_1d", "pl_scalar", ("pl-1d",)),
    "drift-quad": ("drift_quad", "quad_vector", SETTINGS),
    "fed-lasso": (
        "fed_lasso",
        "lasso_vector",
        ("fed-lasso", "fed-lasso-smooth", "fed-lasso-l2"),
    ),
    "simplex-lsq": (
        "simplex_lsq",
        "simplex_vector",
        ("simplex-lsq", "simplex-lsq-feasible"),
    ),
    "nonconvex-simplex": ("nonconvex_simplex", "simplex_point", ("nonconvex-simplex",)),
}


def _generator_config(setting: str) -> dict:
    path = REPO_ROOT / "data" / "configs" / "examples" / f"{setting}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _arm_paths(setting: str) -> list[Path]:
    return sorted((REPO_ROOT / "configs" / "examples" / setting).glob("*.yaml"))


class ShippedConfigsTest(unittest.TestCase):
    @pytest.mark.fast
    def test_every_setting_has_a_generator_config_and_eight_arms(self) -> None:
        for setting in SETTINGS:
            with self.subTest(setting=setting):
                self.assertTrue(
                    (REPO_ROOT / "data" / "configs" / "examples" / f"{setting}.yaml").is_file()
                )
                self.assertEqual({path.stem for path in _arm_paths(setting)}, set(ARMS))

    def test_every_arm_config_loads_and_names_the_extension(self) -> None:
        for setting in SETTINGS:
            for path in _arm_paths(setting):
                with self.subTest(config=str(path.relative_to(REPO_ROOT))):
                    config = load_config(path)
                    self.assertEqual(config.experiment.extensions, [EXTENSION])
                    self.assertEqual(config.model.name, "quad_vector")
                    # Derived from the model's registration, not stated.
                    self.assertEqual(config.task.name, "drift_quad")
                    self.assertEqual(
                        config.data.path,
                        f"data/generated/examples/{setting}/manifest.json",
                    )

    @pytest.mark.fast
    def test_every_generator_config_names_the_extension_and_its_own_output(self) -> None:
        for setting in SETTINGS:
            with self.subTest(setting=setting):
                config = _generator_config(setting)
                self.assertEqual(config["dataset"]["name"], "drift_quad")
                self.assertEqual(config["dataset"]["extensions"], [EXTENSION])
                self.assertEqual(
                    config["dataset"]["output_dir"], f"data/generated/examples/{setting}"
                )
                # The ratios describe a cut and there is none; the generator
                # writes all three splits itself.
                self.assertNotIn("client_splits", config)

    @pytest.mark.fast
    def test_the_dials_a_config_states_are_the_dials_its_arms_state(self) -> None:
        """The one pair of numbers that could describe two problems in one run."""

        for setting in SETTINGS:
            problem = _generator_config(setting)["problem"]
            for path in _arm_paths(setting):
                with self.subTest(config=str(path.relative_to(REPO_ROOT))):
                    model = yaml.safe_load(path.read_text(encoding="utf-8"))["model"]
                    self.assertEqual(model["input_dim"], problem["dim"])
                    self.assertEqual(model["condition_number"], problem["condition_number"])


class ImportRegistersNothingTest(unittest.TestCase):
    def test_importing_the_problem_touches_no_registry(self) -> None:
        """So the sweep script, and this file, can read ProblemSpec freely.

        Imported through the loader's own file-import helper rather than by
        hand: a module holding ``@dataclass(slots=True)`` has to be in
        ``sys.modules`` before its body runs, because dataclasses resolves
        annotations through ``sys.modules[cls.__module__]``. That is why
        ``extensions._import_file`` registers it there first, and importing
        it any other way here would be exercising a different import.
        """

        import sys

        # A snapshot rather than "these names are absent": another test in
        # this process may already have loaded the extension through a
        # config, and the claim is about what the *import* does, not about
        # what the registries happen to hold.
        before = {
            attribute: list(getattr(registry, attribute).list())
            for attribute in ("generators", "tasks", "models", "server_strategies", "datasets")
        }
        module = extensions._import_file(REPO_ROOT / EXTENSION)
        self.addCleanup(sys.modules.pop, module.__name__, None)

        after = {attribute: list(getattr(registry, attribute).list()) for attribute in before}
        self.assertEqual(after, before, "importing problem.py registered something")
        self.assertTrue(callable(module.register))
        # And the self-check ran at import, which is what makes the closed
        # forms above load-bearing rather than decorative.
        self.assertEqual(module.F_STAR, 0.0)


class GeneratedDataTest(unittest.TestCase):
    """What the generator writes, from a run of it into a scratch tree."""

    @classmethod
    def setUpClass(cls) -> None:
        import tempfile

        from fedbrew.data.generate import generate_from_config

        cls._scratch = tempfile.TemporaryDirectory()
        config = _generator_config("drift-quad")
        config["dataset"]["output_dir"] = cls._scratch.name
        config["dataset"]["extensions"] = [str(REPO_ROOT / EXTENSION)]
        path = Path(cls._scratch.name) / "generator.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        cls.manifest_path = generate_from_config(path)
        cls.manifest = json.loads(Path(cls.manifest_path).read_text(encoding="utf-8"))
        cls.addClassCleanup(extensions._loaded.pop, str(REPO_ROOT / EXTENSION), None)
        for component, name in (
            (registry.generators, "drift_quad"),
            (registry.tasks, "drift_quad"),
            (registry.models, "quad_vector"),
        ):
            cls.addClassCleanup(component._items.pop, name, None)
            cls.addClassCleanup(component._origins.pop, name, None)
            cls.addClassCleanup(component._config_keys.pop, name, None)
        cls.addClassCleanup(registry.MODEL_TASKS.pop, "quad_vector", None)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._scratch.cleanup()

    def test_the_manifest_validates(self) -> None:
        from fedbrew.data.manifest_validation import validate_manifest

        errors = [
            issue for issue in validate_manifest(self.manifest_path) if issue.severity == "error"
        ]
        self.assertEqual(errors, [])

    def test_the_reference_optimum_travels_with_the_data(self) -> None:
        """It is a property of the shards, so run.json records what the run
        was scored against without anyone restating it."""

        reference = self.manifest["reference"]
        self.assertEqual(reference["f_star"], 0.0)
        self.assertEqual(reference["x_star"], [0.0] * 16)
        self.assertEqual(
            reference["problem"],
            {"clients": 8, "dim": 16, "condition_number": 100.0, "dissimilarity": 1.0},
        )
        # zeta is the dial itself, not a consequence of it.
        self.assertAlmostEqual(reference["realised_dissimilarity"], 1.0, places=12)
        self.assertAlmostEqual(reference["max_stable_learning_rate"], 0.02, places=15)

    def test_the_splits_are_declared_as_the_training_data_they_are(self) -> None:
        from fedbrew.data.manifest_validation import IDENTICAL_TO_TRAIN

        self.assertEqual(self.manifest["client_test_source"], IDENTICAL_TO_TRAIN)

    def test_the_offsets_sum_to_exactly_zero_on_disk(self) -> None:
        """What makes x* = 0 and F* = 0 exact for the federated objective, and
        the one claim a round trip through torch.save could have broken."""

        import torch

        from fedbrew.data.writers.torch_shards import load_client_shard

        shard = load_client_shard(Path(self.manifest_path).parent / self.manifest["global_test"])
        offsets = shard["x"]
        self.assertEqual(offsets.dtype, torch.float64)
        self.assertEqual(float(offsets.sum(dim=0).abs().max()), 0.0)


class CrossCheckTest(unittest.TestCase):
    """The two channels the problem is split across cannot silently disagree."""

    def setUp(self) -> None:
        self.path = REPO_ROOT / "configs" / "examples" / "drift-quad" / "fedavg.yaml"
        if not (REPO_ROOT / "data" / "generated" / "examples" / "drift-quad").is_dir():
            self.skipTest("the drift-quad dataset has not been generated")

    def test_the_shipped_arm_builds(self) -> None:
        config = load_config(self.path)
        components = build_components(config)
        self.assertEqual(len(components.dataset.list_clients()), 8)
        self.assertEqual(type(components.task).__name__, "DriftQuadTask")

    def test_a_curvature_the_data_was_not_generated_with_is_refused(self) -> None:
        """A kappa mismatch would fail on nothing and draw a plausible curve
        for a condition number the config does not name."""

        config = load_config(self.path)
        config.model.extra["condition_number"] = 50.0
        with self.assertRaises(ValueError) as caught:
            build_components(config)
        message = str(caught.exception)
        self.assertIn("kappa=50.0", message)
        self.assertIn("manifest reference", message)

    def test_preflight_notes_that_the_test_split_is_the_training_data(self) -> None:
        issues = run_checks(load_config(self.path))
        codes = [issue.code for issue in issues]
        self.assertIn("data.test_is_training_data", codes)
        self.assertEqual([issue.code for issue in issues if issue.severity == "error"], [])


class EveryExampleIsAnExtensionTest(unittest.TestCase):
    """The five are one shape, checked as one shape.

    `drift-quad`'s own classes below go deeper on one example; this goes
    across all of them, on the properties the migration was for.
    """

    @pytest.mark.fast
    def test_every_example_directory_has_a_row(self) -> None:
        """The other direction: an example nothing here knows about."""

        on_disk = {
            path.name
            for path in (REPO_ROOT / "examples").iterdir()
            if path.is_dir() and (path / "problem.py").is_file()
        }
        self.assertEqual(sorted(on_disk), sorted(EXAMPLES))

    @pytest.mark.fast
    def test_no_example_ships_a_private_spec_file(self) -> None:
        """`config.yaml` was the composed-in-Python spec the hook replaced."""

        stragglers = sorted(
            str(path.relative_to(REPO_ROOT))
            for path in (REPO_ROOT / "examples").glob("*/config.yaml")
        )
        self.assertEqual(stragglers, [])

    def test_every_example_registers_a_generator_a_task_and_a_model(self) -> None:
        for name, (dataset, model, _) in EXAMPLES.items():
            with self.subTest(example=name):
                registered = _registrations_of(f"examples/{name}/problem.py")
                self.assertEqual(
                    registered,
                    {("generators", dataset), ("tasks", dataset), ("models", model)},
                )

    def test_importing_any_example_touches_no_registry(self) -> None:
        """So the sweep scripts, and this file, can read a ProblemSpec freely."""

        import sys

        for name in EXAMPLES:
            with self.subTest(example=name):
                before = _registry_snapshot()
                module = extensions._import_file(REPO_ROOT / "examples" / name / "problem.py")
                self.addCleanup(sys.modules.pop, module.__name__, None)
                self.assertEqual(_registry_snapshot(), before)
                self.assertTrue(callable(module.register))

    @pytest.mark.fast
    def test_every_generator_config_names_its_own_extension(self) -> None:
        for name, (dataset, _, settings) in EXAMPLES.items():
            for setting in settings:
                path = REPO_ROOT / "data" / "configs" / "examples" / f"{setting}.yaml"
                with self.subTest(config=str(path.relative_to(REPO_ROOT))):
                    config = yaml.safe_load(path.read_text(encoding="utf-8"))
                    self.assertEqual(config["dataset"]["name"], dataset)
                    self.assertEqual(
                        config["dataset"]["extensions"], [f"examples/{name}/problem.py"]
                    )
                    self.assertEqual(
                        config["dataset"]["output_dir"],
                        f"data/generated/examples/{setting}",
                    )
                    # The ratios describe a cut and there is none; every
                    # generator here writes all three splits itself.
                    self.assertNotIn("client_splits", config)

    def test_every_arm_config_loads_and_names_its_own_extension(self) -> None:
        for name, (dataset, model, settings) in EXAMPLES.items():
            for setting in settings:
                directory = REPO_ROOT / "configs" / "examples" / setting
                self.assertTrue(directory.is_dir(), f"{setting} ships no arm configs")
                for path in sorted(directory.glob("*.yaml")):
                    with self.subTest(config=str(path.relative_to(REPO_ROOT))):
                        config = load_config(path)
                        self.assertEqual(
                            config.experiment.extensions, [f"examples/{name}/problem.py"]
                        )
                        self.assertEqual(config.model.name, model)
                        # Derived from the model's registration, not stated.
                        self.assertEqual(config.task.name, dataset)
                        self.assertEqual(
                            config.data.path,
                            f"data/generated/examples/{setting}/manifest.json",
                        )

    @pytest.mark.fast
    def test_every_arm_config_sets_its_own_experiment_name(self) -> None:
        """The five ship the same arm names; the stem alone would collide."""

        seen: dict[str, str] = {}
        for _, (_, _, settings) in EXAMPLES.items():
            for setting in settings:
                for path in sorted((REPO_ROOT / "configs" / "examples" / setting).glob("*.yaml")):
                    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                    name = raw["experiment"].get("name")
                    relative = str(path.relative_to(REPO_ROOT))
                    with self.subTest(config=relative):
                        self.assertIsNotNone(name, "an arm config must set experiment.name")
                        self.assertNotIn(
                            name,
                            seen,
                            f"{relative} and {seen.get(name)} would share "
                            f"runs_index.jsonl entry {name!r}",
                        )
                    seen[name] = relative


def _registry_snapshot() -> dict[str, list[str]]:
    return {
        attribute: list(getattr(registry, attribute).list())
        for attribute in ("generators", "tasks", "models", "server_strategies", "datasets")
    }


def _registrations_of(entry: str) -> set[tuple[str, str]]:
    """Load one extension and return what it registered.

    An extension already loaded in this process -- by an arm config a test
    above loaded, or by another test module -- returns its existing record and
    is *not* torn down here. Undoing a load this module did not make would
    leave the registry names in place with the loader's memo gone, so the next
    config naming that file would call `register()` a second time and hit the
    duplicate-name refusal. Only what this function loads is undone.
    """

    resolved = str((REPO_ROOT / entry).resolve())
    existing = extensions._loaded.get(resolved)
    if existing is not None:
        return set(existing.registered)
    loaded = extensions.load_extensions([str(REPO_ROOT / entry)])[0]
    _CLEANUP.append(loaded)
    return set(loaded.registered)


#: The loads this module made, undone at teardown so the suite's other modules
#: see the registries they expect.
_CLEANUP: list[Any] = []


def tearDownModule() -> None:
    for loaded in _CLEANUP:
        for label, name in loaded.registered:
            component = getattr(registry, label)
            component._items.pop(name, None)
            component._origins.pop(name, None)
            component._config_keys.pop(name, None)
            registry.MODEL_TASKS.pop(name, None)
        extensions._loaded.pop(loaded.resolved, None)
    _CLEANUP.clear()


if __name__ == "__main__":
    unittest.main()
