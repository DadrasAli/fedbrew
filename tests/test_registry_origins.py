"""Every registered name carries where it came from.

Two readers need that. The documentation guards diff a chapter against what
``register_builtin_components`` makes, and once an extension can register
into the same registries in the same process, "everything registered" and
"what the package ships" stop being the same set; ``builtin()`` is the second
one. And a duplicate registration is a conflict between two sources, so the
error has to name both -- "already registered" on its own tells a reader who
loaded one file that some other file they did not name got there first, and
nothing else.

The model registry carries one more fact per name: the task adapter the model
needs. It used to be a static dict beside the registry, filled by hand in the
same commit as the registration and by nothing else, which is a closed table
an out-of-tree model cannot appear in. Registration now requires it.

The generator registry is the sixth. ``fedbrew generate`` dispatched through
two closed dicts -- the module per name and the sections per name -- and so
could not reach a generator defined outside the package; it now dispatches
through ``GeneratorSpec`` entries the same way the other five dispatch.
"""

from __future__ import annotations

import unittest

import pytest

from fedbrew.core import registry
from fedbrew.core.registry import (
    BUILTIN,
    GeneratorSpec,
    ModelRegistry,
    Registry,
    register_builtin_components,
    registering_from,
    task_for_model,
)

pytestmark = pytest.mark.fast


class OriginTest(unittest.TestCase):
    def test_a_name_registered_outside_any_block_is_builtin(self) -> None:
        items: Registry[int] = Registry("items")
        items.register("one", 1)
        self.assertEqual(items.origin("one"), BUILTIN)
        self.assertEqual(items.builtin(), ["one"])

    def test_a_name_registered_inside_a_block_carries_the_block_origin(self) -> None:
        items: Registry[int] = Registry("items")
        with registering_from("examples/thing/problem.py") as recorded:
            items.register("two", 2)
        self.assertEqual(items.origin("two"), "examples/thing/problem.py")
        self.assertEqual(recorded, [("items", "two")])
        self.assertEqual(items.list(), ["two"])
        self.assertEqual(items.builtin(), [])

    def test_an_explicit_origin_wins_over_the_block(self) -> None:
        items: Registry[int] = Registry("items")
        with registering_from("outer.py"):
            items.register("three", 3, origin="inner.py")
        self.assertEqual(items.origin("three"), "inner.py")

    def test_blocks_nest_and_unwind(self) -> None:
        items: Registry[int] = Registry("items")
        with registering_from("outer.py"):
            with registering_from("inner.py"):
                items.register("a", 1)
            items.register("b", 2)
        items.register("c", 3)
        self.assertEqual(
            [items.origin(name) for name in ("a", "b", "c")],
            ["inner.py", "outer.py", BUILTIN],
        )

    def test_the_block_unwinds_on_an_exception(self) -> None:
        items: Registry[int] = Registry("items")
        with self.assertRaises(RuntimeError):
            with registering_from("failing.py"):
                raise RuntimeError("boom")
        items.register("after", 1)
        self.assertEqual(items.origin("after"), BUILTIN)

    def test_a_blank_origin_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            with registering_from("  "):
                pass

    def test_a_blank_name_is_refused(self) -> None:
        items: Registry[int] = Registry("items")
        with self.assertRaises(ValueError) as caught:
            items.register("  ", 1)
        self.assertIn("items", str(caught.exception))


class DeclaredConfigKeysTest(unittest.TestCase):
    def test_keys_are_stored_and_empty_by_default(self) -> None:
        items: Registry[int] = Registry("server_strategies", config_section="server")
        items.register("plain", 1)
        items.register("knobbed", 2, config_keys={"knob", "dial"})
        self.assertEqual(items.config_keys("plain"), frozenset())
        self.assertEqual(items.config_keys("knobbed"), frozenset({"knob", "dial"}))

    def test_a_registry_with_no_config_section_refuses_a_declaration(self) -> None:
        """A key declared where nothing forwards it is the failure chapter 12
        calls the worst one; the registry refuses to hold it."""

        items: Registry[int] = Registry("tasks")
        with self.assertRaises(ValueError) as caught:
            items.register("t", 1, config_keys={"knob"})
        self.assertIn("nothing would forward", str(caught.exception))

    def test_a_bare_string_is_not_a_key_set(self) -> None:
        items: Registry[int] = Registry("server_strategies", config_section="server")
        with self.assertRaises(ValueError):
            items.register("s", 1, config_keys="knob")

    def test_the_three_registries_with_a_free_block_have_a_section(self) -> None:
        self.assertEqual(registry.server_strategies.config_section, "server")
        self.assertEqual(registry.client_updates.config_section, "client")
        self.assertEqual(registry.datasets.config_section, "data")
        for component in (registry.tasks, registry.models, registry.generators):
            self.assertIsNone(component.config_section)


class DuplicateTest(unittest.TestCase):
    def test_the_error_names_both_origins(self) -> None:
        items: Registry[int] = Registry("tasks")
        items.register("dup", 1)
        with registering_from("second.py"):
            with self.assertRaises(ValueError) as caught:
                items.register("dup", 2)
        message = str(caught.exception)
        self.assertIn("tasks already holds 'dup'", message)
        self.assertIn("registered by the package", message)
        self.assertIn("from second.py", message)
        self.assertEqual(items.get("dup"), 1, "the first registration stands")

    def test_two_extensions_colliding_name_each_other(self) -> None:
        items: Registry[int] = Registry("models")
        with registering_from("first.py"):
            items.register("dup", 1)
        with registering_from("second.py"):
            with self.assertRaises(ValueError) as caught:
                items.register("dup", 2)
        message = str(caught.exception)
        self.assertIn("registered by first.py", message)
        self.assertIn("from second.py", message)


class ListingTest(unittest.TestCase):
    def test_builtins_first_then_each_extension_under_its_origin(self) -> None:
        items: Registry[int] = Registry("items")
        items.register("b", 1)
        items.register("a", 2)
        with registering_from("x.py"):
            items.register("z", 3)
            items.register("y", 4)
        with registering_from("w.py"):
            items.register("q", 5)
        self.assertEqual(items.listing(), "a, b; from x.py: y, z; from w.py: q")

    def test_an_empty_registry_lists_nothing(self) -> None:
        self.assertEqual(Registry("items").listing(), "")

    def test_the_unknown_name_message_lists_extensions_with_their_origin(self) -> None:
        """The message a reader who mistyped sees says where each name came from."""

        from fedbrew.core.config import _validate_registered_names, load_config

        register_builtin_components()
        with registering_from("examples/thing/problem.py"):
            registry.tasks.register("origin_probe_task", lambda **kwargs: None)
        self.addCleanup(registry.tasks._items.pop, "origin_probe_task", None)
        self.addCleanup(registry.tasks._origins.pop, "origin_probe_task", None)

        config = load_config("configs/dev/smoke.yaml")
        config.task.name = "definitely_not_registered"
        with self.assertRaises(ValueError) as caught:
            _validate_registered_names(config)
        message = str(caught.exception)
        self.assertIn("classification", message)
        self.assertIn("from examples/thing/problem.py: origin_probe_task", message)


class ModelRegistryTest(unittest.TestCase):
    def test_registration_requires_the_task(self) -> None:
        models = ModelRegistry("models")
        with self.assertRaises(TypeError):
            models.register("m", lambda config: None)  # type: ignore[call-arg]
        with self.assertRaises(ValueError):
            models.register("m", lambda config: None, task="")

    def test_registration_fills_model_tasks(self) -> None:
        register_builtin_components()
        with registering_from("probe.py"):
            registry.models.register("model_tasks_probe", lambda config: None, task="probe_task")
        self.addCleanup(registry.models._items.pop, "model_tasks_probe", None)
        self.addCleanup(registry.models._origins.pop, "model_tasks_probe", None)
        self.addCleanup(registry.MODEL_TASKS.pop, "model_tasks_probe", None)
        self.assertEqual(task_for_model("model_tasks_probe"), "probe_task")
        self.assertNotIn("model_tasks_probe", registry.models.builtin())

    def test_the_builtin_pairings_are_the_documented_eight(self) -> None:
        register_builtin_components()
        pairings = {name: registry.MODEL_TASKS[name] for name in registry.models.builtin()}
        self.assertEqual(
            pairings,
            {
                "mlp": "classification",
                "cnn": "classification",
                "small_cnn": "classification",
                "femnist_resnet18": "classification",
                "openimage_shufflenet": "classification",
                "tiny_gpt2": "causal_lm",
                "hf_causal_lm": "causal_lm",
                "hf_causal_lm_lora": "causal_lm",
            },
        )

    def test_task_for_model_registers_the_builtins_itself(self) -> None:
        """A caller needs no ordering knowledge: the lookup fills the table."""

        self.assertEqual(task_for_model("mlp"), "classification")

    def test_an_unknown_model_names_the_known_ones(self) -> None:
        with self.assertRaises(ValueError) as caught:
            task_for_model("definitely_not_a_model")
        self.assertIn("mlp", str(caught.exception))


class GeneratorSpecTest(unittest.TestCase):
    def test_a_target_string_resolves_lazily(self) -> None:
        spec = GeneratorSpec("json:dumps", {"a"})
        import json

        self.assertIs(spec.resolve(), json.dumps)

    def test_a_callable_target_is_returned_as_is(self) -> None:
        def generate() -> None:
            pass

        self.assertIs(GeneratorSpec(generate, set()).resolve(), generate)

    def test_the_sections_are_frozen(self) -> None:
        spec = GeneratorSpec("json:dumps", ["a", "b"])
        self.assertEqual(spec.sections, frozenset({"a", "b"}))

    def test_a_bad_kind_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            GeneratorSpec("json:dumps", set(), kind="delegated")

    def test_a_bad_target_string_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            GeneratorSpec("json.dumps", set())

    def test_a_target_that_is_not_callable_is_refused_at_resolve(self) -> None:
        with self.assertRaises(TypeError):
            GeneratorSpec("json:__name__", set()).resolve()

    def test_a_generator_is_registered_from_its_parts(self) -> None:
        """The line an extension author writes: the callable and its sections."""

        registry.register_builtin_components()

        def generate() -> None:
            pass

        with registering_from("probe.py"):
            registry.generators.register("spec_probe", generate, sections={"probe"})
        self.addCleanup(registry.generators._items.pop, "spec_probe", None)
        self.addCleanup(registry.generators._origins.pop, "spec_probe", None)
        self.addCleanup(registry.generators._config_keys.pop, "spec_probe", None)

        spec = registry.generators.get("spec_probe")
        self.assertIs(spec.resolve(), generate)
        self.assertEqual(spec.sections, frozenset({"probe"}))
        self.assertEqual(spec.kind, "shards")

    def test_a_generator_declares_its_sections_keys_with_them(self) -> None:
        """The out-of-tree form: a mapping, section -> the keys it reads."""

        from fedbrew.core.registry import GeneratorRegistry

        def generate() -> None:
            pass

        registered = GeneratorRegistry("generators")
        registered.register("keyed", generate, sections={"problem": {"dim", "kappa"}})

        spec = registered.get("keyed")
        self.assertEqual(spec.sections, frozenset({"problem"}))
        self.assertEqual(spec.section_keys, {"problem": frozenset({"dim", "kappa"})})
        with self.assertRaises(TypeError):
            spec.section_keys["problem"] = frozenset()  # type: ignore[index]

    def test_a_section_with_no_keys_is_refused_at_registration(self) -> None:
        """`{"problem": set()}` would declare a section that reads nothing."""

        from fedbrew.core.registry import GeneratorRegistry

        registered = GeneratorRegistry("generators")
        for bad in ({"problem": set()}, {"problem": "dim"}, {"problem": {""}}):
            with self.subTest(sections=bad), self.assertRaises(ValueError) as caught:
                registered.register("keyed", "json:dumps", sections=bad)  # type: ignore[arg-type]
            self.assertIn("problem", str(caught.exception))

    def test_a_spec_and_loose_sections_together_are_refused(self) -> None:
        from fedbrew.core.registry import GeneratorRegistry

        registered = GeneratorRegistry("generators")
        with self.assertRaises(ValueError) as caught:
            registered.register("both", GeneratorSpec("json:dumps", {"a"}), sections={"b"})
        self.assertIn("already carries them", str(caught.exception))

    def test_a_generator_declares_no_run_config_keys(self) -> None:
        from fedbrew.core.registry import GeneratorRegistry

        registered = GeneratorRegistry("generators")
        with self.assertRaises(ValueError):
            registered.register("g", "json:dumps", config_keys={"knob"})

    def test_the_builtin_eight_resolve_and_declare_the_sections_they_read(self) -> None:
        """The sections were a dict beside the dispatch; this pins them.

        `client_splits` appears on the three `shards` generators that cut by it
        and on neither SFT one, whose splits come from `tree_splits`. It is
        absent from all four `tensors` rows on purpose: the shared writer cuts
        those splits, not the generator, so `_sections_read_by` grants it there
        rather than each spec repeating it.
        """

        register_builtin_components()
        expected = {
            "synthetic_classification": ({"synthetic", "splits"}, "tensors"),
            "mnist": ({"mnist"}, "tensors"),
            "cifar10": ({"cifar10"}, "tensors"),
            "femnist": ({"femnist", "client_splits"}, "shards"),
            "tiny_causal_lm": ({"causal_lm", "splits", "client_splits"}, "shards"),
            "generic_sft": ({"generic_sft", "caps", "tree_splits"}, "shards"),
            "hf_causal_lm_text": (
                {"hf_causal_lm_text", "causal_lm", "splits", "source_splits", "client_splits"},
                "shards",
            ),
            "oasst1_sft": ({"oasst1_sft", "sft", "pilot_caps", "tree_splits", "splits"}, "shards"),
        }
        self.assertEqual(set(registry.generators.builtin()), set(expected))
        for name, (sections, kind) in expected.items():
            with self.subTest(generator=name):
                spec = registry.generators.get(name)
                self.assertEqual(spec.sections, frozenset(sections))
                self.assertEqual(spec.kind, kind)
                self.assertTrue(callable(spec.resolve()))


if __name__ == "__main__":
    unittest.main()
