"""Two paths that were inert, and would have misled the next edit.

**One optimizer, one home.** `_server_optimizer` returned `config.server
.strategy` for `fedadam` and its three siblings, so the factory decided the
optimizer for a named strategy; the `kwargs.setdefault("server_optimizer",
"fedadam")` in each alias builder could therefore never fire, and
`_validate_fedopt` warned when `server.server_optimizer` *differed* from the
strategy -- a warning about a value nothing read, and silence when it agreed.
Three pieces of machinery for one decision, two of them dead.

Now `_build_server` passes `server_optimizer` only for `strategy: fedopt`,
where the config genuinely chooses it, and each alias builder's `setdefault` is
the one place its optimizer is named. `server_optimizer` under a named strategy
is refused rather than warned about: a key its component never receives is what
`UNHONOURED_CLIENT_OPTIONS` already refuses on the client side.

**One registration, one origin.** `_register_once` skipped any name already
present, which bought idempotence -- `register_builtin_components()` is called
from a dozen entry points -- by also swallowing a registration made by someone
else. Extensions load at `load_config` time, *before* the builtins, so an
extension registering `"fedavg"` silently kept it and the package's own
strategy never appeared. Measured: the stub survived and `origin("fedavg")` read
`extensions[0]:my_ext.py`.

`Registry.register` already refuses a duplicate and names both origins -- its
contract is "names are registered once and never overwritten" -- so the fix is
to stop routing around it. Idempotence needs only `origin(name) == BUILTIN`.

Neither item changed a number: nothing shipped sets `server_optimizer`, and no
extension in this repo registers a built-in name. Both are guarded here because
what was wrong was the code reading as though it did something. See
FINDINGS.csv P10-F33.
"""

from __future__ import annotations

import copy
import glob
import inspect
import unittest

import pytest

from fedbrew.core import factory as factory_module
from fedbrew.core import registry as registry_module
from fedbrew.core.config import load_config
from fedbrew.core.factory import FEDOPT_SERVER_STRATEGIES, _server_optimizer
from fedbrew.core.registry import (
    BUILTIN,
    Registry,
    register_builtin_components,
    server_strategies,
)
from fedbrew.core.validation import validate_full_config
from fedbrew.servers.fedopt import unread_fedopt_hyperparameters

#: The four strategies that name their own optimizer.
NAMED = sorted(FEDOPT_SERVER_STRATEGIES - {"fedopt"})


def _stub(*args: object, **kwargs: object) -> None:
    raise AssertionError("an extension stub was built")


def _builtin_marker(*args: object, **kwargs: object) -> None:
    raise AssertionError("a stand-in for a package builder was built")


def _codes(config: object) -> list[tuple[str, str]]:
    return [(issue.severity, issue.code) for issue in validate_full_config(config).issues]


@pytest.mark.fast
class TheOptimizerHasOneHomeTest(unittest.TestCase):
    def test_each_alias_builder_names_its_own(self) -> None:
        for name in NAMED:
            with self.subTest(strategy=name):
                source = inspect.getsource(getattr(registry_module, f"_build_{name}_server"))
                self.assertIn(f'setdefault("server_optimizer", "{name}")', source)

    def test_the_setdefault_is_reached(self) -> None:
        """It was dead: the factory passed the key, so the default never fired."""

        source = inspect.getsource(factory_module._build_server)
        self.assertIn('config.server.strategy == "fedopt"', source)

    def test_building_a_named_strategy_gets_its_own_optimizer(self) -> None:
        register_builtin_components()
        for name in NAMED:
            with self.subTest(strategy=name):
                # None for the hyperparameters this optimizer never reads; the
                # constructor refuses a value it would ignore. P01-F07.
                _, unread = unread_fedopt_hyperparameters(name)
                server = server_strategies.get(name)(
                    participation_rate=1.0,
                    seed=0,
                    server_learning_rate=0.01,
                    beta1=0.9,
                    beta2=None if "beta2" in unread else 0.99,
                    tau=None if "tau" in unread else 1e-3,
                )
                self.assertEqual(server.server_optimizer, name)

    def test_the_factory_helper_now_answers_only_for_fedopt(self) -> None:
        config = load_config("configs/femnist/fedadam.yaml")
        config.server.strategy = "fedopt"
        config.server.extra["server_optimizer"] = "fedyogi"
        self.assertEqual(_server_optimizer(config), "fedyogi")

    def test_fedopt_without_one_is_still_refused(self) -> None:
        config = load_config("configs/femnist/fedadam.yaml")
        config.server.strategy = "fedopt"
        config.server.extra.pop("server_optimizer", None)
        with self.assertRaises(ValueError):
            _server_optimizer(config)


class AnUnreadOptimizerIsRefusedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config("configs/femnist/fedadam.yaml")

    def test_the_shipped_config_is_clean(self) -> None:
        self.assertNotIn("server_optimizer", self.config.server.extra)
        self.assertEqual([code for _, code in _codes(self.config) if "optimizer" in code], [])

    def test_setting_it_under_a_named_strategy_is_an_error(self) -> None:
        """Including when it agrees -- it is unread either way."""

        for value in ("fedadam", "fedyogi"):
            with self.subTest(server_optimizer=value):
                config = copy.deepcopy(self.config)
                config.server.extra["server_optimizer"] = value
                self.assertIn(("error", "algorithm.fedopt_optimizer_unread"), _codes(config))

    def test_the_old_warning_was_silent_when_it_agreed(self) -> None:
        """Which is what made it a warning about nothing: the mismatch was not
        the problem, the key being unread was."""

        config = copy.deepcopy(self.config)
        config.server.extra["server_optimizer"] = "fedadam"
        codes = [code for _, code in _codes(config)]
        self.assertNotIn("algorithm.fedopt_optimizer_mismatch", codes)

    def test_fedopt_still_requires_it(self) -> None:
        config = copy.deepcopy(self.config)
        config.server.strategy = "fedopt"
        config.server.extra.pop("server_optimizer", None)
        self.assertIn(("error", "algorithm.fedopt_optimizer_invalid"), _codes(config))

    def test_no_shipped_config_sets_it(self) -> None:
        offenders = [
            path
            for path in sorted(glob.glob("configs/**/*.yaml", recursive=True))
            if "server_optimizer" in _server_extra(path)
        ]
        self.assertEqual(offenders, [])


def _server_extra(path: str) -> dict:
    try:
        return dict(load_config(path).server.extra)
    except Exception:
        return {}


@pytest.mark.fast
class RegistrationIsIdempotentNotPermissiveTest(unittest.TestCase):
    """The narrow skip, and the shadow it used to swallow."""

    def setUp(self) -> None:
        register_builtin_components()
        self._added: list[str] = []

    def tearDown(self) -> None:
        for name in self._added:
            server_strategies._items.pop(name, None)
            server_strategies._origins.pop(name, None)
            server_strategies._config_keys.pop(name, None)

    def _register_stub(self, name: str) -> None:
        server_strategies.register(name, _stub, origin="extensions[0]:my_ext.py")
        self._added.append(name)

    def test_calling_it_twice_is_still_a_no_op(self) -> None:
        before = server_strategies.get("fedavg")
        register_builtin_components()
        self.assertIs(server_strategies.get("fedavg"), before)
        self.assertEqual(server_strategies.origin("fedavg"), BUILTIN)

    def test_an_extension_shadowing_a_builtin_is_refused(self) -> None:
        """On a fresh registry, because the order is the whole point.

        An extension registers at `load_config` time and the builtins after
        it, so the shadow can only be reproduced with the built-in *not yet*
        registered. By the time any test runs, the process-wide registry is
        already populated, so this uses its own.
        """

        registry: Registry[object] = Registry("server_strategies")
        registry.register("fedavg", _stub, origin="extensions[0]:my_ext.py")
        with self.assertRaises(ValueError) as caught:
            registry_module._register_once(registry, "fedavg", _builtin_marker)
        message = str(caught.exception)
        self.assertIn("fedavg", message)
        self.assertIn("my_ext.py", message)
        self.assertIn("the package", message)
        self.assertIs(registry.get("fedavg"), _stub, "the stub is still there to be seen")

    def test_the_old_form_would_have_kept_the_stub_silently(self) -> None:
        """`if not registry.exists(name)`, rebuilt, so the note is measured."""

        registry: Registry[object] = Registry("server_strategies")
        registry.register("fedavg", _stub, origin="extensions[0]:my_ext.py")
        if not registry.exists("fedavg"):  # pragma: no cover - the old body
            registry.register("fedavg", _builtin_marker)
        self.assertIs(registry.get("fedavg"), _stub)
        self.assertEqual(registry.origin("fedavg"), "extensions[0]:my_ext.py")

    def test_a_fresh_registry_takes_the_builtin_and_is_idempotent(self) -> None:
        registry: Registry[object] = Registry("server_strategies")
        registry_module._register_once(registry, "fedavg", _builtin_marker)
        registry_module._register_once(registry, "fedavg", _builtin_marker)
        self.assertIs(registry.get("fedavg"), _builtin_marker)
        self.assertEqual(registry.origin("fedavg"), BUILTIN)

    def test_an_extension_adding_a_new_name_is_untouched(self) -> None:
        self._register_stub("a_strategy_the_package_does_not_ship")
        register_builtin_components()
        self.assertEqual(
            server_strategies.origin("a_strategy_the_package_does_not_ship"),
            "extensions[0]:my_ext.py",
        )

    def test_the_generator_registry_survives_a_second_call(self) -> None:
        """It wraps what it is given, so an identity check would have raised.

        The first form of this fix compared `registry.get(name) is obj` as
        well; `GeneratorRegistry.register` stores a fresh `GeneratorSpec`, so
        all seven generators re-registered and the second call raised.
        """

        from fedbrew.core.registry import generators

        before = {name: generators.get(name) for name in generators.builtin()}
        self.assertTrue(before, "no built-in generators; the premise is untested")
        register_builtin_components()
        for name, spec in before.items():
            with self.subTest(generator=name):
                self.assertIs(generators.get(name), spec)


if __name__ == "__main__":
    unittest.main()
