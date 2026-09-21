"""Every rule refuses the engine options it would silently ignore.

`_training_client_kwargs` hands each of the nine engine options only to the
rules whose local step reads it. A rule outside the gate never receives the
key, so the option went nowhere and the run -- and its run.json -- reported a
momentum or a cosine decay that never happened.

This was guarded for `fedprox` and `scaffold` only, on the premise that every
other rule raises in its constructor instead. Four of them cannot: the
constructor tests an attribute the factory sets only for the rules that honour
the option, so for the rules that do not it is permanently the base class's
default and the check never fires. `delta_sgd + momentum: 0.9` and
`fedlalr + max_grad_norm: 10.0` both loaded and trained, the second leaving an
arm unclipped while its artifact claimed otherwise.

The right-hand column of `UNHONOURED_CLIENT_OPTIONS` is not asserted against a
copy written here. It is derived from `_training_client_kwargs` itself, for
every registered rule, so a gate that changes without the table changing fails.
"""

from __future__ import annotations

import copy
import pathlib
import unittest
from typing import Any

import pytest
import torch

from fedbrew.clients.torch_fedprox_client import TorchFedProxClient
from fedbrew.clients.torch_scaffold_client import TorchScaffoldClient
from fedbrew.core.config import (
    ENGINE_CLIENT_OPTIONS,
    UNHONOURED_CLIENT_OPTIONS,
    load_config,
    validate_config,
)
from fedbrew.core.factory import _training_client_kwargs
from fedbrew.core.protocol import FitRequest
from fedbrew.core.registry import client_updates, register_builtin_components
from tests.test_empty_training_batches import _data, _Task

#: One value per option, of the type the option takes. The value never reaches
#: a client -- validation is what is under test -- but it has to survive the
#: type checks that run before the refusal.
OPTION_VALUES: dict[str, Any] = {
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "nesterov": True,
    "learning_rate_schedule": "cosine",
    "min_learning_rate": 0.0,
    "update_mode": "sequential_epoch",
    "frozen_gradient_weighting": "examples",
    "max_local_steps": 3,
    "max_grad_norm": 1.0,
}

#: A shipped config per registered rule. Every rule needs one: the derivation
#: below runs the factory's gate on a real config, and a rule with no config
#: here is a rule this guard would otherwise skip in silence.
RULE_CONFIGS: dict[str, str] = {
    "centralized": "configs/femnist/centralized.yaml",
    "delta_sgd": "configs/femnist/delta_sgd.yaml",
    "fedavg": "configs/femnist/fedavg.yaml",
    "fedavg_ft": "configs/femnist/fedavg_ft.yaml",
    "fedlalr": "configs/femnist/fedlalr.yaml",
    "fedprox": "configs/femnist/fedprox.yaml",
    "local_adamw": "configs/dev/tiny_causal_lm.yaml",
    "local_sgd": "configs/dev/synthetic.yaml",
    "scaffold": "configs/femnist/scaffold.yaml",
}


class _StubDataset:
    """Only `get_client_data` is reached; the gate branches on the config."""

    def get_client_data(self, client_id: str) -> dict[str, Any]:
        return {}


def _forwarded(config: Any) -> set[str]:
    """The kwargs the factory would hand this config's client."""

    return set(_training_client_kwargs(config, object(), _StubDataset(), "c0", {}))


def _config(rule: str) -> Any:
    config = load_config(RULE_CONFIGS[rule])
    assert config.client.update_rule == rule, f"{RULE_CONFIGS[rule]} is not a {rule} config"
    return config


@pytest.mark.fast
class TableMatchesTheFactoryTest(unittest.TestCase):
    """The table says what the factory does, or the guard fails."""

    def test_every_registered_rule_has_a_config_to_check(self) -> None:
        register_builtin_components()
        self.assertEqual(
            set(client_updates.builtin()) - set(RULE_CONFIGS),
            set(),
            "a registered rule has no config here, so nothing checks its row",
        )

    def test_the_table_is_derived_from_the_gates(self) -> None:
        for rule in sorted(RULE_CONFIGS):
            with self.subTest(rule=rule):
                ignored = tuple(
                    name for name in ENGINE_CLIENT_OPTIONS if name not in _forwarded(_config(rule))
                )
                declared = UNHONOURED_CLIENT_OPTIONS.get(rule, ("", ()))[1]
                self.assertEqual(
                    tuple(declared),
                    ignored,
                    f"UNHONOURED_CLIENT_OPTIONS[{rule!r}] disagrees with what "
                    "_training_client_kwargs forwards",
                )

    def test_a_rule_that_honours_everything_has_no_row(self) -> None:
        """An empty row would read as "checked and found nothing"."""

        for rule, (_, ignored) in UNHONOURED_CLIENT_OPTIONS.items():
            with self.subTest(rule=rule):
                self.assertTrue(ignored, f"{rule} has an empty row; delete it instead")

    def test_the_table_names_only_engine_options(self) -> None:
        for rule, (_, ignored) in UNHONOURED_CLIENT_OPTIONS.items():
            with self.subTest(rule=rule):
                self.assertEqual(set(ignored) - set(ENGINE_CLIENT_OPTIONS), set())

    def test_every_engine_option_has_a_test_value(self) -> None:
        self.assertEqual(set(OPTION_VALUES), set(ENGINE_CLIENT_OPTIONS))


@pytest.mark.fast
class IgnoredOptionsAreRefusedTest(unittest.TestCase):
    def test_the_shipped_configs_still_load(self) -> None:
        """They document the omission in a comment; this makes it a check."""

        for rule in sorted(RULE_CONFIGS):
            with self.subTest(rule=rule):
                validate_config(_config(rule))

    def test_each_ignored_option_is_refused(self) -> None:
        for rule, (_, ignored) in sorted(UNHONOURED_CLIENT_OPTIONS.items()):
            for name in ignored:
                with self.subTest(rule=rule, option=name):
                    config = _config(rule)
                    config.client.extra[name] = OPTION_VALUES[name]
                    with self.assertRaises(ValueError) as caught:
                        validate_config(config)
                    message = str(caught.exception)
                    self.assertIn(f"client.{name}", message)
                    self.assertIn(rule, message)

    def test_the_options_a_rule_does_honour_are_untouched(self) -> None:
        """The guard must not reject what the rule's engine actually reads."""

        for rule in sorted(RULE_CONFIGS):
            honoured = set(ENGINE_CLIENT_OPTIONS) - set(
                UNHONOURED_CLIENT_OPTIONS.get(rule, ("", ()))[1]
            )
            for name in sorted(honoured):
                with self.subTest(rule=rule, option=name):
                    config = _config(rule)
                    config.client.extra[name] = OPTION_VALUES[name]
                    try:
                        validate_config(config)
                    except ValueError as error:
                        # A rule may still refuse an option it receives --
                        # fedavg requires momentum: 0.0 at the client. What it
                        # must not say is that the option is unhonourable.
                        self.assertNotIn("cannot honour", str(error))

    def test_the_pairs_that_used_to_load(self) -> None:
        """One per rule the constructor check could not reach.

        Each of these loaded, trained and was recorded in run.json as
        configured. `fedlalr + max_grad_norm` is the one that changes a number:
        the arm trained unclipped while its artifact said it was clipped.
        """

        for rule, option, value in (
            ("delta_sgd", "momentum", 0.9),
            ("fedlalr", "max_grad_norm", 10.0),
        ):
            with self.subTest(rule=rule, option=option):
                config = _config(rule)
                config.client.extra[option] = value
                with self.assertRaises(ValueError):
                    validate_config(config)

    def test_every_ignored_option_is_named_at_once(self) -> None:
        config = _config("fedprox")
        _, ignored = UNHONOURED_CLIENT_OPTIONS["fedprox"]
        config.client.extra.update({name: OPTION_VALUES[name] for name in ignored})
        with self.assertRaises(ValueError) as caught:
            validate_config(config)
        for name in ignored:
            self.assertIn(f"client.{name}", str(caught.exception))

    def test_the_non_engine_options_these_rules_read_are_untouched(self) -> None:
        config = _config("fedprox")
        config.client.extra.update(
            {"train_shuffle": False, "eval_shuffle": False, "drop_last": True}
        )
        validate_config(config)

    def test_fedavg_still_accepts_the_options_it_implements(self) -> None:
        config = load_config("configs/femnist/fedavg.yaml")
        config.client.extra["momentum"] = 0.9
        validate_config(config)


class _ScalerTask(_Task):
    """A task whose AMP path is engaged, as TorchClassificationTask's is."""

    _scaler = object()


def _request() -> FitRequest:
    torch.manual_seed(0)
    model = _Task().build_model()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return FitRequest(
        round_id=1,
        client_id="c0",
        payload={
            "model_state": state,
            "server_control": {k: torch.zeros_like(v) for k, v in state.items()},
        },
    )


class AmpIsNoLongerRefusedTest(unittest.TestCase):
    """Neither refuses it any more, and that is the change worth pinning.

    Both corrections live in a wrapper's `.step()`, and `GradScaler.step`
    unscales `.grad` before delegating to a wrapped optimizer -- so the
    correction sees true-scale gradients.
    `tests/test_amp_composes_with_wrapped_optimizers.py` records the
    measurement that lifted the refusal.
    """

    def _kwargs(self, task: Any) -> dict[str, Any]:
        return {
            "client_id": "c0",
            "task": task,
            "model_config": {},
            "client_data": copy.deepcopy(_data(8)),
            "local_iterations": 1,
            "batch_size": 4,
            "learning_rate": 0.1,
            "train_shuffle": False,
        }

    def test_fedprox_no_longer_refuses_amp(self) -> None:
        kwargs = self._kwargs(_ScalerTask()) | {"proximal_mu": 0.01}
        TorchFedProxClient(**kwargs).fit(_request())

    def test_scaffold_no_longer_refuses_amp(self) -> None:
        TorchScaffoldClient(**self._kwargs(_ScalerTask())).fit(_request())

    @pytest.mark.fast
    def test_neither_client_still_reads_the_scaler(self) -> None:
        """The backstop is gone from both, not merely bypassed."""

        for module in ("torch_scaffold_client", "torch_fedprox_client"):
            with self.subTest(module=module):
                source = (
                    pathlib.Path(__file__).resolve().parent.parent
                    / "fedbrew"
                    / "clients"
                    / f"{module}.py"
                ).read_text(encoding="utf-8")
                self.assertNotIn('getattr(self.task, "_scaler"', source)

    def test_both_train_normally_without_it(self) -> None:
        fedprox = TorchFedProxClient(**(self._kwargs(_Task()) | {"proximal_mu": 0.01}))
        self.assertGreater(fedprox.fit(_request()).num_examples, 0)
        scaffold = TorchScaffoldClient(**self._kwargs(_Task()))
        self.assertGreater(scaffold.fit(_request()).num_examples, 0)


@pytest.mark.fast
class EveryForwardedKeywordIsAcceptedTest(unittest.TestCase):
    """The mirror of this module's other guard, and the same defect class.

    `UNHONOURED_CLIENT_OPTIONS` covers the option a rule never receives. This
    covers the option a rule always receives and cannot take: the factory
    passes `eval_batch_size` to every rule unconditionally, and
    `TorchAdamWClient` declared neither the parameter nor a forward to the base
    class that does. Every other rule accepted it and this one raised TypeError
    before the first round, so `update_rule: local_adamw` could not start a run at all --
    including the two shipped configs that select it.

    It went unseen because the only tests that construct a client through the
    factory on that rule are gated behind the `llm` extra, which no CI job and
    no default environment installed. The `llm` job now runs them; this guard
    is the cheap check that does not need the extra at all.

    Nothing here is hand-listed. The keywords come from
    `_training_client_kwargs` for each rule's own shipped config, so a keyword
    added to the factory tomorrow is checked against all nine rules without an
    edit here.
    """

    def _reject_reason(self, rule: str, keywords: set[str]) -> str | None:
        """The TypeError text if the rule cannot bind these keywords."""

        try:
            client_updates.get(rule)(**dict.fromkeys(keywords))
        except TypeError as error:
            if "unexpected keyword argument" in str(error):
                return str(error)
        except Exception:
            # Bound fine and then failed on the sentinel values, which is the
            # expected outcome: this guard is about the signature, not the body.
            return None
        return None

    def test_every_rule_accepts_every_keyword_the_factory_sends_it(self) -> None:
        register_builtin_components()
        for rule in sorted(RULE_CONFIGS):
            with self.subTest(rule=rule):
                forwarded = _forwarded(_config(rule))
                self.assertTrue(forwarded, "the scan produced no keywords to check")
                reason = self._reject_reason(rule, forwarded)
                self.assertIsNone(
                    reason,
                    f"the factory sends {rule} a keyword its client cannot take, "
                    f"so no run on that rule can start: {reason}",
                )

    def test_eval_batch_size_is_one_of_them(self) -> None:
        """Named directly, because it is the keyword this guard was written for."""

        for rule in sorted(RULE_CONFIGS):
            with self.subTest(rule=rule):
                self.assertIn("eval_batch_size", _forwarded(_config(rule)))

    def test_the_check_would_notice_a_rule_that_dropped_one(self) -> None:
        """Guards the guard: a narrowed constructor must be reported."""

        forwarded = _forwarded(_config("local_adamw"))
        narrowed = forwarded - {"eval_batch_size"}
        self.assertIsNone(self._reject_reason("local_adamw", narrowed))
        self.assertIsNotNone(
            self._reject_reason("local_adamw", forwarded | {"a_keyword_no_client_takes"}),
            "the check cannot detect a keyword the client refuses",
        )


if __name__ == "__main__":
    unittest.main()
