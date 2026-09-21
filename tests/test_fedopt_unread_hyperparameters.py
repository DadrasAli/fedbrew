"""The FedOpt hyperparameters each optimizer reads, measured rather than listed.

`UNREAD_FEDOPT_HYPERPARAMETERS` claims that `fedavgm` never reads `beta2` or
`tau` and that `fedadagrad` never reads `beta2`. `validate_config` refuses a
config that sets one of them, so that table decides what a config may say. A
table that drifted from the updates would refuse a knob that matters, or keep
requiring one that does not -- which is the finding it exists to close.

So it is not read back here. Every optimizer is run twice under two values of
every hyperparameter, and the table is checked against which pairs moved the
model state. P01-F07.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from fedbrew.core.config import (
    FEDOPT_STRATEGIES,
    fedopt_optimizer_name,
    load_config,
    validate_config,
)
from fedbrew.core.validation import validate_full_config
from fedbrew.servers.fedopt import (
    FEDOPT_HYPERPARAMETERS,
    SUPPORTED_FEDOPT_OPTIMIZERS,
    UNREAD_FEDOPT_HYPERPARAMETERS,
    FedOptServer,
    unread_fedopt_hyperparameters,
)

BASE = {"server_learning_rate": 0.1, "beta1": 0.5, "beta2": 0.9, "tau": 0.01}
OTHER = {"server_learning_rate": 0.3, "beta1": 0.8, "beta2": 0.4, "tau": 0.05}

# Two rounds, because m starts at zeros: fedavgm's first update is delta
# whatever beta1 is, so a one-round probe would report beta1 unread.
DELTAS = (
    {"w": torch.tensor([0.3, -0.7, 0.2])},
    {"w": torch.tensor([-0.1, 0.4, 0.9])},
)

#: One shipped config per named FedOpt strategy, so the refusal is checked
#: against the schema a real arm goes through rather than a minimal one
#: written here.
STRATEGY_CONFIGS: dict[str, str] = {
    "fedavgm": "configs/examples/pl-1d/fedavgm.yaml",
    "fedadagrad": "configs/femnist/fedadagrad.yaml",
    "fedadam": "configs/femnist/fedadam.yaml",
    "fedyogi": "configs/femnist/fedyogi.yaml",
}


def _run(optimizer: str, values: dict[str, float]) -> torch.Tensor:
    """Two FedOpt updates under `values`, returning the resulting state."""

    _, unread = unread_fedopt_hyperparameters(optimizer)
    server = FedOptServer(
        server_optimizer=optimizer,
        participation_rate=1.0,
        seed=0,
        **{name: (None if name in unread else values[name]) for name in FEDOPT_HYPERPARAMETERS},
    )
    # What the table says nothing reads is set past the constructor's refusal:
    # the claim under test is about the update, not about the door.
    for name in unread:
        setattr(server, name, values[name])
    server._model_state = {"w": torch.zeros(3)}
    for delta in DELTAS:
        server._model_state = server._apply_fedopt_update(delta)
    return server._model_state["w"]


def _config(strategy: str) -> Any:
    config = load_config(STRATEGY_CONFIGS[strategy])
    assert config.server.strategy == strategy, f"{STRATEGY_CONFIGS[strategy]} is not {strategy}"
    return config


@pytest.mark.fast
class UnreadHyperparametersAreMeasuredTest(unittest.TestCase):
    """The table says what the updates do, or the guard fails."""

    def test_the_table_names_every_hyperparameter_that_changes_nothing(self) -> None:
        for optimizer in sorted(SUPPORTED_FEDOPT_OPTIMIZERS):
            _, unread = unread_fedopt_hyperparameters(optimizer)
            baseline = _run(optimizer, BASE)
            for name in FEDOPT_HYPERPARAMETERS:
                with self.subTest(optimizer=optimizer, hyperparameter=name):
                    moved = _run(optimizer, {**BASE, name: OTHER[name]})
                    if name in unread:
                        self.assertTrue(
                            torch.equal(baseline, moved),
                            f"{optimizer} is listed as never reading {name}, but "
                            f"{OTHER[name]} moved the state to {moved.tolist()} "
                            f"from {baseline.tolist()}",
                        )
                    else:
                        self.assertFalse(
                            torch.equal(baseline, moved),
                            f"{optimizer} is listed as reading {name}, but "
                            f"{OTHER[name]} left the state at {baseline.tolist()}",
                        )

    def test_every_row_names_a_supported_optimizer(self) -> None:
        self.assertLessEqual(set(UNREAD_FEDOPT_HYPERPARAMETERS), SUPPORTED_FEDOPT_OPTIMIZERS)
        for reason, unread in UNREAD_FEDOPT_HYPERPARAMETERS.values():
            self.assertTrue(reason)
            self.assertTrue(unread, "an empty row would read as checked-and-found-nothing")
            self.assertLessEqual(set(unread), set(FEDOPT_HYPERPARAMETERS))


@pytest.mark.fast
class TheServerRefusesWhatItWouldNotReadTest(unittest.TestCase):
    def _kwargs(self, optimizer: str, **overrides: float | None) -> dict[str, Any]:
        _, unread = unread_fedopt_hyperparameters(optimizer)
        kwargs: dict[str, Any] = {
            "server_optimizer": optimizer,
            "participation_rate": 1.0,
            "seed": 0,
        }
        kwargs.update(
            {name: (None if name in unread else BASE[name]) for name in FEDOPT_HYPERPARAMETERS}
        )
        kwargs.update(overrides)
        return kwargs

    def test_a_value_for_an_unread_hyperparameter_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            FedOptServer(**self._kwargs("fedavgm", beta2=0.9))
        self.assertIn("beta2", str(caught.exception))

    def test_a_missing_hyperparameter_it_reads_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            FedOptServer(**self._kwargs("fedadam", tau=None))
        self.assertIn("tau", str(caught.exception))

    def test_a_checkpoint_carries_only_what_the_optimizer_reads(self) -> None:
        state = FedOptServer(**self._kwargs("fedavgm")).save_state()
        self.assertNotIn("beta2", state)
        self.assertNotIn("tau", state)
        self.assertEqual(state["beta1"], BASE["beta1"])

    def test_a_stale_placeholder_in_an_old_checkpoint_neither_blocks_nor_loads(self) -> None:
        # Checkpoints written before this change carry beta2/tau for fedavgm.
        # Nothing read them then either, so a resume must not refuse over one.
        server = FedOptServer(**self._kwargs("fedavgm"))
        state = dict(server.save_state())
        state.update({"beta2": 0.99, "tau": 1e-8})
        server.load_state(state)
        self.assertIsNone(server.beta2)
        self.assertIsNone(server.tau)


class TheConfigRefusesAPlaceholderTest(unittest.TestCase):
    def test_every_shipped_fedopt_config_sets_exactly_what_it_reads(self) -> None:
        seen = 0
        for path in sorted(Path("configs").rglob("*.yaml")):
            # A run config is one with a runtime block; configs/llm_assets/
            # holds asset-preparation configs, a different schema entirely.
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict) or "runtime" not in loaded:
                continue
            config = load_config(str(path))
            if config.server.strategy not in FEDOPT_STRATEGIES:
                continue
            seen += 1
            _, unread = unread_fedopt_hyperparameters(fedopt_optimizer_name(config))
            with self.subTest(config=str(path)):
                self.assertEqual(
                    [name for name in unread if name in config.server.extra],
                    [],
                    "a shipped config carries a hyperparameter its optimizer never reads",
                )
                self.assertEqual(
                    [
                        name
                        for name in FEDOPT_HYPERPARAMETERS
                        if name not in unread and name not in config.server.extra
                    ],
                    [],
                    "a shipped config omits one its optimizer does read",
                )
        self.assertGreater(seen, 0)

    @pytest.mark.fast
    def test_every_fedopt_strategy_has_a_config_to_check(self) -> None:
        self.assertEqual(set(STRATEGY_CONFIGS) | {"fedopt"}, set(FEDOPT_STRATEGIES))

    def test_a_hyperparameter_the_optimizer_never_reads_is_refused(self) -> None:
        for strategy, unread in (("fedavgm", ("beta2", "tau")), ("fedadagrad", ("beta2",))):
            for name in unread:
                with self.subTest(strategy=strategy, hyperparameter=name):
                    config = _config(strategy)
                    config.server.extra[name] = 0.99
                    with self.assertRaises(ValueError) as caught:
                        validate_config(config)
                    self.assertIn(f"server.{name}", str(caught.exception))

    def test_a_hyperparameter_the_optimizer_reads_is_still_required(self) -> None:
        for strategy, read in (
            ("fedavgm", "beta1"),
            ("fedadagrad", "tau"),
            ("fedadam", "beta2"),
            ("fedyogi", "tau"),
        ):
            with self.subTest(strategy=strategy, hyperparameter=read):
                config = _config(strategy)
                del config.server.extra[read]
                with self.assertRaises(ValueError) as caught:
                    validate_config(config)
                self.assertIn(read, str(caught.exception))

    @pytest.mark.fast
    def test_the_bare_fedopt_spelling_is_judged_by_the_optimizer_it_names(self) -> None:
        config = _config("fedadam")
        config.server.strategy = "fedopt"
        # Case and whitespace are normalised the way FedOptServer normalises
        # them, so a loud spelling is judged as the same optimizer.
        config.server.extra["server_optimizer"] = " FedAvgM "
        self.assertEqual(fedopt_optimizer_name(config), "fedavgm")
        with self.assertRaises(ValueError) as caught:
            validate_config(config)
        self.assertIn("server.beta2", str(caught.exception))
        for name in ("beta2", "tau"):
            del config.server.extra[name]
        validate_config(config)

    @pytest.mark.fast
    def test_an_unnamed_optimizer_keeps_every_hyperparameter_required(self) -> None:
        """`fedopt` with no `server_optimizer` must not become the lenient case."""

        config = _config("fedadam")
        config.server.strategy = "fedopt"
        self.assertEqual(fedopt_optimizer_name(config), "fedopt")
        validate_config(config)
        del config.server.extra["beta2"]
        with self.assertRaises(ValueError) as caught:
            validate_config(config)
        self.assertIn("beta2", str(caught.exception))


class ValidateOnlyAgreesWithTheRunPathTest(unittest.TestCase):
    def test_preflight_reports_what_validate_config_refuses(self) -> None:
        config = _config("fedavgm")
        config.server.extra["beta2"] = 0.99
        report = validate_full_config(config)
        codes = [issue.code for issue in report.issues if issue.severity == "error"]
        self.assertIn("algorithm.fedopt_hyperparameter_unread", codes)

    def test_preflight_is_silent_on_a_shipped_config(self) -> None:
        codes = [
            issue.code
            for issue in validate_full_config(_config("fedadagrad")).issues
            if issue.code == "algorithm.fedopt_hyperparameter_unread"
        ]
        self.assertEqual(codes, [])


if __name__ == "__main__":
    unittest.main()
