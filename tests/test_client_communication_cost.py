"""What a client meters must be what its payload holds, for every rule.

`communicated_bytes` is the only quantity that makes two arms with different
per-round payloads comparable, and the failure mode is silent by construction:
counting one state out of two produces a number of the right order of
magnitude, in the right units, that nothing else contradicts.

The SCAFFOLD client says so in a comment and counts both of its states.
FedLALR counts all three.

So the assertion here is not a table of expected multipliers. It is the
standard itself, applied to every registered rule: run `fit`, find every
model-shaped state in the payload it returned, and require the meter to equal
their sum. A rule that grows a second state and does not count it fails
without anyone remembering to add a row.
"""

from __future__ import annotations

import copy
import unittest
from collections.abc import Mapping
from typing import Any

import pytest
import torch
from torch import Tensor

from fedbrew.clients.fedavg_client import FedAvgClient
from fedbrew.clients.torch_adamw_client import TorchAdamWClient
from fedbrew.clients.torch_delta_sgd_client import (
    DEFAULT_DELTA,
    DEFAULT_GAMMA,
    DEFAULT_THETA_0,
    TorchDeltaSGDClient,
)
from fedbrew.clients.torch_fedlalr_client import (
    DEFAULT_BETA1,
    DEFAULT_BETA2,
    DEFAULT_EPSILON,
    TorchFedLALRClient,
)
from fedbrew.clients.torch_fedprox_client import TorchFedProxClient
from fedbrew.clients.torch_scaffold_client import TorchScaffoldClient
from fedbrew.clients.torch_sgd_client import TorchSGDClient
from fedbrew.core.federated_state import model_state_size
from fedbrew.core.protocol import FitRequest
from fedbrew.core.registry import client_updates, register_builtin_components
from tests.test_empty_training_batches import _data, _Task

COST_METRICS = ("communicated_parameters", "communicated_bytes")

#: The base client refuses to build without these, and every rule that reads
#: them wants the neutral values: the point here is the payload, not the step.
SGD_SETTINGS: dict[str, Any] = {
    "momentum": 0.0,
    "weight_decay": 0.0,
    "nesterov": False,
    "learning_rate_schedule": "constant",
    "min_learning_rate": 0.0,
}
UPDATE_MODE_SETTINGS: dict[str, Any] = {
    "update_mode": "sequential_epoch",
    "frozen_gradient_weighting": "examples",
}


def _kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "client_id": "c0",
        "task": _Task(),
        "model_config": {},
        "client_data": copy.deepcopy(_data(8)),
        "local_iterations": 1,
        "batch_size": 4,
        "learning_rate": 0.1,
        "train_shuffle": False,
        "metrics": list(COST_METRICS),
    }
    kwargs.update(overrides)
    return kwargs


#: One builder per registered update rule. `centralized` and `fedavg_ft` build
#: the same clients as `fedavg` through the registry, so they are named here
#: against the same class rather than skipped.
BUILDERS = {
    "local_sgd": lambda: TorchSGDClient(**_kwargs(**SGD_SETTINGS)),
    "fedavg": lambda: FedAvgClient(**_kwargs(**SGD_SETTINGS), **UPDATE_MODE_SETTINGS),
    "centralized": lambda: FedAvgClient(**_kwargs(**SGD_SETTINGS), **UPDATE_MODE_SETTINGS),
    "fedavg_ft": lambda: FedAvgClient(**_kwargs(**SGD_SETTINGS), **UPDATE_MODE_SETTINGS),
    "local_adamw": lambda: TorchAdamWClient(
        **_kwargs(
            weight_decay=0.0,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            learning_rate_schedule="constant",
            min_learning_rate=0.0,
            total_rounds=1,
        )
    ),
    "fedprox": lambda: TorchFedProxClient(**_kwargs(), proximal_mu=0.01),
    "scaffold": lambda: TorchScaffoldClient(**_kwargs()),
    "delta_sgd": lambda: TorchDeltaSGDClient(
        **_kwargs(),
        **UPDATE_MODE_SETTINGS,
        eta_0=0.2,
        theta_0=DEFAULT_THETA_0,
        gamma=DEFAULT_GAMMA,
        delta=DEFAULT_DELTA,
    ),
    "fedlalr": lambda: TorchFedLALRClient(
        **_kwargs(), beta1=DEFAULT_BETA1, beta2=DEFAULT_BETA2, epsilon=DEFAULT_EPSILON
    ),
}


def _request() -> FitRequest:
    """One request every rule accepts: the model, plus the states each reads.

    A rule that reads none of the extras ignores them, and the payload it
    *returns* is what the assertions look at.
    """

    torch.manual_seed(0)
    model = _Task().build_model()
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    zeros = {name: torch.zeros_like(value) for name, value in state.items()}
    return FitRequest(
        round_id=1,
        client_id="c0",
        payload={
            "model_state": state,
            "server_control": zeros,
            "momentum_state": zeros,
            "second_moment_state": zeros,
        },
    )


def model_shaped_states(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Tensor]]:
    """Every value in the payload that is a state dict of tensors.

    Structural, not a list of known key names: a new state named anything at
    all is counted, which is the whole point. `model_state_metadata` and
    `num_examples_by_split` are mappings too and are excluded because their
    values are not tensors.
    """

    states: dict[str, Mapping[str, Tensor]] = {}
    for name, value in payload.items():
        if not isinstance(value, Mapping) or not value:
            continue
        if all(isinstance(tensor, Tensor) for tensor in value.values()):
            states[name] = value
    return states


class MeterMatchesPayloadTest(unittest.TestCase):
    @pytest.mark.fast
    def test_every_registered_rule_has_a_builder(self) -> None:
        register_builtin_components()
        self.assertEqual(
            set(client_updates.builtin()) - set(BUILDERS),
            set(),
            "a registered rule builds no client here, so its meter is unchecked",
        )

    def test_the_meter_equals_every_state_in_the_payload(self) -> None:
        for rule, build in sorted(BUILDERS.items()):
            with self.subTest(rule=rule):
                result = build().fit(_request())
                states = model_shaped_states(result.payload)
                self.assertTrue(states, f"{rule} returned no model-shaped state")

                sizes = [model_state_size(state) for state in states.values()]
                self.assertEqual(
                    result.metrics["communicated_parameters"],
                    float(sum(parameters for parameters, _ in sizes)),
                    f"{rule} meters a different parameter count than it uploads",
                )
                self.assertEqual(
                    result.metrics["communicated_bytes"],
                    float(sum(num_bytes for _, num_bytes in sizes)),
                    f"{rule} meters {sorted(states)} as though it uploaded fewer",
                )

    def test_the_multipliers_chapter_07_lists_are_measured(self) -> None:
        """Section 5's table, against what a round actually moves."""

        expected = {
            "local_sgd": 1,
            "fedavg": 1,
            "centralized": 1,
            "fedavg_ft": 1,
            "local_adamw": 1,
            "fedprox": 1,
            "delta_sgd": 1,
            "scaffold": 2,
            "fedlalr": 3,
        }
        self.assertEqual(set(expected), set(BUILDERS))

        _, model_bytes = model_state_size(_request().payload["model_state"])
        for rule, multiplier in sorted(expected.items()):
            with self.subTest(rule=rule):
                result = BUILDERS[rule]().fit(_request())
                self.assertEqual(
                    result.metrics["communicated_bytes"],
                    float(multiplier * model_bytes),
                    f"chapter 07 section 5 says {rule} moves {multiplier}x the model",
                )


if __name__ == "__main__":
    unittest.main()
