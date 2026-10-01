"""A FedOpt server's update is a function of the round's fold, equal to the one it always made.

``FedOptServer.update_from_fold`` is what the resident round runs where the fold
is (``fedbrew/core/resident.py``), and what ``aggregate_stream`` runs on the host's
tensors: one set of functions, written without a move to the CPU. This module holds
it to the update as it was written before -- every step through ``torch_utils``' CPU
helpers, kept below as ``reference`` -- bit for bit, for every optimizer in both
float widths over several rounds on tensors of several shapes, and holds what the
resident round relies on of a function: it reads and writes no server state, leaves
its arguments alone, and starts from no moments as the host's first update does.
"""

from __future__ import annotations

import unittest
from typing import cast

import pytest
import torch

from fedbrew.core.torch_utils import (
    StateDict,
    add_model_states,
    divide_model_states,
    scale_model_state,
    sqrt_model_state,
    subtract_model_states,
    zeros_like_model_state,
)
from fedbrew.servers.fedavg import FedAvgServer
from fedbrew.servers.fedopt import FedOptServer, unread_fedopt_hyperparameters

pytestmark = pytest.mark.fast

OPTIMIZERS = ("fedavgm", "fedadam", "fedyogi", "fedadagrad")
SHAPES = {"weight": (5, 3), "bias": (3,), "scalar": ()}


def _server(optimizer: str) -> FedOptServer:
    _, unread = unread_fedopt_hyperparameters(optimizer)
    return FedOptServer(
        server_optimizer=optimizer,
        server_learning_rate=0.7,
        beta1=0.9,
        beta2=None if "beta2" in unread else 0.95,
        tau=None if "tau" in unread else 0.01,
        participation_rate=1.0,
        seed=0,
    )


def _state(generator: torch.Generator, dtype: torch.dtype) -> StateDict:
    return {
        name: torch.randn(shape, generator=generator, dtype=dtype) for name, shape in SHAPES.items()
    }


def reference(
    server: FedOptServer,
    model: StateDict,
    fold: StateDict,
    m: StateDict | None,
    v: StateDict | None,
) -> tuple[StateDict, StateDict, StateDict | None]:
    """The update as written before it was a function: its own helpers, on the CPU."""

    delta = subtract_model_states(fold, model)
    if m is None:
        m = zeros_like_model_state(delta)
    beta1, lr = server.beta1, server.server_learning_rate
    if server.server_optimizer == "fedavgm":
        m = add_model_states(scale_model_state(m, beta1), delta)
        return add_model_states(model, scale_model_state(m, lr)), m, None
    tau = cast(float, server.tau)
    if v is None:
        v = {key: value + tau**2 for key, value in zeros_like_model_state(delta).items()}
    squared = {key: value * value for key, value in delta.items()}
    m = add_model_states(scale_model_state(m, beta1), scale_model_state(delta, 1.0 - beta1))
    if server.server_optimizer == "fedadagrad":
        v = add_model_states(v, squared)
    elif server.server_optimizer == "fedadam":
        beta2 = cast(float, server.beta2)
        v = add_model_states(scale_model_state(v, beta2), scale_model_state(squared, 1.0 - beta2))
    else:
        beta2 = cast(float, server.beta2)
        signed = {
            key: squared[key] * torch.sign(value)
            for key, value in subtract_model_states(v, squared).items()
        }
        v = subtract_model_states(v, scale_model_state(signed, 1.0 - beta2))
    step = divide_model_states(m, sqrt_model_state(v), eps=tau)
    return add_model_states(model, scale_model_state(step, lr)), m, v


class TheUpdateIsWhatItWasTest(unittest.TestCase):
    def test_every_optimizer_in_both_widths_over_rounds(self) -> None:
        for optimizer in OPTIMIZERS:
            for dtype in (torch.float32, torch.float64):
                with self.subTest(optimizer=optimizer, dtype=dtype):
                    generator = torch.Generator().manual_seed(11)
                    server = _server(optimizer)
                    model = reference_model = _state(generator, dtype)
                    carried = None
                    m = v = None
                    for _ in range(5):
                        fold = _state(generator, dtype)
                        model, carried = server.update_from_fold(model, fold, carried)
                        reference_model, m, v = reference(server, reference_model, fold, m, v)
                        self.assertEqual(list(model), list(reference_model))
                        for key in model:
                            self.assertTrue(torch.equal(model[key], reference_model[key]), key)
                            self.assertTrue(torch.equal(carried["m"][key], m[key]), key)
                            if v is not None:
                                self.assertTrue(torch.equal(carried["v"][key], v[key]), key)
                        self.assertEqual(set(carried), {"m"} if v is None else {"m", "v"})

    def test_the_host_path_is_the_same_update(self) -> None:
        for optimizer in OPTIMIZERS:
            with self.subTest(optimizer=optimizer):
                generator = torch.Generator().manual_seed(3)
                host, functional = _server(optimizer), _server(optimizer)
                model = _state(generator, torch.float64)
                host._model_state = dict(model)
                carried = None
                for step in range(4):
                    fold = _state(generator, torch.float64)
                    delta = subtract_model_states(fold, host._model_state)
                    host._model_state = host._apply_fedopt_update(delta)
                    model, carried = functional.update_from_fold(model, fold, carried)
                    for key in model:
                        self.assertTrue(torch.equal(host._model_state[key], model[key]), key)
                    self.assertEqual(host._update_step, step + 1)
                    self.assertEqual(host.carried_state().keys(), carried.keys())
                    for name in carried:
                        for key in model:
                            self.assertTrue(
                                torch.equal(host.carried_state()[name][key], carried[name][key])
                            )


class AFunctionTest(unittest.TestCase):
    def test_it_reads_and_writes_no_server_state_and_leaves_its_arguments(self) -> None:
        for optimizer in OPTIMIZERS:
            with self.subTest(optimizer=optimizer):
                generator = torch.Generator().manual_seed(5)
                server = _server(optimizer)
                model, fold = _state(generator, torch.float64), _state(generator, torch.float64)
                _, carried = server.update_from_fold(model, fold, None)
                kept = {
                    "model": {k: t.clone() for k, t in model.items()},
                    "fold": {k: t.clone() for k, t in fold.items()},
                    "carried": {
                        n: {k: t.clone() for k, t in s.items()} for n, s in carried.items()
                    },
                }
                server.update_from_fold(model, fold, carried)
                self.assertIsNone(server._model_state)
                self.assertIsNone(server._m)
                self.assertIsNone(server._v)
                self.assertEqual(server._update_step, 0)
                for key in model:
                    self.assertTrue(torch.equal(model[key], kept["model"][key]))
                    self.assertTrue(torch.equal(fold[key], kept["fold"][key]))
                    for name in carried:
                        self.assertTrue(torch.equal(carried[name][key], kept["carried"][name][key]))

    def test_the_second_update_starts_from_what_the_first_returned(self) -> None:
        generator = torch.Generator().manual_seed(9)
        server = _server("fedadam")
        model, fold = _state(generator, torch.float64), _state(generator, torch.float64)
        once, carried = server.update_from_fold(model, fold, None)
        again, _ = server.update_from_fold(once, fold, carried)
        fresh, _ = server.update_from_fold(once, fold, None)
        self.assertFalse(all(torch.equal(again[key], fresh[key]) for key in again))

    def test_the_server_adopts_a_round_computed_elsewhere(self) -> None:
        generator = torch.Generator().manual_seed(2)
        server = _server("fedyogi")
        model, fold = _state(generator, torch.float32), _state(generator, torch.float32)
        new, carried = server.update_from_fold(model, fold, None)
        server.adopt_update(new, carried)
        self.assertIs(server._model_state, new)
        self.assertIs(server._m, carried["m"])
        self.assertIs(server._v, carried["v"])
        self.assertEqual(server._update_step, 1)
        self.assertEqual(server.carried_state(), carried)


class FedAvgAdoptsTheFoldTest(unittest.TestCase):
    def test_the_model_is_the_fold_and_nothing_is_carried(self) -> None:
        server = FedAvgServer(participation_rate=1.0, seed=0)
        generator = torch.Generator().manual_seed(1)
        model, fold = _state(generator, torch.float64), _state(generator, torch.float64)
        new, carried = server.update_from_fold(model, fold, None)
        self.assertIsNone(carried)
        self.assertIsNone(server.carried_state())
        for key in fold:
            self.assertTrue(torch.equal(new[key], fold[key]))
        server.adopt_update(new, carried)
        self.assertIs(server._model_state, new)


if __name__ == "__main__":
    unittest.main()
