"""The four optimizer wrappers are usable where a torch optimizer is, checkably.

SCAFFOLD's control-variate correction and FedProx's proximal term only ever
mutate `.grad`, so both are expressed as an object that wraps a real optimizer
and adjusts gradients inside `.step()`. That keeps every caller on the plain
`task.train_step(model, batch, optimizer)` signature instead of pushing
hook plumbing into the task adapters for two of nine rules.

Typed as `torch.optim.Optimizer`, that shape was five mypy errors stating one
decision five times -- and two of the suppressions written against them sat on
the call line while mypy blames the argument line, so they suppressed nothing
and `warn_unused_ignores = false` kept that invisible. `OptimizerLike` states
the actual requirement instead: `zero_grad` and `step`, which is exactly what
the two `train_step` implementations call.

mypy checks that at the call sites and mypy does not gate CI, so the claim is
pinned here too. The protocol is `runtime_checkable`, which tests for the
presence of the members and not their signatures -- enough to catch a wrapper
losing a method or the protocol growing a member that excludes one.
"""

from __future__ import annotations

import unittest
from typing import Any, get_type_hints

import pytest
import torch
from torch import nn

from fedbrew.clients.local_update_modes import _ClippingOptimizer, _GradientOnlyOptimizer
from fedbrew.clients.torch_fedprox_client import _FedProxCorrectingOptimizer
from fedbrew.clients.torch_scaffold_client import _ScaffoldCorrectingOptimizer
from fedbrew.core.torch_utils import OptimizerLike
from fedbrew.tasks.base import TaskAdapter
from fedbrew.tasks.classification.torch_classification import TorchClassificationTask

pytestmark = pytest.mark.fast

#: What a training step is allowed to ask of the object it is handed. Written
#: here rather than read off the protocol: a test that derives its expectation
#: from the thing under test cannot notice the thing changing.
REQUIRED_MEMBERS = ("zero_grad", "step")


def _model() -> nn.Module:
    return nn.Linear(4, 2)


def _real_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=0.1)


def _wrappers() -> dict[str, Any]:
    model = _model()
    inner = _real_optimizer(model)
    zeros = {name: torch.zeros_like(tensor) for name, tensor in model.state_dict().items()}
    return {
        "_ClippingOptimizer": _ClippingOptimizer(inner, model, 1.0),
        "_GradientOnlyOptimizer": _GradientOnlyOptimizer(model.parameters()),
        "_ScaffoldCorrectingOptimizer": _ScaffoldCorrectingOptimizer(inner, model, zeros, zeros),
        "_FedProxCorrectingOptimizer": _FedProxCorrectingOptimizer(inner, model, zeros, 0.1),
    }


class EveryWrapperSatisfiesTheProtocolTest(unittest.TestCase):
    def test_each_one_is_an_instance(self) -> None:
        for name, wrapper in _wrappers().items():
            with self.subTest(wrapper=name):
                self.assertIsInstance(wrapper, OptimizerLike)

    def test_a_real_torch_optimizer_is_too(self) -> None:
        """The protocol has to keep admitting what it replaced."""

        self.assertIsInstance(_real_optimizer(_model()), OptimizerLike)

    def test_something_that_is_neither_is_refused(self) -> None:
        """Anti-vacuity: a protocol with no members admits everything."""

        self.assertNotIsInstance(object(), OptimizerLike)


class TheProtocolAsksForNoMoreThanAStepNeedsTest(unittest.TestCase):
    def test_it_requires_exactly_the_two_members(self) -> None:
        """`param_groups` is the one that would break a caller that works.

        Three of the four wrappers expose it and `_GradientOnlyOptimizer` does
        not, because nothing in a training step reads it. Adding it here would
        exclude that wrapper while every runtime path kept working, so the
        member list is pinned rather than left to whoever edits the protocol.
        """

        # Read off the class rather than through `typing`'s private
        # `_get_protocol_attrs`, and off both halves: a method appears in
        # `vars`, a bare `name: Any` member only in `__annotations__`, and
        # `param_groups` would most naturally be written as the second.
        public = {name for name in vars(OptimizerLike) if not name.startswith("_")}
        public |= {
            name
            for name in getattr(OptimizerLike, "__annotations__", {})
            if not name.startswith("_")
        }
        self.assertEqual(sorted(public), sorted(REQUIRED_MEMBERS))

    def test_the_contract_is_annotated_with_it(self) -> None:
        """`train_step` is where the widening had to land to be worth anything."""

        annotation = get_type_hints(TaskAdapter.train_step)["optimizer"]
        self.assertIn(OptimizerLike, getattr(annotation, "__args__", ()))


class AmpNeedsParamGroupsNotTheNominalTypeTest(unittest.TestCase):
    """What AMP actually requires of an optimizer-shaped object.

    Not the torch type. `GradScaler.unscale_` reads `param_groups` and tracks
    per-device inf counts against the object itself, and that is all it needs.
    `train_step` tested `isinstance` against the nominal type until it was
    measured with the guard lifted: `_ClippingOptimizer`,
    `_ScaffoldCorrectingOptimizer` and `_FedProxCorrectingOptimizer` each took
    an AMP training step, and `_GradientOnlyOptimizer` raised `AttributeError`
    on `param_groups` -- the one wrapper with no inner optimizer to expose.

    So the refusal survives only where the requirement is genuinely unmet, and
    this pins both sides of that.
    """

    def _task_with_a_scaler(self) -> TorchClassificationTask:
        task = TorchClassificationTask(
            model_config={"name": "mlp", "input_dim": 4, "num_classes": 2},
            batch_size=2,
        )
        # use_amp resolves to False off CUDA, so the branch is unreachable on a
        # CPU host; a non-None scaler is what selects it, and the refusal fires
        # before the scaler is touched.
        task._scaler = object()
        return task

    def test_a_wrapper_with_no_param_groups_is_refused_by_name(self) -> None:
        task = self._task_with_a_scaler()
        model = _model()
        batch = (torch.zeros(2, 4), torch.zeros(2, dtype=torch.long))
        with self.assertRaises(TypeError) as raised:
            task.train_step(model, batch, _GradientOnlyOptimizer(model.parameters()))
        self.assertIn("_GradientOnlyOptimizer", str(raised.exception))
        self.assertIn("param_groups", str(raised.exception))

    def test_a_wrapper_that_exposes_them_gets_past_the_guard(self) -> None:
        """It used to be refused here. The guard was stricter than GradScaler.

        The scaler is a bare object, so the autocast call after the guard
        fails on something else entirely -- which is the point: the failure is
        no longer this guard's.
        """

        task = self._task_with_a_scaler()
        model = _model()
        batch = (torch.zeros(2, 4), torch.zeros(2, dtype=torch.long))
        wrapper = _ClippingOptimizer(_real_optimizer(model), model, 1.0)
        with self.assertRaises(Exception) as raised:  # noqa: B017
            task.train_step(model, batch, wrapper)
        self.assertNotIn("param_groups", str(raised.exception))
        self.assertNotIsInstance(raised.exception, TypeError)


if __name__ == "__main__":
    unittest.main()
